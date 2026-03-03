"""
This script contains all adapter classes for our GPT2 model
"""
import torch
from torch import nn
from peft import LoraConfig, get_peft_model
from config import GPT2Config
from models.gpt2 import GPT2Model
from modules.attention import CausalSelfAttention
from modules.gpt2_layer import GPT2Layer

###########################
#### Bottleneck Adapter ###
###########################

class BottleneckAdapter(nn.Module):
    """
    An sequential bottleneck adapter class proposed by Houlsby et. al.
    Baasically a small FFN with a residual add.
    """
    def __init__(self, input_dim, intermediate_dim):
        super().__init__()
        self.bottleneck_down = nn.Linear(input_dim, intermediate_dim)
        self.act = nn.GELU()
        self.bottleneck_up = nn.Linear(intermediate_dim, input_dim)
        
    def _init_weights(self, scale, identity = True):
        """
        Initialize the adapter projection layers

        Args:
            scale: Std Dev of random normal weights
            identity: If True, the adapter will be initialized to the identity function. Defaults to True.
        """
        self.bottleneck_down.weight.data.normal_(mean = 0.0, std = scale)
        # If identity == True, the up projection weights will not be initialized
        if identity:
            self.bottleneck_up.weight.data.zero_()
        else:
            self.bottleneck_up.weight.data.normal_(mean = 0.0, std = scale)
        # Initialize both bias vectors to 0
        self.bottleneck_down.bias.data.zero_()
        self.bottleneck_up.bias.data.zero_()
        
    def forward(self, x):
        return x + self.bottleneck_up(self.act(self.bottleneck_down(x)))

# Adapter Transformer Block:
class AdapterGPT2Layer(nn.Module):
    def __init__(self, base_layer: GPT2Layer, adapter_dim: int, initializer_range = 0.02, identity_init = True):
        super().__init__()
        # Base gpt2layer
        self.base_layer = base_layer
        # Initialize the Adapter Modules
        self.attention_adapter = BottleneckAdapter(base_layer.model_dim, adapter_dim)
        self.ffn_adapter = BottleneckAdapter(base_layer.model_dim, adapter_dim)
        # Initialize weights
        self._initialize_adapter_weights(initializer_range, identity_init)
        
    def _initialize_adapter_weights(self, initializer_range, identity):
        self.attention_adapter._init_weights(initializer_range, identity)
        self.ffn_adapter._init_weights(initializer_range, identity)

    def forward(self, x, attention_mask):
        """
        Overrides the standard GPT2 Layer forward function
        """
        # Adapter Layer is applied before residual add.
        attn_out = x + self.attention_adapter(self.base_layer._attn(x, attention_mask))
        return attn_out + self.ffn_adapter(self.base_layer._ffn(attn_out))
    
def apply_adapter(model: GPT2Model, adapter_dim):
    """
    Apply adapter PEFT to our GPT2 Model

    Args:
        model: GPT2 Model
        adapter_dim: Intermediate dimensions of bottleneck adapters
    """
    new_blocks = []
    for b in model.blocks:
        new_blocks.append(AdapterGPT2Layer(b, adapter_dim, initializer_range=model.config.initializer_range))
    model.blocks = nn.ModuleList(new_blocks)
    # Freeze the non-adapter parameters
    for n, p in model.named_parameters():
        p.requires_grad = ("bottleneck" in n)
    
####################
#### IA3 Adapter ###
####################

class IA3SelfAttention(nn.Module):
    def __init__(self, base_attention: CausalSelfAttention):
        super().__init__()
        self.attn = base_attention
        # Additionally add the learnable vectors:
        L_k = torch.ones(base_attention.all_head_size)
        L_v = torch.ones(base_attention.all_head_size)
        self.task_vec_k = nn.Parameter(L_k) # Shape (D,)
        self.task_vec_v = nn.Parameter(L_v) # Shape (D,)
    
    def forward(self, x, attention_mask):
        # Apply Projection First, Shape: (b, t, D)
        K, Q, V = self.attn._forward_first(x)
        # Apply element wise multiplication
        K, V = K * self.task_vec_k, V * self.task_vec_v
        # Complete the rest of the attention routime:
        return self.attn._forward_rest(K, Q, V, attention_mask)
        
# GPT2 Layer with IA3 Adapter
class IA3GPT2Layer(nn.Module):
    def __init__(self, base_layer:GPT2Layer):
        super().__init__()
        # Compose on the original base GPT2Layer
        self.base_layer = base_layer
        # New IA3 Attention Module
        self.self_attention = IA3SelfAttention(base_layer.self_attention)
        # Learable vector for FFN activation
        self.task_vec_ff = nn.Parameter(torch.ones(base_layer.ffn_act_dim))
    
    # Subroutines of forward:
    def _attn(self, x, attn_mask):
        attn_in = self.base_layer.attention_layer_norm(x)
        # Attention has been swapped out wiht IA3SelfAttention
        attn_out = self.base_layer.attention_dense(self.self_attention(attn_in, attn_mask))    
        return self.base_layer.attention_dropout(attn_out)
        
    def _ffn(self, x):
        ffn_in = self.base_layer.ffn_layer_norm(x)
        ffn_act = self.base_layer.ffn_act_fn(self.base_layer.ffn_in_dense(ffn_in))
        # Apply multiplication to activations
        ffn_out = self.base_layer.ffn_out_dense(self.task_vec_ff * ffn_act)
        return self.base_layer.ffn_dropout(ffn_out)

    def forward(self, x, attention_mask):
        """
        The same forward function of GPT2Layer works here.
        """
        attn_out = x + self._attn(x, attention_mask)
        return attn_out + self._ffn(attn_out)

def apply_IA3(model: GPT2Model):
    """
    Apply IA3 Adapters to GPT2 model

    Args:
        model: GPT2 Model
    """
    new_blocks = []
    for b in model.blocks:
        new_blocks.append(IA3GPT2Layer(b))
    model.blocks = nn.ModuleList(new_blocks)
    # Freeze the non-adapter parameters
    for n, p in model.named_parameters():
        p.requires_grad = ("task_vec" in n)

#####################
#### LoRA Adapter ###
#####################

def apply_lora(model:GPT2Model, target_modules:list[str], 
               rank = 8, dropout = 0.05, alpha = None, **lora_config_kwargs):
    """
    Apply Lora to GPT2 Model. Based on HuggingFace's PEFT library.

    Args:
        model: GPT2 Model
        target_modules: Attention Modules to inject LoRA to. Select from ['query', 'key', 'value', 'attention_dense']
        rank: LoRA Rank. Defaults to 8.
        dropout: Dropout Probability applied at LoRA input. Defaults to 0.05.
        alpha: LoRA scaling factor. Defaults to None.
    
    Returns:
    GPT-2 Model with LoRA Modules applied.    
    """
    if alpha is None: alpha = 2 * rank
    lora_config = LoraConfig(
        r = rank,
        lora_alpha = alpha,
        target_modules = target_modules,
        lora_dropout = dropout,
        task_type = "CAUSAL_LM",
        **lora_config_kwargs
    )
    return get_peft_model(model, lora_config) #type: ignore
    