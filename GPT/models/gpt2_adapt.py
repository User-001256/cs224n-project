"""
This script contains all adapter classes for our GPT2 model
"""
import torch
from torch import nn
from models.gpt2 import GPT2Model
from modules.attention import CausalSelfAttention, MASK_VALUE
from modules.gpt2_layer import GPT2Layer
from typing import List, Union, Dict
from einops import rearrange

# Parameter Keywords
LORA_PARAM_KWD = "lora_"
ADAPTER_PARAM_KWD = "bottleneck_"
IA3_PARAM_KWD = "task_vec_"
PREFIX_PARAM_KWD = "prefix_"

##################################
#### Gated Network for UniPELT ###
##################################
class GatedNetwork(nn.Module):
    """
    Gated Network Logic proposed by Mao et.al.
    https://arxiv.org/abs/2110.07577
    """
    def __init__(self, input_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, 1)
        self.sigma = nn.Sigmoid()
        # Initialize gated network weights
        nn.init.normal_(self.dense.weight, std = 1e-4)
        nn.init.zeros_(self.dense.bias)
        
    def forward(self, x):
        return self.sigma(self.dense(x))
    
######################
#### Prefix Tuning ###
######################
class PrefixMLP(nn.Module):
    def __init__(self, bottleneck_dim, intermediate_dim, model_dim):
        super().__init__()
        self.linear0 = nn.Linear(bottleneck_dim, intermediate_dim)
        self.act = nn.Tanh()
        self.linear1 = nn.Linear(intermediate_dim, model_dim)
        self._init_weights()
        
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.linear0.weight.data)
        nn.init.zeros_(self.linear0.bias.data)
        nn.init.zeros_(self.linear1.weight.data)
        nn.init.zeros_(self.linear1.bias.data)
    
    def forward(self, x):
        return self.linear1(self.act(self.linear0(x)))

class PrefixSelfAttention(nn.Module):
    """
    Prefix Tuning Modified Self Attention Module Proposed by Xiang Li and Percy Liang.
    https://arxiv.org/abs/2101.00190
    """
    def __init__(self, base_attention:CausalSelfAttention, num_prefixes = 10, 
                 bottleneck_dim = 64, intermediate_dim = 512):
        super().__init__()
        # Inherit and instantiate all attributes from the base attention module
        self.num_attention_heads = base_attention.num_attention_heads
        self.attention_head_size = base_attention.attention_head_size
        self.all_head_size = base_attention.all_head_size
        self.query = base_attention.query
        self.key = base_attention.key
        self.value = base_attention.value
        self.dropout = base_attention.dropout
        self.softmax = base_attention.softmax
        self.causal_mask = base_attention.causal_mask
        # Also inheritate the same methods
        self._split_heads = base_attention._split_heads
        self._merge_heads = base_attention._merge_heads
        self._forward_first = base_attention._forward_first
        # Need to override the forward rest method
        
        # We do not mask the prefix tokens.
        self.register_buffer(
            "pre_mask",
            torch.ones((self.causal_mask.shape[-1], num_prefixes), dtype = torch.bool).view(1, 1, self.causal_mask.shape[-1], num_prefixes),
            persistent=False
        )
        # Concatenate both masks together.
        self.all_mask = torch.cat([self.pre_mask, self.causal_mask], dim = -1)
        # We also do not want to treat the prefix tokens as dummy tokens.
        self.register_buffer(
            "pre_attention_mask",
            torch.zeros(num_prefixes),
            persistent=False
        )

        # Initialize the learnable bottleneck vectors:
        self.prefix_vec_k = nn.Parameter(torch.randn((num_prefixes, bottleneck_dim)))
        self.prefix_vec_v = nn.Parameter(torch.randn((num_prefixes, bottleneck_dim)))
        # Initialize the learnable MLP Projection Networks
        self.prefix_projection_k = PrefixMLP(bottleneck_dim, intermediate_dim, self.all_head_size)
        self.prefix_projection_v = PrefixMLP(bottleneck_dim, intermediate_dim, self.all_head_size)
    
    def _get_prefixes(self):
        prefix_k = self.prefix_projection_k(self.prefix_vec_k)
        prefix_v = self.prefix_projection_v(self.prefix_vec_v)
        return prefix_k, prefix_v
    
    def _forward_rest(self, K, Q, V, attention_mask):
        """
        Overrides the default _forward_rest for Causal Self Attention.
        Input K, Q, V are assumed to have shape (b, t, D)
        """
        batch = K.shape[0] # Used for manual broadcasting...
        # Apply reshape to the key, query and value tensors.
        K, Q, V = self._split_heads(K), self._split_heads(Q), self._split_heads(V)
            # Shape: (b, h, t, d)
        # Obtain the Prefixes. Add a batch dimension
        prefix_k, prefix_v = self._get_prefixes() # Shape (p, D)
        prefix_k = prefix_k.expand(batch, -1, -1) # Manual broadcasting
        prefix_v = prefix_v.expand(batch, -1, -1)
        # Use split_heads to get the prefixes to the right shape (b, h, p, d)
        prefix_k, prefix_v = self._split_heads(prefix_k), self._split_heads(prefix_v)
        
        # Concatenate the prefix onto the K and V tensors
        Kp = torch.cat([prefix_k, K], dim = -2) # Shape (b, h, p + t, d)
        Vp = torch.cat([prefix_v, V], dim = -2) # Shape (b, h, p + t, d)
        
        # Now carry on with our attention routine:
        # Apply scaled dot-product. Result shape: (b, h, t, t + p)
        attn_weights = torch.matmul(Q, rearrange(Kp, "b h t d -> b h d t")) / (self.attention_head_size ** 0.5)
        # Only mask the correct things
        mask = self.all_mask[..., :attn_weights.shape[-2], :attn_weights.shape[-1]] # Shape (..., t, t+p)
        attn_weights = torch.masked_fill(attn_weights, mask == 0, MASK_VALUE)
        # Add attention mask. More manual broadcasting...
        prefix_mask = self.pre_attention_mask.expand((batch, 1, 1, -1))
        attn_weights = attn_weights + torch.cat([prefix_mask, attention_mask], dim = -1)
        
        # Apply softmax (row wise), then apply dropout on normalised attn scores
        attn_probs = self.dropout(self.softmax(attn_weights))    
        # Apply value projection and merge heads. Shape: (b, t, D)
        res = self._merge_heads(torch.matmul(attn_probs, Vp))
        return res 
    
    def forward(self, x, attention_mask):
        K, Q, V = self._forward_first(x) # Shape: (b, t, D)
        return self._forward_rest(K, Q, V, attention_mask)

####################
#### IA3 Adapter ###
####################
class IA3SelfAttention(nn.Module):
    """
    IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations) 
    Adapter self-attention module. Proposed by Liu et.al:
    https://arxiv.org/pdf/2205.05638
    
    Injects 2 vectors of dimension d_model, which is elementwise multiplied
    to the key and value vectors.
    """
    def __init__(self, base_attention: Union[CausalSelfAttention, PrefixSelfAttention]):
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
        # Apply element wise multiplication.
        K, V = K * self.task_vec_k, V * self.task_vec_v
        # Complete the rest of the attention routime:
        return self.attn._forward_rest(K, Q, V, attention_mask)
        
# GPT2 Layer with IA3 Adapter
class IA3GPT2Layer(nn.Module):
    """
    IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations) Adapter GPT Block.

    Besides the learnable vectors injected at self-attention, an additional 
    learnable vector of dimension 4 * d_model is injected at the intermediate
    activations of the FFN module. Elementwise mult is applied after activation.
    """
    def __init__(self, base_layer:GPT2Layer, bias = False):
        super().__init__()
        # Compose on the original base GPT2Layer
        self.base_layer = base_layer
        self.model_dim = base_layer.model_dim
        # New IA3 Attention Module
        self.self_attention = IA3SelfAttention(base_layer.self_attention, bias)
        # Learable vector for FFN activation
        self.task_vec_ff = nn.Parameter(torch.ones(base_layer.ffn_act_dim))
        # Optional bias for the ffn task vector
        self.task_vec_bias_ff = nn.Parameter(torch.zeros(base_layer.ffn_act_dim)) if bias else None
    
    # Subroutines of forward: Compatible with bottleneck adapters
    def _attn(self, x, attn_mask):
        attn_in = self.base_layer.attention_layer_norm(x)
        # Attention has been swapped out wiht IA3SelfAttention
        attn_out = self.base_layer.attention_dense(self.self_attention(attn_in, attn_mask))    
        return self.base_layer.attention_dropout(attn_out)
        
    def _ffn(self, x):
        ffn_in = self.base_layer.ffn_layer_norm(x)
        ffn_act = self.base_layer.ffn_act_fn(self.base_layer.ffn_in_dense(ffn_in))
        # Apply multiplication to activations
        if self.task_vec_bias_ff is not None:
            ffn_out = self.base_layer.ffn_out_dense(self.task_vec_ff * ffn_act + self.task_vec_bias_ff)
        else:
            ffn_out = self.base_layer.ffn_out_dense(self.task_vec_ff * ffn_act)
        return self.base_layer.ffn_dropout(ffn_out)

    def forward(self, x, attention_mask):
        """
        The same forward function of GPT2Layer works here.
        """
        attn_out = x + self._attn(x, attention_mask)
        return attn_out + self._ffn(attn_out)
    
#####################
#### LoRA Adapter ###
#####################
class LoRAProjWrapper(nn.Module):
    def __init__(self, in_features, out_features, zero_init = False):
        """Just an Engineering Hack for ensuring HuggingFace weight loading compatibility."""
        super().__init__()
        self.default = nn.Linear(in_features, out_features, bias = False)
        if zero_init:
            nn.init.zeros_(self.default.weight)
        else:
            nn.init.kaiming_uniform_(self.default.weight)
                
    def forward(self, x):
        return self.default(x)
    
class LoRALinear(nn.Module):
    """
    A LoRA Wrapper for nn.Linear.
    """
    def __init__(self, layer: nn.Linear, rank:int = 8, alpha:int = 16, dropout_prob:float = 0.05, gated = False):
        super().__init__()
        self.base = layer

        self.rank = rank
        self.alpha = alpha
        
        # Construct LoRA Layer Projection Matrices (Compatible with HuggingFace Weight Loading)
        self.lora_A = LoRAProjWrapper(layer.in_features, self.rank, False)
        self.lora_B = LoRAProjWrapper(self.rank, layer.out_features, True)
        self.dropout = nn.Dropout(dropout_prob)
        
        # Construct an optional gated logic according to UniPELT
        self.lora_gate = GatedNetwork(layer.in_features) if gated else None
            
    def forward(self, x):
        # Base Layer Output
        base_out = self.base(x)
        # Compute the LoRA additive update (Scaled by alpha / r)
        lora_add = self.lora_B(self.lora_A(self.dropout(x))) * (self.alpha / self.rank)
        # Optional gate for Gated LoRA:
        if self.lora_gate is not None:
            return base_out + lora_add * self.lora_gate(x)
        else:
            return base_out + lora_add
        
def apply_lora_to_attention(attn:Union[CausalSelfAttention, PrefixSelfAttention], 
                            target_modules: List[str], rank = 8, dropout = 0.05, alpha = 16, gated = False):
    for name, module in attn.named_children():
        if name in target_modules and isinstance(module, nn.Linear):
            # Only apply LoRA to linear modules
            setattr(attn, name, LoRALinear(module, rank, alpha, dropout, gated))  
               
###########################
#### Bottleneck Adapter ###
###########################
class BottleneckAdapter(nn.Module):
    """
    An sequential bottleneck adapter class proposed by Houlsby et. al.
    Baasically a small FFN with residual add.
    
    Additionally added a gated network for UniPELT.
    """
    def __init__(self, input_dim, intermediate_dim, init_scale = 0.02, gated = False):
        super().__init__()
        self.bottleneck_down = nn.Linear(input_dim, intermediate_dim)
        self.act = nn.GELU()
        self.bottleneck_up = nn.Linear(intermediate_dim, input_dim)
        self.bottleneck_down.weight.data.normal_(mean = 0.0, std = init_scale)
        self.bottleneck_up.weight.data.zero_()
        # Initialize both bias vectors to 0
        self.bottleneck_down.bias.data.zero_()
        self.bottleneck_up.bias.data.zero_()
        
        # Initialize the optional gated logic
        self.bottleneck_gate = GatedNetwork(input_dim) if gated else None
            
    def forward(self, x):
        if self.bottleneck_gate is not None:
            return x + self.bottleneck_gate(x) *  self.bottleneck_up(self.act(self.bottleneck_down(x)))
        else:
            return x +  self.bottleneck_up(self.act(self.bottleneck_down(x)))

# Adapter Transformer Block:
class AdapterGPT2Layer(nn.Module):
    """
    GPT-2 block with injected bottleneck adapters. Adapters will be placed 
    after the multiheaded attention and FFN but before their residual adds.
    """
    def __init__(self, base_layer: Union[GPT2Layer, IA3GPT2Layer], adapter_dim: int, 
                 target_modules = ["attn", "ffn"],
                 initializer_range = 0.02, gated = False):
        super().__init__()
        # Base gpt2layer
        self.base_layer = base_layer
        # Query model dimension
        d_model = base_layer.model_dim
        
        # Initialize the Adapter Modules
        self.attention_adapter = BottleneckAdapter(d_model, adapter_dim, initializer_range, gated) if "attn" in target_modules else nn.Identity()
        self.ffn_adapter = BottleneckAdapter(d_model, adapter_dim, initializer_range, gated) if "ffn" in target_modules else nn.Identity()

    def forward(self, x, attention_mask):
        """
        Overrides the standard GPT2 Layer forward function
        """
        # Adapter Layer is applied before residual add.
        attn_out = x + self.attention_adapter(self.base_layer._attn(x, attention_mask))
        return attn_out + self.ffn_adapter(self.base_layer._ffn(attn_out))
            
##############################################
#### Application Function for PEFT Modules ###
##############################################
def apply_IA3(model: GPT2Model, **IA3_kwargs):
    """
    Apply IA3 adapter to all GPT-2 blocks in our GPT2 Model.
    
    No Tunable Parameters
    """
    new_blocks = []
    for b in model.blocks:
        new_blocks.append(IA3GPT2Layer(b, **IA3_kwargs))
    model.blocks = nn.ModuleList(new_blocks)
    # Freeze the non-adapter parameters
    for n, p in model.named_parameters():
        p.requires_grad = ("task_vec" in n)  
        
def apply_prefix_tuning(model: GPT2Model, **PFT_kwargs):
    """
    Apply adapters to all GPT2 Blocks within our GPT2 model.
    Prefix Tuning will be compatible with all other PEFT methods since it only changes the attention block.
    The attention block follows the same API as the GPT2's vanilla CausalSelfAttention.

    Args:
        model: GPT2 Model
        num_prefixes: Number of learnable prefix tokens (Defaults to 10)
        bottleneck_dim: The base dimension of learnable feature vector. (Defaults to 64)
    """
    for b in model.blocks:
        attn = b.self_attention
        b.self_attention = PrefixSelfAttention(attn, **PFT_kwargs)
        
def apply_lora(model:GPT2Model, target_modules: List[str] = ["query", "value"],
               rank = 8, dropout = 0.05, alpha = 16, gated = False):
    """
    Apply LoRA to GPT-2 Model.
    
    Args:
        model: GPT2Model
        target_modules: Choices ["key", "query", "value", "attention_dense"]
        rank: Defaults to 8
        dropout: Dropout probability applied before LoRA projection. Defaults to 0.05.
        alpha: LoRA scaling factor. Defaults to 16.
        gated: Adds the gating logic in UniPELT. Defaults to False.
    """
    for b in model.blocks:
        apply_lora_to_attention(b.self_attention, target_modules, rank, dropout, alpha, gated)
        
def apply_adapter(model: GPT2Model, **Adapter_kwargs):
    """
    Apply adapters to all GPT2 Blocks within our GPT2 model.

    Args:
        model: GPT2 Model
        adapter_dim: Intermediate dimensions of bottleneck adapters
        target_modules: Inject adapter in which modules? Options are "attn" and "ffn".
        gated: Add the gated PEFT logic in UniPELT.
    """
    new_blocks = []
    for b in model.blocks:
        new_blocks.append(AdapterGPT2Layer(b, **Adapter_kwargs ,initializer_range=model.config.initializer_range))
    model.blocks = nn.ModuleList(new_blocks)
    # Freeze the non-adapter parameters
    for n, p in model.named_parameters():
        p.requires_grad = ("bottleneck" in n)
    
#################################
#### Master Function for PEFT ###
#################################
def apply_gpt_peft(model: GPT2Model, peft_config:Dict):
    """
    Apply all parameter efficient adaptation modules for our GPT2 model.

    Args:
        model: GPT2 Model
        peft_config: A dictionary.
            keys: "lora", "ia3", "adapter", or "prefix"
            values: A kwargs dictionary with peft hyperparameters.
    
    Returns:
        This function modifies the GPT2 model in-place.
    """
    # First Apply Prefix Tuning
    if "prefix" in peft_config:
        apply_prefix_tuning(model, **peft_config["prefix"])
    # Apply IA3
    if 'ia3' in peft_config:
        apply_IA3(model, **peft_config["ia3"])
    # Apply LoRA
    if 'lora' in peft_config:
        apply_lora(model, **peft_config["lora"])
    # Finally Apply Adapter
    if 'adapter' in peft_config:
        apply_adapter(model, **peft_config["adapter"])
    
    # Set base network parameters to frozen:
    for n, p in model.named_parameters():
        p.requires_grad = (IA3_PARAM_KWD in n) or (LORA_PARAM_KWD in n) or (ADAPTER_PARAM_KWD in n) or (PREFIX_PARAM_KWD in n)