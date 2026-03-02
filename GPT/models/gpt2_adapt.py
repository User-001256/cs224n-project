"""
This script contains all adapter classes for our GPT2 model
"""
import torch
from collections import OrderedDict
from torch import nn
from peft import LoraConfig, get_peft_model
from config import GPT2Config
from models.gpt2 import GPT2Model
from modules.gpt2_layer import GPT2Layer

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
class GPT2LayerWithAdapter(GPT2Layer):
    def __init__(self, config: GPT2Config, adapter_dim, identity_init = True):
        # Initialize the basic GPT2 Block
        super().__init__(config)
        # Initialize the Adapter Projections (Named so that enabling gradient is easier)
        self.attention_adapter = BottleneckAdapter(config.hidden_size, adapter_dim)
        self.ffn_adapter = BottleneckAdapter(config.hidden_size, adapter_dim)
        # Initialize weights
        self._initialize_adapter_weights(identity_init)
        
    def _initialize_adapter_weights(self, identity):
        self.attention_adapter._init_weights(self.config.initializer_range, identity)
        self.ffn_adapter._init_weights(self.config.initializer_range, identity)
        
    def forward(self, x, attention_mask):
        """
        Overrides the standard GPT2Layer Forward Function.
        """
        # Attention module. Apply adapter before residual add.
        attn_out = x + self.attention_adapter(self._attn(x, attention_mask))
        # print(torch.max(self.attention_adapter(self._attn(x, attention_mask)) - self._attn(x, attention_mask)))
        # FFN module. Apply adapter before residual add.
        # print(torch.max(self.ffn_adapter(self._ffn(attn_out)) - self._ffn(attn_out)))
        return attn_out + self.ffn_adapter(self._ffn(attn_out))

    @classmethod
    def from_gpt2layer(cls, layer:GPT2Layer, adapter_dim, init_identity = True):
        out = GPT2LayerWithAdapter(layer.config, adapter_dim)
        # Copy non adapter modules and weights
        for name, module in layer.named_children():
            setattr(out, name, module)
        # Initialize adapter weights
        out._initialize_adapter_weights(init_identity)
        return out
    
def apply_adapter(model: GPT2Model, adapter_dim):
    """
    Apply adapter PEFT to our GPT2 Model

    Args:
        model: GPT2 Model
        adapter_dim: Intermediate dimensions of bottleneck adapters
    """
    new_blocks = []
    for b in model.blocks:
        new_blocks.append(GPT2LayerWithAdapter.from_gpt2layer(b, adapter_dim))
    model.blocks = nn.ModuleList(new_blocks)

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
    