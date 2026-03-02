from torch import nn

import torch.nn.functional as F

from modules.attention import CausalSelfAttention
from config import GPT2Config

## Note:
## config.hidden_size should be D (d_model) according to our convention.

class GPT2Layer(nn.Module):
  def __init__(self, config:GPT2Config):
    super().__init__()
    # Save config
    self.config = config

    # Multi-head attention.
    self.self_attention = CausalSelfAttention(config)
    # Auxillary Modules for multi-headed attention
      # Output projection
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
      # Layer Norm (Applied before attention)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
      # Drop-out (Applied after output projection, before residual add)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    
    # Feed-forward network layers
    self.ffn_in_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.ffn_act_fn = F.gelu
    self.ffn_out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    # Auxillary Modules for FFN
      # Layer Norm (Applied before FFN)
    self.ffn_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
      # Drop-out (Applied after FFN, before residual add)
    self.ffn_dropout = nn.Dropout(config.hidden_dropout_prob)
    
  def _attn(self, x, attn_mask):
    """
    Subroutine of attn_forward without residual add.
    """
    # Apply Layernorm before going into the attention layer
    attn_in = self.attention_layer_norm(x)
    # Apply scaled dot-product attention with output projection
    attn_out = self.attention_dense(self.self_attention(attn_in, attn_mask))
    # Apply drop-out
    return self.attention_dropout(attn_out)
  
  def _ffn(self, x):
    """
    Subroutine of ffn_forward without residual add.
    """
    # Apply layer norm for FFN input
    ffn_in = self.ffn_layer_norm(x)
    # Obtain FFN output
    ffn_out = self.ffn_out_dense(self.ffn_act_fn(self.ffn_in_dense(ffn_in)))
    # Apply dropout
    return self.ffn_dropout(ffn_out)
    
    
  def attn_forward(self, x, attn_mask):
    """
    Attention component of the GPT-2 block

    Args:
        x: Input to the attention module. Shape: (b, t, D)
        attn_mask: Attention mask. See CausalSelfAttention documentation for details.

    Returns:
        Output. Same shape as input.
    """
    return x + self._attn(x, attn_mask)
  
  def ffn_forward(self, x):
    """
    Feed-forward component of the GPT-2 block

    Args:
        x: Input to the FFN module. Shape: (b, t, D)
        
    Returns:
        Output. Same shape as the input.
    """
    return x + self._ffn(x)

  def forward(self, x, attention_mask):
    """
    Forward method of attention block. x -> self attention -> ffn -> out.
    """
    return self.ffn_forward(self.attn_forward(x, attention_mask))
