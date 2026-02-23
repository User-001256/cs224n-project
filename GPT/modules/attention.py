import torch

from einops import rearrange
from torch import nn

# This file has been modified to make things a bit easier to follow.
# Shape Abbreviations:
# b: Batch Size
# h: Number Heads
# t: Temporal Dimension / Sequence Length
# D: Model Dimension (total, d_model)
# d: Model Dimension (per head, d_head)

MASK_VALUE = -10000.0 # Large negative number for masking (Alternatively, use -torch.inf)

class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # Initialize the linear transformation layers for key, value, query.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer. Although it is a bit unusual, we empirically
    # observe that it yields better performance.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    self.softmax = nn.Softmax(dim = -1)
    
    # Also needs a causal mask: (Homework 3 Code)
    max_pos = config.max_position_embeddings
    self.register_buffer(
        "causal_mask",
        torch.tril(torch.ones((max_pos, max_pos), dtype=torch.bool)).view(
                1, 1, max_pos, max_pos
            ),
        persistent=False,
    )
    
  def split_heads(self, x):
    """
    Separate a projection tensor into heads

    Args:
        x: Input tensor. Assumped to be shape (b, t, D)
        
    Returns:
        Splitted tensor of shape (b, h, t, d)
    """
    # Produce multiple heads by splitting in the model dimension
    x = rearrange(x, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # Transpose the resulting tensor to the correct dimension specification
    return rearrange(x, 'b t h d -> b h t d')
  
  def merge_heads(self, x):
    """
    Merge a multi-headed attention tensor into an output tensor

    Args:
        x: Input tensor. Assumed to be shape (b, h, t, d)
    
    Returns:
        Concatenated tensor with shape (b, t, D)
    """
    # Reverse the operations of split heads:
    x = rearrange(x, 'b h t d -> b t h d') # Transpose
    return rearrange(x, 'b t h d -> b t (h d)') # Return shape (b, t, D)
    
    
  def forward(self, x, attention_mask):
    """
    Masked Multi-headed Self-attention Module

    Args:
        x: Input to the self-attention module. Shape (b, t, D)
        attention_mask: Mask for padding tokens. Shape (b, 1, 1, t)
        
    Returns:
        Output for the multi-headed self-attention. Same shape as x
    """
    # First, generate the key, query, value tensor, then re-shape
    K, Q, V = self.key(x), self.query(x), self.value(x) # Shape: (b, t, D)
    # Apply reshape to the key, query and value tensors.
    K, Q, V = self.split_heads(K), self.split_heads(Q), self.split_heads(V)
        # Shape: (b, h, t, d)
    # Apply scaled dot-product. Result shape: (b, h, t, t)
    attn_weights = torch.matmul(Q, rearrange(K, "b h t d -> b h d t")) / (self.attention_head_size ** 0.5)
    # Apply causal mask (Homework 3 code)
    mask = self.causal_mask[..., :x.shape[1], :x.shape[1]].to(attn_weights.device, dtype = attn_weights.dtype)
    attn_weights = torch.masked_fill(attn_weights, mask == 0, MASK_VALUE)
    
    #! attention_mask is used to mask out padding tokens. This is not the same
    #! as our causal mask, which is used to prevent the model from seeing the future.
    attn_weights = attn_weights + attention_mask
    
    # Apply softmax (row wise), then apply dropout on normalised attn scores
    attn_probs = self.dropout(self.softmax(attn_weights))    
    # Apply value projection and merge heads. Shape: (b, t, D)
    out = self.merge_heads(torch.matmul(attn_probs, V))
    
    #! Output projection (W_o) is applied in the parent module (GPT2Layer)
    
    return out
    
    