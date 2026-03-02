"""
This file has been re-written for better pre-trained weights integration.
"""

import torch
from torch import nn
from transformers import GPT2Model as HFGPT2Model
from transformers import GPT2Config as HFGPT2Config

from config import GPT2Config
from models.base_gpt import GPTPreTrainedModel
from modules.gpt2_layer import GPT2Layer
from utils import get_extended_attention_mask

class GPT2Model(GPTPreTrainedModel):
  """
  The GPT model returns the final embeddings for each token in a sentence.

  The model consists of:
  1. Embedding layers (used in self.embed).
  2. A stack of n GPT layers (used in self.encode).
  3. A linear transformation layer for the [CLS] token (used in self.forward, as given).
  """

  def __init__(self, config):
    super().__init__(config)
    self.config = config

    # Embedding layers.
    self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
    self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
    self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)

    # Register position_ids (1, len position emb) to buffer because it is a constant.
    position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
    self.register_buffer('position_ids', position_ids)

    # GPT-2 layers.
    self.gpt_layers = nn.ModuleList([GPT2Layer(config) for _ in range(config.num_hidden_layers)])

    # Final layer norm.
    self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    self.init_weights()

  def embed(self, input_ids):
    """
    Input Embedding for the GPT-2 Model

    Args:
        input_ids: Tensor. Shape: (b, t). Stores token ids.

    """    
    # Generate word embeddings and positional embeddings 
    w_emb = self.word_embedding(input_ids) # (b, t, d_model)
      # Truncate the position ids to the input batch's sequence length.
    p_emb = self.pos_embedding(self.position_ids[..., :input_ids.size(1)]) # (b, t, d_model)
    # Simply add the two embeddings together as final input embeddings
    emb = w_emb + p_emb
    # Apply dropout during training
    return self.embed_dropout(emb)

  def encode(self, hidden_states, attention_mask):
    """
    hidden_states: the output from the embedding layer [batch_size, seq_len, hidden_size]
    attention_mask: [batch_size, seq_len]
    """
    # Get the extended attention mask for self-attention.
    # Returns extended_attention_mask of size [batch_size, 1, 1, seq_len].
    # Distinguishes between non-padding tokens (with a value of 0) and padding tokens
    # (with a value of a large negative number).
    extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, self.dtype)
    # Pass the hidden states through the encoder layers.
    for i, layer_module in enumerate(self.gpt_layers):
      # Feed the encoding from the last bert_layer to the next.
      hidden_states = layer_module(hidden_states, extended_attention_mask)

    return hidden_states

  def forward(self, input_ids, attention_mask):
    """
    input_ids: [batch_size, seq_len], seq_len is the max length of the batch
    attention_mask: same size as input_ids, 1 represents non-padding tokens, 0 represents padding tokens
    """
    # Get the embedding for each input token.
    embedding_output = self.embed(input_ids=input_ids)

    # Feed to a transformer (a stack of GPTLayers).
    sequence_output = self.encode(embedding_output, attention_mask=attention_mask)
    sequence_output = self.final_layer_norm(sequence_output)

    # Get the hidden state of the final token.
    last_non_pad_idx = attention_mask.sum(dim=1) - 1  # Subtract 1 to get last index
    last_token = sequence_output[torch.arange(sequence_output.shape[0]), last_non_pad_idx]

    return {'last_hidden_state': sequence_output, 'last_token': last_token}

  def hidden_state_to_token(self, hidden_state):
    """
    GPT-2 uses weight tying with the input word embeddings. The logits are the dot product between output hidden states
    and the word embedding weights:

      return hidden_state(s) * E^T
    """
    return hidden_state @ self.word_embedding.weight.T

  def transfer_weights(self, target_model:HFGPT2Model):
    """
    Transfer weights from HuggingFace's 'transformer' GPT2 model to our GPT2 Implementation
    """
    target_model = target_model.eval()
    # Load embedding layers
    self.word_embedding.load_state_dict(target_model.wte.state_dict())
    self.pos_embedding.load_state_dict(target_model.wpe.state_dict())
    # Load all transformer block weights:
    target_weights = target_model.state_dict()
    d = self.config.hidden_size #d_model
    for i, layer in enumerate(self.gpt_layers):
      # Remap the Q,K,V weights from a conv1d to 3 linear projections
      layer.self_attention.query.weight.data = target_weights[f'h.{i}.attn.c_attn.weight'][:, :d].T
      layer.self_attention.query.bias.data = target_weights[f'h.{i}.attn.c_attn.bias'][:d]
      layer.self_attention.key.weight.data = target_weights[f'h.{i}.attn.c_attn.weight'][:, d:d*2].T
      layer.self_attention.key.bias.data = target_weights[f'h.{i}.attn.c_attn.bias'][d:d*2]
      layer.self_attention.value.weight.data = target_weights[f'h.{i}.attn.c_attn.weight'][:, d*2:].T
      layer.self_attention.value.bias.data = target_weights[f'h.{i}.attn.c_attn.bias'][d*2:]

      # Remap final dense layer in Multiheaded Attention.
      layer.attention_dense.weight.data = target_weights[f'h.{i}.attn.c_proj.weight'].T
      layer.attention_dense.bias.data = target_weights[f'h.{i}.attn.c_proj.bias']

      # Remap attention layer norm.
      layer.attention_layer_norm.weight.data = target_weights[f'h.{i}.ln_1.weight']
      layer.attention_layer_norm.bias.data = target_weights[f'h.{i}.ln_1.bias']

      # Remap post-attention MLP layers.
      layer.ffn_in_dense.weight.data = target_weights[f'h.{i}.mlp.c_fc.weight'].T
      layer.ffn_in_dense.bias.data = target_weights[f'h.{i}.mlp.c_fc.bias']
      layer.ffn_out_dense.weight.data = target_weights[f'h.{i}.mlp.c_proj.weight'].T
      layer.ffn_out_dense.bias.data = target_weights[f'h.{i}.mlp.c_proj.bias']

      # Remap second layer norm weights.
      layer.ffn_layer_norm.weight.data = target_weights[f'h.{i}.ln_2.weight']
      layer.ffn_layer_norm.bias.data = target_weights[f'h.{i}.ln_2.bias']
    # Remap the final layer norm weights.
    self.final_layer_norm.weight.data = target_weights['ln_f.weight']
    self.final_layer_norm.bias.data = target_weights['ln_f.bias']

  @classmethod
  def from_pretrained(cls, model = "gpt2", **kwargs):
    """
    Pretrained weights loading from HuggingFace's 'transformers' library.

    Args:
        model: Variant of the pre-trained weights: 
          - 'gpt2' - Base Model, 124M Params
          - 'gpt2-medium' - Medium Model, 355M Params
          - 'gpt2-large' - Large Model, 774M Params
          - 'gpt2-xl' - Extra Large Model, 1.5B Params
          - 'distilgpt2' - Distilled Base Model, 82M Params
        kwargs: Kept for compatibility reasons. This implementation automatically infers hyperparameters from HF pretrained weights.

    Returns:
        Our GPT2 implementation loaded with HF pre-trained weights.
    """
    # Loads hugging face GPT2 model and model config
    hf_model = HFGPT2Model.from_pretrained(model).eval()
    hf_config = hf_model.config
    # Log Model Architecture Configurations
    config = GPT2Config(
      vocab_size = hf_config.vocab_size,                  # Vocab Size
      hidden_size = hf_config.n_embd,                     # d_model
      num_hidden_layers = hf_config.n_layer,              # transformer blocks
      num_attention_heads = hf_config.n_head,             # Attn heads
      intermediate_size = 3072 if hf_config.n_inner is None else hf_config.n_inner,
                                                          # FFN Activation Size
      hidden_dropout_prob = hf_config.resid_pdrop,        # Dropout after Attn and FFN
      attention_probs_dropout_prob= hf_config.attn_pdrop, # Dropout within Attn
      max_position_embeddings = hf_config.n_ctx,          # Max context length
      initializer_range = hf_config.initializer_range,    # std for weight initialization
      layer_norm_eps = hf_config.layer_norm_epsilon,      # Layer Norm
    )
    our_model = GPT2Model(config).eval()
    # Transfer weights
    our_model.transfer_weights(hf_model) # type:ignore
    return our_model
