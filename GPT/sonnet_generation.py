'''
Sonnet generation starter code.

Running:
  `python sonnet_generation.py --use_gpu`

trains your SonnetGPT model and writes the required submission files.
'''

from program_config import SONNET_ARGS, DATA_PATHS
from peft_config import SONNET_PEFT
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange

from datasets import SonnetsDataset
from models.gpt2 import GPT2Model
from models.gpt2_adapt import *

from torch_utils import *
from generation_utils import *

class SonnetGPT(nn.Module):
  def __init__(self, gpt_backbone:GPT2Model, gpt_tokenizer:GPT2Tokenizer):
    super().__init__()
    self.gpt = gpt_backbone
    self.tokenizer = gpt_tokenizer
    self.tokenizer.pad_token = self.tokenizer.eos_token

    # Define the generation logit processors
    self.temp_process = None
    self.topk_process = None
    self.topp_process = None

  def forward(self, input_ids, attention_mask):
    """
    This is similar to the forward for ParaphraseGPT, but we now want to produce a logit for each token in our sequence;
    not just the last token! This will allow our model to learn the natural language distribution that composes sonnets,
    not just the distribution over next tokens for the last token!
    """
    outputs = self.gpt(input_ids, attention_mask)
    hidden_states = outputs['last_hidden_state']
    logits = self.gpt.hidden_state_to_token(hidden_states)
    return logits

  @property
  def device(self):
    return next(self.gpt.parameters()).device
    
  def update_logit_processors(self, temperature = None, k = None, p = None):
    """
    Update the generation logit processors

    Args:
        temperature: For temperature scaling.
        k: For top K sampling
        p: For top P sampling
    """
    if temperature is not None:
      self.temp_process = TemperatureScaling(temperature)
    if k is not None:
      self.topk_process = TopKFilter(k)
    if p is not None:
      self.topp_process = TopPFilter(p)
    
  @torch.no_grad
  def logit_process(self, logits:torch.Tensor):
    """
    Process the input logits with the model's logit processors

    Args:
        logits: Tensor. Shape (batch, vocab_size)

    Returns:
        Processed logits
    """
    # We shall apply the logit processors in the canonical order
    # temperature -> top k -> top p
    x = logits
    if self.temp_process is not None:
      x = self.temp_process(x)
    if self.topk_process is not None:
      x = self.topk_process(x)
    if self.topp_process is not None:
      x = self.topp_process(x)
    return x

  @torch.no_grad
  def generate(self, encoding, max_length=128, **kwargs):
    token_ids = encoding.to(self.device)
    attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.device)
    
    for _ in range(max_length):
      # Forward pass to get logits. 
      logits_sequence = self.forward(token_ids, attention_mask)
      # Only obtain the logits of the last predicted token. Shape (batch, vocab)
      logits = logits_sequence[:, -1, :]
      
      # Process the logits using temperature, topk, or topp filters
      logits = self.logit_process(logits)
      # Sample a token
      sampled_token = sample_from_logits(logits)
      
      # Stop if end-of-sequence token is reached
      if sampled_token.item() == self.tokenizer.eos_token_id:
        break

      # Append sampled token
      token_ids = torch.cat([token_ids, sampled_token], dim=1)
      attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=torch.int64).to(self.device)], dim=1
      )

    generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())[3:]
    return token_ids, generated_output

def train(args):
  device = select_device()
  # Create the data and its corresponding datasets and dataloader.
  sonnet_dataset = SonnetsDataset(DATA_PATHS.sonnet_train)
  sonnet_dataloader = DataLoader(sonnet_dataset, shuffle=True, batch_size=args.batch_size,
                                 collate_fn=sonnet_dataset.collate_fn, num_workers=args.num_workers)

  # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
  held_out_sonnet_dataset = SonnetsDataset(DATA_PATHS.sonnet_test)

  # Initialize model_config for checkpoints
  model_config = SimpleNamespace(
    gpt_kind = args.gpt_kind,
    fine_tune_mode = args.fine_tune_mode,
  )
  if args.fine_tune_mode == "peft":
    model_config.peft_config = SONNET_PEFT

  # Obtain the GPT2 Model and Tokenizers
  model = GPT2Model.from_pretrained(args.gpt_kind)
  tokenizer = GPT2Tokenizer.from_pretrained(args.gpt_kind)
  # Apply peft:
  if args.fine_tune_mode == "peft":
    apply_gpt_peft(model, SONNET_PEFT)
  # Apply PEFT if required
  model = SonnetGPT(model, tokenizer)
  model = model.to(device)
  summarize_peft(model)

  # If we want to preview validation set, also need to initialize sampling hyperparameters
  if args.preview_valid:
    model.update_logit_processors(args.temperature, args.top_k, args.top_p)

  # Initialize optimizer:
  optim_config = {
    "lr" : args.lr,
    "weight_decay": args.weight_decay
  }
  trainable_params = [p for p in model.parameters() if p.requires_grad]
  optimizer = set_optimizer(trainable_params, **optim_config)

  # Run for the specified number of epochs.
  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0

    for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch}', leave = False):
      # Get the input and move it to the gpu (I do not recommend training this model on CPU).
      b_ids, b_mask = batch['token_ids'], batch['attention_mask']
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)

      # Compute the loss, gradients, and update the model's parameters. (Training for next token prediction)
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')  # Ignore the last prediction in the sequence.
      labels = b_ids[:, 1:].contiguous().flatten()  # Ignore the first token to compose the labels.
      loss = F.cross_entropy(logits, labels, reduction='mean')
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / num_batches
    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}.")
    if args.preview_valid:
      print('Generating several output sonnets...')
      model.eval()
      
      for batch in held_out_sonnet_dataset:
        encoding = model.tokenizer(batch[1], return_tensors='pt', padding=True, truncation=True).to(device)
        output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)
        print(f'{batch[1]}{output[1]}\n\n')

    # Save model at each epoch
    save_name = f'{epoch}_{args.filepath}'
    save_fine_tune_checkpoint(save_name, model.gpt, model_config)


@torch.no_grad()
def generate_submission_sonnets(args):
  device = select_device()
  # Most recent filepath
  load_name = f'{args.epochs-1}_{args.filepath}'
  gpt = load_fine_tune_checkpoint(load_name, device, model = None)
  # Load tokenizer
  tokenizer = GPT2Tokenizer.from_pretrained(args.gpt_kind)
  # Instantiate Sonnet GPT:
  model = SonnetGPT(gpt, tokenizer)
  model = model.to(device)

  model.eval()
  model.update_logit_processors(args.temperature, args.top_k, args.top_p)

  # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
  held_out_sonnet_dataset = SonnetsDataset(DATA_PATHS.sonnet_test)

  generated_sonnets = []
  for batch in held_out_sonnet_dataset:
    sonnet_id = batch[0]
    encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True).to(device)
    output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)[0][0]
    decoded_output = model.tokenizer.decode(output)
    full_sonnet = f'{decoded_output}\n\n'
    generated_sonnets.append((sonnet_id, full_sonnet))

    print(f'{decoded_output}\n\n')

  with open(args.sonnet_out, "w+") as f:
    f.write(f"--Generated Sonnets-- \n\n")
    for sonnet in generated_sonnets:
      f.write(f"\n{sonnet[0]}\n")
      f.write(sonnet[1])

if __name__ == "__main__":
  args = SONNET_ARGS
  args.filepath = f'{args.epochs}-{args.lr}-sonnet.pt'  # Save path.
  set_seed(args.seed)  # Fix the seed for reproducibility.
  train(args)
  generate_submission_sonnets(args)