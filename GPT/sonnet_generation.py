'''
Sonnet generation starter code.

Running:
  `python sonnet_generation.py --use_gpu`

trains your SonnetGPT model and writes the required submission files.
'''

import argparse
import random
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

from torch_utils import *
from generation_utils import *

class SonnetGPT(nn.Module):
  """Your GPT-2 Model designed for paraphrase detection."""

  def __init__(self, args):
    super().__init__()
    # TODO: Maybe add a PEFT for self.gpt?
    self.gpt = GPT2Model.from_pretrained(args.gpt_kind)
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

    # Define the generation logit processors
    self.temp_process = None
    self.topk_process = None
    self.topp_process = None
    
    for param in self.gpt.parameters():
      param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    """
    This is similar to the forward for ParaphraseGPT, but we now want to produce a logit for each token in our sequence;
    not just the last token! This will allow our model to learn the natural language distribution that composes sonnets,
    not just the distribution over next tokens for the last token!
    """
    ### YOUR CODE HERE
    outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
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
    """
    Generates an original sonnet using top-p sampling and softmax temperature.

    TODO: this is probably not ideal. You can look at hugging face's model.generate(...) function for inspiration.
    In particular, generating multiple sequences and choosing the best with beam search is one avenue. Top_k is another;
    there are many.
    """
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
      
      # logits_last_token = logits_sequence[:, -1, :] / temperature  # Apply temperature scaling

      # # Convert logits to probabilities
      # probs = torch.nn.functional.softmax(logits_last_token, dim=-1)

      # # Top-p (nucleus) sampling
      # sorted_probs, sorted_indices = torch.sort(probs, descending=True)
      # cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
      # top_p_mask = cumulative_probs <= top_p
      # top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()  # Shift mask right for proper thresholding
      # top_p_mask[..., 0] = True  # Always include the highest probability token
      # filtered_probs = sorted_probs * top_p_mask  # Zero out unlikely tokens
      # filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)  # Normalize probabilities

      # # Sample from filtered distribution
      # sampled_index = torch.multinomial(filtered_probs, 1)
      # sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)

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


def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")


def train(args, preview_valid = True):
  device = select_device()
  # Create the data and its corresponding datasets and dataloader.
  sonnet_dataset = SonnetsDataset(args.sonnet_path)
  sonnet_dataloader = DataLoader(sonnet_dataset, shuffle=True, batch_size=args.batch_size,
                                 collate_fn=sonnet_dataset.collate_fn, num_workers=args.num_workers)

  # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  model = SonnetGPT(args)
  model = model.to(device)
  
  # If we want to preview validation set, also need to initialize sampling hyperparameters
  if preview_valid:
    model.update_logit_processors(args.temperature, args.top_k, args.top_p)

  # Initialize optimizer:
  optim_config = {
    "lr" : args.lr,
    "weight_decay": args.weight_decay
  }
  optimizer = set_optimizer(model.parameters(), **optim_config)

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
    if preview_valid:
      print('Generating several output sonnets...')
      model.eval()
      
      for batch in held_out_sonnet_dataset:
        encoding = model.tokenizer(batch[1], return_tensors='pt', padding=True, truncation=True).to(device)
        output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)
        print(f'{batch[1]}{output[1]}\n\n')

    # TODO: consider a stopping condition to prevent overfitting on the small dataset of sonnets.
    save_model(model, optimizer, args, f'{epoch}_{args.filepath}')


@torch.no_grad()
def generate_submission_sonnets(args):
  device = select_device()
  saved = torch.load(f'{args.epochs-1}_{args.filepath}', weights_only=False)

  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

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


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--sonnet_path", type=str, default="../data/sonnets.txt")
  parser.add_argument("--held_out_sonnet_path", type=str, default="../data/sonnets_held_out.txt")
  parser.add_argument("--sonnet_out", type=str, default="../predictions/generated_sonnets.txt")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  
  parser.add_argument("--gpt_kind", type=str, help="The model size as specified on hugging face.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2')

  # Generation parameters.
  parser.add_argument("--temperature", type=float, help="softmax temperature.", default=1.2)
  parser.add_argument("--top_p", type=float, help="Cumulative probability distribution for nucleus sampling.",
                      default=0.9)
  parser.add_argument("--top_k", type=int, help="Thresholding count for TopK Sampling.", default = 50)

  parser.add_argument("--batch_size", help='The training batch size.', type=int, default=8)
  parser.add_argument("--num_workers", help = 'For dataloader. Use 0 for CPU and 2-8 for GPU.', type = int, default = 0)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--weight_decay", type = float, default = 0.0, help = "Weight Decay for Optimizer.")

  args = parser.parse_args()
  return args

if __name__ == "__main__":
  args = get_args()
  args.filepath = f'{args.epochs}-{args.lr}-sonnet.pt'  # Save path.
  set_seed(args.seed)  # Fix the seed for reproducibility.
  train(args)
  generate_submission_sonnets(args)