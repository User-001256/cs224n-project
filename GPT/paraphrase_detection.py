'''
Paraphrase detection for GPT starter code.

Consider:
 - ParaphraseGPT: Your implementation of the GPT-2 classification model.
 - train: Training procedure for ParaphraseGPT on the Quora paraphrase detection dataset.
 - test: Test procedure. This function generates the required files for your submission.

Running:
  `python paraphrase_detection.py --use_gpu`
trains and evaluates your ParaphraseGPT model and writes the required submission files.
'''

import argparse
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from tqdm import tqdm

from datasets import get_paraphrase_dataloader
from evaluation import model_eval_paraphrase, model_test_paraphrase
from models.gpt2 import GPT2Model

from torch_utils import *

class ParaphraseGPT(nn.Module):
  """Your GPT-2 Model designed for paraphrase detection."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(args.gpt_kind)
    # self.paraphrase_detection_head = nn.Linear(args.d, 2)  # Paraphrase detection has two outputs: 1 (yes) or 0 (no).

    # By default, fine-tune the full model.
    for param in self.gpt.parameters():
      param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    """
    TODO: Predict the label of the token using the paraphrase_detection_head Linear layer.

    We structure the input as:

      'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": '

    So you want to find the prediction for the next token at the end of this sentence. Optimistically, it will be the
    token "yes" (byte pair encoding index of 8505) for examples that are paraphrases or "no" (byte pair encoding index
     of 3919) for examples that are not paraphrases.
    """

    'Takes a batch of sentences and produces embeddings for them.'
    ### YOUR CODE HERE
    outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
    last_token = outputs['last_token']
    logits = self.gpt.hidden_state_to_token(last_token)
    return logits

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


def train(args):
  """Train GPT-2 for paraphrase detection on the Quora dataset."""
  device = select_device()
  
  train_dl = get_paraphrase_dataloader(args.para_train, "train", batch_size = args.batch_size, num_workers = args.num_workers)
  valid_dl = get_paraphrase_dataloader(args.para_dev,   "valid", batch_size = args.batch_size, num_workers = args.num_workers)

  model = ParaphraseGPT(args)
  model = model.to(device)
  
  optim_config = {
    "lr" : args.lr,
    "weight_decay": args.weight_decay
  }
  optimizer = set_optimizer(model.parameters(), **optim_config)
  best_dev_acc = 0

  # Run for the specified number of epochs.
  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0
    for batch in tqdm(train_dl, desc=f'train-{epoch}', leave = False):
      # Get the input and move it to the gpu (I do not recommend training this model on CPU).
      b_ids, b_mask, labels = batch['token_ids'], batch['attention_mask'], batch['labels'].flatten()
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)
      labels = labels.to(device)

      # Compute the loss, gradients, and update the model's parameters.
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      preds = torch.argmax(logits, dim=1)
      loss = F.cross_entropy(logits, labels, reduction='mean')
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / num_batches

    dev_acc, dev_f1, *_ = model_eval_paraphrase(valid_dl, model, device)

    if dev_acc > best_dev_acc:
      best_dev_acc = dev_acc
      save_model(model, optimizer, args, args.filepath)

    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, dev acc :: {dev_acc :.3f}")


@torch.no_grad()
def test(args):
  """Evaluate your model on the dev and test datasets; save the predictions to disk."""
  device = select_device()
  saved = torch.load(args.filepath)

  model = ParaphraseGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()
  print(f"Loaded model to test from {args.filepath}")

  dev_dl =  get_paraphrase_dataloader(args.para_dev, "valid", batch_size = args.batch_size, num_workers = args.num_workers)
  test_dl = get_paraphrase_dataloader(args.para_test, "test", batch_size = args.batch_size, num_workers = args.num_workers)

  dev_para_acc, _, dev_para_y_pred, _, dev_para_sent_ids = model_eval_paraphrase(dev_dl, model, device)
  print(f"dev paraphrase acc :: {dev_para_acc :.3f}")
  test_para_y_pred, test_para_sent_ids = model_test_paraphrase(test_dl, model, device)

  with open(args.para_dev_out, "w+", encoding='utf-8') as f:
    f.write(f"id \t Predicted_Is_Paraphrase \n")
    for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
      f.write(f"{p}, {s} \n")

  with open(args.para_test_out, "w+", encoding='utf-8') as f:
    f.write(f"id \t Predicted_Is_Paraphrase \n")
    for p, s in zip(test_para_sent_ids, test_para_y_pred):
      f.write(f"{p}, {s} \n")


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--para_train", type=str, default="../data/quora-train.csv")
  parser.add_argument("--para_dev", type=str, default="../data/quora-dev.csv")
  parser.add_argument("--para_test", type=str, default="../data/quora-test-student.csv")
  parser.add_argument("--para_dev_out", type=str, default="../predictions/para-dev-output.csv")
  parser.add_argument("--para_test_out", type=str, default="../predictions/para-test-output.csv")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--gpt_kind", type=str,
                      help="The model size as specified on hugging face.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2')

  parser.add_argument("--batch_size", help='Batch Size. Recommend 64 for GPU.', type=int, default=8)
  parser.add_argument("--num_workers", help = 'For dataloader. Use 0 for CPU and 2-8 for GPU.', type = int, default = 0)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--weight_decay", type = float, default = 0.0, help = "Weight Decay for Optimizer.")


  args = parser.parse_args()
  return args

# GPT2Model.from_pretrained automatically infers hyperparameters.
# No need of this.
# def add_arguments(args):
#   """Add arguments that are deterministic on model size."""
#   if args.model_size == 'gpt2':
#     args.d = 768
#     args.l = 12
#     args.num_heads = 12
#   elif args.model_size == 'gpt2-medium':
#     args.d = 1024
#     args.l = 24
#     args.num_heads = 16
#   elif args.model_size == 'gpt2-large':
#     args.d = 1280
#     args.l = 36
#     args.num_heads = 20
#   else:
#     raise Exception(f'{args.model_size} is not supported.')
#   return args


if __name__ == "__main__":
  args = get_args()
  args.filepath = f'{args.epochs}-{args.lr}-paraphrase.pt'  # Save path.
  set_seed(args.seed)  # Fix the seed for reproducibility.
  train(args)
  test(args)
