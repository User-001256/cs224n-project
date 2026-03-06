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

from program_config import PARAPHRASE_ARGS, DATA_PATHS
from peft_config import PARAPHRASE_PEFT
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from tqdm import tqdm

from datasets import get_paraphrase_dataloader
from evaluation import model_eval_paraphrase, model_test_paraphrase
from models.gpt2 import GPT2Model
from models.gpt2_adapt import apply_adapter, apply_IA3, apply_lora

from torch_utils import *

class ParaphraseGPT(nn.Module):
  """Your GPT-2 Model designed for paraphrase detection."""

  def __init__(self, gpt_backbone:GPT2Model):
    super().__init__()
    self.gpt = gpt_backbone
    # self.paraphrase_detection_head = nn.Linear(args.d, 2)  # Paraphrase detection has two outputs: 1 (yes) or 0 (no).

  def forward(self, input_ids, attention_mask):
    """
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

def train(args):
  """Train GPT-2 for paraphrase detection on the Quora dataset."""
  device = select_device()
  
  train_dl = get_paraphrase_dataloader(DATA_PATHS.para_train, "train", batch_size = args.batch_size, num_workers = args.num_workers)
  valid_dl = get_paraphrase_dataloader(DATA_PATHS.para_valid, "valid", batch_size = args.batch_size, num_workers = args.num_workers)

  # Initialize model_config for checkpoint saving
  model_config = SimpleNamespace(
    gpt_kind = args.gpt_kind,
    fine_tune_mode = args.fine_tune_mode,
  )
  if args.fine_tune_mode == "peft":
    model_config.peft_config = PARAPHRASE_PEFT
    
  # Initialize GPT2 model:
  model = GPT2Model.from_pretrained(args.gpt_kind)
  model = model.to(device)
  # Apply PEFT:
  if args.fine_tune_mode == "peft":
    apply_gpt_peft(model, PARAPHRASE_PEFT)
  
  # Adds the Paraphrase GPT Wrapper
  model = ParaphraseGPT(model)
  model = model.to(device)
  summarize_peft(model)

  optim_config = {
    "lr" : args.lr,
    "weight_decay": args.weight_decay
  }
  trainable_params = [p for p in model.parameters() if p.requires_grad]
  optimizer = set_optimizer(trainable_params, **optim_config)
  best_dev_acc = 0

  # Run for the specified number of epochs.
  for epoch in range(args.epochs):
    model.train()
    train_loss = 0.0
    num_obs = 0
    # Early Stopping
    es = EarlyStoppingScheduler()
    for batch in tqdm(train_dl, desc=f'train-{epoch}', leave = False):
      # Get the input and move it to the gpu (I do not recommend training this model on CPU).
      b_ids, b_mask, labels = batch['token_ids'], batch['attention_mask'], batch['labels'].flatten()
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)
      labels = labels.to(device)

      # Compute the loss, gradients, and update the model's parameters.
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      loss = F.cross_entropy(logits, labels, reduction='sum')
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_obs += labels.shape[0]

    train_loss = train_loss / num_obs

    dev_acc, dev_f1, dev_loss, *_ = model_eval_paraphrase(valid_dl, model, device)

    if dev_acc > best_dev_acc:
      best_dev_acc = dev_acc
      save_fine_tune_checkpoint(args.filepath, model.gpt, model_config)
      
    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, dev loss/acc/f1 :: ({dev_loss:.3f}, {dev_acc :.3f}, {dev_f1:.3f})")
    es.step(dev_loss)
    if es.stopped:
      print(f"Early Stopping Activated at Epoch {epoch}. Halting Training.")


@torch.no_grad()
def test(args):
  """Evaluate your model on the dev and test datasets; save the predictions to disk."""
  device = select_device()
  
  # Load our checkpoint. Infer from the saved configs. Returns a GPT2Model
  gpt = load_fine_tune_checkpoint(args.filepath, device, model = None)
  model = ParaphraseGPT(gpt) # type: ignore

  dev_dl =  get_paraphrase_dataloader(args.para_valid,"valid", batch_size = args.batch_size, num_workers = args.num_workers)
  test_dl = get_paraphrase_dataloader(args.para_test,  "test", batch_size = args.batch_size, num_workers = args.num_workers)

  dev_para_acc, _, _, dev_para_y_pred, _, dev_para_sent_ids = model_eval_paraphrase(dev_dl, model, device)
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

if __name__ == "__main__":
  args = PARAPHRASE_ARGS
  args.filepath = f'{args.epochs}-{args.lr}-paraphrase.pt'  # Save path.
  set_seed(args.seed)  # Fix the seed for reproducibility.
  train(args)
  test(args)
