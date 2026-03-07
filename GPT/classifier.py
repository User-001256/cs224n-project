#!/usr/bin/env python3

'''
Trains and evaluates GPT2SentimentClassifier on SST and CFIMDB

Codebase heavily modified by User001256 for clarity and additional functionality.
'''

from program_config import CLASSIFIER_ARGS, DATA_PATHS
from peft_config import CLASSIFIER_PEFT

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score

from models.gpt2 import GPT2Model
from models.gpt2_adapt import *
from models.gpt2_classifier import GPT2Classifier
from tqdm import tqdm

from torch_utils import *
from datasets import get_sentiment_dataloader

# Evaluate the model on test examples.
def model_test_eval(dataloader, model, device):
  model.eval()  # Switch to eval model, will turn off randomness like dropout.
  y_pred = []
  sents = []
  sent_ids = []
  for step, batch in enumerate(tqdm(dataloader, desc=f'eval')):
    b_ids, b_mask, b_sents, b_sent_ids = batch['token_ids'], batch['attention_mask'], \
                                         batch['sents'], batch['sent_ids']

    b_ids = b_ids.to(device)
    b_mask = b_mask.to(device)

    logits = model(b_ids, b_mask)
    logits = logits.detach().cpu().numpy()
    preds = np.argmax(logits, axis=1).flatten()

    y_pred.extend(preds)
    sents.extend(b_sents)
    sent_ids.extend(b_sent_ids)

  return y_pred, sents, sent_ids

def train_one_epoch(model:GPT2Classifier,
                    train_dataloader,
                    valid_dataloader,
                    optimizer:torch.optim.Optimizer,
                    device = torch.device("cpu"),
                    tqdm_title = "Training Batches"):
  """
  Wrapper function for one-epoch training loop. Used for the GPT2 Sentiment Classifier.

  Args:
      model: GPT2 Model with Prediction Head
      train_dataloader: Train Dataloader
      valid_dataloader: Valid / Dev Dataloader
      optimizer: Model Optimizer
      device: Torch Device. Defaults to "cpu".
      eval_train: Evaluate on Training Set? Defaults to False.
      eval_valid: Evaluate on Dev Set? Defaults to True.
  """
  model.train()
  for batch in tqdm(train_dataloader, desc=tqdm_title, leave = False):
      b_ids, b_mask, b_labels = (batch['token_ids'],
                                 batch['attention_mask'], batch['labels'])

      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)
      b_labels = b_labels.to(device)
      
      # Typical Training Loop...
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      loss = F.cross_entropy(logits, b_labels.view(-1), reduction='mean')

      loss.backward()
      optimizer.step()
  # Evaluate Model on both training and validaiton set
  out = dict()
  model.eval()
  train_acc, train_f1, train_loss, *_ = model_eval(train_dataloader, model, device, "Eval Train")
  out["train_loss"] = train_loss
  out["train_acc"] = train_acc
  out["train_f1"] = train_f1
  dev_acc, dev_f1, dev_loss, *_ = model_eval(valid_dataloader, model, device, "Eval Dev")
  out["valid_loss"] = dev_loss
  out["valid_acc"] = dev_acc
  out["valid_f1"] = dev_f1
  return out

# Evaluate the model on dev examples. Don't track gradients.
@torch.no_grad
def model_eval(dataloader, model, device, tqdm_title = "Eval Batches"):
  model.eval()  # Switch to eval model, will turn off randomness like dropout.
  y_true = []
  y_pred = []
  sents = []
  sent_ids = []
  loss = 0.0
  for batch in tqdm(dataloader, desc=tqdm_title, leave = False):
    b_ids, b_mask, b_labels, b_sents, b_sent_ids = batch['token_ids'], batch['attention_mask'], \
                                                   batch['labels'], batch['sents'], batch['sent_ids']

    b_ids = b_ids.to(device)
    b_mask = b_mask.to(device)

    logits = model(b_ids, b_mask)
    loss += F.cross_entropy(logits, b_labels.to(device).view(-1), reduction='sum')
    logits = logits.detach().cpu().numpy()
    preds = np.argmax(logits, axis=1).flatten()

    b_labels = b_labels.flatten()
    y_true.extend(b_labels)
    y_pred.extend(preds)
    sents.extend(b_sents)
    sent_ids.extend(b_sent_ids)

  f1 = f1_score(y_true, y_pred, average='macro')
  acc = accuracy_score(y_true, y_pred)

  return acc, f1, loss / len(y_true), y_pred, y_true, sents, sent_ids

def train_main(model, optimizer, train_dataloader, valid_dataloader, trainer_args, model_config):
  """
  Main routine for classifier training

  Args:
      model: Classifier model
      optimizer: Model Optimizer
      train_dataloader: Training Dataloader
      valid_dataloader: Validation Dataloader.
      trainer_args: Arguments for the trainer. Contains attributes: epochs, filepath (for checkpoints)
      model_config: Config for the model.
      train_metric: Report traning set metrics. Default to False
      valid_metric: Report validation set metrics. Defaults to True
  """
  device = select_device()
  model = model.to(device)
  best_dev_acc = 0.0
  es = EarlyStoppingScheduler()
  for epoch in range(trainer_args.epochs):
      metrics = train_one_epoch(model, train_dataloader, valid_dataloader,
                                optimizer, device, f"Epoch {epoch}")
      # Report training metrics
      print(f"Epoch {epoch}: ", end = "")
      train_loss = metrics["train_loss"]
      train_acc = metrics["train_acc"]
      print(f"train loss/acc :: ({train_loss:.3f}, {train_acc:.3f}),  ", end = "")
      valid_loss = metrics["valid_loss"]
      valid_acc = metrics["valid_acc"]
      print(f"dev loss/acc :: ({valid_loss:.3f}, {valid_acc:.3f})")
      # Save model checkpoint if needed.
      if valid_acc > best_dev_acc:
        best_dev_acc = valid_acc
        save_fine_tune_checkpoint(trainer_args.filepath, model, model_config)
      # Update Early Stopping:
      es.step(valid_loss)
      if es.stopped:
        print(f"Early Stopping Activated at Epoch {epoch}. Halting Training.")
        break
      
def train_init(program_args):
  """
  Initialize the trainer object.

  Args:
      program_args: Program Arguments
  """
  a = program_args
  # Loads the train and valid dataloaders
  train_dl, num_labels = get_sentiment_dataloader(a.train, "train", batch_size = a.batch_size,
                                      num_workers = a.num_workers)
  valid_dl = get_sentiment_dataloader(a.dev, "valid", batch_size = a.batch_size, 
                                      num_workers = a.num_workers)
  # Construct the model config to be written to checkpoints
  def _get_n_embd(gpt_kind):
    if gpt_kind == "gpt2":
      return 768
    if gpt_kind == "gpt2-medium":
      return 1024
    if gpt_kind == "gpt2-large":
      return 1280
    if gpt_kind == "gpt2-xl":
      return 1600
  clf_config = SimpleNamespace(
    hidden_dropout_prob = a.hidden_dropout_prob,
    hidden_size = _get_n_embd(a.gpt_kind),
    num_labels = num_labels,
    clf_kind = a.prediction_head,
    mlp_act_dim = None # Always use default
  )
  model_config = SimpleNamespace(
    gpt_kind = a.gpt_kind,
    fine_tune_mode = a.fine_tune_mode,
    clf_config = clf_config
  )
  if a.fine_tune_mode == "peft":
    model_config.peft_config = CLASSIFIER_PEFT
  
  # Now initialize our model:
  gpt = GPT2Model.from_pretrained(a.gpt_kind)
  if a.fine_tune_mode == "peft": # PEFT Fine Tune Mode
    apply_gpt_peft(gpt, CLASSIFIER_PEFT)
  elif a.fine_tune_mode == "last_layer": # Only Last Layer
    for p in gpt.parameters(): p.requires_grad = False
  # Add prediction head
  clf = GPT2Classifier(clf_config, gpt)
  summarize_peft(clf)
  
  # Initialize optimizer:
  optim_config = {
    "lr" : a.lr,
    "weight_decay": a.weight_decay
  }
  trainable = [p for p in clf.parameters() if p.requires_grad]
  opt = set_optimizer(trainable, **optim_config)
  # Generate Return object:
  return {
    "model": clf,
    "optimizer": opt,
    "train_dataloader": train_dl,
    "valid_dataloader": valid_dl,
    "trainer_args": a,
    "model_config": model_config
  } # Simply unpack this return object as train_main's arguments

@torch.no_grad
def test(program_args):
    device = select_device()
    a = program_args
    # Load our checkpoint. Infer from the saved configs
    model = load_fine_tune_checkpoint(a.filepath, device = device, model = None)
    model = model.to(device)
    # Obtain dataloader for dev and test sets
    dev_dl = get_sentiment_dataloader(a.dev, "valid", batch_size = a.batch_size,
                                      num_workers = a.num_workers)
    test_dl = get_sentiment_dataloader(a.test, "test", batch_size = a.batch_size, 
                                      num_workers = a.num_workers)

    # Evaluate
    dev_acc, dev_f1, dev_loss, dev_pred, _, _, dev_sent_ids = model_eval(
      dev_dl, model, device)
    print(f'Dev Set: acc :: {dev_acc:.3f}, macro f1 :: {dev_f1:.3f}, loss :: {dev_loss:.3f}')

    test_pred, test_sents, test_sent_ids = model_test_eval(test_dl, model, device)
    print('DONE Test')

    with open(a.dev_out, "w+") as f:
      f.write(f"id \t Predicted_Sentiment \n")
      for p, s in zip(dev_sent_ids, dev_pred):
        f.write(f"{p}, {s} \n")

    with open(a.test_out, "w+") as f:
      f.write(f"id \t Predicted_Sentiment \n")
      for p, s in zip(test_sent_ids, test_pred):
        f.write(f"{p}, {s} \n")
        
def run_everything(args):
  set_seed(args.seed)

  print('Training Sentiment Classifier on SST...')
  config = SimpleNamespace(
    filepath='sst-classifier.pt',
    lr=args.lr,
    weight_decay = args.weight_decay,
    prediction_head=args.prediction_head,
    epochs=args.epochs,
    batch_size=args.batch_size,
    num_workers = args.num_workers,
    gpt_kind = args.gpt_kind,
    hidden_dropout_prob=args.hidden_dropout_prob,
    fine_tune_mode=args.fine_tune_mode,
    train=DATA_PATHS.sst_train,
    dev=DATA_PATHS.sst_valid,
    test=DATA_PATHS.sst_test,
    dev_out='predictions/' + args.fine_tune_mode + '-sst-dev-out.csv',
    test_out='predictions/' + args.fine_tune_mode + '-sst-test-out.csv'
  )

  train_main(**train_init(config))

  print('Evaluating on SST...')
  test(config)

  print('Training Sentiment Classifier on cfimdb...')
  config = SimpleNamespace(
    filepath='cfimdb-classifier.pt',
    lr=args.lr,
    weight_decay = args.weight_decay,
    prediction_head=args.prediction_head,
    epochs=args.epochs,
    batch_size=args.batch_size,
    num_workers = args.num_workers,
    gpt_kind = args.gpt_kind,
    hidden_dropout_prob=args.hidden_dropout_prob,
    fine_tune_mode=args.fine_tune_mode,
    train=DATA_PATHS.imdb_train,
    dev=DATA_PATHS.imdb_valid,
    test=DATA_PATHS.imdb_test,
    dev_out='predictions/' + args.fine_tune_mode + '-cfimdb-dev-out.csv',
    test_out='predictions/' + args.fine_tune_mode + '-cfimdb-test-out.csv'
  )

  train_main(**train_init(config))

  print('Evaluating on cfimdb...')
  test(config)  
        


if __name__ == "__main__":
  run_everything(CLASSIFIER_ARGS)

