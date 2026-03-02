# !/usr/bin/env python3


"""
This file contains our Dataset class for Quora paraphrase detection. You may want to modify this file to train on
additional sources of data, or if you change how the Quora dataset is processed (i.e. data augmentation, etc.).
"""

import csv

import re
import torch

from transformers import GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader

################################
### Sentiment Classificaiton ###
################################

# Training Dataset, with Labels
class SentimentDataset(Dataset):
  def __init__(self, dataset):
    self.dataset = dataset
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return self.dataset[idx]

  def pad_data(self, data):
    sents = [x[0] for x in data]
    labels = [x[1] for x in data]
    sent_ids = [x[2] for x in data]

    encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])
    labels = torch.LongTensor(labels)

    return token_ids, attention_mask, labels, sents, sent_ids

  def collate_fn(self, all_data):
    token_ids, attention_mask, labels, sents, sent_ids = self.pad_data(all_data)

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'labels': labels,
      'sents': sents,
      'sent_ids': sent_ids
    }

    return batched_data

# Test Datasets. Without labels
class SentimentTestDataset(Dataset):
  def __init__(self, dataset):
    self.dataset = dataset
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return self.dataset[idx]

  def pad_data(self, data):
    sents = [x[0] for x in data]
    sent_ids = [x[1] for x in data]

    encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])

    return token_ids, attention_mask, sents, sent_ids

  def collate_fn(self, all_data):
    token_ids, attention_mask, sents, sent_ids = self.pad_data(all_data)

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'sents': sents,
      'sent_ids': sent_ids
    }

    return batched_data

def _load_data(filename, is_test = False):
    """
    Subroutine of get_sentiment_dataloader. Copied from starter code.
    """
    num_labels = {}
    data = []
    if is_test:
        with open(filename, 'r') as fp:
            for record in csv.DictReader(fp, delimiter='\t'):
                sent = record['sentence'].lower().strip()
                sent_id = record['id'].lower().strip()
                data.append((sent, sent_id))
    else:
        with open(filename, 'r') as fp:
            for record in csv.DictReader(fp, delimiter='\t'):
                sent = record['sentence'].lower().strip()
                sent_id = record['id'].lower().strip()
                label = int(record['sentiment'].strip())
                if label not in num_labels:
                    num_labels[label] = len(num_labels)
                data.append((sent, label, sent_id))

    if is_test:
        return data
    else:
        return data, len(num_labels)
    
def get_sentiment_dataloader(filename, fold = "train", **dataloader_kwargs):
    """
    Obtain the sentiment classification dataloader.

    Args:
        filename: Input CSV file name. Format: (ID, Sentence, Sentiment Label)
        fold: Fold of the dataset. Accepts ('train', 'valid', 'test'). Defaults to 'train'.
        dataloader_kwargs: Dataloader arguments, can contain optional items like 'batch_size' and 'num_workers'.
    
    Returns:
        If loading train set, returns the dataloader and the number of labels.
        Otherwise, returns the dataloader only.
    """
    if fold == "train":
        data, num_labels = _load_data(filename, False)
        dataset = SentimentDataset(data)
        dataloader = DataLoader(dataset, shuffle = True, collate_fn=dataset.collate_fn, **dataloader_kwargs)
        return dataloader, num_labels
    elif fold == "valid":
        data, _ = _load_data(filename, False)
        dataset = SentimentDataset(data)
        dataloader = DataLoader(dataset, shuffle = False, collate_fn=dataset.collate_fn, **dataloader_kwargs)
        return dataloader
    elif fold == "test":
        data = _load_data(filename, True)
        dataset = SentimentTestDataset(data)
        dataloader = DataLoader(dataset, shuffle = False, collate_fn=dataset.collate_fn, **dataloader_kwargs)
        return dataloader
    else:
        raise RuntimeError(f"Dataset split '{fold}' is not recognized. Please use 'train', 'valid' or 'test'.")
        
############################
### Paraphrase Detection ###
############################

def preprocess_string(s):
  return ' '.join(s.lower()
                  .replace('.', ' .')
                  .replace('?', ' ?')
                  .replace(',', ' ,')
                  .replace('\'', ' \'')
                  .split())


class ParaphraseDetectionDataset(Dataset):
  def __init__(self, dataset, args):
    self.dataset = dataset
    self.p = args
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return self.dataset[idx]

  def collate_fn(self, all_data):
    sent1 = [x[0] for x in all_data]
    sent2 = [x[1] for x in all_data]
    # labels = torch.LongTensor([x[2] for x in all_data])
    labels = ['yes' if label == 1 else 'no' for label in [x[2] for x in all_data]]
    labels = self.tokenizer(labels, return_tensors='pt', padding=True, truncation=True)['input_ids']
    sent_ids = [x[3] for x in all_data]

    cloze_style_sents = [f'Question 1: "{s1}"\nQuestion 2: "{s2}\nAre these questions asking the same thing?\n' for
                         (s1, s2) in zip(sent1, sent2)]
    encoding = self.tokenizer(cloze_style_sents, return_tensors='pt', padding=True, truncation=True)

    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'labels': labels,
      'sent_ids': sent_ids
    }

    return batched_data


class ParaphraseDetectionTestDataset(Dataset):
  def __init__(self, dataset, args):
    self.dataset = dataset
    self.p = args
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return self.dataset[idx]

  def collate_fn(self, all_data):
    sent1 = [x[0] for x in all_data]
    sent2 = [x[1] for x in all_data]
    sent_ids = [x[2] for x in all_data]

    cloze_style_sents = [f'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": ' for (s1, s2) in
                         zip(sent1, sent2)]

    encoding = self.tokenizer(cloze_style_sents, return_tensors='pt', padding=True, truncation=True)

    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'sent_ids': sent_ids
    }

    return batched_data


def load_paraphrase_data(paraphrase_filename, split='train'):
  paraphrase_data = []
  if split == 'test':
    with open(paraphrase_filename, 'r', encoding="utf-8") as fp:
      for record in csv.DictReader(fp, delimiter='\t'):
        sent_id = record['id'].lower().strip()
        paraphrase_data.append((preprocess_string(record['sentence1']),
                                preprocess_string(record['sentence2']),
                                sent_id))

  else:
    with open(paraphrase_filename, 'r', encoding="utf-8") as fp:
      for record in csv.DictReader(fp, delimiter='\t'):
        try:
          sent_id = record['id'].lower().strip()
          paraphrase_data.append((preprocess_string(record['sentence1']),
                                  preprocess_string(record['sentence2']),
                                  int(float(record['is_duplicate'])), sent_id))
        except:
          pass

  print(f"Loaded {len(paraphrase_data)} {split} examples from {paraphrase_filename}")
  return paraphrase_data

#########################
### Sonnet Generation ###
#########################

class SonnetsDataset(Dataset):
  def __init__(self, file_path):
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.sonnets = self._load_sonnets(file_path)

  def _load_sonnets(self, file_path):
    """Reads the file and extracts individual sonnets."""
    with open(file_path, 'r', encoding='utf-8') as f:
      text = f.read()

    # Split sonnets based on numbering pattern (e.g., "\n\n1\n\n")
    sonnets = re.split(r'\n\s*\d+\s*\n', text)[1:]  # Remove header text

    # Strip leading/trailing spaces
    return [s.strip() for s in sonnets]

  def __len__(self):
    return len(self.sonnets)

  def __getitem__(self, idx):
    return (idx, self.sonnets[idx])

  def collate_fn(self, all_data):
    idx = [example[0] for example in all_data]
    sonnets = [example[1] for example in all_data]

    encoding = self.tokenizer(sonnets, return_tensors='pt', padding=True, truncation=True)
    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'sent_ids': idx
    }

    return batched_data
