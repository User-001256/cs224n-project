"""
DPO preference dataset implementation for the sonnet generation task.
- loads JSONL preference records
- tokenizes prompt + chosen and prompt + rejected
- builds completion_mask so DPO only scores continuation tokens and not the prompt

"""

import json
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer


class SonnetPreferenceDataset(Dataset):
    def __init__(self, file_path, tokenizer_name="gpt2-medium"):
        # Load saved DPO preference pairs and prepare a tokenizer for collation.
        self.file_path = Path(file_path)
        self.records = self._load_records(self.file_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_records(self, file_path: Path) -> List[Dict]:
        # Read one JSON object per line from the preference file.
        records = []
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def _ensure_prompt_prefix(self, prompt: str, text: str) -> str:
        # Make sure each candidate sequence starts with the prompt so masking is aligned.
        if text.startswith(prompt):
            return text
        separator = "" if prompt.endswith("\n") else "\n"
        return f"{prompt}{separator}{text.lstrip()}"

    def _build_branch(
        self,
        prompts: Sequence[str],
        texts: Sequence[str],
    ) -> Dict[str, torch.Tensor]:
        # Tokenize one branch of the preference data and mark which tokens belong to the completion.
        prefixed_texts = [
            self._ensure_prompt_prefix(prompt, text) for prompt, text in zip(prompts, texts)
        ]
        encoding = self.tokenizer(
            prefixed_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        prompt_tokens = self.tokenizer(
            list(prompts),
            add_special_tokens=False,
            padding=False,
            truncation=True,
        )["input_ids"]

        input_ids = torch.LongTensor(encoding["input_ids"])
        attention_mask = torch.LongTensor(encoding["attention_mask"])
        completion_mask = torch.zeros_like(attention_mask)

        seq_lens = attention_mask.sum(dim=1).tolist()
        prompt_lens = [len(tokens) for tokens in prompt_tokens]
        for row_idx, (prompt_len, seq_len) in enumerate(zip(prompt_lens, seq_lens)):
            if seq_len > prompt_len:
                completion_mask[row_idx, prompt_len:seq_len] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "completion_mask": completion_mask,
        }

    def __len__(self):
        # Return the number of stored preference examples.
        return len(self.records)

    def __getitem__(self, idx):
        # Return the raw JSONL record so collation can tokenize both branches together.
        return self.records[idx]

    def collate_fn(self, batch: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Convert a batch of prompt/chosen/rejected text triples into DPO-ready tensors.
        prompts = [item["prompt"] for item in batch]
        chosen_texts = [item["chosen"] for item in batch]
        rejected_texts = [item["rejected"] for item in batch]

        chosen = self._build_branch(prompts, chosen_texts)
        rejected = self._build_branch(prompts, rejected_texts)

        return {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_attention_mask": chosen["attention_mask"],
            "chosen_completion_mask": chosen["completion_mask"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
            "rejected_completion_mask": rejected["completion_mask"],
        }
