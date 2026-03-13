"""
Train a GPT-2 sonnet model with Direct Preference Optimization.
Pipeline:
- split the data into train and pairs files with build_sonnet_preferences.py and --split-only
- train SFT sonnet model with sonnet_generation.py
- generate scored pairs (sonnet_dpo_prefs.jsonl) with model checkpoint with build_sonnet_preferences.py --checkpoint
- run train_dpo_sonnets.py with the SFT checkpoint to train on preference triplets
    - uses dpo_dataset.py to tokenize and batch that data
    - computes DPO loss with dpo_model.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
GPT_DIR = SCRIPT_DIR.parent
REPO_ROOT = GPT_DIR.parent
for path in (GPT_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from peft_config import SONNET_PEFT
from program_config import DPO_ARGS
from datasets import SonnetsDataset
from sonnet_generation import SonnetGPT
from sonnet_testing import compute_chrf_score
from torch_utils import (
    load_fine_tune_checkpoint,
    save_fine_tune_checkpoint,
    select_device,
    set_optimizer,
    set_seed,
    summarize_peft,
)

from dpo_dataset import SonnetPreferenceDataset
from dpo_model import GPT2DPOWrapper


DEFAULT_PREFS_PATH = Path(DPO_ARGS.prefs_path)
DEFAULT_OUTPUT_DIR = Path(DPO_ARGS.output_dir)
DEFAULT_EVAL_PATH = Path(DPO_ARGS.eval_path)
DEFAULT_GENERATIONS_OUT = Path(DPO_ARGS.generations_out)

def move_batch_to_device(batch, device):
    # Move every tensor in the batch dictionary onto the active device.
    return {key: value.to(device) for key, value in batch.items()}


def build_model_config(args):
    # Recreate the checkpoint metadata expected by the existing save/load helpers.
    model_config = SimpleNamespace(
        gpt_kind=args.gpt_kind,
        fine_tune_mode=args.fine_tune_mode,
    )
    if args.fine_tune_mode == "peft":
        model_config.peft_config = SONNET_PEFT
    return model_config


@torch.no_grad()
def evaluate_dpo_checkpoint(checkpoint_path, args):
    # Load the best DPO checkpoint, generate held-out sonnets, and score them with CHRF.
    device = select_device()
    gpt = load_fine_tune_checkpoint(checkpoint_path, device=device, model=None)
    dataset = SonnetPreferenceDataset(args.prefs_path, tokenizer_name=args.gpt_kind)
    model = SonnetGPT(gpt, dataset.tokenizer).to(device)
    model.eval()
    model.update_logit_processors(args.temperature, args.top_k, args.top_p)

    held_out_sonnet_dataset = SonnetsDataset(args.eval_path)
    args.generations_out.parent.mkdir(parents=True, exist_ok=True)

    generated_sonnets = []
    for batch in held_out_sonnet_dataset:
        sonnet_id = batch[0]
        encoding = model.tokenizer(batch[1], return_tensors="pt", padding=False, truncation=True).to(device)
        output = model.generate(encoding["input_ids"], max_length=args.max_new_tokens)[0][0]
        decoded_output = model.tokenizer.decode(output)
        generated_sonnets.append((sonnet_id, f"{decoded_output}\n\n"))

    with args.generations_out.open("w+", encoding="utf-8") as handle:
        handle.write("--Generated Sonnets-- \n\n")
        for sonnet_id, sonnet in generated_sonnets:
            handle.write(f"\n{sonnet_id}\n")
            handle.write(sonnet)

    score = compute_chrf_score(args.generations_out, args.eval_path)
    print(f"DPO CHRF: {score:.4f}")
    print(f"Saved generated sonnets to {args.generations_out}")


def parse_args():
    # Expose the DPO data, checkpoint, and optimization settings as CLI arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefs-path", type=Path, default=DEFAULT_PREFS_PATH)
    parser.add_argument("--checkpoint", type=Path, default=Path(DPO_ARGS.checkpoint))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DPO_ARGS.seed)
    parser.add_argument("--batch-size", type=int, default=DPO_ARGS.batch_size)
    parser.add_argument("--num-workers", type=int, default=DPO_ARGS.num_workers)
    parser.add_argument("--epochs", type=int, default=DPO_ARGS.epochs)
    parser.add_argument("--lr", type=float, default=DPO_ARGS.lr)
    parser.add_argument("--weight-decay", type=float, default=DPO_ARGS.weight_decay)
    parser.add_argument("--beta", type=float, default=DPO_ARGS.beta)
    parser.add_argument("--grad-clip", type=float, default=DPO_ARGS.grad_clip)
    parser.add_argument("--gpt-kind", type=str, default=DPO_ARGS.gpt_kind)
    parser.add_argument("--fine-tune-mode", type=str, default=DPO_ARGS.fine_tune_mode)
    parser.add_argument("--temperature", type=float, default=DPO_ARGS.temperature)
    parser.add_argument("--top-k", type=int, default=DPO_ARGS.top_k)
    parser.add_argument("--top-p", type=float, default=DPO_ARGS.top_p)
    parser.add_argument("--max-new-tokens", type=int, default=DPO_ARGS.max_new_tokens)
    parser.add_argument("--eval-path", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--generations-out", type=Path, default=DEFAULT_GENERATIONS_OUT)
    parser.add_argument("--save-every-epoch", action="store_true", default=DPO_ARGS.save_every_epoch)
    return parser.parse_args()


def train_dpo(args):
    # Run end-to-end DPO training from a pretrained SFT checkpoint.
    set_seed(args.seed)
    device = select_device()

    dataset = SonnetPreferenceDataset(args.prefs_path, tokenizer_name=args.gpt_kind)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
    )

    policy = load_fine_tune_checkpoint(args.checkpoint, device=device, model=None)
    dpo_model = GPT2DPOWrapper(policy, beta=args.beta).to(device)
    # Print how many policy parameters remain trainable under the current fine-tuning setup
    summarize_peft(dpo_model.policy)

    # Reuse the usual optimizer setup, but only for policy weights that DPO is allowed to update.
    optimizer = set_optimizer(
        [param for param in dpo_model.policy.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_config = build_model_config(args)
    best_epoch_loss = None
    best_checkpoint_path = None

    for epoch in range(args.epochs):
        dpo_model.train()
        total_loss = 0.0
        num_batches = 0

        progress = tqdm(dataloader, desc=f"dpo-train-{epoch}", leave=False)
        for batch in progress:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad()

            loss = dpo_model.dpo_loss(
                chosen_input_ids=batch["chosen_input_ids"],
                chosen_attention_mask=batch["chosen_attention_mask"],
                chosen_completion_mask=batch["chosen_completion_mask"],
                rejected_input_ids=batch["rejected_input_ids"],
                rejected_attention_mask=batch["rejected_attention_mask"],
                rejected_completion_mask=batch["rejected_completion_mask"],
            )
            loss.backward()
            clip_grad_norm_(dpo_model.policy.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(1, num_batches)
        print(f"Epoch {epoch}: dpo loss :: {avg_loss:.4f}")

        checkpoint_name = f"dpo_epoch{epoch}.pt"
        checkpoint_path = args.output_dir / checkpoint_name
        if args.save_every_epoch:
            save_fine_tune_checkpoint(str(checkpoint_path), dpo_model.policy, model_config)

        if best_epoch_loss is None or avg_loss < best_epoch_loss:
            best_epoch_loss = avg_loss
            best_checkpoint_path = args.output_dir / "dpo_best.pt"
            save_fine_tune_checkpoint(str(best_checkpoint_path), dpo_model.policy, model_config)

    if best_checkpoint_path is not None:
        print(f"Best DPO checkpoint saved to {best_checkpoint_path}")
        evaluate_dpo_checkpoint(best_checkpoint_path, args)


if __name__ == "__main__":
    train_dpo(parse_args())
