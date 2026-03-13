"""
Build a small sonnet preference dataset for DPO.

Workflow:
1. Deterministically split the real sonnet corpus into an SFT split and a
   prompt-source split.
2. Use a trained sonnet checkpoint to sample multiple continuations for each
   held-out prompt.
3. Rank the candidates with a lightweight structural heuristic.
4. Save best-vs-worst preference pairs as JSONL for DPO training.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from tqdm import tqdm
from transformers import GPT2Tokenizer

from program_config import SONNET_ARGS
from sonnet_generation import SonnetGPT
from torch_utils import load_fine_tune_checkpoint, select_device, set_seed


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
DEFAULT_REAL_SONNETS = DATA_DIR / "sonnets.txt"
DEFAULT_SFT_SPLIT = DATA_DIR / "sonnets_sft_train_split.txt"
DEFAULT_PROMPT_SPLIT = DATA_DIR / "sonnets_dpo_prompt_source.txt"
DEFAULT_PREFS_JSONL = DATA_DIR / "sonnet_dpo_prefs.jsonl"

TARGET_LINES = 14


def read_numbered_sonnets(path: Path) -> List[str]:
    # Parse the numbered corpus format into a plain list of sonnet strings.
    text = path.read_text(encoding="utf-8")
    parts = re.split(r"\n\s*\d+\s*\n", text)[1:]
    return [part.strip() for part in parts if part.strip()]


def write_numbered_sonnets(path: Path, sonnets: Sequence[str]) -> None:
    # Write sonnets back out in the same numbered format expected elsewhere in the project.
    with path.open("w", encoding="utf-8") as handle:
        for idx, sonnet in enumerate(sonnets, start=1):
            handle.write(f"{idx}\n\n{sonnet.strip()}\n\n")


def split_real_sonnets(
    sonnets: Sequence[str],
    heldout_count: int = 26,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    # Make a deterministic train/prompt-source split from the real sonnet corpus.
    indices = list(range(len(sonnets)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    heldout = set(indices[:heldout_count])
    sft_split = [sonnets[idx] for idx in range(len(sonnets)) if idx not in heldout]
    prompt_split = [sonnets[idx] for idx in range(len(sonnets)) if idx in heldout]
    return sft_split, prompt_split


def nonempty_lines(text: str) -> List[str]:
    # Normalize a sonnet into its non-blank lines.
    return [line.strip() for line in text.splitlines() if line.strip()]


def prompt_from_sonnet(sonnet: str, prompt_lines: int = 1) -> str:
    # Use the first few lines of a source sonnet as the generation prompt.
    lines = nonempty_lines(sonnet)
    prompt = "\n".join(lines[:prompt_lines]).strip()
    return prompt


def normalize_word(word: str) -> str:
    # Strip punctuation so simple overlap and rhyme checks are less noisy.
    return re.sub(r"[^a-z]", "", word.lower())


def count_repeated_ngrams(tokens: Sequence[str], n: int = 3) -> int:
    # Count repeated n-grams as a lightweight signal for degenerate generation loops.
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    return sum(count - 1 for count in counts.values() if count > 1)


def rhyme_score(lines: Sequence[str]) -> float:
    # Reward rough Shakespearean rhyme-pattern matches using final two-letter endings.
    if len(lines) < TARGET_LINES:
        return 0.0
    rhyme_pairs = [(0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 13)]
    score = 0.0
    for i, j in rhyme_pairs:
        end_i = normalize_word(lines[i].split()[-1]) if lines[i].split() else ""
        end_j = normalize_word(lines[j].split()[-1]) if lines[j].split() else ""
        if len(end_i) >= 2 and len(end_j) >= 2 and end_i[-2:] == end_j[-2:]:
            score += 0.4
    return score


def score_sonnet(prompt: str, sonnet: str) -> Tuple[float, Dict[str, float]]:
    # Combine several structural heuristics into one score and keep the breakdown for inspection.
    lines = nonempty_lines(sonnet)
    text = "\n".join(lines)
    lower_text = text.lower()
    lower_prompt = prompt.strip().lower()
    tokens = re.findall(r"[a-zA-Z']+", lower_text)
    unique_lines = len(set(line.lower() for line in lines))

    components: Dict[str, float] = {}
    components["line_count"] = max(0.0, 3.0 - 0.75 * abs(TARGET_LINES - len(lines)))

    if lines:
        word_counts = [len(re.findall(r"[a-zA-Z']+", line)) for line in lines]
        well_formed = sum(4 <= count <= 12 for count in word_counts)
        components["line_length"] = 2.0 * well_formed / len(lines)
        punctuation = sum(line.endswith((",", ".", ";", ":", "?", "!")) for line in lines)
        components["line_end_punct"] = 0.75 * punctuation / len(lines)
    else:
        components["line_length"] = 0.0
        components["line_end_punct"] = 0.0

    components["prompt_match"] = 1.5 if lines and lines[0].strip().lower() == lower_prompt else 0.0
    duplicate_count = len(lines) - unique_lines
    components["duplicate_penalty"] = -1.5 * duplicate_count

    repeated_trigrams = count_repeated_ngrams(tokens, n=3)
    components["repetition_penalty"] = -0.2 * repeated_trigrams

    alpha_chars = sum(char.isalpha() for char in text)
    nonspace_chars = sum(not char.isspace() for char in text)
    alpha_ratio = alpha_chars / nonspace_chars if nonspace_chars else 0.0
    components["alphabetic_ratio"] = 1.0 if alpha_ratio >= 0.7 else 0.0

    prompt_words = {normalize_word(word) for word in prompt.split()}
    prompt_words.discard("")
    candidate_words = {normalize_word(word) for word in tokens}
    overlap = len(prompt_words & candidate_words)
    components["prompt_overlap"] = min(1.0, 0.25 * overlap)

    components["rhyme_hint"] = rhyme_score(lines)

    total_words = len(tokens)
    components["total_length"] = 1.0 if 80 <= total_words <= 160 else 0.0

    total_score = sum(components.values())
    return total_score, components


@torch.no_grad()
def sample_sonnet(
    model: SonnetGPT,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    max_new_tokens: int,
) -> str:
    # Generate one continuation from the current sonnet model for a single prompt.
    encoding = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True)
    input_ids = encoding["input_ids"].to(model.device)
    token_ids, _ = model.generate(input_ids, max_length=max_new_tokens)
    return tokenizer.decode(token_ids[0].detach().cpu().tolist(), skip_special_tokens=True).strip()


def unique_candidates(
    model: SonnetGPT,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    num_candidates: int,
    max_new_tokens: int,
    max_attempts: int,
) -> List[str]:
    # Sample until we collect enough distinct candidates or hit the retry budget.
    seen = set()
    candidates: List[str] = []
    attempts = 0
    while len(candidates) < num_candidates and attempts < max_attempts:
        attempts += 1
        candidate = sample_sonnet(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        normalized = candidate.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            candidates.append(normalized)
    return candidates


def load_sonnet_model(checkpoint_path: Path, args) -> SonnetGPT:
    # Restore a trained sonnet checkpoint and configure its sampling settings.
    device = select_device()
    gpt = load_fine_tune_checkpoint(checkpoint_path, device, model=None)
    tokenizer = GPT2Tokenizer.from_pretrained(args.gpt_kind)
    model = SonnetGPT(gpt, tokenizer).to(device)
    model.eval()
    model.update_logit_processors(args.temperature, args.top_k, args.top_p)
    return model


def build_preferences(
    model: SonnetGPT,
    tokenizer: GPT2Tokenizer,
    prompt_sonnets: Sequence[str],
    prompt_lines: int,
    num_candidates: int,
    max_new_tokens: int,
) -> List[Dict[str, object]]:
    # Generate candidates per prompt, rank them, and keep the best-vs-worst pair.
    records: List[Dict[str, object]] = []
    for source_idx, source_sonnet in enumerate(tqdm(prompt_sonnets, desc="building-prefs")):
        prompt = prompt_from_sonnet(source_sonnet, prompt_lines=prompt_lines)
        if not prompt:
            continue

        candidates = unique_candidates(
            model,
            tokenizer,
            prompt=f"{prompt}\n",
            num_candidates=num_candidates,
            max_new_tokens=max_new_tokens,
            max_attempts=num_candidates * 4,
        )
        if len(candidates) < 2:
            continue

        scored = []
        for candidate in candidates:
            score, breakdown = score_sonnet(prompt, candidate)
            scored.append(
                {
                    "text": candidate,
                    "score": score,
                    "breakdown": breakdown,
                }
            )
        scored.sort(key=lambda item: item["score"], reverse=True)

        chosen = scored[0]
        rejected = scored[-1]
        if chosen["score"] <= rejected["score"]:
            continue

        records.append(
            {
                "prompt": prompt,
                "chosen": chosen["text"],
                "rejected": rejected["text"],
                "meta": {
                    "source_index": source_idx,
                    "prompt_lines": prompt_lines,
                    "chosen_score": chosen["score"],
                    "rejected_score": rejected["score"],
                    "chosen_breakdown": chosen["breakdown"],
                    "rejected_breakdown": rejected["breakdown"],
                    "candidate_count": len(scored),
                    "source_sonnet": source_sonnet,
                },
            }
        )
    return records


def write_jsonl(path: Path, records: Sequence[Dict[str, object]]) -> None:
    # Persist preference records in a line-delimited JSON format for DPO training.
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def parse_args() -> argparse.Namespace:
    # Expose the split, generation, and output settings as CLI flags.
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-sonnets", type=Path, default=DEFAULT_REAL_SONNETS)
    parser.add_argument("--sft-split-out", type=Path, default=DEFAULT_SFT_SPLIT)
    parser.add_argument("--prompt-split-out", type=Path, default=DEFAULT_PROMPT_SPLIT)
    parser.add_argument("--prefs-out", type=Path, default=DEFAULT_PREFS_JSONL)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--heldout-count", type=int, default=26)
    parser.add_argument("--prompt-lines", type=int, default=1)
    parser.add_argument("--num-candidates", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=140)
    parser.add_argument("--gpt-kind", type=str, default=SONNET_ARGS.gpt_kind)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--split-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    # Run the end-to-end split/build workflow from the command line.
    args = parse_args()
    set_seed(args.seed)

    sonnets = read_numbered_sonnets(args.real_sonnets)
    if len(sonnets) != 131:
        print(f"Found {len(sonnets)} real sonnets; default split still uses heldout_count={args.heldout_count}.")

    sft_split, prompt_split = split_real_sonnets(
        sonnets,
        heldout_count=args.heldout_count,
        seed=args.seed,
    )
    write_numbered_sonnets(args.sft_split_out, sft_split)
    write_numbered_sonnets(args.prompt_split_out, prompt_split)

    print(f"Wrote {len(sft_split)} SFT sonnets to {args.sft_split_out}")
    print(f"Wrote {len(prompt_split)} prompt-source sonnets to {args.prompt_split_out}")

    if args.split_only:
        return
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required unless --split-only is set.")

    model = load_sonnet_model(args.checkpoint, args)
    tokenizer = model.tokenizer
    prefs = build_preferences(
        model=model,
        tokenizer=tokenizer,
        prompt_sonnets=prompt_split,
        prompt_lines=args.prompt_lines,
        num_candidates=args.num_candidates,
        max_new_tokens=args.max_new_tokens,
    )
    write_jsonl(args.prefs_out, prefs)
    print(f"Wrote {len(prefs)} preference pairs to {args.prefs_out}")


if __name__ == "__main__":
    main()
