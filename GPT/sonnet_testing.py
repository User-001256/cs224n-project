import re
from pathlib import Path
from sacrebleu.metrics import CHRF


BASE_DIR = Path(__file__).resolve().parent

# File structure:
# data/
#   sonnets_held_out_dev.txt
#   TRUE_sonnets_held_out_dev.txt
# GPT/
#   sonnet_testing.py
#   predictions/
#     generated_sonnets_dev.txt

PRED_FILE = BASE_DIR / "predictions" / "generated_sonnets_dev_BEST.txt"
DEV_FILE = BASE_DIR.parent / "data" / "sonnets_held_out_dev.txt"
TRUE_FILE = BASE_DIR.parent / "data" / "TRUE_sonnets_held_out_dev.txt"
FINALIZE_MODE = "strip"


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def clean_line(line: str) -> str:
    return line.replace("\xa0", " ").strip()


def extract_numbered_sonnets(text: str):
    text = normalize_newlines(text)
    parts = re.split(r"^\s*(\d{1,3})\s*$", text, flags=re.MULTILINE)

    sonnets = {}
    for i in range(1, len(parts), 2):
        sonnet_num = int(parts[i])
        content = parts[i + 1]

        lines = []
        for line in content.splitlines():
            line = clean_line(line)
            if line:
                lines.append(line)

        sonnets[sonnet_num] = lines

    return sonnets


def load_dev_order(dev_path: Path):
    text = dev_path.read_text(encoding="utf-8", errors="replace")
    dev_sonnets = extract_numbered_sonnets(text)
    return list(dev_sonnets.keys())


def load_true_sonnets(true_path: Path):
    text = true_path.read_text(encoding="utf-8", errors="replace")
    return extract_numbered_sonnets(text)


def build_full_gold_sonnets(dev_order, true_sonnets):
    gold = []

    for sonnet_num in dev_order:
        if sonnet_num not in true_sonnets:
            raise ValueError(f"Sonnet {sonnet_num} found in dev set but not in true file.")

        lines = true_sonnets[sonnet_num]
        if len(lines) < 14:
            raise ValueError(
                f"Sonnet {sonnet_num} in true file has {len(lines)} lines, expected at least 14."
            )

        gold.append(finalize_sonnet_text("\n".join(lines[:14]), FINALIZE_MODE))

    return gold


def parse_generated_file(pred_path: Path):
    text = pred_path.read_text(encoding="utf-8", errors="replace")
    text = normalize_newlines(text)
    text = text.lstrip("\ufeff")

    text = re.sub(r"^\s*--Generated Sonnets--\s*", "", text, flags=re.IGNORECASE)
    parts = re.split(r"^\s*(\d+)\s*$", text, flags=re.MULTILINE)

    preds = {}
    for i in range(1, len(parts), 2):
        idx = int(parts[i])
        content = parts[i + 1]

        lines = []
        for line in content.splitlines():
            line = clean_line(line)
            if line:
                lines.append(line)

        preds[idx] = lines

    return preds


def extract_predicted_full_sonnets(dev_order, pred_blocks):
    preds = []
    missing = []

    for i in range(len(dev_order)):
        if i not in pred_blocks:
            missing.append(i)
            preds.append("")
            continue

        lines = pred_blocks[i][:14]
        preds.append(finalize_sonnet_text("\n".join(lines), FINALIZE_MODE))

    return preds, missing


def compute_chrf_score(pred_file, dev_file):
    """
    Compute CHRF score for a generated sonnet file.

    Args:
        pred_file: Path (or str) to the model's generated sonnets file.
        dev_file: Path (or str) to the dev file with partial sonnets
                  (used to determine ordering and to locate the TRUE_ file).
    """
    pred_path = Path(pred_file)
    dev_path = Path(dev_file)

    # Derive the corresponding TRUE_ file from the dev file name
    true_path = dev_path.with_name("TRUE_" + dev_path.name)

    dev_order = load_dev_order(dev_path)
    true_sonnets = load_true_sonnets(true_path)
    gold_full_sonnets = build_full_gold_sonnets(dev_order, true_sonnets)

    pred_blocks = parse_generated_file(pred_path)
    preds_full, missing = extract_predicted_full_sonnets(dev_order, pred_blocks)

    if missing:
        print(f"Warning: missing predictions for indices: {missing}")

    metric = CHRF(
        word_order=0,
        lowercase=False,
        whitespace=False,
        eps_smoothing=False,
    )
    return metric.corpus_score(preds_full, [gold_full_sonnets]).score

def finalize_sonnet_text(text: str, mode: str) -> str:
    if mode == "none":
        return text
    if mode == "strip":
        return text.strip()
    if mode == "collapse_spaces":
        return re.sub(r"[ \t]+", " ", text).strip()
    if mode == "strip_each_line":
        return "\n".join(line.strip() for line in text.split("\n")).strip()
    raise ValueError(f"Unknown finalize mode: {mode}")

def main():
    score = compute_chrf_score(PRED_FILE, DEV_FILE)
    print(f"Score: {score:.4f}")


if __name__ == "__main__":
    main()