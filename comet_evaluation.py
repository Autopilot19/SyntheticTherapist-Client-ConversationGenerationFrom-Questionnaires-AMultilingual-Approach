#!/usr/bin/env python
import os
from pathlib import Path
from itertools import zip_longest
from statistics import mean
import torch
from comet import download_model, load_from_checkpoint

SRC_EN_DIR = Path(
    "/storage/ukp/work/boudabous/questionnaire2dialogue/corpora/gemma_en/test"
)
SRC_DE_DIR = Path(
    "/storage/ukp/work/boudabous/questionnaire2dialogue/translation_pipeline/corpora/translated_german/sentence_level/test"
)

OUT_DIR = Path(
    "/storage/ukp/work/boudabous/questionnaire2dialogue/translation_pipeline/evaluation"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault(
    "COMET_CACHE",
    "/storage/ukp/work/boudabous/.cache/comet"
)

MODEL_NAME = "Unbabel/wmt22-cometkiwi-da"


def collect_pairs():
    en_files = sorted(SRC_EN_DIR.glob("*.txt"))
    print(f"Found {len(en_files)} English files in {SRC_EN_DIR}")

    pairs = []
    for en_path in en_files:
        de_path = SRC_DE_DIR / en_path.name
        if not de_path.exists():
            print(f"Missing french translation for {en_path.name}.")
            continue

        try:
            src_text = en_path.read_text(encoding="utf-8").strip()
            mt_text = de_path.read_text(encoding="utf-8").strip()
        except Exception as e:
            print(f"Could not read {en_path.name} or its translation: {e}")
            continue

        if not src_text or not mt_text:
            print(f"Empty file in pair {en_path.name}.")
            continue

        pairs.append(
            {
                "basename": en_path.name,
                "src": src_text,
                "mt": mt_text,
            }
        )

    print(f"Using {len(pairs)} aligned dialogue pairs for COMET.")
    return pairs


def _chunk_pair(src_text: str, mt_text: str, tokenizer, max_positions: int):
    """
    Split the (src, mt) dialogue into multiple chunks so COMET reads the full text
    without truncation. Keeps tags unchanged.
    """
    chunk_max = max(64, max_positions - 50)

    src_lines = src_text.splitlines()
    mt_lines = mt_text.splitlines()

    chunks = []
    cur_src = ""
    cur_mt = ""

    for s_line, m_line in zip_longest(src_lines, mt_lines, fillvalue=""):
        s_line = s_line.rstrip("\n")
        m_line = m_line.rstrip("\n")

        cand_src = (cur_src + "\n" + s_line).strip() if cur_src else s_line.strip()
        cand_mt = (cur_mt + "\n" + m_line).strip() if cur_mt else m_line.strip()

        pair_len = len(tokenizer(cand_src, cand_mt, truncation=False)["input_ids"])

        if pair_len <= chunk_max:
            cur_src = cand_src
            cur_mt = cand_mt
        else:
            if cur_src or cur_mt:
                chunks.append({"src": cur_src, "mt": cur_mt})
                cur_src, cur_mt = "", ""

            # start new chunk with this line pair
            cand_src = s_line.strip()
            cand_mt = m_line.strip()
            chunks.append({"src": cand_src, "mt": cand_mt})

    if cur_src or cur_mt:
        chunks.append({"src": cur_src, "mt": cur_mt})

    return chunks


def run_comet(pairs):
    """Run COMET QE (reference-free) on src–mt pairs."""
    if "HUGGINGFACE_HUB_TOKEN" not in os.environ:
        print(
            "HUGGINGFACE_HUB_TOKEN is not set. "
        )

    print(f"Downloading/loading COMET model: {MODEL_NAME}")
    model_path = download_model(MODEL_NAME)
    model = load_from_checkpoint(model_path)

    tokenizer = model.encoder.tokenizer
    max_positions = getattr(model.encoder, "max_positions", 512)
    print(f"max_positions: {max_positions}")

    data = []
    doc_meta = []  # (basename, start_idx, end_idx)
    for p in pairs:
        start = len(data)
        chunks = _chunk_pair(p["src"], p["mt"], tokenizer, max_positions)
        data.extend(chunks)
        end = len(data)
        doc_meta.append((p["basename"], start, end))

    model.cuda()
    model_output = model.predict(
        data,
        batch_size=8,
        gpus=1,
        progress_bar=True,
    )


    chunk_scores = [float(s) for s in model_output.scores]

    seg_scores = []
    for (_basename, start, end) in doc_meta:
        scores = chunk_scores[start:end]
        seg_scores.append(mean(scores) if scores else float("nan"))

    sys_score = mean(seg_scores) if seg_scores else float("nan")
    return seg_scores, sys_score


def save_scores(pairs, seg_scores, sys_score):
    """Save per-dialogue scores and print summary."""
    out_tsv = OUT_DIR / "comet_evaluation_german.tsv"

    with out_tsv.open("w", encoding="utf-8") as f:
        f.write("filename\tcomet_evaluation_german.tsv\n")
        for pair, score in zip(pairs, seg_scores):
            f.write(f"{pair['basename']}\t{score:.6f}\n")

    print(f"\nSystem-level COMET score: {sys_score:.4f}")
    print(f"Per-dialogue scores written to: {out_tsv}")


def main():
    pairs = collect_pairs()
    if not pairs:
        print("No aligned pairs found.")
        return

    seg_scores, sys_score = run_comet(pairs)
    save_scores(pairs, seg_scores, sys_score)


if __name__ == "__main__":
    main()
