#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Iterable

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None


UTTERANCE_KEYS = {
    "text", "content", "utterance", "message", "response", "prompt"
}


def flatten_strings(obj: Any) -> list[str]:
   
    results: list[str] = []

    if obj is None:
        return results

    if isinstance(obj, str):
        s = obj.strip()
        if s:
            results.append(s)
        return results

    if isinstance(obj, list):
        for item in obj:
            results.extend(flatten_strings(item))
        return results

    if isinstance(obj, dict):
        # Prefer common utterance fields if present
        found_direct = False
        for k, v in obj.items():
            if k.lower() in UTTERANCE_KEYS and isinstance(v, str):
                s = v.strip()
                if s:
                    results.append(s)
                    found_direct = True

        # If no direct utterance key matched, recurse through all values
        if not found_direct:
            for v in obj.values():
                results.extend(flatten_strings(v))
        return results

    return results


def extract_utterances_from_txt(path: Path) -> list[str]:

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    tagged = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith("Therapist:") or s.startswith("Client:"):
            tagged.append(s)

    if tagged:
        return tagged

    return [line.strip() for line in lines if line.strip()]


def extract_utterances_from_json(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return flatten_strings(data)


def extract_utterances_from_jsonl(path: Path) -> list[str]:
    utterances: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            utterances.extend(flatten_strings(obj))
    return utterances


def count_tokens(texts: Iterable[str], tokenizer=None) -> int:
    if tokenizer is None:
        return sum(len(t.split()) for t in texts)
    total = 0
    for t in texts:
        total += len(tokenizer.encode(t, add_special_tokens=False))
    return total


def process_file(path: Path, tokenizer=None) -> tuple[int, int]:
    suffix = path.suffix.lower()

    if suffix == ".json":
        utterances = extract_utterances_from_json(path)
    elif suffix == ".jsonl":
        utterances = extract_utterances_from_jsonl(path)
    elif suffix == ".txt":
        utterances = extract_utterances_from_txt(path)
    else:
        return 0, 0

    num_utterances = len(utterances)
    num_tokens = count_tokens(utterances, tokenizer=tokenizer)
    return num_utterances, num_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Path to translation_pipeline"
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["arabic", "german", "french"],
        help="Language folders to process"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemma",
        help="Model folder inside synthetic_dialogue"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Optional HF tokenizer path for model-token counting"
    )
    args = parser.parse_args()

    tokenizer = None
    if args.tokenizer_path:
        if AutoTokenizer is None:
            raise RuntimeError("transformers is not available, but --tokenizer_path was provided.")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    root = Path(args.root)

    print("=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)

    for lang in args.languages:
        lang_dir = root / lang / "synthetic_dialogue" / args.model
        if not lang_dir.exists():
            print(f"Missing folder: {lang_dir}")
            continue

        files = []
        for ext in ("*.json", "*.jsonl", "*.txt"):
            files.extend(lang_dir.rglob(ext))

        total_dialogues = 0
        total_utterances = 0
        total_tokens = 0

        for fp in sorted(files):
            n_utts, n_toks = process_file(fp, tokenizer=tokenizer)
            if n_utts == 0:
                continue
            total_dialogues += 1
            total_utterances += n_utts
            total_tokens += n_toks

        avg_utterances = (
            total_utterances / total_dialogues if total_dialogues > 0 else 0.0
        )

        print(f"\nLanguage: {lang}")
        print(f"  Dialogues              : {total_dialogues}")
        print(f"  Total utterances       : {total_utterances}")
        print(f"  Average utterances/dialogue : {avg_utterances:.3f}")
        print(f"  Total tokens           : {total_tokens}")


if __name__ == "__main__":
    main()
