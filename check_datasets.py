import argparse
import json
import random
from datasets import load_from_disk
import os


def inspect(lang, num_samples, base_dir="./processed_dataset"):
    path = os.path.join(base_dir, lang)
    if not os.path.exists(path):
        print(f"  [{lang.upper()}] Not found at {path}, skipping.\n")
        return

    dataset = load_from_disk(path)

    print(f"{'='*70}")
    print(f"  [{lang.upper()}] Loaded from: {path}")
    print(f"{'='*70}")
    print(f"  Splits   : {list(dataset.keys())}")
    for split_name, split_data in dataset.items():
        print(f"  {split_name:8s} : {len(split_data)} samples")
    print(f"  Columns  : {dataset['train'].column_names}")
    print()

    # Pick random samples from train
    train = dataset["train"]
    indices = random.sample(range(len(train)), min(num_samples, len(train)))

    for idx in indices:
        sample = train[idx]
        messages = json.loads(sample["messages"])

        print(f"  ── Sample index={idx} | lang={sample['lang']} ──")
        print(f"  Num messages: {len(messages)}")
        print(f"  Roles: {[m['role'] for m in messages]}")
        print()

        for j, msg in enumerate(messages):
            content = msg["content"]
            if len(content) > 150:
                content = content[:150] + "..."
            print(f"    [{msg['role']:>9s}] {content}")

        print()
        print(f"  {'─'*60}")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default=None, choices=["ar", "fr", "de"],
                        help="Inspect one language only (default: all)")
    parser.add_argument("--num", type=int, default=3, help="Number of random samples to show (default: 3)")
    parser.add_argument("--base_dir", type=str, default="./processed_dataset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    langs = [args.lang] if args.lang else ["ar", "fr", "de"]
    for lang in langs:
        inspect(lang, args.num, args.base_dir)


if __name__ == "__main__":
    main()