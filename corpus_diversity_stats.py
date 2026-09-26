#!/usr/bin/env python
import math
import re
from pathlib import Path
from statistics import mean, stdev

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


OUT_DIR = Path(
    "/storage/ukp/work/boudabous/questionnaire2dialogue/translation_pipeline/evaluation"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Tokenizer: words and punctuation
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

# Extract a short ID from filenames
ID_PATTERN = re.compile(r"(notactive|active|control)\d+", re.IGNORECASE)


def short_id(filename: str) -> str:
    m = ID_PATTERN.search(filename)
    if m:
        return m.group(0).lower()
    return Path(filename).stem


def tokenize(text: str):
    """Lowercase and split into word and punctuation tokens."""
    return TOKEN_PATTERN.findall(text.lower())


def safe_std(values):
    """
    Compute sample standard deviation.
    Returns NaN if fewer than 2 valid values exist.
    """
    clean = [v for v in values if not math.isnan(v)]
    if len(clean) < 2:
        return float("nan")
    return stdev(clean)


def fmt(v, digits=4):
    """Format floats safely."""
    return "nan" if math.isnan(v) else f"{v:.{digits}f}"


def _mtld_direction(tokens, ttr_threshold=0.72):
    types = set()
    token_count = 0
    factor_count = 0.0

    for token in tokens:
        token_count += 1
        types.add(token)
        ttr = len(types) / float(token_count)

        if ttr < ttr_threshold:
            factor_count += 1.0
            types = set()
            token_count = 0

    # Handle leftover partial factor
    if token_count > 0:
        ttr = len(types) / float(token_count)
        factor_count += (1.0 - ttr) / (1.0 - ttr_threshold)

    if factor_count == 0:
        return float("nan")

    return len(tokens) / factor_count


def mtld(tokens, ttr_threshold=0.72, min_segment_length=10):
    """
    Bidirectional MTLD: average of forward and backward passes.
    """
    if len(tokens) < min_segment_length:
        return float("nan")

    forward = _mtld_direction(tokens, ttr_threshold)
    backward = _mtld_direction(list(reversed(tokens)), ttr_threshold)

    return (forward + backward) / 2.0


def self_bleu_per_doc(docs_tokens, max_ngram=4):
    """
    Compute Self-BLEU-4 for each document in a list of tokenized documents.
    """
    n = len(docs_tokens)
    if n < 2:
        return [float("nan")] * n, float("nan")

    smoothie = SmoothingFunction().method1
    weights = tuple(1.0 / max_ngram for _ in range(max_ngram))

    scores = []

    for i, hyp in enumerate(docs_tokens):
        refs = [
            docs_tokens[j]
            for j in range(n)
            if j != i and len(docs_tokens[j]) > 0
        ]

        if len(hyp) == 0 or len(refs) == 0:
            scores.append(float("nan"))
            continue

        score = sentence_bleu(
            refs,
            hyp,
            weights=weights,
            smoothing_function=smoothie,
        )
        scores.append(score)

    clean = [s for s in scores if not math.isnan(s)]
    mean_score = float(mean(clean)) if clean else float("nan")

    return scores, mean_score


def load_one_folder(corpus_dir: Path, max_files=None):
    files = sorted(corpus_dir.glob("*.txt"))

    if max_files is not None:
        files = files[:max_files]

    docs_text = []
    docs_tokens = []

    for path in files:
        try:
            text = path.read_text(encoding="utf-8").strip()
        except Exception as e:
            print(f"Could not read {path.name}: {e}")
            continue

        if not text:
            print(f"Empty file: {path.name}.")
            continue

        tokens = tokenize(text)

        if not tokens:
            print(f"No tokens after tokenization: {path.name}.")
            continue

        docs_text.append((path.name, text))
        docs_tokens.append(tokens)

    return docs_text, docs_tokens


def load_two_corpora(corpus_dir_1: Path, corpus_dir_2: Path, max_files=None):
    docs_text_1, docs_tokens_1 = load_one_folder(corpus_dir_1, max_files=max_files)
    docs_text_2, docs_tokens_2 = load_one_folder(corpus_dir_2, max_files=max_files)

    docs_text = docs_text_1 + docs_text_2
    docs_tokens = docs_tokens_1 + docs_tokens_2

    return docs_text, docs_tokens


def corpus_stats(docs_tokens):
    all_tokens = [tok for doc in docs_tokens for tok in doc]
    vocab = set(all_tokens)

    total_tokens = len(all_tokens)
    total_types = len(vocab)
    ttr = total_types / total_tokens if total_tokens > 0 else float("nan")

    return total_tokens, total_types, ttr


def save_per_dialogue_scores(
    run_name: str,
    filenames,
    docs_tokens_all,
    mtld_scores,
    selfbleu_scores,
):
    out_path = OUT_DIR / f"mtld_selfbleu_{run_name}.tsv"

    def fmt_tsv(v):
        return "" if math.isnan(v) else f"{v:.6f}"

    with out_path.open("w", encoding="utf-8") as f:
        f.write("filename\tn_tokens\tmtld\tself_bleu_0_1\n")

        for full_name, toks_all, m, sb in zip(
            filenames,
            docs_tokens_all,
            mtld_scores,
            selfbleu_scores,
        ):
            f.write(
                f"{short_id(full_name)}\t"
                f"{len(toks_all)}\t"
                f"{fmt_tsv(m)}\t"
                f"{fmt_tsv(sb)}\n"
            )

    print(f"\nPer-dialogue MTLD + Self-BLEU written to: {out_path}")


def main():
    # config
    corpus_1 = "/storage/ukp/work/boudabous/questionnaire2dialogue/translation_pipeline/arabic/synthetic_dialogue/gemma/mdd/dev"
    corpus_2 = "/storage/ukp/work/boudabous/questionnaire2dialogue/translation_pipeline/arabic/synthetic_dialogue/gemma/control/dev"

    # output file
    run_name = "arabic_gemma_dev"

    max_files = None
    

    corpus_dir_1 = Path(corpus_1)
    corpus_dir_2 = Path(corpus_2)

    if not corpus_dir_1.exists():
        raise SystemExit(f"Corpus directory does not exist: {corpus_dir_1}")

    if not corpus_dir_2.exists():
        raise SystemExit(f"Corpus directory does not exist: {corpus_dir_2}")

    print(f"Loading corpus 1 from: {corpus_dir_1}")
    print(f"Loading corpus 2 from: {corpus_dir_2}")

    docs_text, docs_tokens_all = load_two_corpora(
        corpus_dir_1,
        corpus_dir_2,
        max_files=max_files,
    )

    print(f"Loaded {len(docs_tokens_all)} non-empty dialogue files in total.\n")

    filenames = [name for (name, _) in docs_text]

    total_tokens, total_types, ttr = corpus_stats(docs_tokens_all)

    print("Basic corpus statistics (incl. punctuation)")
    print(f"Total tokens: {total_tokens}")
    print(f"Total types (vocabulary size): {total_types}")
    print(f"Type–Token Ratio (TTR): {ttr:.4f}\n")

    # Word-only tokens for computing MTLD and Self-BLEU
    docs_tokens_words = [
        [t for t in doc if re.match(r"^\w+$", t)]
        for doc in docs_tokens_all
    ]

    print("Computing MTLD per dialogue")
    mtld_scores = [mtld(doc) for doc in docs_tokens_words]
    mtld_clean = [s for s in mtld_scores if not math.isnan(s)]

    if mtld_clean:
        mtld_mean = mean(mtld_clean)
        mtld_std = safe_std(mtld_clean)

        print("MTLD summary")
        print(f"Documents with valid MTLD: {len(mtld_clean)}")
        print(f"Mean MTLD: {fmt(mtld_mean)}")
        print(f"Std  MTLD: {fmt(mtld_std)}")
        print(f"Min  MTLD: {min(mtld_clean):.4f}")
        print(f"Max  MTLD: {max(mtld_clean):.4f}\n")
    else:
        print("No valid MTLD scores.\n")

    print("Computing Self-BLEU-4")
    sb_scores, sb_mean = self_bleu_per_doc(docs_tokens_words, max_ngram=4)
    sb_clean = [s for s in sb_scores if not math.isnan(s)]

    if sb_clean:
        sb_std = safe_std(sb_clean)

        print("Self-BLEU summary")
        print(f"Documents used: {len(sb_clean)}")
        print(f"Mean Self-BLEU-4 (0–1): {fmt(sb_mean)}")
        print(f"Std  Self-BLEU-4 (0–1): {fmt(sb_std)}\n")
    else:
        print("Could not compute Self-BLEU.\n")

    save_per_dialogue_scores(
        run_name=run_name,
        filenames=filenames,
        docs_tokens_all=docs_tokens_all,
        mtld_scores=mtld_scores,
        selfbleu_scores=sb_scores,
    )


if __name__ == "__main__":
    main()