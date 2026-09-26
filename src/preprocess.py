import os
import re
import json
from datasets import Dataset, DatasetDict



LANGUAGES = ["ar", "de", "fr"]

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

FOLDERS = {
    "translation_pipeline/french/synthetic_dialogue/gemma":  ("Thérapeute:", "Client:", "fr"),
    "translation_pipeline/arabic/synthetic_dialogue/gemma":  ("المعالج:",    "العميل:", "ar"),
    "translation_pipeline/german/synthetic_dialogue/gemma":  ("Therapeut:",  "Klient:", "de"),
}

CONTEXT_TURNS = 6

OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "processed_dataset")


SYSTEM_PROMPTS = {
    "ar": "أنت معالج نفسي متعاطف ومحترف. استمع بعناية إلى العميل وقدم ردودًا داعمة وعلاجية.",
    "fr": "Vous êtes un thérapeute empathique et professionnel. Écoutez attentivement le client et fournissez des réponses de soutien thérapeutique.",
    "de": "Sie sind ein einfühlsamer und professioneller Therapeut. Hören Sie dem Klienten aufmerksam zu und geben Sie unterstützende therapeutische Antworten.",
}


def parse_file(filepath, therapist_label, client_label, lang):
    """
    Parse a dialogue .txt file and produce multiple training samples,
    each with a sliding context window of multi-turn chat messages.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read().strip()


    pattern = rf'(?=(?:{re.escape(therapist_label)}|{re.escape(client_label)}))'
    turns = [t.strip() for t in re.split(pattern, raw) if t.strip()]

    samples = []
    system_msg = {"role": "system", "content": SYSTEM_PROMPTS[lang]}

    for i, turn in enumerate(turns):
        if not turn.startswith(therapist_label):
            continue

        # Extract therapist response and handle [END] markers
        therapist_response = turn.replace(therapist_label, "").strip()
        therapist_response = re.sub(r"\[/?END\]", "", therapist_response).strip()

        if not therapist_response:
            continue

        # Build context as proper multi-turn messages 
        context_turns = turns[max(0, i - CONTEXT_TURNS):i]

        messages = [system_msg]

        for ct in context_turns:
            if ct.startswith(therapist_label):
                content = ct.replace(therapist_label, "").strip()
                content = re.sub(r"\[/?END\]", "", content).strip()
                if content:
                    messages.append({"role": "assistant", "content": content})
            elif ct.startswith(client_label):
                content = ct.replace(client_label, "").strip()
                if content:
                    messages.append({"role": "user", "content": content})

        messages.append({"role": "assistant", "content": therapist_response})

        samples.append({
            "messages": json.dumps(messages, ensure_ascii=False),
            "lang": lang,
        })

    return samples


def get_split(filepath):
    normalized = filepath.replace("\\", "/")
    if "/train/" in normalized:
        return "train"
    if "/dev/" in normalized:
        return "dev"
    if "/test/" in normalized:
        return "test"
    return "train"


def build_dataset():
    # Collect samples per language and per split
    lang_splits = {
        lang: {"train": [], "dev": []}
        for lang in LANGUAGES
    }

    for folder, (therapist_label, client_label, lang) in FOLDERS.items():
        folder_path = os.path.join(PROJECT_ROOT, folder)

        if lang not in LANGUAGES:
            print(f"  [{lang.upper()}] Skipped (not in LANGUAGES)")
            continue

        if not os.path.exists(folder_path):
            print(f"  [{lang.upper()}] Folder not found, skipping: {folder_path}")
            continue

        counts = {
            "mdd/train": 0,
            "mdd/dev": 0,
            "control/train": 0,
            "control/dev": 0,
            "test_skipped": 0,
        }

        for dirpath, _, filenames in os.walk(folder_path):
            for fname in filenames:
                if not fname.endswith(".txt"):
                    continue

                fpath = os.path.join(dirpath, fname)
                split = get_split(fpath)
                norm = fpath.replace("\\", "/")

                if split == "test":
                    counts["test_skipped"] += 1
                    continue

                samples = parse_file(fpath, therapist_label, client_label, lang)
                subset = "mdd" if "/mdd/" in norm else "control"
                counts[f"{subset}/{split}"] += len(samples)

                if split == "train":
                    lang_splits[lang]["train"].extend(samples)
                elif split == "dev":
                    lang_splits[lang]["dev"].extend(samples)

        print(
            f"  [{lang.upper()}] "
            f"mdd/train={counts['mdd/train']} | "
            f"mdd/dev={counts['mdd/dev']} | "
            f"control/train={counts['control/train']} | "
            f"control/dev={counts['control/dev']} | "
            f"test skipped={counts['test_skipped']}"
        )

    for lang in LANGUAGES:
        train_samples = lang_splits[lang]["train"]
        dev_samples = lang_splits[lang]["dev"]

        if not train_samples:
            print(f"\n  [{lang.upper()}] No training samples found, skipping.")
            continue

        lang_output = os.path.join(OUTPUT_ROOT, lang)

        splits = {"train": Dataset.from_list(train_samples)}
        if dev_samples:
            splits["dev"] = Dataset.from_list(dev_samples)
        else:
            print(f"  [{lang.upper()}] Warning: no dev samples, saving train only.")

        dataset_dict = DatasetDict(splits)
        dataset_dict.save_to_disk(lang_output)

        print(f"\n  [{lang.upper()}] Dataset saved to: {lang_output}")
        print(f"    Train samples : {len(splits['train'])}")
        if dev_samples:
            print(f"    Dev samples   : {len(splits['dev'])}")

        # Preview one sample
        sample_messages = json.loads(train_samples[0]["messages"])
        print(f"    Sample preview ({train_samples[0]['lang']}):")
        for msg in sample_messages[:4]:
            preview = msg["content"][:80] + "..." if len(msg["content"]) > 80 else msg["content"]
            print(f"      [{msg['role']}] {preview}")

    print(f"\n  All datasets saved under: {OUTPUT_ROOT}")
    print(f"    processed_dataset/ar/")
    print(f"    processed_dataset/fr/")
    print(f"    processed_dataset/de/")


if __name__ == "__main__":
    build_dataset()
