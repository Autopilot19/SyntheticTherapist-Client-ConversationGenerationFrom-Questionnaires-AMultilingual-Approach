import os
import re
import json
import time
import argparse
from typing import Any, Dict, List, Tuple, Optional

import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


BASE_MODEL_DEFAULT = "meta-llama/Meta-Llama-3.1-8B-Instruct"



# Task definitions

TASKS = {
    2: {
        "task_name": "task2_distortions",
        "gold_fields": ["distortions"],
        "letters": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        "labels_en": [
            "all-or-nothing thinking",
            "overgeneralization",
            "mental filter",
            "should statements",
            "labeling",
            "personalization",
            "magnification",
            "emotional reasoning",
            "mind reading",
            "fortune-telling",
        ],
    },
    3: {
        "task_name": "task3_core_major",
        "gold_fields": ["core_belief_major", "core_belief_major_grained"],
        "letters": ["a", "b", "c"],
        "labels_en": [
            "helpless",
            "unlovable",
            "worthless",
        ],
    },
    4: {
        "task_name": "task4_core_fine",
        "gold_fields": ["core_belief_fine_grained"],
        "letters": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "u"],
        "labels_en": [
            "I am incompetent",
            "I am helpless",
            "I am powerless, weak, vulnerable",
            "I am a victim",
            "I am needy",
            "I am trapped",
            "I am out of control",
            "I am a failure, loser",
            "I am defective",
            "I am unlovable",
            "I am unattractive",
            "I am undesirable, unwanted",
            "I am bound to be rejected",
            "I am bound to be abandoned",
            "I am bound to be alone",
            "I am worthless, waste",
            "I am immoral",
            "I am bad - dangerous, toxic, evil",
            "I don't deserve to live",
        ],
    },
}


DISPLAY_LABELS = {
    2: {
        "en": [
            "all-or-nothing thinking",
            "overgeneralization",
            "mental filter",
            "should statements",
            "labeling",
            "personalization",
            "magnification",
            "emotional reasoning",
            "mind reading",
            "fortune-telling",
        ],
        "fr": [
            "pensée tout ou rien",
            "surgénéralisation",
            "filtre mental",
            "affirmations du type « je dois »",
            "étiquetage",
            "personnalisation",
            "amplification",
            "raisonnement émotionnel",
            "lecture de pensée",
            "prédiction de l’avenir",
        ],
        "de": [
            "Alles-oder-nichts-Denken",
            "Übergeneralisierung",
            "mentaler Filter",
            "Sollte-Aussagen",
            "Etikettierung",
            "Personalisierung",
            "Vergrößerung",
            "emotionales Schlussfolgern",
            "Gedankenlesen",
            "Zukunftsdeutung",
        ],
        "ar": [
            "التفكير الكلّي أو لا شيء",
            "التعميم المفرط",
            "الفلتر الذهني",
            "عبارات «يجب»",
            "إطلاق التسميات",
            "التخصيص الشخصي",
            "التهويل",
            "الاستدلال العاطفي",
            "قراءة الأفكار",
            "التنبؤ بالمستقبل",
        ],
    },
    3: {
        "en": ["helpless", "unlovable", "worthless"],
        "fr": ["impuissant", "impossible à aimer", "sans valeur"],
        "de": ["hilflos", "nicht liebenswert", "wertlos"],
        "ar": ["عاجز", "غير جدير بالحب", "عديم القيمة"],
    },
    4: {
        "en": [
            "I am incompetent",
            "I am helpless",
            "I am powerless, weak, vulnerable",
            "I am a victim",
            "I am needy",
            "I am trapped",
            "I am out of control",
            "I am a failure, loser",
            "I am defective",
            "I am unlovable",
            "I am unattractive",
            "I am undesirable, unwanted",
            "I am bound to be rejected",
            "I am bound to be abandoned",
            "I am bound to be alone",
            "I am worthless, waste",
            "I am immoral",
            "I am bad - dangerous, toxic, evil",
            "I don't deserve to live",
        ],
        "fr": [
            "je suis incompétent",
            "je suis impuissant",
            "je suis sans pouvoir, faible, vulnérable",
            "je suis une victime",
            "je suis dans le besoin",
            "je suis piégé",
            "je suis hors de contrôle",
            "je suis un échec, un perdant",
            "je suis défectueux",
            "je suis impossible à aimer",
            "je suis peu attirant",
            "je suis indésirable, non désiré",
            "je vais être rejeté",
            "je vais être abandonné",
            "je vais être seul",
            "je suis sans valeur, un déchet",
            "je suis immoral",
            "je suis mauvais — dangereux, toxique, malveillant",
            "je ne mérite pas de vivre",
        ],
        "de": [
            "ich bin unfähig",
            "ich bin hilflos",
            "ich bin machtlos, schwach, verletzlich",
            "ich bin ein Opfer",
            "ich bin bedürftig",
            "ich bin gefangen",
            "ich bin außer Kontrolle",
            "ich bin ein Versager, Verlierer",
            "ich bin fehlerhaft",
            "ich bin nicht liebenswert",
            "ich bin unattraktiv",
            "ich bin unerwünscht, ungewollt",
            "ich werde sicher zurückgewiesen",
            "ich werde sicher verlassen",
            "ich werde sicher allein sein",
            "ich bin wertlos, Abfall",
            "ich bin unmoralisch",
            "ich bin schlecht — gefährlich, toxisch, böse",
            "ich verdiene es nicht zu leben",
        ],
        "ar": [
            "أنا غير كفء",
            "أنا عاجز",
            "أنا بلا قوة، ضعيف، هش",
            "أنا ضحية",
            "أنا محتاج",
            "أنا محاصر",
            "أنا خارج السيطرة",
            "أنا فاشل، خاسر",
            "أنا معيب",
            "أنا غير جدير بالحب",
            "أنا غير جذاب",
            "أنا غير مرغوب فيه، غير مطلوب",
            "سوف أُرفَض حتمًا",
            "سوف أُهجَر حتمًا",
            "سوف أبقى وحيدًا",
            "أنا عديم القيمة، نفاية",
            "أنا غير أخلاقي",
            "أنا سيئ — خطير، سام، شرير",
            "أنا لا أستحق أن أعيش",
        ],
    },
}


SYSTEM_PROMPTS = {
    "en": "You are a helpful assistant.",
    "fr": "Vous êtes un assistant utile.",
    "de": "Sie sind ein hilfreicher Assistent.",
    "ar": "أنت مساعد مفيد.",
}



# Utility


def normalize_text(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("’", "'").replace("`", "'").replace("“", '"').replace("”", '"')
    s = s.replace("—", "-").replace("–", "-")
    s = re.sub(r"\s+", " ", s)
    return s


def shorten(text: str, n: int = 300) -> str:
    text = str(text).replace("\n", "\\n")
    return text if len(text) <= n else text[:n] + "...[TRUNCATED]"


def print_block(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def ensure_dir_for_file(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def ensure_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        if "," in value:
            return [x.strip() for x in value.split(",") if x.strip()]
        return [value]
    return [str(value)]


def get_model_device(model) -> torch.device:
    return next(model.parameters()).device


def get_active_adapter_name(model) -> str:
    try:
        if hasattr(model, "active_adapter"):
            return str(model.active_adapter)
        if hasattr(model, "active_adapters"):
            return str(model.active_adapters)
    except Exception:
        pass
    return "base_or_unknown"



# Gold-label normalization


def build_alias_map(task_id: int) -> Dict[str, str]:
    alias_to_canonical = {}
    canonical = TASKS[task_id]["labels_en"]

    for i, canon in enumerate(canonical):
        alias_to_canonical[normalize_text(canon)] = canon
        for lang in ["fr", "de", "ar"]:
            alias_to_canonical[normalize_text(DISPLAY_LABELS[task_id][lang][i])] = canon

    return alias_to_canonical


def get_gold_labels(item: Dict[str, Any], task_id: int) -> List[str]:
    alias_map = build_alias_map(task_id)

    raw_values = []
    for key in TASKS[task_id]["gold_fields"]:
        if key in item and item[key] is not None:
            raw_values = ensure_list(item[key])
            if raw_values:
                break

    normalized = []
    for v in raw_values:
        canon = alias_map.get(normalize_text(v))
        if canon is not None:
            normalized.append(canon)

    return sorted(set(normalized))



# Prompt construction


def build_choices_text(task_id: int, language: str) -> str:
    letters = TASKS[task_id]["letters"]
    labels = DISPLAY_LABELS[task_id][language]
    return "\n".join(f"{letter}: {label}" for letter, label in zip(letters, labels))


def build_user_prompt(task_id: int, language: str, situation: str, thoughts: str) -> str:
    choices = build_choices_text(task_id, language)

    if task_id == 2:
        templates = {
            "en": (
                "You are acting as a CBT therapist. Based on the patient's current situation and thoughts, "
                "identify the cognitive distortions. A patient can have up to 3 distortions.\n\n"
                "Situation: {situation}\n\n"
                "Thoughts: {thoughts}\n\n"
                "Question: Which distortions are present?\n\n"
                "Choices:\n{choices}\n\n"
                "Answer with letters only. If there is more than one answer, separate them with commas."
            ),
            "fr": (
                "Vous agissez comme un thérapeute TCC. À partir de la situation actuelle du patient et de ses pensées, "
                "identifiez les distorsions cognitives. Un patient peut avoir jusqu’à 3 distorsions.\n\n"
                "Situation : {situation}\n\n"
                "Pensées : {thoughts}\n\n"
                "Question : Quelles distorsions sont présentes ?\n\n"
                "Choix :\n{choices}\n\n"
                "Répondez uniquement avec les lettres. S’il y a plusieurs réponses, séparez-les par des virgules."
            ),
            "de": (
                "Sie handeln als KVT-Therapeut. Bestimmen Sie anhand der aktuellen Situation und der Gedanken des Patienten "
                "die kognitiven Verzerrungen. Ein Patient kann bis zu 3 Verzerrungen haben.\n\n"
                "Situation: {situation}\n\n"
                "Gedanken: {thoughts}\n\n"
                "Frage: Welche Verzerrungen liegen vor?\n\n"
                "Auswahl:\n{choices}\n\n"
                "Antworten Sie nur mit Buchstaben. Wenn es mehrere Antworten gibt, trennen Sie sie durch Kommas."
            ),
            "ar": (
                "أنت تعمل كمعالج بالعلاج المعرفي السلوكي. اعتمادًا على الوضع الحالي للمريض وأفكاره، "
                "حدّد التشوهات المعرفية. قد يكون لدى المريض حتى 3 تشوهات.\n\n"
                "الموقف: {situation}\n\n"
                "الأفكار: {thoughts}\n\n"
                "السؤال: ما التشوهات الموجودة؟\n\n"
                "الخيارات:\n{choices}\n\n"
                "أجب بالحروف فقط. وإذا وُجد أكثر من جواب، افصل بينها بفواصل."
            ),
        }
    elif task_id == 3:
        templates = {
            "en": (
                "You are acting as a CBT therapist. Based on the patient's current situation and thoughts, "
                "identify the major core beliefs. A patient may have multiple core beliefs.\n\n"
                "Situation: {situation}\n\n"
                "Thoughts: {thoughts}\n\n"
                "Question: Which major core beliefs are present?\n\n"
                "{choices}\n\n"
                "Answer with letters only. If there is more than one answer, separate them with commas."
            ),
            "fr": (
                "Vous agissez comme un thérapeute TCC. À partir de la situation actuelle du patient et de ses pensées, "
                "identifiez les croyances centrales majeures. Un patient peut avoir plusieurs croyances centrales.\n\n"
                "Situation : {situation}\n\n"
                "Pensées : {thoughts}\n\n"
                "Question : Quelles croyances centrales majeures sont présentes ?\n\n"
                "{choices}\n\n"
                "Répondez uniquement avec les lettres. S’il y a plusieurs réponses, séparez-les par des virgules."
            ),
            "de": (
                "Sie handeln als KVT-Therapeut. Bestimmen Sie anhand der aktuellen Situation und der Gedanken des Patienten "
                "die übergeordneten Grundüberzeugungen. Ein Patient kann mehrere Grundüberzeugungen haben.\n\n"
                "Situation: {situation}\n\n"
                "Gedanken: {thoughts}\n\n"
                "Frage: Welche übergeordneten Grundüberzeugungen liegen vor?\n\n"
                "{choices}\n\n"
                "Antworten Sie nur mit Buchstaben. Wenn es mehrere Antworten gibt, trennen Sie sie durch Kommas."
            ),
            "ar": (
                "أنت تعمل كمعالج بالعلاج المعرفي السلوكي. اعتمادًا على الوضع الحالي للمريض وأفكاره، "
                "حدّد المعتقدات الجوهرية الكبرى. قد يكون لدى المريض عدة معتقدات جوهرية.\n\n"
                "الموقف: {situation}\n\n"
                "الأفكار: {thoughts}\n\n"
                "السؤال: ما المعتقدات الجوهرية الكبرى الموجودة؟\n\n"
                "{choices}\n\n"
                "أجب بالحروف فقط. وإذا وُجد أكثر من جواب، افصل بينها بفواصل."
            ),
        }
    else:
        templates = {
            "en": (
                "You are acting as a CBT therapist. Based on the patient's current situation and thoughts, "
                "identify the fine-grained beliefs. A patient can have up to 9 fine-grained beliefs.\n\n"
                "Situation: {situation}\n\n"
                "Thoughts: {thoughts}\n\n"
                "Question: Which fine-grained beliefs are present?\n\n"
                "Choices:\n{choices}\n\n"
                "Answer with letters only. If there is more than one answer, separate them with commas."
            ),
            "fr": (
                "Vous agissez comme un thérapeute TCC. À partir de la situation actuelle du patient et de ses pensées, "
                "identifiez les croyances fines. Un patient peut avoir jusqu’à 9 croyances fines.\n\n"
                "Situation : {situation}\n\n"
                "Pensées : {thoughts}\n\n"
                "Question : Quelles croyances fines sont présentes ?\n\n"
                "Choix :\n{choices}\n\n"
                "Répondez uniquement avec les lettres. S’il y a plusieurs réponses, séparez-les par des virgules."
            ),
            "de": (
                "Sie handeln als KVT-Therapeut. Bestimmen Sie anhand der aktuellen Situation und der Gedanken des Patienten "
                "die feingranularen Überzeugungen. Ein Patient kann bis zu 9 feingranulare Überzeugungen haben.\n\n"
                "Situation: {situation}\n\n"
                "Gedanken: {thoughts}\n\n"
                "Frage: Welche feingranularen Überzeugungen liegen vor?\n\n"
                "Auswahl:\n{choices}\n\n"
                "Antworten Sie nur mit Buchstaben. Wenn es mehrere Antworten gibt, trennen Sie sie durch Kommas."
            ),
            "ar": (
                "أنت تعمل كمعالج بالعلاج المعرفي السلوكي. اعتمادًا على الوضع الحالي للمريض وأفكاره، "
                "حدّد المعتقدات الدقيقة. قد يكون لدى المريض حتى 9 معتقدات دقيقة.\n\n"
                "الموقف: {situation}\n\n"
                "الأفكار: {thoughts}\n\n"
                "السؤال: ما المعتقدات الدقيقة الموجودة؟\n\n"
                "الخيارات:\n{choices}\n\n"
                "أجب بالحروف فقط. وإذا وُجد أكثر من جواب، افصل بينها بفواصل."
            ),
        }

    return templates[language].format(
        situation=situation,
        thoughts=thoughts,
        choices=choices,
    )



# Model loading


def load_model_and_tokenizer(
    adapter_path: str,
    base_model: str,
    load_in_4bit: bool,
    use_lora: bool,
):
    print_block("MODEL LOADING")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    print(f"Base model     : {base_model}")
    print(f"Use LoRA       : {use_lora}")
    print(f"Adapter path   : {adapter_path}")
    print(f"Load in 4-bit  : {load_in_4bit}")

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        quantization_config=quant_config,
    )

    if use_lora:
        if not os.path.isdir(adapter_path):
            raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")
        if not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
            raise FileNotFoundError(f"Missing adapter_config.json in: {adapter_path}")
        if not os.path.exists(os.path.join(adapter_path, "adapter_model.safetensors")):
            raise FileNotFoundError(f"Missing adapter_model.safetensors in: {adapter_path}")
        model = PeftModel.from_pretrained(base, adapter_path)
    else:
        model = base

    model.eval()

    print_block("MODEL DEBUG INFO")
    print(f"Model type             : {type(model)}")
    print(f"Active adapter         : {get_active_adapter_name(model)}")
    print(f"First parameter device : {get_model_device(model)}")
    print(f"HF device map          : {getattr(model, 'hf_device_map', None)}")
    print(f"CUDA available         : {torch.cuda.is_available()}")
    print(f"CUDA device count      : {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            except Exception:
                pass

    return model, tokenizer



# Generation


@torch.inference_mode()
def generate_letters(
    model,
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    max_input_tokens: int,
    max_new_tokens: int,
    use_system_role: bool,
) -> Tuple[str, Dict[str, Any]]:
    if use_system_role:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    else:
        messages = [
            {"role": "user", "content": user_prompt},
        ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    tokenized = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )

    prompt_tokens = int(tokenized["input_ids"].shape[1])
    device = get_model_device(model)
    tokenized = {k: v.to(device) for k, v in tokenized.items()}

    start = time.perf_counter()
    outputs = model.generate(
        **tokenized,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    elapsed_sec = time.perf_counter() - start

    gen_ids = outputs[0][tokenized["input_ids"].shape[1]:]
    generated_tokens = int(gen_ids.shape[0])
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    meta = {
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "elapsed_sec": elapsed_sec,
    }
    return text, meta


def run_smoke_test(
    model,
    tokenizer,
    language: str,
    max_input_tokens: int,
    use_system_role: bool,
):
    smoke_prompts = {
        "en": "Reply with exactly this letter and nothing else: a",
        "fr": "Réponds exactement avec la lettre suivante et rien d'autre : a",
        "de": "Antworte genau mit folgendem Buchstaben und nichts anderem: a",
        "ar": "أجب بالحرف التالي فقط دون أي شيء آخر: a",
    }

    raw_output, meta = generate_letters(
        model=model,
        tokenizer=tokenizer,
        system_prompt=SYSTEM_PROMPTS[language],
        user_prompt=smoke_prompts[language],
        max_input_tokens=max_input_tokens,
        max_new_tokens=8,
        use_system_role=use_system_role,
    )

    print_block("SMOKE TEST")
    print(f"Prompt tokens    : {meta['prompt_tokens']}")
    print(f"Generated tokens : {meta['generated_tokens']}")
    print(f"Elapsed sec      : {meta['elapsed_sec']:.2f}")
    print(f"Raw output       : {raw_output}")



# Prediction parsing


def parse_predicted_letters(text: str, allowed_letters: List[str]) -> Tuple[List[str], bool]:
    cleaned = text.lower()
    cleaned = cleaned.replace("\n", ",")
    cleaned = cleaned.replace(";", ",")
    cleaned = cleaned.replace("，", ",")
    cleaned = cleaned.replace("answer:", "")
    cleaned = cleaned.replace("réponse:", "")
    cleaned = cleaned.replace("antwort:", "")

    parts = [p.strip() for p in re.split(r"[,\s]+", cleaned) if p.strip()]
    preds = []

    for p in parts:
        if p in allowed_letters and p not in preds:
            preds.append(p)

    follow_format = len(preds) > 0
    return preds, follow_format


def predicted_letters_to_labels(task_id: int, letters: List[str]) -> List[str]:
    letter_to_label = {
        letter: label for letter, label in zip(TASKS[task_id]["letters"], TASKS[task_id]["labels_en"])
    }
    return [letter_to_label[x] for x in letters if x in letter_to_label]


def labels_to_binary(task_id: int, labels: List[str]) -> List[int]:
    canonical = TASKS[task_id]["labels_en"]
    return [1 if label in labels else 0 for label in canonical]



# Evaluation


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate(
    task_id: int,
    language: str,
    data_file: str,
    model,
    tokenizer,
    output_file: str,
    max_input_tokens: int,
    max_new_tokens: int,
    debug_first_n: int,
    limit_samples: Optional[int],
    use_system_role: bool,
):
    data = load_json(data_file)
    if limit_samples is not None:
        data = data[:limit_samples]

    allowed_letters = TASKS[task_id]["letters"]
    y_true = []
    y_pred = []
    follow_format_flags = []
    dumped = []

    print_block("EVALUATION CONFIG")
    print(f"Task ID          : {task_id}")
    print(f"Task name        : {TASKS[task_id]['task_name']}")
    print(f"Language         : {language}")
    print(f"Data file        : {data_file}")
    print(f"Samples          : {len(data)}")
    print(f"Use system role  : {use_system_role}")

    for idx, item in enumerate(tqdm(data, desc=f"Task {task_id}")):
        situation = str(item.get("situation", "")).strip()
        thoughts = str(item.get("thoughts", "")).strip()

        gold_labels = get_gold_labels(item, task_id)
        gold_bin = labels_to_binary(task_id, gold_labels)

        user_prompt = build_user_prompt(
            task_id=task_id,
            language=language,
            situation=situation,
            thoughts=thoughts,
        )

        if idx < debug_first_n:
            raw_token_count = tokenizer(user_prompt, return_tensors="pt", truncation=False)["input_ids"].shape[1]
            trunc_token_count = tokenizer(
                user_prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens
            )["input_ids"].shape[1]

            print_block(f"DEBUG SAMPLE {idx}")
            print(f"Situation chars        : {len(situation)}")
            print(f"Thoughts chars         : {len(thoughts)}")
            print(f"Raw prompt tokens      : {raw_token_count}")
            print(f"Truncated prompt tokens: {trunc_token_count}")
            print(f"Gold labels            : {gold_labels}")
            print(f"Prompt preview         : {shorten(user_prompt, 1200)}")

        raw_output, gen_meta = generate_letters(
            model=model,
            tokenizer=tokenizer,
            system_prompt=SYSTEM_PROMPTS[language],
            user_prompt=user_prompt,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
            use_system_role=use_system_role,
        )

        pred_letters, follow_format = parse_predicted_letters(raw_output, allowed_letters)
        pred_labels = predicted_letters_to_labels(task_id, pred_letters)
        pred_bin = labels_to_binary(task_id, pred_labels)

        y_true.append(gold_bin)
        y_pred.append(pred_bin)
        follow_format_flags.append(follow_format)

        dumped.append({
            "idx": idx,
            "situation": situation,
            "thoughts": thoughts,
            "gold_labels": gold_labels,
            "pred_letters": pred_letters,
            "pred_labels": pred_labels,
            "follow_format": follow_format,
            "prompt_tokens": gen_meta["prompt_tokens"],
            "generated_tokens": gen_meta["generated_tokens"],
            "elapsed_sec": gen_meta["elapsed_sec"],
            "raw_output": raw_output,
        })

        if idx < debug_first_n:
            print(f"Generation elapsed sec : {gen_meta['elapsed_sec']:.2f}")
            print(f"Prompt tokens          : {gen_meta['prompt_tokens']}")
            print(f"Generated tokens       : {gen_meta['generated_tokens']}")
            print(f"Raw output             : {raw_output}")
            print(f"Pred letters           : {pred_letters}")
            print(f"Pred labels            : {pred_labels}")

    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    avg_prompt_tokens = sum(x["prompt_tokens"] for x in dumped) / len(dumped) if dumped else 0.0
    avg_generated_tokens = sum(x["generated_tokens"] for x in dumped) / len(dumped) if dumped else 0.0
    avg_elapsed_sec = sum(x["elapsed_sec"] for x in dumped) / len(dumped) if dumped else 0.0

    results = {
        "task_id": task_id,
        "task_name": TASKS[task_id]["task_name"],
        "language": language,
        "data_file": data_file,
        "n_samples": len(y_true),
        "follow_format_count": int(sum(follow_format_flags)),
        "follow_format_ratio": float(sum(follow_format_flags) / len(follow_format_flags)) if follow_format_flags else 0.0,
        "weighted_f1_official_style": float(weighted_f1),
        "macro_f1": float(macro_f1),
        "macro_recall": float(macro_recall),
        "avg_prompt_tokens": float(avg_prompt_tokens),
        "avg_generated_tokens": float(avg_generated_tokens),
        "avg_elapsed_sec": float(avg_elapsed_sec),
        "labels_en": TASKS[task_id]["labels_en"],
    }

    ensure_dir_for_file(output_file)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "results": results,
                "predictions": dumped,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print_block("FINAL RESULTS")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nSaved to: {output_file}")



# Main


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=int, required=True, choices=[2, 3, 4])
    parser.add_argument("--language", type=str, required=True, choices=["en", "fr", "de", "ar"])
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)

    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--base_model", type=str, default=BASE_MODEL_DEFAULT)

    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--no_system_role", action="store_true")

    parser.add_argument("--max_input_tokens", type=int, default=1800)
    parser.add_argument("--max_new_tokens", type=int, default=24)
    parser.add_argument("--debug_first_n", type=int, default=2)
    parser.add_argument("--limit_samples", type=int, default=None)

    args = parser.parse_args()

    print_block("RUN CONFIG")
    print(json.dumps(vars(args), ensure_ascii=False, indent=2))

    model, tokenizer = load_model_and_tokenizer(
        adapter_path=args.adapter_path,
        base_model=args.base_model,
        load_in_4bit=args.load_in_4bit,
        use_lora=args.use_lora,
    )

    run_smoke_test(
        model=model,
        tokenizer=tokenizer,
        language=args.language,
        max_input_tokens=args.max_input_tokens,
        use_system_role=not args.no_system_role,
    )

    evaluate(
        task_id=args.task,
        language=args.language,
        data_file=args.data_file,
        model=model,
        tokenizer=tokenizer,
        output_file=args.output_file,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        debug_first_n=args.debug_first_n,
        limit_samples=args.limit_samples,
        use_system_role=not args.no_system_role,
    )


if __name__ == "__main__":
    main()
