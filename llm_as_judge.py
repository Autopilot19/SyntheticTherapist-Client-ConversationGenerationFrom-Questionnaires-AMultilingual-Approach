import argparse
import json
import os
import re
import time
import textwrap

import google.generativeai as genai


# Dialogues to evaluate 

DIALOGUES = {
    "Control_40": {
        "fr": "/storage/ukp/work/boudabous/questionnaire2dialogue/translation_pipeline/french/synthetic_dialogue/gemma/control/dev/control40.txt",
        "de": "/storage/ukp/work/boudabous/questionnaire2dialogue/translation_pipeline/german/synthetic_dialogue/gemma/control/dev/control40.txt",
        "ar": "/storage/ukp/work/boudabous/questionnaire2dialogue/translation_pipeline/arabic/synthetic_dialogue/gemma/control/dev/control40.txt",
    },
    "Control_144": {
        "fr": "/storage/ukp/work/boudabous/questionnaire2dialogue/translation_pipeline/french/synthetic_dialogue/gemma/control/train/control144.txt",
        "de": "/storage/ukp/work/boudabous/questionnaire2dialogue/translation_pipeline/german/synthetic_dialogue/gemma/control/train/control144.txt",
        "ar": "/storage/ukp/work/boudabous/questionnaire2dialogue/translation_pipeline/arabic/synthetic_dialogue/gemma/control/train/control144.txt",
    },
    "Mdd_active_684": {
        "fr": "/storage/ukp/work/boudabous/questionnaire2dialogue/translation_pipeline/french/synthetic_dialogue/gemma/mdd/dev/mdd_active684.txt",
        "de": "/storage/ukp/work/boudabous/questionnaire2dialogue/translation_pipeline/german/synthetic_dialogue/gemma/mdd/dev/mdd_active684.txt",
        "ar": "/storage/ukp/work/boudabous/questionnaire2dialogue/translation_pipeline/arabic/synthetic_dialogue/gemma/mdd/dev/mdd_active684.txt",
    },
    "Mdd_active_25": {
        "fr": "/storage/ukp/work/boudabous/questionnaire2dialogue/translation_pipeline/french/synthetic_dialogue/gemma/mdd/train/mdd_active25.txt",
        "de": "/storage/ukp/work/boudabous/questionnaire2dialogue/translation_pipeline/german/synthetic_dialogue/gemma/mdd/train/mdd_active25.txt",
        "ar": "/storage/ukp/work/boudabous/questionnaire2dialogue/translation_pipeline/arabic/synthetic_dialogue/gemma/mdd/train/mdd_active25.txt",
    },
}

LANGUAGE_NAMES = {
    "fr": "French",
    "de": "German",
    "ar": "Arabic",
}

CRITERIA = [
    "Fluency",
    "Context Fit & Coherence",
    "Naturalness / Realism",
    "Empathy",
    "Appropriateness",
    "Engagement",
]


# Judge prompt

JUDGE_PROMPT = textwrap.dedent("""\
You are an expert evaluator of multilingual therapy dialogues. You will be given a therapist-client conversation that was originally generated in English and translated into {language_name}.

Evaluate the dialogue on the following 6 criteria using a 1-5 scale:

Scoring Scale:
1 – Very Poor: Serious problems. Unnatural, confusing, or inappropriate.
2 – Poor: Several noticeable issues, but meaning can still be understood.
3 – Acceptable: Understandable but has clear weaknesses.
4 – Good: Mostly natural and appropriate with only minor issues.
5 – Excellent: Very natural, coherent, and realistic.

=== A. Multilingual Language Quality ===

1. Fluency / Language Quality
- Are the sentences grammatically correct and natural in {language_name}?
- Are there strange phrases or translation artifacts?
Give lower scores for grammar errors, unnatural wording, or literal translations.

2. Context Fit & Coherence
- Does each response logically follow the previous turn?
- Does the conversation make logical sense throughout?
Give lower scores for generic/unrelated responses or sudden topic jumps.

3. Naturalness / Realism
- Does the conversation sound like a real human conversation?
- Does the dialogue feel robotic or scripted?
Give lower scores for artificial dialogue, repetitive structures, or overly formal/mechanical responses.

=== B. Therapy-Related Dialogue Quality ===

4. Empathy
- Does the therapist show understanding and emotional awareness?
- Does the therapist acknowledge the client's feelings?
- Does the response sound supportive and respectful?
Give lower scores when the therapist sounds cold, dismissive, or ignores emotional concerns.

5. Appropriateness
- Is the therapist's response appropriate for the situation?
- Does the response feel respectful and reasonable?
Give lower scores for strange, unsuitable, socially awkward, or insensitive responses.

6. Engagement
- Does the therapist help move the conversation forward?
- Does the therapist encourage the client to share more?
- Does the therapist ask thoughtful questions?
Give higher scores when the conversation progresses naturally.
Give lower scores for short/unhelpful responses or when the dialogue feels stuck/repetitive.

=== Dialogue to Evaluate ===

Language: {language_name}
Dialogue ID: {dialogue_id}

{dialogue_text}

=== Instructions ===

Score each criterion from 1 to 5. Respond ONLY in this exact JSON format:
{{
    "fluency": <1-5>,
    "coherence": <1-5>,
    "naturalness": <1-5>,
    "empathy": <1-5>,
    "appropriateness": <1-5>,
    "engagement": <1-5>,
    "brief_justification": "<2-3 sentences explaining your overall assessment>"
}}""")


# Functions 

def load_dialogue(filepath):
    """Load a dialogue text file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()


def parse_scores(text):
    """Parse JSON scores from output."""
    text = text.strip()
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text).strip()

    parsed = json.loads(text)

    scores = {}
    key_map = {
        "fluency": "fluency",
        "coherence": "coherence",
        "context fit & coherence": "coherence",
        "context_fit": "coherence",
        "naturalness": "naturalness",
        "naturalness / realism": "naturalness",
        "naturalness_realism": "naturalness",
        "empathy": "empathy",
        "appropriateness": "appropriateness",
        "engagement": "engagement",
    }

    for key, value in parsed.items():
        normalized = key.lower().strip()
        if normalized in key_map:
            scores[key_map[normalized]] = max(1, min(5, int(value)))
        elif normalized == "brief_justification":
            scores["brief_justification"] = str(value)

    if "brief_justification" not in scores:
        scores["brief_justification"] = ""

    return scores


def judge_single(model, dialogue_text, dialogue_id, language, max_retries=3):
    """Judge a single dialogue."""
    prompt = JUDGE_PROMPT.format(
        language_name=LANGUAGE_NAMES[language],
        dialogue_id=dialogue_id,
        dialogue_text=dialogue_text,
    )

    for attempt in range(max_retries):
        try:
            result = model.generate_content(prompt)
            text = result.text.strip()
            scores = parse_scores(text)
            return scores, None

        except Exception as e:
            error_msg = str(e)[:200]
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"    Retry {attempt + 1}/{max_retries} ({error_msg}), waiting {wait}s...")
                time.sleep(wait)
            else:
                return None, error_msg


def main():
    parser = argparse.ArgumentParser(description="Evaluate therapy dialogues with Gemini")
    parser.add_argument("--output", type=str, default="./results/dialogue_eval.json")
    parser.add_argument("--gemini_model", type=str, default="gemini-2.5-flash")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between API calls")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print(" Missing API_key")
        exit(1)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(args.gemini_model)

    print(f"Judge model: {args.gemini_model}")
    print(f"Evaluating {len(DIALOGUES)} dialogues × {len(LANGUAGE_NAMES)} languages = {len(DIALOGUES) * len(LANGUAGE_NAMES)} evaluations")
    print()

    all_results = []
    score_keys = ["fluency", "coherence", "naturalness", "empathy", "appropriateness", "engagement"]

    call_count = 0
    total_calls = len(DIALOGUES) * len(LANGUAGE_NAMES)

    for dialogue_id, lang_paths in DIALOGUES.items():
        for lang, filepath in lang_paths.items():
            call_count += 1
            print(f"[{call_count}/{total_calls}] {dialogue_id} / {LANGUAGE_NAMES[lang]}...", end=" ")

            if not os.path.exists(filepath):
                print(f"FILE NOT FOUND: {filepath}")
                all_results.append({
                    "dialogue_id": dialogue_id,
                    "language": lang,
                    "error": f"File not found: {filepath}",
                })
                continue

            dialogue_text = load_dialogue(filepath)
            scores, error = judge_single(model, dialogue_text, dialogue_id, lang)

            if scores:
                result = {
                    "dialogue_id": dialogue_id,
                    "language": lang,
                    "language_name": LANGUAGE_NAMES[lang],
                    **scores,
                }
                avg = sum(scores.get(k, 0) for k in score_keys) / len(score_keys)
                result["average"] = round(avg, 2)
                all_results.append(result)

                scores_str = " | ".join(f"{k}={scores.get(k, '?')}" for k in score_keys)
                print(f"avg={avg:.1f} | {scores_str}")
            else:
                all_results.append({
                    "dialogue_id": dialogue_id,
                    "language": lang,
                    "error": error,
                })
                print(f" {error}")

            time.sleep(args.delay)

    # Compute aggregated scores 
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    valid_results = [r for r in all_results if "error" not in r]

    # Per language averages
    print(f"\n{'Language':<12s}", end="")
    for k in score_keys:
        print(f"  {k[:10]:>10s}", end="")
    print(f"  {'Average':>8s}")
    print("-" * 80)

    per_lang = {}
    for lang in LANGUAGE_NAMES:
        lang_results = [r for r in valid_results if r["language"] == lang]
        if lang_results:
            avgs = {}
            for k in score_keys:
                vals = [r[k] for r in lang_results if k in r]
                avgs[k] = sum(vals) / len(vals) if vals else 0
            overall = sum(avgs.values()) / len(avgs)
            per_lang[lang] = {**avgs, "average": overall, "n": len(lang_results)}

            print(f"{LANGUAGE_NAMES[lang]:<12s}", end="")
            for k in score_keys:
                print(f"  {avgs[k]:>10.2f}", end="")
            print(f"  {overall:>8.2f}")

    # Per dialogue averages
    print(f"\n{'Dialogue':<20s}", end="")
    for k in score_keys:
        print(f"  {k[:10]:>10s}", end="")
    print(f"  {'Average':>8s}")
    print("-" * 80)

    per_dialogue = {}
    for did in DIALOGUES:
        did_results = [r for r in valid_results if r["dialogue_id"] == did]
        if did_results:
            avgs = {}
            for k in score_keys:
                vals = [r[k] for r in did_results if k in r]
                avgs[k] = sum(vals) / len(vals) if vals else 0
            overall = sum(avgs.values()) / len(avgs)
            per_dialogue[did] = {**avgs, "average": overall, "n": len(did_results)}

            print(f"{did:<20s}", end="")
            for k in score_keys:
                print(f"  {avgs[k]:>10.2f}", end="")
            print(f"  {overall:>8.2f}")

    # Overall average
    if valid_results:
        overall_avgs = {}
        for k in score_keys:
            vals = [r[k] for r in valid_results if k in r]
            overall_avgs[k] = sum(vals) / len(vals) if vals else 0
        grand_avg = sum(overall_avgs.values()) / len(overall_avgs)

        print(f"\n{'OVERALL':<20s}", end="")
        for k in score_keys:
            print(f"  {overall_avgs[k]:>10.2f}", end="")
        print(f"  {grand_avg:>8.2f}")

   
    output_data = {
        "judge_model": args.gemini_model,
        "criteria": score_keys,
        "per_language": per_lang,
        "per_dialogue": per_dialogue,
        "overall": overall_avgs if valid_results else {},
        "overall_average": grand_avg if valid_results else 0,
        "individual_results": all_results,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nFull results saved to: {args.output}")


if __name__ == "__main__":
    main()