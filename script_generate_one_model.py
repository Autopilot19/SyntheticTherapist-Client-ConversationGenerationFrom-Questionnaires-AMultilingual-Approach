import os
import argparse
from datetime import datetime
import time  

from langchain_core.prompts import PromptTemplate
from langchain_community.llms import VLLMOpenAI


NODE = os.getenv("NODE", "bob")
SET_NAME = os.getenv("SET_NAME", "blank")

MODEL_MAP = {
    "gemma": "google/gemma-3-27b-it",
    "qwen-2.5": "Qwen/Qwen2.5-32B-Instruct",
}


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_llm(model_key: str, max_tokens: int, temperature: float, top_p: float) -> VLLMOpenAI:
    if model_key not in MODEL_MAP:
        raise ValueError(f"Unsupported model key '{model_key}'. Use one of: {list(MODEL_MAP.keys())}")

    return VLLMOpenAI(
        openai_api_key="EMPTY",
        openai_api_base=f"http://{NODE}:9000/v1",
        model=MODEL_MAP[model_key],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )


def main(model: str, questionnaire_path: str, prompt_script_path: str, turns: int,
         max_tokens: int, temperature: float, top_p: float) -> None:

    prompt_text = open(prompt_script_path, "r", encoding="utf-8").read()
    questionnaire_text = open(questionnaire_path, "r", encoding="utf-8").read()

    prompt_tmpl = PromptTemplate(
        input_variables=["questionnaire", "history", "turns"],
        template=prompt_text
    )

    prompt = prompt_tmpl.format(
        questionnaire=questionnaire_text,
        history="",
        turns=turns
    )

    llm = build_llm(model, max_tokens=max_tokens, temperature=temperature, top_p=top_p)

    #  measure generation time 
    t0 = time.time()
    script = str(llm.invoke(prompt)).strip()
    t1 = time.time()
   

    # Save output 
    questionnaire_name = os.path.basename(questionnaire_path).split(".")[0]
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    out_dir = "/storage/ukp/work/boudabous/questionnaire2dialogue/one_model/german"
    os.makedirs(out_dir, exist_ok=True)

    file_path = f"{out_dir}/{current_time}_script_{questionnaire_name}_{model}.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(script + "\n")

    print(f"\nSaved to: {file_path}\n")
    print(script)
    print(f"\033[94m  Generation time: {t1 - t0:.2f} seconds\033[0m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-q", "--questionnaire", type=str, required=True)
    parser.add_argument("--prompt_script", type=str, required=True)
    parser.add_argument("--turns", type=int, default=20)

   
    parser.add_argument("--max_tokens", type=int, default=4096)

    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)

    args = parser.parse_args()

    main(
        model=args.model,
        questionnaire_path=args.questionnaire,
        prompt_script_path=args.prompt_script,
        turns=args.turns,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
