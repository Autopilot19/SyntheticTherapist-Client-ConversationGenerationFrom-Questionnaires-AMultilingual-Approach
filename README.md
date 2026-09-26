# Synthetic Therapist-Client Dialogues — Multilingual Approach

This repository contains the code used to generate multilingual therapist-client dialogues, fine-tune language models on the generated data, and evaluate the resulting models with automatic metrics and counseling-related benchmarks.

The project supports the main target languages used in the thesis pipeline:

- German: `de`
- French: `fr`
- Arabic: `ar`
- English: `en` where needed for source data or benchmark baselines

## 1. Project structure

```text
questionnaire2dialogue/
├── src/
│   ├── agents.py                         # Client, therapist, and dialogue-session logic
│   ├── llms.py                           # Model wrappers, e.g. Llama, Qwen, Gemma, Mistral
│   ├── main.py                           # Entry point for dialogue generation
│   ├── preprocess.py                     # Dataset preprocessing before fine-tuning
│   ├── train.py                          # LoRA / QLoRA fine-tuning script
│   ├── merge_lora.py                     # Merge LoRA adapter into the base model
│   ├── script_generate_one_model.py      # One-model dialogue generation setup
│   └── prompts/                          # General English prompt templates if used
│
├── benchmarks/
│   ├── cbt.py                            # CBT-related benchmark helpers
│   ├── constants.py                      # Label lists, prompt constants, mappings
│   ├── eval_task2.py                     # cbt bench task evaluation script
│   ├── eval_task3.py                     # cbt bench task evaluation script
│   ├── eval_task4.py                     # cbt bench task evaluation script
│   ├── generate_adv.py                   # Generate CounselBench-Adv responses
│   ├── judge_adv.py                      # Judge CounselBench-Adv responses with an LLM judge
│   ├── preference.py                     # Preference / comparison evaluation utilities
│   ├── run_counselingBench.py            # CounselingBench evaluation entry point
│   └── utils.py                          # Shared cbt bench utilities
│
├── translation_pipeline/
│   ├── arabic/
│   ├── french/
│   ├── german/
│   │   ├── patient_information/          # Translated patient/case descriptions
│   │   ├── prompts/                      # Language- and model-specific prompts
│   │   ├── synthetic_dialogue/           # Generated dialogues
│   │   └── benchmark/                    # Translated benchmark files if used
│   ├── corpora/                          # translated corpora
│   ├── evaluation/                       # Metric outputs and evaluation scripts
│   └── scripts/                          # Translation and corpus preparation scripts

Important: the `benchmarks/` folder contains evaluation code. It should usually be tracked in Git. Generated benchmark outputs, logs, model checkpoints, and large corpora should be ignored instead.

---

## 2. Environment setup

Create or activate the environment used for this project:

```bash
conda activate q2d
```

Then install the project dependencies:

```bash
pip install -r requirements.txt
```

## 3. Starting a vLLM server

Most generation and evaluation scripts call models through an OpenAI-compatible vLLM server. Start the server first, then run the generation/evaluation code.

Example for Gemma:

```bash
vllm serve google/gemma-3-27b-it \
  --host 0.0.0.0 \
  --port 9000 \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --max-num-batched-tokens 4096 \
  --enforce-eager \
  --served-model-name google/gemma-3-27b-it
```

In the model wrapper in `src/llms.py`, the Gemma  class should point to this server.

If the server is running on the same node as the generation script, `NODE` should resolve to the node hostname. If the server is on another node, replace it with the real hostname.
---

## 4. Dialogue generation

Dialogue generation is controlled mainly by these files:

```text
src/main.py
src/agents.py
src/llms.py
```

The prompts and patient descriptions are stored by language:

```text
translation_pipeline/german/prompts/
translation_pipeline/german/patient_information/
translation_pipeline/french/prompts/
translation_pipeline/french/patient_information/
translation_pipeline/arabic/prompts/
translation_pipeline/arabic/patient_information/
```

Generated dialogues are written to:

```text
translation_pipeline/<language>/synthetic_dialogue/<model>/<split>/<group>/
```

Example German output path:

```text
translation_pipeline/german/synthetic_dialogue/gemma/dev/control/
```

### 4.1 Generate one dialogue manually

Example command:

```bash
cd src

python main.py \
  -m gemma \
  -pc ../translation_pipeline/german/prompts/mdd_gemma_client.txt \
  -pt ../translation_pipeline/german/prompts/gemma_therapist.txt \
  -ir 15 \
  -q ../translation_pipeline/german/patient_information/control_patient/dev/control144.md
```

Arguments:

```text
-m   model name used by LLMsType.get_llm() in src/agents.py
-pc  client prompt file
-pt  therapist prompt file
-ir  number of iterative rewrites / retry rounds
-q   patient information file
```

Make sure the model name is registered in:

```text
src/agents.py
```

inside:

```python
LLMsType.get_llm()
```

For example:

```python
case "gemma":
    return Gemma()
```
---

## 5. Important generation settings

Before launching a large generation run, check these files:

```text
src/agents.py
src/llms.py
run_mdd.sh
run_multi.sh
run_one_model.sh
```

In `src/agents.py`, check:

```python
lang = "de"
```

Set it according to the target language:

```python
lang = "de"  # German
lang = "fr"  # French
lang = "ar"  # Arabic
lang = "en"  # English
```

Also check the output path near the end of `TherapySession.run()`. For example:

```python
file_path = (
    "/PATH/questionnaire2dialogue/translation_pipeline/german/synthetic_dialogue/gemma/dev/control/"
    f"{self.therapist.questionnaire_name}.txt"
)
```

For a different language, split, model, or group, update this output path before running.

---

## 6. Fine-tuning

Fine-tuning is done from the `src/` folder.

Main files:

```text
src/preprocess.py
src/train.py
src/merge_lora.py
```

The general workflow is:

```text
Generated dialogues → preprocessing → LoRA/QLoRA fine-tuning → merge adapter → evaluation
```

### 6.1 Preprocess the generated dialogues

Use:

```bash
cd src
python preprocess.py
```

The processed data is usually saved under:

```text
src/processed_dataset/<language>/
```

Example:

```text
src/processed_dataset/de/
src/processed_dataset/fr/
src/processed_dataset/ar/
```

### 6.2 Train a LoRA / QLoRA model

Example:

```bash
cd src

python train.py \
  --language de \
  --dataset_dir ./processed_dataset/de \
  --output_dir ./models/llama-3.1-8b-therapist-de
```

For French:

```bash
python train.py \
  --language fr \
  --dataset_dir ./processed_dataset/fr \
  --output_dir ./models/llama-3.1-8b-therapist-fr
```

For Arabic:

```bash
python train.py \
  --language ar \
  --dataset_dir ./processed_dataset/ar \
  --output_dir ./models/llama-3.1-8b-therapist-ar
```

The fine-tuning script is responsible for LoRA / QLoRA training with TRL `SFTTrainer`.

### 6.3 Merge the LoRA adapter

After training, merge the adapter into the base model:

```bash
cd src

python merge_lora.py \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --adapter_dir ./models/llama-3.1-8b-therapist-de \
  --output_dir ./models/llama-3.1-8b-therapist-de-merged
```

The merged model is the model used for inference and benchmark evaluation.

---

## 7. Automatic metrics

Metric scripts are located mainly in:

```text
translation_pipeline/evaluation/scripts/
```

Useful files include:

```text
translation_pipeline/evaluation/scripts/corpus_diversity_stats.py
translation_pipeline/evaluation/scripts/compute_diversity_stats.py
translation_pipeline/evaluation/scripts/run_div.sh
translation_pipeline/evaluation/scripts/run_comet_eval.sh
translation_pipeline/evaluation/scripts/llm_as_judge.sh
```

Typical metrics include:

```text
MTLD
Self-BLEU
COMET
LLM-as-judge scores
```

Before running a metric script, open the corresponding `.py` file and check:

```text
input dialogue directory
language
model name
split: train / dev / test
output file path
```

Example:

```bash
cd translation_pipeline/evaluation/scripts
bash run_div.sh
```

For COMET:

```bash
cd translation_pipeline/evaluation/scripts
bash run_comet_eval.sh
```

For LLM-as-judge evaluation:

```bash
cd translation_pipeline/evaluation/scripts
bash llm_as_judge.sh
```

Metric outputs are usually written under:

```text
translation_pipeline/evaluation/
translation_pipeline/evaluation/results/
translation_pipeline/evaluation/Results/
translation_pipeline/evaluation/qwen_llm_judge/
```

---

## 8. Benchmark evaluation

Benchmark code is located in:

```text
benchmarks/
```

Important files:

```text
benchmarks/run_counselingBench.py
benchmarks/generate_adv.py
benchmarks/judge_adv.py
benchmarks/eval_task2.py
benchmarks/eval_task3.py
benchmarks/eval_task4.py
benchmarks/cbt.py
benchmarks/constants.py
benchmarks/utils.py
```

### 8.1 CounselingBench

Use:

```bash
cd benchmarks

python run_counselingBench.py \
  --model_name ../src/models/llama-3.1-8b-therapist-de-merged \
  --dataset ./counselingbench_de.csv \
  --lang de \
  --mode zero-shot \
  --tp 2
```

For chain-of-thought prompting mode:

```bash
python run_counselingBench.py \
  --model_name ../src/models/llama-3.1-8b-therapist-de-merged \
  --dataset ./counselingbench_de.csv \
  --lang de \
  --mode zero-shot-cot \
  --tp 2
```

Supported benchmark modes:

```text
zero-shot
zero-shot-cot
```

The prompt templates and answer parsing logic are handled inside the benchmark scripts and constants.

### 8.2 CounselBench-Adv generation

Generate model responses:

```bash
cd benchmarks

python generate_adv.py \
  --model ../src/models/llama-3.1-8b-therapist-de-merged \
  --csv ./counselbench_adv_de.csv \
  --language de \
  --output ./results/counselbench/adv_responses_de.json
```

For French:

```bash
python generate_adv.py \
  --model ../src/models/llama-3.1-8b-therapist-fr-merged \
  --csv ./counselbench_adv_fr.csv \
  --language fr \
  --output ./results/counselbench/adv_responses_fr.json
```

For Arabic:

```bash
python generate_adv.py \
  --model ../src/models/llama-3.1-8b-therapist-ar-merged \
  --csv ./counselbench_adv_ar.csv \
  --language ar \
  --output ./results/counselbench/adv_responses_ar.json
```

### 8.3 CounselBench-Adv judging

First start the judge model as a vLLM server, for example Qwen:

```text
Qwen/Qwen2.5-32B-Instruct
```

Then judge the generated responses:

```bash
cd benchmarks

python judge_adv.py \
  --input ./results/counselbench/adv_responses_de.json \
  --examples ./adv_examples.json \
  --output ./results/counselbench/adv_judged_de.json \
  --api_base http://node:9000/v1 \
  --model Qwen/Qwen2.5-32B-Instruct
```

The important point is that `judge_adv.py` does not load the judge model itself. It calls an already running OpenAI-compatible vLLM server.

### 8.4 CBT benchmark tasks

The CBT-related benchmark scripts are:

```text
benchmarks/cbt.py
benchmarks/eval_task2.py
benchmarks/eval_task3.py
benchmarks/eval_task4.py
benchmarks/constants.py
```

Use the relevant task script depending on the benchmark file:

```bash
cd benchmarks
python eval_task2.py
python eval_task3.py
python eval_task4.py
```

Before running, check the dataset path, model path, language, and output path inside the script.

---

## 9. Recommended full workflow

A complete experiment usually follows this order:

1. Prepare or translate patient descriptions.

   ```text
   translation_pipeline/<language>/patient_information/
   ```

2. Prepare language-specific client and therapist prompts.

   ```text
   translation_pipeline/<language>/prompts/
   ```

3. Start the vLLM server for the chosen generator model.

4. Generate dialogues.

   ```bash
   bash run_mdd.sh
   # or
   bash run_multi.sh
   # or
   bash run_one_model.sh
   ```

5. Check generated dialogue files.

   ```text
   translation_pipeline/<language>/synthetic_dialogue/<model>/<split>/<group>/
   ```

6. Compute automatic corpus metrics.

   ```bash
   cd translation_pipeline/evaluation/scripts
   bash run_div.sh
   bash run_comet_eval.sh
   bash llm_as_judge.sh
   ```

7. Preprocess generated dialogues for training.

   ```bash
   cd src
   python preprocess.py
   ```

8. Fine-tune the model.

   ```bash
   python train.py --language de --dataset_dir ./processed_dataset/de --output_dir ./models/llama-3.1-8b-therapist-de
   ```

9. Merge the LoRA adapter.

   ```bash
   python merge_lora.py --base_model meta-llama/Llama-3.1-8B-Instruct --adapter_dir ./models/llama-3.1-8b-therapist-de --output_dir ./models/llama-3.1-8b-therapist-de-merged
   ```

10. Evaluate the merged model on benchmarks.

   ```bash
   cd benchmarks
   python run_counselingBench.py --model_name ../src/models/llama-3.1-8b-therapist-de-merged --dataset ./counselingbench_de.csv --lang de --mode zero-shot --tp 2
   ```

---


## 10. Reproducibility checklist

For every experiment, record:

```text
language
model name
base model checkpoint
fine-tuned adapter path
merged model path
prompt files
patient-information split
generation script
number of iterative rewrites
sampling parameters: temperature, top_p, top_k, min_p, max_tokens
benchmark mode: zero-shot or zero-shot-cot
judge model if LLM-as-judge was used
output directory
```

This makes it easier to reproduce similar tables and figures to those of the thesis.
