# Sanad — Qwen2.5-7B-Instruct 4-bit QLoRA Fine-Tuning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/3isa2002/qwen2.5-sanad-arabic-qlora/blob/main/notebooks/qwen25_7b_sanad_qlora_colab.ipynb)
[![Hugging Face Adapter](https://img.shields.io/badge/Hugging%20Face-3isa2002%2Fqwen2.5--7b--sanad--arabic--qlora-yellow?logo=huggingface)](https://huggingface.co/3isa2002/qwen2.5-7b-sanad-arabic-qlora)

Bilingual **Egyptian Arabic & English** 4-bit **QLoRA** fine-tuning pipeline built as the model-adaptation layer for **[Sanad (`sanad-support-rag`)](https://github.com/3isa2002/sanad-support-rag)**.

> **Project Status:** Complete — dataset pipeline, validation suite, 4-bit QLoRA training (`NVIDIA L4 22.03 GB`, `Unsloth + TRL`, `bfloat16`), held-out Before-vs-After evaluation (`n=45`, `28.89% -> 93.33%` strict end-to-end pass), and live LoRA adapter + Model Card published on **[Hugging Face Hub (`3isa2002/qwen2.5-7b-sanad-arabic-qlora`)](https://huggingface.co/3isa2002/qwen2.5-7b-sanad-arabic-qlora)**.

---

## Why This Repo Exists (`Sanad v1 Prompted RAG` → `v2 QLoRA Adaptation`)

In **[sanad-support-rag](https://github.com/3isa2002/sanad-support-rag)** (`v1`), `Qwen2.5-7B` runs locally via Ollama using prompt engineering and hybrid retrieval (`NumPy` dense embeddings + Arabic-aware `BM25` + `RRF`). While retrieval brings the right passages, relying solely on long system prompts in a 7B model leaves three reliability gaps:

1. **Dialect Drift:** Replies to Egyptian Arabic queries can slip into stiff MSA or inconsistent tone.
2. **Citation Drift:** Occasional omission of inline `[1]`, `[2]` source markers required by Sanad's verification layer.
3. **Refusal / Handoff Hallucination:** Guessing answers when retrieved chunks are only topically related instead of emitting the deterministic `HANDOFF |` escalation token used to open human tickets in SQLite.

This repository isolates the **training, evaluation, and adapter-export pipeline** (`PyTorch`, `Hugging Face`, `PEFT`, `TRL`, `bitsandbytes`, optional `Unsloth`) from the `FastAPI` serving app so the base `Sanad v1` codebase remains clean while producing a drop-in LoRA adapter / GGUF model (`v2`).

---

## Architecture & Engineering Decisions

| Component | Choice | Why |
| --- | --- | --- |
| **Base Model** | `Qwen/Qwen2.5-7B-Instruct` | Strong bilingual Arabic/English tokenization and instruction following; directly matches `Sanad`'s Ollama deployment (`qwen2.5:7b`). |
| **Quantization** | 4-bit `NF4` + Double Quantization | Shrinks 7B base weights from ~14 GB (FP16) to ~4 GB, fitting comfortably inside a free **Google Colab `T4 16 GB`** GPU alongside LoRA states and activations. |
| **Compute Dtype** | `auto` (`fp16` on T4 / `bf16` on L4/A100) | Colab T4 (Turing) lacks native `bf16` tensor cores, so selecting `float16` avoids slow emulation or dtype errors. |
| **LoRA Config** | `r=16`, `lora_alpha=32`, all attention + MLP projections |Adapts `q/k/v/o_proj` and `gate/up/down_proj` for stronger reasoning and formatting adherence across dialects. |
| **Loss Masking** | `completion_only_loss: true` | Computes cross-entropy loss **only** on the assistant response rather than retrieved context chunks (~80% of tokens), preventing chunk memorization. |
| **Training Engines** | Standard HF (`PEFT + TRL SFTTrainer`) + `--use-unsloth` | Standard HF path teaches/verifies the canonical PyTorch stack; `--use-unsloth` enables custom CUDA kernels for ~2x faster T4 runs. |
| **Tracking** | `Weights & Biases` (default) + `MLflow` option | Tracks `train_loss`, `eval_loss`, and Before-vs-After metrics; falls back cleanly to offline mode when no `WANDB_API_KEY` is set. |

---

## Dataset Design (`549` Verified Examples, Zero Cross-Split Leakage)

Generated deterministically (`seed=42`) via [`scripts/build_dataset.py`](scripts/build_dataset.py) and audited by [`scripts/validate_dataset.py`](scripts/validate_dataset.py) in standard `messages` chat format (`system`, `user`, `assistant`):

- **3-Way Split (Disjoint Scenario Groups & Context Chunks):**
  - `data/processed/train.jsonl`: **464** examples (**84.52%**)
  - `data/processed/val.jsonl`: **40** examples (**7.29%**) — used only for `eval_loss` checkpointing during `SFTTrainer`
  - `data/processed/test.jsonl`: **45** examples (**8.20%**) — strictly held-out policies never seen in `train` or `val`
- **Behavior Breakdown:**
  - `train`: `245` single-chunk grounded | `74` multi-chunk synthesis | `145` strict `HANDOFF |`
  - `val`: `22` single-chunk grounded | `6` multi-chunk synthesis | `12` strict `HANDOFF |`
  - `test`: `24` single-chunk grounded | `7` multi-chunk synthesis | `14` strict `HANDOFF |`
- **Language Mix (incl. Cross-Lingual RAG):**
  - `train`: `258` `ar_eg`, `24` `en_to_ar` (English policy chunks -> Egyptian Arabic answer), `174` `en`, `8` `ar_to_en`
  - `val`: `18` `ar_eg`, `2` `en_to_ar`, `20` `en`
  - `test`: `19` `ar_eg`, `3` `en_to_ar`, `23` `en`
- **Anti-Shortcut Safeguards Verified in [`data/processed/dataset_report.json`](data/processed/dataset_report.json):**
  - Randomized chunk positions + distractor injection in `train` single-citation targets: `[1]: 91`, `[2]: 58`, `[3]: 67`, `[4]: 29`.
  - Adversarial keyword-overlap `HANDOFF |` traps where chunks share topic keywords (e.g., shipping or installments) yet lack the asked detail.
  - Zero shared `scenario_group`, context chunks, or exact questions across splits; max cross-split word Jaccard similarity = **`0.4737`**.

---

## Repository Structure

```text
qwen2.5-sanad-arabic-qlora/
├── configs/
│   └── qlora_qwen25_7b.yaml           # Unified model, QLoRA, LoRA, SFT & eval hyperparameters
├── data/processed/
│   ├── train.jsonl                    # 464 training examples
│   ├── val.jsonl                      # 40 validation examples
│   ├── test.jsonl                     # 45 held-out benchmark examples
│   └── dataset_report.json            # Automated leakage & distribution audit
├── notebooks/
│   └── qwen25_7b_sanad_qlora_colab.ipynb  # End-to-end Google Colab T4 / Kaggle notebook
├── scripts/
│   ├── build_dataset.py               # Reproducible dataset generator (seed=42)
│   ├── validate_dataset.py            # Schema, citation & zero-leakage validator
│   ├── train.py                       # HF PEFT+TRL SFTTrainer & optional --use-unsloth trainer
│   ├── evaluate.py                    # Before vs After evaluation on test.jsonl
│   └── export_hf_and_gguf.py          # Push LoRA adapter to HF Hub + Ollama Modelfile/GGUF export
├── src/
│   ├── data/                          # Schema, disjoint train/eval scenario banks, validator
│   ├── eval/                          # Deterministic citation, HANDOFF & language scoring metrics
│   └── training/                      # YAML config, dataset prep, 4-bit loader, W&B/MLflow tracker
└── tests/
    ├── test_dataset.py                # Offline tests for dataset schema & anti-leakage checks
    └── test_eval_metrics.py           # Offline tests for config, formatting & eval scoring
```

---

## Before vs After Benchmark (`data/processed/test.jsonl`, `n=45`)

Executed deterministically (`do_sample=false`, greedy decoding) on Google Colab Pro (`NVIDIA L4`, `22.03 GB VRAM`, `torch.bfloat16`) via `scripts/evaluate.py --mode both --split test` (`outputs/eval/comparison_table.md` & `outputs/eval/comparison_report.json`):

| Metric (`n=45` held-out test) | Base `Qwen2.5-7B-Instruct` | Sanad QLoRA (Ours) | Delta |
| --- | ---: | ---: | ---: |
| Strict End-to-End Pass % | 28.89% | **93.33%** | **+64.44 pp (improved)** |
| Strict Format Compliance % | 35.56% | **95.56%** | **+60.00 pp (improved)** |
| Citation Compliance % (Grounded) | 77.42% | **100.00%** | **+22.58 pp (improved)** |
| Exact Citation Match % (Grounded) | 67.74% | **96.77%** | **+29.03 pp (improved)** |
| `HANDOFF` Overall Accuracy % | 75.56% | **95.56%** | **+20.00 pp (improved)** |
| `HANDOFF` Recall % (Anti-Hallucination) | 42.86% | **85.71%** | **+42.85 pp (improved)** |
| `HANDOFF` Precision % | 66.67% | **100.00%** | **+33.33 pp (improved)** |
| `HANDOFF` F1 % | 52.18% | **92.31%** | **+40.13 pp (improved)** |
| False Handoff Rate % (Lower is better) | 9.68% | **0.00%** | **-9.68 pp (improved)** |
| Phantom Citation Rate % (Lower is better) | 0.00% | **0.00%** | `0.00 pp` |
| Language Match % | 53.33% | **100.00%** | **+46.67 pp (improved)** |
| Arabic Strict Pass % | 40.91% | **86.36%** | **+45.45 pp (improved)** |
| English Strict Pass % | 17.39% | **100.00%** | **+82.61 pp (improved)** |

**Per-Behavior Strict End-to-End Pass Breakdown (`n=45`):**
- `grounded_single` (`n=24`): `33.33%` -> **`100.00%`** (`+66.67 pp`)
- `grounded_multi` (`n=7`): `14.29%` -> **`85.71%`** (`+71.42 pp`)
- `handoff` (`n=14`): `28.57%` -> **`85.71%`** (`+57.14 pp`)

### Training Run Summary (`outputs/qwen2.5-7b-sanad-qlora/training_summary.json`)
- **Backend & Hardware:** `unsloth` + `TRL SFTTrainer` on `NVIDIA L4` (`22.03 GB` VRAM), `compute_dtype=torch.bfloat16`
- **Trainable Parameters:** `40,370,176` / `4,931,917,312` (**`0.8185%`**)
- **Loss & Runtime:** Final `train_loss = 0.7895` | Best `eval_loss = 0.9168` (`87` optimizer steps / `3` epochs, `798.3s` total runtime)
- **Hugging Face Adapter Repo:** Published live at **[huggingface.co/3isa2002/qwen2.5-7b-sanad-arabic-qlora](https://huggingface.co/3isa2002/qwen2.5-7b-sanad-arabic-qlora)** (`adapter_model.safetensors`, `adapter_config.json`, tokenizer, `chat_template.jinja`, Model Card `README.md`)
- **Experiment Tracker (`W&B`):** Logged via non-interactive `WANDB_MODE=offline` (`qwen25-sanad-arabic-qlora`)

### Qualitative Before vs After Examples (`outputs/eval/side_by_side_examples.md`)
- **`sanad_test_0012` (`grounded_single` | `en`) — Language Drift Fixed:** Base replied in Arabic (`FAIL`); Fine-tuned replied in English (`Fake product reports trigger an immediate 6-hour ad freeze pending supplier invoices [2], and if confirmed fake, buyers get double their paid amount [2].`, `PASS`).
- **`sanad_test_0040` (`handoff` | `ar_eg`) — Anti-Hallucination Fixed:** Base fabricated a case status (`تحقيق البلاغ رقم #CF-9031 وصل إلى أن البائع لم يقدم فواتير المورد الأصلي حتى الآن [2].`, `FAIL`); Fine-tuned emitted strict `HANDOFF | حالات البلاغات عن المنتجات المقلدة بتتطلب فتح تحقيق مع البائع ومراجعة مستندات المورد، وهحوّلك فوراً لموظف خدمة العملاء.` (`PASS`).
- **`sanad_test_0033` (`handoff` | `ar_eg`) — Mixed Arabic/Chinese Hallucination Fixed:** Base drifted mid-sentence into Chinese + `[1]` + `HANDOFF |` (`FAIL`); Fine-tuned emitted clean Egyptian Arabic `HANDOFF | معنديش في السياسة المتاحة بيانات عن خصم تقييمات الأجهزة القديمة، وهحوّلك فوراً لموظف خدمة العملاء يراجعلك السعر.` (`PASS`).

---

## How to Run

### 1. Locally (CPU / Windows — Dataset, Validation & Unit Tests Only)
> Do **not** run 7B QLoRA training on an 8 GB local GPU. Use local execution for dataset generation, validation, and offline unit tests:

```bash
pip install -r requirements.txt
python scripts/build_dataset.py
python scripts/validate_dataset.py
pytest tests/ -v
```

### 2. Google Colab (`T4 16 GB`) / Kaggle (Training, Eval & HF Upload)
Open [`notebooks/qwen25_7b_sanad_qlora_colab.ipynb`](notebooks/qwen25_7b_sanad_qlora_colab.ipynb) on Colab with a **T4 GPU**, or run the CLI workflow:

```bash
pip install -r requirements-train.txt
# Optional 2x Colab speedup: pip install unsloth

# 1) Train (add --use-unsloth for Unsloth kernels)
python scripts/train.py --config configs/qlora_qwen25_7b.yaml --tracker wandb

# 2) Run deterministic Before vs After evaluation on held-out test set
python scripts/evaluate.py --config configs/qlora_qwen25_7b.yaml --mode both --split test --tracker wandb

# 3) Push LoRA adapter + Model Card to Hugging Face Hub
python scripts/export_hf_and_gguf.py --hf-repo 3isa2002/qwen2.5-7b-sanad-arabic-qlora --push-to-hub
```

### 3. Integrating Back into `Sanad` (`Ollama` GGUF / `vLLM`)
- **Ollama (`Sanad v1` drop-in swap):**
  1. Export a quantized `Q4_K_M` GGUF and generate `outputs/ollama/Modelfile`:
     ```bash
     python scripts/export_hf_and_gguf.py --export-gguf
     ```
  2. Register locally with Ollama:
     ```bash
     ollama create sanad-qwen25-ft -f outputs/ollama/Modelfile
     ```
  3. Point `Sanad`'s model configuration from `qwen2.5:7b` to `sanad-qwen25-ft`.
- **vLLM (Next Milestone):** Serve the merged 16-bit / quantized model behind an OpenAI-compatible `vLLM` endpoint with LoRA hot-swapping or AWQ quantization.
