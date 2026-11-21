# Sanad — Qwen2.5-7B-Instruct 4-bit QLoRA Fine-Tuning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/3isa2002/qwen2.5-sanad-arabic-qlora/blob/main/notebooks/qwen25_7b_sanad_qlora_colab.ipynb)

Bilingual **Egyptian Arabic & English** 4-bit **QLoRA** fine-tuning pipeline built as the model-adaptation layer for **[Sanad (`sanad-support-rag`)](https://github.com/3isa2002/sanad-support-rag)**.

> **Project Status:** Dataset pipeline, validation suite, QLoRA training scripts, Before/After evaluator, and Colab T4 notebook are complete and unit-tested locally. **GPU training and benchmark numbers below are marked pending** until executed on Google Colab (`T4 16 GB`) / Kaggle.

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

> **Pending Execution:** Run `notebooks/qwen25_7b_sanad_qlora_colab.ipynb` (or `scripts/evaluate.py --mode both --split test`) on a `16 GB` GPU (`Colab T4` / `Kaggle`), then replace this table with the generated `outputs/eval/comparison_table.md`.

| Metric (`n=45` held-out test) | Base `Qwen2.5-7B-Instruct` | Sanad QLoRA (Ours) | Delta |
| --- | ---: | ---: | ---: |
| Strict End-to-End Pass % | _Pending run_ | _Pending run_ | _Pending run_ |
| Strict Format Compliance % | _Pending run_ | _Pending run_ | _Pending run_ |
| Citation Compliance % (Grounded) | _Pending run_ | _Pending run_ | _Pending run_ |
| Exact Citation Match % (Grounded) | _Pending run_ | _Pending run_ | _Pending run_ |
| `HANDOFF` Overall Accuracy % | _Pending run_ | _Pending run_ | _Pending run_ |
| `HANDOFF` Recall % (Anti-Hallucination) | _Pending run_ | _Pending run_ | _Pending run_ |
| `HANDOFF` F1 % | _Pending run_ | _Pending run_ | _Pending run_ |
| False Handoff Rate % (Lower is better) | _Pending run_ | _Pending run_ | _Pending run_ |
| Phantom Citation Rate % (Lower is better) | _Pending run_ | _Pending run_ | _Pending run_ |
| Language Match % | _Pending run_ | _Pending run_ | _Pending run_ |

- **Training Run Summary:** _Pending Colab T4 run_ (`outputs/qwen2.5-7b-sanad-qlora/training_summary.json`)
- **Hugging Face Adapter Repo:** _Pending upload_ (`3isa2002/qwen2.5-7b-sanad-arabic-qlora`)
- **Experiment Tracker Run (`W&B`):** _Pending run link_

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
