"""Push the trained QLoRA adapter to Hugging Face Hub and prepare Sanad Ollama/GGUF artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.schema import SANAD_SYSTEM_PROMPT  # noqa: E402
from src.training.config import load_config, resolve_project_path  # noqa: E402


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_hf_model_card(
    *,
    hf_repo: str,
    base_model: str,
    eval_table_md: str | None,
    training_summary: dict[str, Any] | None,
) -> str:
    """Generate a factual Hugging Face Model Card README.md without invented benchmarks."""
    if training_summary:
        train_block = (
            f"- **Trainable Parameters:** {training_summary['trainable_params']:,} "
            f"({training_summary['trainable_pct']}% of {training_summary['total_params']:,})\n"
            f"- **Final Train Loss / Eval Loss:** {training_summary['train_loss']} / {training_summary['eval_loss']}\n"
            f"- **Hardware / Runtime:** {training_summary['gpu_name']} ({training_summary['runtime_seconds']}s)\n"
        )
    else:
        train_block = "- **Training Run Metrics:** Pending final run metrics.\n"

    eval_block = (
        eval_table_md.strip()
        if eval_table_md
        else "_Evaluation benchmark pending run of `scripts/evaluate.py --mode both` on held-out `test.jsonl`._"
    )

    return f"""---
base_model: {base_model}
library_name: peft
tags:
  - qwen2.5
  - qlora
  - arabic
  - egyptian-arabic
  - rag
  - customer-support
  - sanad
license: apache-2.0
---

# {hf_repo}

4-bit **QLoRA** adapter fine-tuned on `{base_model}` for **Sanad** ([github.com/3isa2002/sanad-support-rag](https://github.com/3isa2002/sanad-support-rag)) — a bilingual Egyptian Arabic & English RAG customer-support assistant.

## Learned Behaviors
1. **Natural Egyptian Arabic & English Support Tone** aligned to customer language (including English KB -> Egyptian Arabic cross-lingual responses).
2. **Strict Context Grounding & Inline Citations** (`[1]`, `[2]`, ...) matched to retrieved chunks.
3. **Zero-Hallucination `HANDOFF |` Protocol** when retrieved chunks lack the required evidence.

## Training Summary
{train_block}
## Before vs After Held-Out Evaluation (`test.jsonl`, `n=45`)
{eval_block}

## Quick Usage (`PEFT` + `Transformers`)
```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

base_id = "{base_model}"
adapter_id = "{hf_repo}"

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(adapter_id)
base_model = AutoModelForCausalLM.from_pretrained(base_id, quantization_config=bnb_config, device_map="auto")
model = PeftModel.from_pretrained(base_model, adapter_id)
```
"""


def write_ollama_modelfile(output_dir: Path, gguf_filename: str = "qwen2.5-7b-sanad-q4_k_m.gguf") -> Path:
    """Create a drop-in Ollama Modelfile compatible with Sanad's local deployment."""
    output_dir.mkdir(parents=True, exist_ok=True)
    modelfile_path = output_dir / "Modelfile"
    content = f'''FROM ./{gguf_filename}

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER num_ctx 2048
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"

SYSTEM """{SANAD_SYSTEM_PROMPT}"""
'''
    modelfile_path.write_text(content, encoding="utf-8")
    return modelfile_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Push QLoRA adapter to Hugging Face & prepare Sanad Ollama Modelfile.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "qlora_qwen25_7b.yaml",
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default="3isa2002/qwen2.5-7b-sanad-arabic-qlora",
        help="Target Hugging Face Hub repo ID (username/repo-name).",
    )
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--push-to-hub", action="store_true", help="Upload adapter + Model Card to HF Hub.")
    parser.add_argument("--private", action="store_true", help="Create HF Hub repo as private.")
    parser.add_argument(
        "--export-gguf",
        action="store_true",
        help="Export merged Q4_K_M GGUF using Unsloth (requires GPU + unsloth installed).",
    )
    parser.add_argument(
        "--gguf-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "ollama",
        help="Directory for Ollama Modelfile and optional GGUF export.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_out = resolve_project_path(PROJECT_ROOT, cfg.training.output_dir)
    eval_out = resolve_project_path(PROJECT_ROOT, cfg.eval.output_dir)
    adapter_dir = args.adapter_path or (train_out / "final_adapter")

    modelfile_path = write_ollama_modelfile(args.gguf_dir)
    print(f"[export] Wrote Ollama Modelfile template: {modelfile_path}")

    eval_table_path = eval_out / "comparison_table.md"
    eval_table_md = eval_table_path.read_text(encoding="utf-8") if eval_table_path.exists() else None
    training_summary = _load_optional_json(train_out / "training_summary.json")

    model_card = build_hf_model_card(
        hf_repo=args.hf_repo,
        base_model=cfg.model.base_model,
        eval_table_md=eval_table_md,
        training_summary=training_summary,
    )

    if adapter_dir.exists():
        (adapter_dir / "README.md").write_text(model_card, encoding="utf-8")
        print(f"[export] Wrote Hugging Face Model Card: {adapter_dir / 'README.md'}")
    else:
        preview_card = args.gguf_dir / "HF_MODEL_CARD_PREVIEW.md"
        preview_card.write_text(model_card, encoding="utf-8")
        print(f"[export] Adapter not present locally yet; wrote card preview to {preview_card}")

    if args.push_to_hub:
        if not adapter_dir.exists():
            raise FileNotFoundError(f"Cannot push missing adapter directory: {adapter_dir}")
        from huggingface_hub import HfApi

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        api = HfApi(token=token)
        api.create_repo(repo_id=args.hf_repo, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(
            folder_path=str(adapter_dir),
            repo_id=args.hf_repo,
            repo_type="model",
            commit_message="Upload Qwen2.5-7B Sanad QLoRA adapter and Model Card",
        )
        print(f"[export] Uploaded adapter to https://huggingface.co/{args.hf_repo}")

    if args.export_gguf:
        if not adapter_dir.exists():
            raise FileNotFoundError(f"Cannot export GGUF from missing adapter directory: {adapter_dir}")
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(adapter_dir),
            max_seq_length=cfg.model.max_seq_length,
            load_in_4bit=True,
        )
        args.gguf_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained_gguf(
            str(args.gguf_dir),
            tokenizer,
            quantization_method="q4_k_m",
        )
        gguf_files = sorted(args.gguf_dir.glob("*.gguf"), key=lambda p: p.stat().st_mtime, reverse=True)
        if gguf_files:
            modelfile_path = write_ollama_modelfile(args.gguf_dir, gguf_filename=gguf_files[0].name)
        print(f"[export] GGUF export finished in {args.gguf_dir}. Run: ollama create sanad-qwen25-ft -f {modelfile_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
