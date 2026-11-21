"""Train Qwen2.5-7B-Instruct with 4-bit QLoRA for Sanad Egyptian Arabic/English RAG support.

Supports both standard Hugging Face (transformers + bitsandbytes + peft + trl SFTTrainer)
and optional --use-unsloth acceleration on Google Colab T4 / Kaggle GPUs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.config import ProjectConfig, load_config, resolve_project_path  # noqa: E402
from src.training.dataset_prep import load_jsonl, render_chat_text, to_prompt_completion  # noqa: E402
from src.training.model_loader import (  # noqa: E402
    assert_gpu_vram,
    filter_supported_kwargs,
    load_hf_4bit_model_and_tokenizer,
    resolve_compute_dtype,
)
from src.training.tracking import (  # noqa: E402
    TrackerKind,
    finish_experiment_tracker,
    init_experiment_tracker,
    log_tracker_metrics,
)


def _count_trainable_parameters(model: Any) -> tuple[int, int, float]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = round((trainable / total) * 100.0, 4) if total else 0.0
    return trainable, total, pct


def _build_hf_datasets(cfg: ProjectConfig, tokenizer: Any, *, use_unsloth: bool) -> tuple[Any, Any]:
    from datasets import Dataset

    train_rows = load_jsonl(resolve_project_path(PROJECT_ROOT, cfg.data.train_file))
    val_rows = load_jsonl(resolve_project_path(PROJECT_ROOT, cfg.data.val_file))

    if use_unsloth:
        train_ds = Dataset.from_list(
            [{"text": render_chat_text(r["messages"], tokenizer, add_generation_prompt=False)} for r in train_rows]
        )
        val_ds = Dataset.from_list(
            [{"text": render_chat_text(r["messages"], tokenizer, add_generation_prompt=False)} for r in val_rows]
        )
        return train_ds, val_ds

    if cfg.data.completion_only_loss:
        train_ds = Dataset.from_list([to_prompt_completion(r) for r in train_rows])
        val_ds = Dataset.from_list([to_prompt_completion(r) for r in val_rows])
    else:
        train_ds = Dataset.from_list([{"messages": r["messages"]} for r in train_rows])
        val_ds = Dataset.from_list([{"messages": r["messages"]} for r in val_rows])
    return train_ds, val_ds


def _train_standard_hf(
    cfg: ProjectConfig,
    output_dir: Path,
    report_to: list[str],
    compute_dtype: Any,
) -> tuple[Any, Any, Any]:
    import torch
    from peft import LoraConfig, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = load_hf_4bit_model_and_tokenizer(
        cfg.model.base_model,
        quant_type=cfg.qlora.quant_type,
        use_double_quant=cfg.qlora.use_double_quant,
        compute_dtype=compute_dtype,
        trust_remote_code=cfg.model.trust_remote_code,
        padding_side="right",
    )
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=cfg.training.gradient_checkpointing,
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        bias=cfg.lora.bias,
        target_modules=list(cfg.lora.target_modules),
        task_type="CAUSAL_LM",
    )

    use_bf16 = compute_dtype == torch.bfloat16
    raw_sft_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": cfg.training.num_train_epochs,
        "per_device_train_batch_size": cfg.training.per_device_train_batch_size,
        "per_device_eval_batch_size": cfg.training.per_device_eval_batch_size,
        "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
        "learning_rate": cfg.training.learning_rate,
        "lr_scheduler_type": cfg.training.lr_scheduler_type,
        "warmup_ratio": cfg.training.warmup_ratio,
        "weight_decay": cfg.training.weight_decay,
        "max_grad_norm": cfg.training.max_grad_norm,
        "optim": cfg.training.optim,
        "fp16": not use_bf16,
        "bf16": use_bf16,
        "gradient_checkpointing": cfg.training.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "logging_steps": cfg.training.logging_steps,
        "eval_strategy": "steps",
        "evaluation_strategy": "steps",
        "eval_steps": cfg.training.eval_steps,
        "save_strategy": "steps",
        "save_steps": cfg.training.save_steps,
        "save_total_limit": cfg.training.save_total_limit,
        "load_best_model_at_end": cfg.training.load_best_model_at_end,
        "metric_for_best_model": cfg.training.metric_for_best_model,
        "greater_is_better": False,
        "max_length": cfg.model.max_seq_length,
        "max_seq_length": cfg.model.max_seq_length,
        "completion_only_loss": cfg.data.completion_only_loss,
        "packing": False,
        "report_to": report_to,
        "seed": cfg.training.seed,
    }
    sft_args = SFTConfig(**filter_supported_kwargs(SFTConfig, raw_sft_kwargs))

    train_ds, val_ds = _build_hf_datasets(cfg, tokenizer, use_unsloth=False)
    raw_trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": sft_args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "peft_config": peft_config,
        "processing_class": tokenizer,
        "tokenizer": tokenizer,
    }
    trainer = SFTTrainer(**filter_supported_kwargs(SFTTrainer, raw_trainer_kwargs))
    return trainer, model, tokenizer


def _train_unsloth(
    cfg: ProjectConfig,
    output_dir: Path,
    report_to: list[str],
    compute_dtype: Any,
) -> tuple[Any, Any, Any]:
    import torch
    from transformers import DataCollatorForSeq2Seq
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model.base_model,
        max_seq_length=cfg.model.max_seq_length,
        dtype=compute_dtype,
        load_in_4bit=cfg.qlora.load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora.r,
        target_modules=list(cfg.lora.target_modules),
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        bias=cfg.lora.bias,
        use_gradient_checkpointing="unsloth",
        random_state=cfg.training.seed,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_bf16 = compute_dtype == torch.bfloat16
    raw_sft_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": cfg.training.num_train_epochs,
        "per_device_train_batch_size": cfg.training.per_device_train_batch_size,
        "per_device_eval_batch_size": cfg.training.per_device_eval_batch_size,
        "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
        "learning_rate": cfg.training.learning_rate,
        "lr_scheduler_type": cfg.training.lr_scheduler_type,
        "warmup_ratio": cfg.training.warmup_ratio,
        "weight_decay": cfg.training.weight_decay,
        "max_grad_norm": cfg.training.max_grad_norm,
        "optim": "adamw_8bit",
        "fp16": not use_bf16,
        "bf16": use_bf16,
        "logging_steps": cfg.training.logging_steps,
        "eval_strategy": "steps",
        "evaluation_strategy": "steps",
        "eval_steps": cfg.training.eval_steps,
        "save_strategy": "steps",
        "save_steps": cfg.training.save_steps,
        "save_total_limit": cfg.training.save_total_limit,
        "load_best_model_at_end": cfg.training.load_best_model_at_end,
        "metric_for_best_model": cfg.training.metric_for_best_model,
        "greater_is_better": False,
        "dataset_text_field": "text",
        "max_length": cfg.model.max_seq_length,
        "max_seq_length": cfg.model.max_seq_length,
        "packing": False,
        "report_to": report_to,
        "seed": cfg.training.seed,
    }
    sft_args = SFTConfig(**filter_supported_kwargs(SFTConfig, raw_sft_kwargs))

    train_ds, val_ds = _build_hf_datasets(cfg, tokenizer, use_unsloth=True)
    raw_trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": sft_args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "data_collator": DataCollatorForSeq2Seq(tokenizer=tokenizer),
        "processing_class": tokenizer,
        "tokenizer": tokenizer,
    }
    trainer = SFTTrainer(**filter_supported_kwargs(SFTTrainer, raw_trainer_kwargs))
    if cfg.data.completion_only_loss:
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )
    return trainer, model, tokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-7B-Instruct for Sanad using QLoRA.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "qlora_qwen25_7b.yaml",
        help="Path to YAML training config.",
    )
    parser.add_argument(
        "--use-unsloth",
        action="store_true",
        help="Use Unsloth kernels for faster training on Colab T4.",
    )
    parser.add_argument(
        "--tracker",
        choices=("wandb", "mlflow", "none"),
        default=None,
        help="Override tracking backend from YAML config.",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Optional experiment run name.")
    parser.add_argument("--epochs", type=int, default=None, help="Optional epoch override.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory override.")
    parser.add_argument(
        "--allow-low-vram",
        action="store_true",
        help="Bypass the 12 GB minimum VRAM safety check.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        from dataclasses import replace

        cfg = replace(cfg, training=replace(cfg.training, num_train_epochs=args.epochs))

    gpu_name, total_vram_gb = assert_gpu_vram(min_vram_gb=12.0, allow_low_vram=args.allow_low_vram)
    compute_dtype = resolve_compute_dtype(cfg.qlora.compute_dtype)

    output_dir = args.output_dir or resolve_project_path(PROJECT_ROOT, cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_adapter_dir = output_dir / "final_adapter"

    tracker: TrackerKind = args.tracker or cfg.tracking.default_tracker  # type: ignore[assignment]
    report_to, tracker_handle = init_experiment_tracker(tracker, cfg, run_name=args.run_name)

    print(f"[train] GPU: {gpu_name} ({total_vram_gb} GB VRAM) | compute_dtype={compute_dtype}")
    print(f"[train] Backend: {'Unsloth + TRL' if args.use_unsloth else 'HF PEFT + TRL'} | tracker={tracker}")

    start_time = time.perf_counter()
    try:
        if args.use_unsloth:
            trainer, _, tokenizer = _train_unsloth(cfg, output_dir, report_to, compute_dtype)
        else:
            trainer, _, tokenizer = _train_standard_hf(cfg, output_dir, report_to, compute_dtype)

        trainable_params, total_params, trainable_pct = _count_trainable_parameters(trainer.model)
        print(
            f"[train] Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_pct:.2f}%)"
        )

        train_result = trainer.train()
        eval_metrics = trainer.evaluate()

        final_adapter_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(str(final_adapter_dir))
        tokenizer.save_pretrained(str(final_adapter_dir))

        elapsed_sec = round(time.perf_counter() - start_time, 2)
        summary = {
            "base_model": cfg.model.base_model,
            "backend": "unsloth" if args.use_unsloth else "hf_peft_trl",
            "gpu_name": gpu_name,
            "gpu_vram_gb": total_vram_gb,
            "compute_dtype": str(compute_dtype),
            "trainable_params": trainable_params,
            "total_params": total_params,
            "trainable_pct": trainable_pct,
            "train_loss": round(float(train_result.training_loss), 4),
            "eval_loss": round(float(eval_metrics.get("eval_loss", 0.0)), 4),
            "runtime_seconds": elapsed_sec,
            "adapter_dir": str(final_adapter_dir),
        }
        (output_dir / "training_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        log_tracker_metrics(tracker, summary, prefix="summary/")
        print(f"[train] Done in {elapsed_sec}s! Adapter saved to {final_adapter_dir}")
        print(json.dumps(summary, indent=2))
    finally:
        finish_experiment_tracker(tracker, tracker_handle)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
