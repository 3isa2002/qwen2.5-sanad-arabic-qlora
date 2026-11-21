"""Before-vs-After evaluation script on held-out Sanad test split.

Measures citation compliance, exact citation match, HANDOFF accuracy/recall/F1,
language matching, and strict formatting for Base Qwen2.5-7B-Instruct vs Fine-Tuned LoRA.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.metrics import (  # noqa: E402
    EvalSummary,
    format_comparison_markdown,
    score_predictions,
)
from src.training.config import load_config, resolve_project_path  # noqa: E402
from src.training.dataset_prep import load_jsonl, render_chat_text  # noqa: E402
from src.training.model_loader import (  # noqa: E402
    assert_gpu_vram,
    load_hf_4bit_model_and_tokenizer,
    resolve_compute_dtype,
)
from src.training.tracking import (  # noqa: E402
    TrackerKind,
    finish_experiment_tracker,
    init_experiment_tracker,
    log_tracker_metrics,
)


def _generate_predictions(
    model: Any,
    tokenizer: Any,
    records: Sequence[dict[str, Any]],
    *,
    max_new_tokens: int,
    batch_size: int,
) -> list[str]:
    import torch

    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True

    prompts = [
        render_chat_text(rec["messages"][:-1], tokenizer, add_generation_prompt=True)
        for rec in records
    ]
    predictions: list[str] = []

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1536,
        ).to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        for row_idx in range(outputs.shape[0]):
            gen_tokens = outputs[row_idx, prompt_len:]
            decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            predictions.append(decoded)

    return predictions


def _save_predictions_jsonl(
    path: Path,
    records: Sequence[dict[str, Any]],
    predictions: Sequence[str],
    details: Sequence[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for rec, pred, diag in zip(records, predictions, details):
            row = {
                "id": rec["id"],
                "question": rec["messages"][1]["content"].split("\n\nQuestion:", maxsplit=1)[-1].strip(),
                "ground_truth": rec["messages"][2]["content"],
                "prediction": pred,
                "diagnostics": diag,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_side_by_side_markdown(
    records: Sequence[dict[str, Any]],
    before_preds: Sequence[str],
    after_preds: Sequence[str],
    before_details: Sequence[dict[str, Any]],
    after_details: Sequence[dict[str, Any]],
    max_examples: int = 8,
) -> str:
    """Highlight real Before-vs-After examples (prioritizing flipped failures -> passes)."""
    ranked: list[tuple[int, int]] = []
    for idx, (b_d, a_d) in enumerate(zip(before_details, after_details)):
        b_ok = bool(b_d["strict_end_to_end_ok"])
        a_ok = bool(a_d["strict_end_to_end_ok"])
        # Priority 0: Before failed & After passed; 1: both passed; 2: After failed
        priority = 0 if (not b_ok and a_ok) else (1 if a_ok else 2)
        ranked.append((priority, idx))
    ranked.sort()

    lines = ["# Qualitative Before vs After Examples (Held-Out Test Set)", ""]
    for _, idx in ranked[:max_examples]:
        rec = records[idx]
        b_ok = before_details[idx]["strict_end_to_end_ok"]
        a_ok = after_details[idx]["strict_end_to_end_ok"]
        q = rec["messages"][1]["content"].split("\n\nQuestion:", maxsplit=1)[-1].strip()
        lines.extend(
            [
                f"## {rec['id']} (`{rec['metadata']['behavior']}` | `{rec['metadata']['language']}`)",
                f"- **Question:** {q}",
                f"- **Expected:** `{rec['messages'][2]['content']}`",
                f"- **Base (`{'PASS' if b_ok else 'FAIL'}`):** `{before_preds[idx]}`",
                f"- **Fine-Tuned (`{'PASS' if a_ok else 'FAIL'}`):** `{after_preds[idx]}`",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Base vs Fine-Tuned Qwen2.5-7B on Sanad.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "qlora_qwen25_7b.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--split",
        choices=("test", "val"),
        default="test",
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--mode",
        choices=("both", "base", "adapter"),
        default="both",
        help="Run base only, adapter only, or both Before vs After in one GPU pass.",
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=None,
        help="Path to trained LoRA adapter directory (defaults to <training.output_dir>/final_adapter).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write evaluation JSON/JSONL/Markdown reports.",
    )
    parser.add_argument(
        "--tracker",
        choices=("wandb", "mlflow", "none"),
        default="none",
        help="Optional tracker to log benchmark comparison metrics.",
    )
    parser.add_argument("--allow-low-vram", action="store_true", help="Override 12 GB GPU check.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    split_rel = cfg.data.test_file if args.split == "test" else cfg.data.val_file
    records = load_jsonl(resolve_project_path(PROJECT_ROOT, split_rel))

    output_dir = args.output_dir or resolve_project_path(PROJECT_ROOT, cfg.eval.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = args.adapter_path or (
        resolve_project_path(PROJECT_ROOT, cfg.training.output_dir) / "final_adapter"
    )

    gpu_name, total_vram_gb = assert_gpu_vram(min_vram_gb=12.0, allow_low_vram=args.allow_low_vram)
    compute_dtype = resolve_compute_dtype(cfg.qlora.compute_dtype)
    print(f"[eval] GPU: {gpu_name} ({total_vram_gb} GB) | split={args.split} (n={len(records)})")

    base_model, tokenizer = load_hf_4bit_model_and_tokenizer(
        cfg.model.base_model,
        quant_type=cfg.qlora.quant_type,
        use_double_quant=cfg.qlora.use_double_quant,
        compute_dtype=compute_dtype,
        trust_remote_code=cfg.model.trust_remote_code,
        padding_side="left",
    )

    before_summary: EvalSummary | None = None
    after_summary: EvalSummary | None = None
    before_preds: list[str] = []
    after_preds: list[str] = []
    before_details: list[dict[str, Any]] = []
    after_details: list[dict[str, Any]] = []

    if args.mode in ("both", "base"):
        print("[eval] Generating Base Qwen2.5-7B-Instruct (Before) predictions...")
        before_preds = _generate_predictions(
            base_model,
            tokenizer,
            records,
            max_new_tokens=cfg.eval.max_new_tokens,
            batch_size=cfg.eval.batch_size,
        )
        before_summary, before_details = score_predictions(
            records,
            before_preds,
            model_label="Base Qwen2.5-7B-Instruct",
        )
        _save_predictions_jsonl(
            output_dir / "before_predictions.jsonl",
            records,
            before_preds,
            before_details,
        )

    if args.mode in ("both", "adapter"):
        if not adapter_dir.exists():
            raise FileNotFoundError(
                f"Trained adapter directory not found at {adapter_dir}. Run scripts/train.py first."
            )
        from peft import PeftModel

        print(f"[eval] Attaching LoRA adapter from {adapter_dir} (After)...")
        ft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        after_preds = _generate_predictions(
            ft_model,
            tokenizer,
            records,
            max_new_tokens=cfg.eval.max_new_tokens,
            batch_size=cfg.eval.batch_size,
        )
        after_summary, after_details = score_predictions(
            records,
            after_preds,
            model_label="Sanad QLoRA (Ours)",
        )
        _save_predictions_jsonl(
            output_dir / "after_predictions.jsonl",
            records,
            after_preds,
            after_details,
        )

    report_payload: dict[str, Any] = {
        "split": args.split,
        "num_examples": len(records),
        "gpu_name": gpu_name,
        "before": asdict(before_summary) if before_summary else None,
        "after": asdict(after_summary) if after_summary else None,
    }
    (output_dir / "comparison_report.json").write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if before_summary and after_summary:
        table_md = format_comparison_markdown(before_summary, after_summary)
        (output_dir / "comparison_table.md").write_text(table_md + "\n", encoding="utf-8")
        side_md = _build_side_by_side_markdown(
            records, before_preds, after_preds, before_details, after_details
        )
        (output_dir / "side_by_side_examples.md").write_text(side_md, encoding="utf-8")
        print("\n=== Before vs After Benchmark ===")
        print(table_md)
    else:
        single = after_summary or before_summary
        print(json.dumps(asdict(single), indent=2) if single else "No metrics computed.")

    tracker: TrackerKind = args.tracker  # type: ignore[assignment]
    if tracker != "none":
        _, handle = init_experiment_tracker(
            tracker,
            cfg,
            run_name=f"{cfg.tracking.run_name}-eval-{args.split}",
            tags=["eval", args.split],
        )
        try:
            if before_summary:
                log_tracker_metrics(tracker, asdict(before_summary), prefix="eval_before/")
            if after_summary:
                log_tracker_metrics(tracker, asdict(after_summary), prefix="eval_after/")
        finally:
            finish_experiment_tracker(tracker, handle)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
