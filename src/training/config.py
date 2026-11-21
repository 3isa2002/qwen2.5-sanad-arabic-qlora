"""Typed YAML configuration loader and validation for Sanad QLoRA."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    base_model: str
    max_seq_length: int
    trust_remote_code: bool = False


@dataclass(frozen=True)
class DataConfig:
    train_file: str
    val_file: str
    test_file: str
    completion_only_loss: bool = True


@dataclass(frozen=True)
class QloraConfig:
    load_in_4bit: bool
    quant_type: str
    use_double_quant: bool
    compute_dtype: str = "auto"


@dataclass(frozen=True)
class LoraSettings:
    r: int
    lora_alpha: int
    lora_dropout: float
    bias: str
    target_modules: tuple[str, ...]


@dataclass(frozen=True)
class TrainingSettings:
    output_dir: str
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    lr_scheduler_type: str
    warmup_ratio: float
    weight_decay: float
    max_grad_norm: float
    optim: str
    gradient_checkpointing: bool
    logging_steps: int
    eval_steps: int
    save_steps: int
    save_total_limit: int
    load_best_model_at_end: bool
    metric_for_best_model: str
    seed: int


@dataclass(frozen=True)
class TrackingSettings:
    default_tracker: str
    wandb_project: str
    mlflow_experiment: str
    run_name: str


@dataclass(frozen=True)
class EvalSettings:
    output_dir: str
    max_new_tokens: int
    do_sample: bool
    temperature: float
    batch_size: int


@dataclass(frozen=True)
class ProjectConfig:
    model: ModelConfig
    data: DataConfig
    qlora: QloraConfig
    lora: LoraSettings
    training: TrainingSettings
    tracking: TrackingSettings
    eval: EvalSettings


def _require_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    val = raw.get(key)
    if not isinstance(val, dict):
        raise ValueError(f"Missing or invalid config section: {key!r}")
    return val


def load_config(path: Path) -> ProjectConfig:
    """Load and validate the YAML training/evaluation configuration."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping in {path}")

    m = _require_mapping(raw, "model")
    d = _require_mapping(raw, "data")
    q = _require_mapping(raw, "qlora")
    l = _require_mapping(raw, "lora")
    t = _require_mapping(raw, "training")
    tr = _require_mapping(raw, "tracking")
    ev = _require_mapping(raw, "eval")

    lora_targets = tuple(str(x) for x in l["target_modules"])
    if not lora_targets:
        raise ValueError("lora.target_modules must be non-empty")

    return ProjectConfig(
        model=ModelConfig(
            base_model=str(m["base_model"]),
            max_seq_length=int(m["max_seq_length"]),
            trust_remote_code=bool(m.get("trust_remote_code", False)),
        ),
        data=DataConfig(
            train_file=str(d["train_file"]),
            val_file=str(d["val_file"]),
            test_file=str(d["test_file"]),
            completion_only_loss=bool(d.get("completion_only_loss", True)),
        ),
        qlora=QloraConfig(
            load_in_4bit=bool(q["load_in_4bit"]),
            quant_type=str(q["quant_type"]),
            use_double_quant=bool(q["use_double_quant"]),
            compute_dtype=str(q.get("compute_dtype", "auto")),
        ),
        lora=LoraSettings(
            r=int(l["r"]),
            lora_alpha=int(l["lora_alpha"]),
            lora_dropout=float(l["lora_dropout"]),
            bias=str(l["bias"]),
            target_modules=lora_targets,
        ),
        training=TrainingSettings(
            output_dir=str(t["output_dir"]),
            num_train_epochs=int(t["num_train_epochs"]),
            per_device_train_batch_size=int(t["per_device_train_batch_size"]),
            per_device_eval_batch_size=int(t["per_device_eval_batch_size"]),
            gradient_accumulation_steps=int(t["gradient_accumulation_steps"]),
            learning_rate=float(t["learning_rate"]),
            lr_scheduler_type=str(t["lr_scheduler_type"]),
            warmup_ratio=float(t["warmup_ratio"]),
            weight_decay=float(t["weight_decay"]),
            max_grad_norm=float(t["max_grad_norm"]),
            optim=str(t["optim"]),
            gradient_checkpointing=bool(t["gradient_checkpointing"]),
            logging_steps=int(t["logging_steps"]),
            eval_steps=int(t["eval_steps"]),
            save_steps=int(t["save_steps"]),
            save_total_limit=int(t["save_total_limit"]),
            load_best_model_at_end=bool(t["load_best_model_at_end"]),
            metric_for_best_model=str(t["metric_for_best_model"]),
            seed=int(t["seed"]),
        ),
        tracking=TrackingSettings(
            default_tracker=str(tr["default_tracker"]),
            wandb_project=str(tr["wandb_project"]),
            mlflow_experiment=str(tr["mlflow_experiment"]),
            run_name=str(tr["run_name"]),
        ),
        eval=EvalSettings(
            output_dir=str(ev["output_dir"]),
            max_new_tokens=int(ev["max_new_tokens"]),
            do_sample=bool(ev["do_sample"]),
            temperature=float(ev["temperature"]),
            batch_size=int(ev["batch_size"]),
        ),
    )


def resolve_project_path(project_root: Path, path_str: str) -> Path:
    """Resolve a relative path against project_root while preserving absolute paths."""
    candidate = Path(path_str)
    return candidate if candidate.is_absolute() else (project_root / candidate)
