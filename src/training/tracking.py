"""Experiment tracking setup for Weights & Biases (W&B) and MLflow."""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any, Literal

from src.training.config import ProjectConfig

TrackerKind = Literal["wandb", "mlflow", "none"]


def init_experiment_tracker(
    tracker: TrackerKind,
    cfg: ProjectConfig,
    *,
    run_name: str | None = None,
    tags: list[str] | None = None,
) -> tuple[list[str], Any]:
    """Initialize W&B or MLflow cleanly and return `(report_to_list, active_handle)`."""
    effective_name = run_name or cfg.tracking.run_name
    hyperparams = {
        "base_model": cfg.model.base_model,
        "max_seq_length": cfg.model.max_seq_length,
        "lora_r": cfg.lora.r,
        "lora_alpha": cfg.lora.lora_alpha,
        "lora_dropout": cfg.lora.lora_dropout,
        "learning_rate": cfg.training.learning_rate,
        "epochs": cfg.training.num_train_epochs,
        "effective_batch_size": (
            cfg.training.per_device_train_batch_size * cfg.training.gradient_accumulation_steps
        ),
        "completion_only_loss": cfg.data.completion_only_loss,
        "quant_type": cfg.qlora.quant_type,
    }

    if tracker == "none":
        os.environ.setdefault("WANDB_DISABLED", "true")
        return [], None

    if tracker == "wandb":
        # Check API key / netrc before wandb.init so Colab never stalls on interactive prompts.
        netrc_path = os.path.expanduser("~/.netrc")
        has_netrc = os.path.exists(netrc_path) and "api.wandb.ai" in open(netrc_path, encoding="utf-8", errors="ignore").read()
        has_key = bool(os.environ.get("WANDB_API_KEY")) or has_netrc
        wandb_mode = "online" if has_key else "offline"
        if not has_key:
            os.environ["WANDB_MODE"] = "offline"
            print("[tracking] WANDB_API_KEY not found; running W&B in non-interactive offline mode.")

        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("wandb is not installed; run pip install wandb or pass --tracker none") from exc

        run = wandb.init(
            project=cfg.tracking.wandb_project,
            name=effective_name,
            mode=wandb_mode,
            config={**hyperparams, "full_config": asdict(cfg)},
            tags=tags or ["qwen2.5-7b", "qlora", "sanad", "arabic"],
            reinit=True,
        )
        return ["wandb"], run

    if tracker == "mlflow":
        os.environ.setdefault("WANDB_DISABLED", "true")
        try:
            import mlflow
        except ImportError as exc:
            raise RuntimeError("mlflow is not installed; run pip install mlflow or pass --tracker none") from exc

        mlflow.set_experiment(cfg.tracking.mlflow_experiment)
        run = mlflow.start_run(run_name=effective_name)
        mlflow.log_params(hyperparams)
        return ["mlflow"], run

    raise ValueError(f"Unsupported tracker: {tracker!r}")


def log_tracker_metrics(tracker: TrackerKind, metrics: dict[str, Any], *, prefix: str = "") -> None:
    """Log flat numeric/string summary metrics to the active tracker."""
    if tracker == "none":
        return

    prefixed = {f"{prefix}{k}": v for k, v in metrics.items() if isinstance(v, (int, float, bool))}
    if not prefixed:
        return

    if tracker == "wandb":
        import wandb

        if wandb.run is not None:
            wandb.log(prefixed)
    elif tracker == "mlflow":
        import mlflow

        if mlflow.active_run() is not None:
            numeric = {k: float(v) for k, v in prefixed.items()}
            mlflow.log_metrics(numeric)


def finish_experiment_tracker(tracker: TrackerKind, handle: Any) -> None:
    """Close active tracking session cleanly."""
    if handle is None or tracker == "none":
        return
    if tracker == "wandb":
        import wandb

        if wandb.run is not None:
            wandb.finish()
    elif tracker == "mlflow":
        import mlflow

        if mlflow.active_run() is not None:
            mlflow.end_run()
