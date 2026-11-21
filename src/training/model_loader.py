"""Shared 4-bit Qwen2.5 loader, GPU VRAM safety check, and TRL version compatibility."""

from __future__ import annotations

import inspect
from typing import Any


def assert_gpu_vram(*, min_vram_gb: float = 12.0, allow_low_vram: bool = False) -> tuple[str, float]:
    """Verify CUDA GPU availability and sufficient VRAM before 7B QLoRA jobs."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU detected. Run 7B QLoRA training/eval on Google Colab (T4 16GB) or Kaggle."
        )

    device_idx = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device_idx)
    total_vram_gb = torch.cuda.get_device_properties(device_idx).total_memory / (1024**3)
    if total_vram_gb < min_vram_gb and not allow_low_vram:
        raise RuntimeError(
            f"Detected {gpu_name} with {total_vram_gb:.1f} GB VRAM (< {min_vram_gb:.1f} GB). "
            "Use Google Colab T4 (16 GB) / Kaggle, or pass --allow-low-vram to override."
        )
    return gpu_name, round(total_vram_gb, 2)


def resolve_compute_dtype(setting: str) -> Any:
    """Resolve compute dtype, automatically choosing float16 on T4/P100 or bfloat16 on Ampere+."""
    import torch

    norm = setting.lower().strip()
    if norm == "bfloat16":
        return torch.bfloat16
    if norm == "float16":
        return torch.float16
    if norm == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    raise ValueError(f"Unsupported compute_dtype: {setting!r}")


def load_hf_4bit_model_and_tokenizer(
    model_name: str,
    *,
    quant_type: str = "nf4",
    use_double_quant: bool = True,
    compute_dtype: Any = None,
    trust_remote_code: bool = False,
    padding_side: str = "right",
) -> tuple[Any, Any]:
    """Load a 4-bit quantized Hugging Face CausalLM and properly configured tokenizer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    dtype = compute_dtype or resolve_compute_dtype("auto")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_use_double_quant=use_double_quant,
        bnb_4bit_compute_dtype=dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype if isinstance(dtype, torch.dtype) else None,
    )
    return model, tokenizer


def filter_supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Filter kwargs against callable signature for smooth TRL/Transformers cross-version support."""
    params = inspect.signature(callable_obj).parameters
    return {k: v for k, v in kwargs.items() if k in params}
