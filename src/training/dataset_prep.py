"""Dataset loading and chat-format conversion helpers for SFTTrainer and inference."""

from __future__ import annotations

from typing import Any, Mapping

from src.data.schema import load_jsonl

__all__ = ["load_jsonl", "to_prompt_completion", "render_chat_text"]


def to_prompt_completion(record: Mapping[str, Any]) -> dict[str, list[dict[str, str]]]:
    """Convert 3-turn `messages` into TRL conversational `prompt`/`completion` turns."""
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("Record must contain at least 2 messages (prompt + assistant)")
    if messages[-1].get("role") != "assistant":
        raise ValueError("Last message role must be 'assistant'")
    return {
        "prompt": [
            {"role": str(m["role"]), "content": str(m["content"])} for m in messages[:-1]
        ],
        "completion": [
            {"role": "assistant", "content": str(messages[-1]["content"])}
        ],
    }


def render_chat_text(
    messages: list[dict[str, str]],
    tokenizer: Any,
    *,
    add_generation_prompt: bool,
) -> str:
    """Render chat turns using the model tokenizer's native chat template."""
    return str(
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    )
