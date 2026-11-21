"""Canonical chat schema, prompt formatting, and deterministic citation/handoff parsers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

SplitName = Literal["train", "val", "test"]
BehaviorType = Literal["grounded_single", "grounded_multi", "handoff"]
LanguageTag = Literal["ar_eg", "en", "en_to_ar", "ar_to_en"]

SANAD_SYSTEM_PROMPT = (
    "You are Sanad, a bilingual customer-support assistant. "
    "Reply in natural Egyptian Arabic for Arabic questions and clear English for English questions. "
    "Answer strictly from the provided numbered context chunks [1], [2], ... and cite every factual claim inline using [n]. "
    "Never invent policies, prices, timeframes, or order details. "
    "If the provided chunks do not contain enough information to answer the question, respond strictly starting with "
    "'HANDOFF | ' followed by a short polite escalation message in the customer's language."
)

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
_CHUNK_LINE_PATTERN = re.compile(r"^\[(\d+)\]\s+(.+)$")
_HANDOFF_PREFIX = "HANDOFF |"


@dataclass(frozen=True)
class TurnVariant:
    """Single customer-question and target-answer pair tied to a scenario's chunks."""

    language: LanguageTag
    question: str
    answer: str


@dataclass(frozen=True)
class ScenarioSpec:
    """Disjoint knowledge-base scenario group kept intact inside one dataset split."""

    scenario_group: str
    split: SplitName
    domain: str
    behavior: BehaviorType
    supporting_chunks: tuple[str, ...]
    variants: tuple[TurnVariant, ...]
    handoff_subtype: str | None = None
    anchor_chunks: tuple[str, ...] = field(default_factory=tuple)


def format_user_prompt(chunks: Sequence[str], question: str) -> str:
    """Format retrieved chunks and a customer question into Sanad's RAG user message."""
    if not chunks:
        raise ValueError("chunks must be non-empty")
    numbered = "\n".join(f"[{idx}] {chunk.strip()}" for idx, chunk in enumerate(chunks, start=1))
    return f"Context:\n{numbered}\n\nQuestion: {question.strip()}"


def parse_user_prompt(user_content: str) -> tuple[dict[int, str], str]:
    """Parse a formatted RAG user message back into numbered chunks and the customer question."""
    if not user_content.startswith("Context:\n"):
        raise ValueError("User prompt must start with 'Context:\\n'")
    parts = user_content.split("\n\nQuestion:", maxsplit=1)
    if len(parts) != 2:
        raise ValueError("User prompt must contain '\\n\\nQuestion:' separator")

    context_block = parts[0][len("Context:\n") :]
    question = parts[1].strip()
    if not question:
        raise ValueError("Parsed question is empty")

    chunks: dict[int, str] = {}
    for line in context_block.splitlines():
        line = line.strip()
        if not line:
            continue
        match = _CHUNK_LINE_PATTERN.match(line)
        if not match:
            raise ValueError(f"Malformed chunk line: {line!r}")
        idx = int(match.group(1))
        text = match.group(2).strip()
        if idx in chunks:
            raise ValueError(f"Duplicate chunk index [{idx}]")
        chunks[idx] = text

    expected_keys = list(range(1, len(chunks) + 1))
    if sorted(chunks.keys()) != expected_keys:
        raise ValueError(f"Chunk indices must be contiguous 1..N, got {sorted(chunks.keys())}")
    return chunks, question


def extract_citations(text: str) -> list[int]:
    """Extract unique sorted integer citation markers [n] from an assistant reply."""
    return sorted({int(match.group(1)) for match in _CITATION_PATTERN.finditer(text)})


def is_handoff_response(text: str) -> bool:
    """Return True if the assistant reply starts with the strict 'HANDOFF |' prefix."""
    return text.lstrip().startswith(_HANDOFF_PREFIX)


def parse_handoff_message(text: str) -> str | None:
    """Return the escalation text after 'HANDOFF |', or None if not a valid handoff reply."""
    stripped = text.strip()
    if not stripped.startswith(_HANDOFF_PREFIX):
        return None
    reason = stripped[len(_HANDOFF_PREFIX) :].strip()
    return reason if reason else None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a UTF-8 JSONL file into a list of parsed JSON objects."""
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_num} invalid JSON: {exc}") from exc
    return records
