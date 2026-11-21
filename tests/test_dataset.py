"""Offline pytest suite for Sanad dataset schema, deterministic generation, and leakage checks."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_dataset import build_all_splits  # noqa: E402
from src.data.schema import (  # noqa: E402
    SANAD_SYSTEM_PROMPT,
    extract_citations,
    format_user_prompt,
    is_handoff_response,
    parse_handoff_message,
    parse_user_prompt,
)
from src.data.validator import validate_dataset_splits  # noqa: E402


def test_prompt_roundtrip_and_citation_parsing() -> None:
    chunks = ["First policy chunk.", "Second policy chunk.", "Third policy chunk."]
    user_msg = format_user_prompt(chunks, "When does my order arrive?")
    parsed_chunks, question = parse_user_prompt(user_msg)

    assert parsed_chunks == {1: "First policy chunk.", 2: "Second policy chunk.", 3: "Third policy chunk."}
    assert question == "When does my order arrive?"
    assert extract_citations("Order arrives in 48 hours [2] with tracking [3][2].") == [2, 3]


def test_handoff_parser() -> None:
    valid = "HANDOFF | المعلومة غير متوفرة حالياً، وهحوّلك لموظف خدمة العملاء."
    assert is_handoff_response(valid)
    assert parse_handoff_message(valid) == "المعلومة غير متوفرة حالياً، وهحوّلك لموظف خدمة العملاء."
    assert not is_handoff_response("تقدر ترجّع المنتج [1].")
    assert parse_handoff_message("HANDOFF |   ") is None


def test_dataset_generation_is_deterministic_and_passes_validation() -> None:
    splits_a = build_all_splits(seed=42)
    splits_b = build_all_splits(seed=42)
    assert splits_a == splits_b

    report = validate_dataset_splits(splits_a)
    assert report.passed, f"Validation errors: {report.errors}"
    assert 400 <= report.total_examples <= 600
    assert 82.0 <= report.split_percentages["train"] <= 87.0
    assert sum(report.train_single_citation_positions.values()) > 0
    assert set(report.train_single_citation_positions.keys()).issuperset({"[1]", "[2]", "[3]"})


def test_validator_rejects_cross_split_chunk_leakage() -> None:
    splits = copy.deepcopy(build_all_splits(seed=42))
    # Inject a train user message into val to simulate chunk + question leakage
    leaked = copy.deepcopy(splits["train"][0])
    leaked["id"] = "sanad_val_9999"
    leaked["metadata"]["split"] = "val"
    splits["val"].append(leaked)

    report = validate_dataset_splits(splits)
    assert not report.passed
    assert any("leakage" in err.lower() for err in report.errors)


def test_validator_rejects_phantom_citation() -> None:
    splits = copy.deepcopy(build_all_splits(seed=42))
    grounded_idx = next(
        idx for idx, row in enumerate(splits["train"]) if not row["metadata"]["expected_handoff"]
    )
    row = splits["train"][grounded_idx]
    row["messages"][2]["content"] = "التفاصيل مذكورة في السياسة [99]."
    row["messages"][0]["content"] = SANAD_SYSTEM_PROMPT

    report = validate_dataset_splits(splits)
    assert not report.passed
    assert any("phantom citations" in err.lower() for err in report.errors)
