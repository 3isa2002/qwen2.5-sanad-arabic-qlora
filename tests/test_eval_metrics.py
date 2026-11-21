"""Offline pytest suite for config loading, prompt-completion prep, and eval metrics."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_dataset import build_all_splits  # noqa: E402
from scripts.export_hf_and_gguf import build_hf_model_card, write_ollama_modelfile  # noqa: E402
from src.eval.metrics import (  # noqa: E402
    format_comparison_markdown,
    matches_expected_language,
    score_predictions,
)
from src.training.config import load_config  # noqa: E402
from src.training.dataset_prep import to_prompt_completion  # noqa: E402


def test_yaml_config_loads_cleanly() -> None:
    cfg = load_config(PROJECT_ROOT / "configs" / "qlora_qwen25_7b.yaml")
    assert cfg.model.base_model == "Qwen/Qwen2.5-7B-Instruct"
    assert cfg.qlora.quant_type == "nf4"
    assert cfg.lora.r == 16
    assert cfg.lora.lora_alpha == 32
    assert cfg.data.completion_only_loss is True
    assert cfg.eval.do_sample is False


def test_to_prompt_completion_preserves_turns() -> None:
    splits = build_all_splits(seed=42)
    sample = splits["train"][0]
    converted = to_prompt_completion(sample)
    assert len(converted["prompt"]) == 2
    assert converted["prompt"][0]["role"] == "system"
    assert converted["prompt"][1]["role"] == "user"
    assert len(converted["completion"]) == 1
    assert converted["completion"][0]["role"] == "assistant"
    assert converted["completion"][0]["content"] == sample["messages"][2]["content"]


def test_language_matching_handles_egyptian_with_tech_terms() -> None:
    assert matches_expected_language("الدفع عبر InstaPay وValU متاح بدون رسوم إضافية [1].", "ar_eg")
    assert matches_expected_language("HANDOFF | المعلومة مش متوفرة حالياً، وهحوّلك لموظف خدمة العملاء.", "ar_eg")
    assert not matches_expected_language("Sorry, I do not have that information.", "ar_eg")
    assert matches_expected_language("Standard delivery takes 24 to 48 hours [2].", "en")
    assert not matches_expected_language("التوصيل خلال 48 ساعة [2].", "en")


def test_score_predictions_perfect_vs_degraded() -> None:
    test_records = build_all_splits(seed=42)["test"]
    perfect_preds = [r["messages"][2]["content"] for r in test_records]

    perfect_summary, _ = score_predictions(
        test_records, perfect_preds, model_label="Sanad QLoRA (Oracle)"
    )
    assert perfect_summary.total_examples == len(test_records)
    assert perfect_summary.strict_end_to_end_pass_pct == 100.0
    assert perfect_summary.handoff_accuracy_pct == 100.0
    assert perfect_summary.citation_compliance_pct == 100.0
    assert perfect_summary.exact_citation_match_pct == 100.0
    assert perfect_summary.phantom_citation_rate_pct == 0.0
    assert perfect_summary.false_handoff_rate_pct == 0.0

    # Degrade a copy: omit citations on grounded #0, hallucinate on first handoff, inject phantom [99]
    degraded_preds = copy.deepcopy(perfect_preds)
    first_grounded = next(i for i, r in enumerate(test_records) if not r["metadata"]["expected_handoff"])
    first_handoff = next(i for i, r in enumerate(test_records) if r["metadata"]["expected_handoff"])
    second_grounded = next(
        i for i, r in enumerate(test_records) if i != first_grounded and not r["metadata"]["expected_handoff"]
    )
    degraded_preds[first_grounded] = "الشحن خلال 48 ساعة بدون ذكر رقم مصدر."
    degraded_preds[first_handoff] = "أيوة طبعاً لدينا فرع في التجمع الخامس مفتوح يومياً."
    degraded_preds[second_grounded] = "موجود في السياسات [99]."

    degraded_summary, _ = score_predictions(
        test_records, degraded_preds, model_label="Base Qwen2.5-7B-Instruct"
    )
    assert degraded_summary.citation_compliance_pct < 100.0
    assert degraded_summary.handoff_recall_pct < 100.0
    assert degraded_summary.phantom_citation_rate_pct > 0.0

    table_md = format_comparison_markdown(degraded_summary, perfect_summary)
    assert "Base Qwen2.5-7B-Instruct" in table_md
    assert "Sanad QLoRA (Oracle)" in table_md
    assert "(improved)" in table_md


def test_export_helpers_generate_clean_files_without_fake_metrics(tmp_path: Path) -> None:
    modelfile = write_ollama_modelfile(tmp_path, gguf_filename="test.gguf")
    text = modelfile.read_text(encoding="utf-8")
    assert "FROM ./test.gguf" in text
    assert "HANDOFF |" in text

    card = build_hf_model_card(
        hf_repo="3isa2002/qwen2.5-7b-sanad-arabic-qlora",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        eval_table_md=None,
        training_summary=None,
    )
    assert "Pending" in card or "pending" in card
