"""Deterministic scoring for citation compliance, HANDOFF accuracy, and formatting."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from src.data.schema import (
    extract_citations,
    is_handoff_response,
    parse_handoff_message,
    parse_user_prompt,
)

_ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF]")
_LATIN_CHAR_RE = re.compile(r"[A-Za-z]")


@dataclass(frozen=True)
class SampleEvaluation:
    """Per-example evaluation diagnostics saved alongside predictions."""

    id: str
    behavior: str
    language: str
    expected_handoff: bool
    expected_citations: list[int]
    pred_is_handoff: bool
    pred_citations: list[int]
    in_range_citations: list[int]
    out_of_range_citations: list[int]
    language_match: bool
    citation_present_valid: bool | None
    exact_citation_match: bool | None
    strict_format_ok: bool
    strict_end_to_end_ok: bool


@dataclass(frozen=True)
class EvalSummary:
    """Aggregate benchmark metrics for a single model run."""

    model_label: str
    total_examples: int
    grounded_examples: int
    handoff_examples: int
    handoff_accuracy_pct: float
    handoff_precision_pct: float
    handoff_recall_pct: float
    handoff_f1_pct: float
    false_handoff_rate_pct: float
    missed_handoff_rate_pct: float
    citation_compliance_pct: float
    exact_citation_match_pct: float
    phantom_citation_rate_pct: float
    language_match_pct: float
    strict_format_compliance_pct: float
    strict_end_to_end_pass_pct: float
    arabic_strict_pass_pct: float
    english_strict_pass_pct: float
    by_behavior_pass_pct: dict[str, float]


def _pct(num: int, den: int) -> float:
    return round((num / den) * 100.0, 2) if den > 0 else 0.0


def matches_expected_language(prediction: str, language_tag: str) -> bool:
    """Check whether the reply body matches the customer's expected reply language."""
    body = prediction.strip()
    if body.startswith("HANDOFF |"):
        body = body[len("HANDOFF |") :].strip()
    if not body:
        return False

    arabic_count = len(_ARABIC_CHAR_RE.findall(body))
    latin_count = len(_LATIN_CHAR_RE.findall(body))

    if language_tag in {"ar_eg", "en_to_ar"}:
        total_alpha = arabic_count + latin_count
        if arabic_count < 3 or total_alpha == 0:
            return False
        # Allow Egyptian Arabic + English brand/tech terms (e.g. InstaPay, ValU, B2B)
        return (arabic_count / total_alpha) >= 0.30

    if language_tag in {"en", "ar_to_en"}:
        return arabic_count == 0 and latin_count >= 3

    return True


def evaluate_sample(record: Mapping[str, Any], prediction: str) -> SampleEvaluation:
    """Score a single model output against ground-truth Sanad metadata."""
    meta = record["metadata"]
    num_chunks = int(
        meta.get("num_chunks")
        or len(parse_user_prompt(record["messages"][1]["content"])[0])
    )
    expected_handoff = bool(meta["expected_handoff"])
    expected_citations = sorted(int(c) for c in meta["expected_citations"])
    behavior = str(meta["behavior"])
    language = str(meta["language"])

    pred_text = prediction.strip()
    pred_is_handoff = is_handoff_response(pred_text)
    pred_handoff_msg = parse_handoff_message(pred_text)
    pred_mentions_handoff = "HANDOFF" in pred_text.upper()

    pred_citations = extract_citations(pred_text)
    in_range = [c for c in pred_citations if 1 <= c <= num_chunks]
    out_of_range = [c for c in pred_citations if c < 1 or c > num_chunks]
    lang_ok = matches_expected_language(pred_text, language)

    if expected_handoff:
        citation_present_valid: bool | None = None
        exact_citation_match: bool | None = None
        strict_format_ok = (
            pred_is_handoff
            and bool(pred_handoff_msg and len(pred_handoff_msg) >= 10)
            and len(pred_citations) == 0
            and lang_ok
        )
        strict_end_to_end_ok = strict_format_ok
    else:
        citation_present_valid = (
            (not pred_mentions_handoff)
            and len(in_range) >= 1
            and len(out_of_range) == 0
        )
        exact_citation_match = (
            (not pred_mentions_handoff) and pred_citations == expected_citations
        )
        strict_format_ok = bool(citation_present_valid and lang_ok)
        strict_end_to_end_ok = bool(exact_citation_match and lang_ok)

    return SampleEvaluation(
        id=str(record["id"]),
        behavior=behavior,
        language=language,
        expected_handoff=expected_handoff,
        expected_citations=expected_citations,
        pred_is_handoff=pred_is_handoff,
        pred_citations=pred_citations,
        in_range_citations=in_range,
        out_of_range_citations=out_of_range,
        language_match=lang_ok,
        citation_present_valid=citation_present_valid,
        exact_citation_match=exact_citation_match,
        strict_format_ok=strict_format_ok,
        strict_end_to_end_ok=strict_end_to_end_ok,
    )


def score_predictions(
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[str],
    *,
    model_label: str,
) -> tuple[EvalSummary, list[dict[str, Any]]]:
    """Compute full Before/After benchmark metrics and per-sample diagnostics."""
    if len(records) != len(predictions):
        raise ValueError(
            f"Record count ({len(records)}) != prediction count ({len(predictions)})"
        )
    if not records:
        raise ValueError("Cannot score an empty dataset split")

    sample_evals = [evaluate_sample(rec, pred) for rec, pred in zip(records, predictions)]

    tp = fp = fn = tn = 0
    grounded_count = 0
    handoff_count = 0
    citation_present_ok = 0
    exact_cit_ok = 0
    phantom_errors = 0
    lang_ok_count = 0
    strict_format_ok_count = 0
    strict_e2e_ok_count = 0

    ar_total = ar_ok = 0
    en_total = en_ok = 0
    behavior_totals: dict[str, int] = {}
    behavior_ok: dict[str, int] = {}

    for rec, pred, ev in zip(records, predictions, sample_evals):
        pred_mentions_handoff = "HANDOFF" in pred.upper()
        if ev.expected_handoff:
            handoff_count += 1
            if ev.pred_is_handoff:
                tp += 1
            else:
                fn += 1
        else:
            grounded_count += 1
            if pred_mentions_handoff:
                fp += 1
            else:
                tn += 1
            if ev.citation_present_valid:
                citation_present_ok += 1
            if ev.exact_citation_match:
                exact_cit_ok += 1

        if ev.out_of_range_citations or (ev.pred_is_handoff and ev.pred_citations):
            phantom_errors += 1
        if ev.language_match:
            lang_ok_count += 1
        if ev.strict_format_ok:
            strict_format_ok_count += 1
        if ev.strict_end_to_end_ok:
            strict_e2e_ok_count += 1

        if ev.language in {"ar_eg", "en_to_ar"}:
            ar_total += 1
            if ev.strict_end_to_end_ok:
                ar_ok += 1
        else:
            en_total += 1
            if ev.strict_end_to_end_ok:
                en_ok += 1

        behavior_totals[ev.behavior] = behavior_totals.get(ev.behavior, 0) + 1
        if ev.strict_end_to_end_ok:
            behavior_ok[ev.behavior] = behavior_ok.get(ev.behavior, 0) + 1

    total = len(sample_evals)
    handoff_acc = _pct(tp + tn, total)
    precision = _pct(tp, tp + fp)
    recall = _pct(tp, tp + fn)
    f1 = (
        round((2 * precision * recall) / (precision + recall), 2)
        if (precision + recall) > 0
        else 0.0
    )

    by_behavior_pct = {
        beh: _pct(behavior_ok.get(beh, 0), cnt)
        for beh, cnt in sorted(behavior_totals.items())
    }

    summary = EvalSummary(
        model_label=model_label,
        total_examples=total,
        grounded_examples=grounded_count,
        handoff_examples=handoff_count,
        handoff_accuracy_pct=handoff_acc,
        handoff_precision_pct=precision,
        handoff_recall_pct=recall,
        handoff_f1_pct=f1,
        false_handoff_rate_pct=_pct(fp, grounded_count),
        missed_handoff_rate_pct=_pct(fn, handoff_count),
        citation_compliance_pct=_pct(citation_present_ok, grounded_count),
        exact_citation_match_pct=_pct(exact_cit_ok, grounded_count),
        phantom_citation_rate_pct=_pct(phantom_errors, total),
        language_match_pct=_pct(lang_ok_count, total),
        strict_format_compliance_pct=_pct(strict_format_ok_count, total),
        strict_end_to_end_pass_pct=_pct(strict_e2e_ok_count, total),
        arabic_strict_pass_pct=_pct(ar_ok, ar_total),
        english_strict_pass_pct=_pct(en_ok, en_total),
        by_behavior_pass_pct=by_behavior_pct,
    )
    details = [asdict(ev) for ev in sample_evals]
    return summary, details


def format_comparison_markdown(before: EvalSummary, after: EvalSummary) -> str:
    """Render a Before-vs-After Markdown benchmark table from real EvalSummary objects."""
    rows: list[tuple[str, float, float, bool]] = [
        ("Strict End-to-End Pass %", before.strict_end_to_end_pass_pct, after.strict_end_to_end_pass_pct, True),
        ("Strict Format Compliance %", before.strict_format_compliance_pct, after.strict_format_compliance_pct, True),
        ("Citation Compliance % (Grounded)", before.citation_compliance_pct, after.citation_compliance_pct, True),
        ("Exact Citation Match % (Grounded)", before.exact_citation_match_pct, after.exact_citation_match_pct, True),
        ("HANDOFF Overall Accuracy %", before.handoff_accuracy_pct, after.handoff_accuracy_pct, True),
        ("HANDOFF Recall % (Anti-Hallucination)", before.handoff_recall_pct, after.handoff_recall_pct, True),
        ("HANDOFF Precision %", before.handoff_precision_pct, after.handoff_precision_pct, True),
        ("HANDOFF F1 %", before.handoff_f1_pct, after.handoff_f1_pct, True),
        ("False Handoff Rate % (Lower is better)", before.false_handoff_rate_pct, after.false_handoff_rate_pct, False),
        ("Phantom Citation Rate % (Lower is better)", before.phantom_citation_rate_pct, after.phantom_citation_rate_pct, False),
        ("Language Match %", before.language_match_pct, after.language_match_pct, True),
        ("Arabic Strict Pass %", before.arabic_strict_pass_pct, after.arabic_strict_pass_pct, True),
        ("English Strict Pass %", before.english_strict_pass_pct, after.english_strict_pass_pct, True),
    ]

    lines = [
        f"| Metric (`n={after.total_examples}` held-out test) | {before.model_label} | {after.model_label} | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, b_val, a_val, higher_is_better in rows:
        diff = round(a_val - b_val, 2)
        sign = "+" if diff > 0 else ""
        trend = ""
        if diff != 0:
            improved = (diff > 0) if higher_is_better else (diff < 0)
            trend = " (improved)" if improved else " (regressed)"
        lines.append(f"| {label} | {b_val:.2f}% | {a_val:.2f}% | {sign}{diff:.2f} pp{trend} |")
    return "\n".join(lines)
