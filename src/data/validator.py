"""Dataset integrity, citation grounding, position-bias, and zero-leakage validation."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.data.schema import (
    SANAD_SYSTEM_PROMPT,
    extract_citations,
    is_handoff_response,
    parse_handoff_message,
    parse_user_prompt,
)

_ARABIC_DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0670\u0640]")
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s\u0600-\u06FF]", re.UNICODE)
_FORBIDDEN_STRINGS = ("{c0}", "{c1}", "<|im_start|>", "<|im_end|>", "<|endoftext|>")


@dataclass(frozen=True)
class ValidationReport:
    """Structured audit summary produced by `validate_dataset_splits`."""

    total_examples: int
    split_counts: dict[str, int]
    split_percentages: dict[str, float]
    behavior_counts_by_split: dict[str, dict[str, int]]
    language_counts_by_split: dict[str, dict[str, int]]
    train_single_citation_positions: dict[str, int]
    max_cross_split_question_jaccard: float
    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0


def normalize_text(text: str) -> str:
    """Normalize Arabic/English text for deterministic deduplication and leakage checks."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = _ARABIC_DIACRITICS_RE.sub("", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ة", "ه").replace("ى", "ي")
    text = _PUNCT_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _word_jaccard(a: str, b: str) -> float:
    tokens_a = set(normalize_text(a).split())
    tokens_b = set(normalize_text(b).split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _validate_record_structure(
    record: Mapping[str, Any],
    expected_split: str,
    seen_ids: set[str],
    errors: list[str],
) -> tuple[dict[int, str], str, list[int]] | None:
    record_id = record.get("id")
    if not isinstance(record_id, str) or not record_id.strip():
        errors.append(f"[{expected_split}] Missing or empty record 'id'.")
        return None
    if record_id in seen_ids:
        errors.append(f"[{expected_split}] Duplicate record id: {record_id}")
    seen_ids.add(record_id)

    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        errors.append(f"[{record_id}] 'messages' must contain exactly 3 turns.")
        return None

    expected_roles = ["system", "user", "assistant"]
    actual_roles = [m.get("role") if isinstance(m, dict) else None for m in messages]
    if actual_roles != expected_roles:
        errors.append(f"[{record_id}] Expected roles {expected_roles}, got {actual_roles}.")
        return None

    contents = [m.get("content") for m in messages]
    if not all(isinstance(c, str) and c.strip() for c in contents):
        errors.append(f"[{record_id}] All message contents must be non-empty strings.")
        return None

    system_msg, user_msg, assistant_msg = contents
    if system_msg != SANAD_SYSTEM_PROMPT:
        errors.append(f"[{record_id}] System prompt does not match canonical SANAD_SYSTEM_PROMPT.")

    for forbidden in _FORBIDDEN_STRINGS:
        if forbidden in user_msg or forbidden in assistant_msg:
            errors.append(f"[{record_id}] Found forbidden unrendered/special token {forbidden!r}.")

    try:
        chunks, question = parse_user_prompt(user_msg)
    except ValueError as exc:
        errors.append(f"[{record_id}] Invalid user prompt format: {exc}")
        return None

    normalized_chunks = [normalize_text(c) for c in chunks.values()]
    if len(set(normalized_chunks)) != len(normalized_chunks):
        errors.append(f"[{record_id}] Duplicate chunks detected inside the same prompt.")

    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        errors.append(f"[{record_id}] Missing or invalid 'metadata' mapping.")
        return None

    if metadata.get("split") != expected_split:
        errors.append(f"[{record_id}] metadata.split={metadata.get('split')!r} != {expected_split!r}.")

    expected_handoff = metadata.get("expected_handoff")
    expected_citations = metadata.get("expected_citations")
    behavior = metadata.get("behavior")
    if not isinstance(expected_handoff, bool) or not isinstance(expected_citations, list):
        errors.append(f"[{record_id}] Invalid expected_handoff or expected_citations in metadata.")
        return None

    found_citations = extract_citations(assistant_msg)
    if expected_handoff:
        if not is_handoff_response(assistant_msg):
            errors.append(f"[{record_id}] Expected HANDOFF response prefix missing.")
        reason = parse_handoff_message(assistant_msg)
        if not reason or len(reason) < 15:
            errors.append(f"[{record_id}] HANDOFF message after '|' is too short or missing.")
        if found_citations:
            errors.append(f"[{record_id}] HANDOFF reply must not contain citations: {found_citations}.")
        if expected_citations:
            errors.append(f"[{record_id}] HANDOFF metadata expected_citations must be empty.")
    else:
        if "HANDOFF" in assistant_msg:
            errors.append(f"[{record_id}] Grounded reply must not contain 'HANDOFF'.")
        if not found_citations:
            errors.append(f"[{record_id}] Grounded reply must contain at least one [n] citation.")
        invalid_refs = [c for c in found_citations if c not in chunks]
        if invalid_refs:
            errors.append(f"[{record_id}] Phantom citations {invalid_refs} not in chunks {sorted(chunks)}.")
        if sorted(found_citations) != sorted(expected_citations):
            errors.append(
                f"[{record_id}] Cited indices {found_citations} != metadata expected {expected_citations}."
            )
        if behavior == "grounded_single" and len(found_citations) != 1:
            errors.append(f"[{record_id}] grounded_single must cite exactly 1 chunk, got {found_citations}.")
        if behavior == "grounded_multi" and len(found_citations) < 2:
            errors.append(f"[{record_id}] grounded_multi must cite >=2 chunks, got {found_citations}.")

    return chunks, question, found_citations


def validate_dataset_splits(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    max_jaccard_threshold: float = 0.72,
    max_single_position_share: float = 0.55,
) -> ValidationReport:
    """Validate schema, grounding, position diversity, and zero leakage across splits."""
    errors: list[str] = []
    seen_ids: set[str] = set()

    split_groups: dict[str, set[str]] = {}
    split_chunks: dict[str, set[str]] = {}
    split_questions: dict[str, list[tuple[str, str]]] = {}

    behavior_counts: dict[str, dict[str, int]] = {}
    language_counts: dict[str, dict[str, int]] = {}
    train_pos_counter: Counter[int] = Counter()

    for split_name in ("train", "val", "test"):
        records = splits.get(split_name, ())
        if not records:
            errors.append(f"Split {split_name!r} is empty or missing.")
            split_groups[split_name] = set()
            split_chunks[split_name] = set()
            split_questions[split_name] = []
            behavior_counts[split_name] = {}
            language_counts[split_name] = {}
            continue

        groups: set[str] = set()
        chunk_norms: set[str] = set()
        seen_questions_in_split: set[str] = set()
        q_list: list[tuple[str, str]] = []
        b_counter: Counter[str] = Counter()
        l_counter: Counter[str] = Counter()

        for record in records:
            parsed = _validate_record_structure(record, split_name, seen_ids, errors)
            if parsed is None:
                continue
            chunks, question, found_citations = parsed
            meta = record["metadata"]
            group = str(meta.get("scenario_group", ""))
            if not group:
                errors.append(f"[{record['id']}] Missing metadata.scenario_group.")
            groups.add(group)

            for chunk_text in chunks.values():
                chunk_norms.add(normalize_text(chunk_text))

            q_norm = normalize_text(question)
            if q_norm in seen_questions_in_split:
                errors.append(f"[{record['id']}] Duplicate normalized question inside {split_name!r}: {question!r}")
            seen_questions_in_split.add(q_norm)
            q_list.append((record["id"], question))

            behavior = str(meta.get("behavior", "unknown"))
            language = str(meta.get("language", "unknown"))
            b_counter[behavior] += 1
            l_counter[language] += 1

            if split_name == "train" and behavior == "grounded_single" and len(found_citations) == 1:
                train_pos_counter[found_citations[0]] += 1

        split_groups[split_name] = groups
        split_chunks[split_name] = chunk_norms
        split_questions[split_name] = q_list
        behavior_counts[split_name] = dict(sorted(b_counter.items()))
        language_counts[split_name] = dict(sorted(l_counter.items()))

    # Cross-split zero-leakage checks
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        shared_groups = split_groups[left] & split_groups[right]
        if shared_groups:
            errors.append(f"Scenario-group leakage between {left} and {right}: {sorted(shared_groups)}")

        shared_chunks = split_chunks[left] & split_chunks[right]
        if shared_chunks:
            errors.append(
                f"Context-chunk leakage between {left} and {right}: {len(shared_chunks)} shared chunk(s)."
            )

        left_q_norms = {normalize_text(q) for _, q in split_questions[left]}
        for rid, q_right in split_questions[right]:
            if normalize_text(q_right) in left_q_norms:
                errors.append(f"Exact question leakage ({left} vs {right}) on {rid}: {q_right!r}")

    # Fuzzy near-duplicate similarity check against train
    max_jaccard = 0.0
    train_qs = split_questions.get("train", [])
    for eval_split in ("val", "test"):
        for eval_id, eval_q in split_questions.get(eval_split, []):
            for train_id, train_q in train_qs:
                sim = _word_jaccard(eval_q, train_q)
                if sim > max_jaccard:
                    max_jaccard = sim
                if sim >= max_jaccard_threshold:
                    errors.append(
                        f"Near-duplicate question leakage (Jaccard={sim:.2f}): "
                        f"{eval_id} ({eval_q!r}) vs {train_id} ({train_q!r})"
                    )

    # Check single-citation position balance in train
    total_single = sum(train_pos_counter.values())
    if total_single > 0:
        for pos, cnt in train_pos_counter.items():
            share = cnt / total_single
            if share > max_single_position_share:
                errors.append(
                    f"Citation position bias in train: [{pos}] has {cnt}/{total_single} ({share:.1%}) > {max_single_position_share:.0%}"
                )

    split_counts = {name: len(splits.get(name, ())) for name in ("train", "val", "test")}
    total = sum(split_counts.values())
    split_percentages = {
        name: round((count / total) * 100, 2) if total else 0.0 for name, count in split_counts.items()
    }

    return ValidationReport(
        total_examples=total,
        split_counts=split_counts,
        split_percentages=split_percentages,
        behavior_counts_by_split=behavior_counts,
        language_counts_by_split=language_counts,
        train_single_citation_positions={f"[{k}]": v for k, v in sorted(train_pos_counter.items())},
        max_cross_split_question_jaccard=round(max_jaccard, 4),
        errors=tuple(errors),
    )
