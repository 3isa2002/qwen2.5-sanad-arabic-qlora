"""Reproducible dataset builder for Sanad Qwen2.5-7B-Instruct QLoRA fine-tuning."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.scenarios_eval import (  # noqa: E402
    TEST_DISTRACTORS,
    TEST_SCENARIOS,
    VAL_DISTRACTORS,
    VAL_SCENARIOS,
)
from src.data.scenarios_train import TRAIN_DISTRACTORS, TRAIN_SCENARIOS  # noqa: E402
from src.data.schema import SANAD_SYSTEM_PROMPT, ScenarioSpec, format_user_prompt  # noqa: E402
from src.data.validator import validate_dataset_splits  # noqa: E402


def _render_grounded_answer(template: str, positions: dict[int, int]) -> str:
    """Substitute {c0}/{c1} placeholders with 1-based [n] citations."""
    rendered = template
    if len(positions) >= 2:
        pair_sorted = f"[{min(positions[0], positions[1])}][{max(positions[0], positions[1])}]"
        rendered = rendered.replace("{c0}{c1}", pair_sorted).replace("{c1}{c0}", pair_sorted)
    for idx, chunk_pos in positions.items():
        rendered = rendered.replace(f"{{c{idx}}}", f"[{chunk_pos}]")
    return rendered


def _sample_distractors(
    scenario: ScenarioSpec,
    split_distractors: tuple[tuple[str, str], ...],
    extra_pool: tuple[tuple[str, str], ...],
    existing_chunks: set[str],
    desired_count: int,
    rng: random.Random,
) -> list[str]:
    """Select split-exclusive distractors from domains different from the scenario's domain."""
    primary_candidates = [
        text
        for dom, text in split_distractors
        if dom != scenario.domain and text not in existing_chunks
    ]
    rng.shuffle(primary_candidates)

    chosen: list[str] = []
    if scenario.behavior != "handoff" and extra_pool:
        secondary_candidates = [
            text
            for dom, text in extra_pool
            if dom != scenario.domain and text not in existing_chunks
        ]
        rng.shuffle(secondary_candidates)
        # Blend at most 1 cross-domain KB chunk with dedicated benign distractors for lexical richness.
        if secondary_candidates and desired_count >= 2:
            chosen.append(secondary_candidates[0])

    for text in primary_candidates:
        if len(chosen) >= desired_count:
            break
        if text not in chosen:
            chosen.append(text)

    if len(chosen) < desired_count and extra_pool and scenario.behavior != "handoff":
        for _, text in extra_pool:
            if text not in existing_chunks and text not in chosen:
                chosen.append(text)
                if len(chosen) >= desired_count:
                    break

    return chosen[:desired_count]


def build_split_records(
    scenarios: tuple[ScenarioSpec, ...],
    split_distractors: tuple[tuple[str, str], ...],
    *,
    seed: int,
    allow_cross_domain_grounded_distractors: bool,
) -> list[dict[str, Any]]:
    """Build deterministic chat-format records for one split without cross-split sharing."""
    rng = random.Random(seed)
    extra_pool: tuple[tuple[str, str], ...] = ()
    if allow_cross_domain_grounded_distractors:
        extra_pool = tuple(
            (s.domain, chunk)
            for s in scenarios
            if s.behavior != "handoff"
            for chunk in s.supporting_chunks
        )

    records: list[dict[str, Any]] = []
    counter = 1

    for scenario in scenarios:
        fixed_chunks = (
            list(scenario.supporting_chunks)
            if scenario.behavior != "handoff"
            else list(scenario.anchor_chunks)
        )
        for v_idx, variant in enumerate(scenario.variants):
            # Target 3-4 total chunks per prompt so gold chunks appear evenly across [1]..[4].
            target_total = 3 if (counter + v_idx) % 2 == 0 else 4
            desired_distractors = max(1, target_total - len(fixed_chunks))
            distractors = _sample_distractors(
                scenario=scenario,
                split_distractors=split_distractors,
                extra_pool=extra_pool,
                existing_chunks=set(fixed_chunks),
                desired_count=desired_distractors,
                rng=rng,
            )

            tagged_chunks: list[tuple[str, str]] = [
                (f"fixed_{idx}", text) for idx, text in enumerate(fixed_chunks)
            ] + [(f"dist_{idx}", text) for idx, text in enumerate(distractors)]

            rng.shuffle(tagged_chunks)
            ordered_texts = [text for _, text in tagged_chunks]

            if scenario.behavior != "handoff":
                positions: dict[int, int] = {}
                for pos, (tag, _) in enumerate(tagged_chunks, start=1):
                    if tag.startswith("fixed_"):
                        fixed_idx = int(tag.split("_", maxsplit=1)[1])
                        positions[fixed_idx] = pos
                used_indices = [
                    idx for idx in positions if f"{{c{idx}}}" in variant.answer
                ]
                if not used_indices:
                    raise ValueError(
                        f"Grounded variant in {scenario.scenario_group!r} has no {{c0}}/{{c1}} placeholders."
                    )
                assistant_reply = _render_grounded_answer(variant.answer, positions)
                expected_citations = sorted(positions[idx] for idx in used_indices)
                expected_handoff = False
                record_behavior = "grounded_multi" if len(expected_citations) >= 2 else "grounded_single"
            else:
                assistant_reply = variant.answer.strip()
                expected_citations = []
                expected_handoff = True
                record_behavior = "handoff"

            user_prompt = format_user_prompt(ordered_texts, variant.question)
            record_id = f"sanad_{scenario.split}_{counter:04d}"
            counter += 1

            records.append(
                {
                    "id": record_id,
                    "messages": [
                        {"role": "system", "content": SANAD_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": assistant_reply},
                    ],
                    "metadata": {
                        "split": scenario.split,
                        "language": variant.language,
                        "behavior": record_behavior,
                        "handoff_subtype": scenario.handoff_subtype,
                        "domain": scenario.domain,
                        "scenario_group": scenario.scenario_group,
                        "num_chunks": len(ordered_texts),
                        "expected_citations": expected_citations,
                        "expected_handoff": expected_handoff,
                    },
                }
            )

    rng.shuffle(records)
    return records


def build_all_splits(seed: int = 42) -> dict[str, list[dict[str, Any]]]:
    """Generate train, val, and test splits deterministically."""
    return {
        "train": build_split_records(
            TRAIN_SCENARIOS,
            TRAIN_DISTRACTORS,
            seed=seed,
            allow_cross_domain_grounded_distractors=True,
        ),
        "val": build_split_records(
            VAL_SCENARIOS,
            VAL_DISTRACTORS,
            seed=seed + 1,
            allow_cross_domain_grounded_distractors=False,
        ),
        "test": build_split_records(
            TEST_SCENARIOS,
            TEST_DISTRACTORS,
            seed=seed + 2,
            allow_cross_domain_grounded_distractors=False,
        ),
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Sanad Arabic/English QLoRA dataset splits.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
        help="Directory to write train.jsonl, val.jsonl, test.jsonl, and dataset_report.json.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed.")
    args = parser.parse_args()

    splits = build_all_splits(seed=args.seed)
    report = validate_dataset_splits(splits)
    if not report.passed:
        print("Dataset validation FAILED before writing:", file=sys.stderr)
        for err in report.errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    for split_name in ("train", "val", "test"):
        write_jsonl(args.output_dir / f"{split_name}.jsonl", splits[split_name])

    report_payload = asdict(report)
    report_payload["seed"] = args.seed
    report_payload["passed"] = report.passed
    (args.output_dir / "dataset_report.json").write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Dataset generated & verified at: {args.output_dir}")
    print(f"Total examples: {report.total_examples} -> {report.split_counts} ({report.split_percentages})")
    print(f"Train single-citation positions: {report.train_single_citation_positions}")
    print(f"Max cross-split question Jaccard: {report.max_cross_split_question_jaccard}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
