"""CLI validator for generated Sanad JSONL splits (schema, grounding, and zero leakage)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.schema import load_jsonl  # noqa: E402
from src.data.validator import validate_dataset_splits  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate generated Sanad dataset JSONL files.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
        help="Directory containing train.jsonl, val.jsonl, and test.jsonl.",
    )
    args = parser.parse_args()

    try:
        splits = {
            name: load_jsonl(args.data_dir / f"{name}.jsonl")
            for name in ("train", "val", "test")
        }
    except (FileNotFoundError, ValueError) as exc:
        print(f"Failed to load dataset splits: {exc}", file=sys.stderr)
        return 1

    report = validate_dataset_splits(splits)
    print(f"Total examples : {report.total_examples}")
    print(f"Split counts   : {report.split_counts} ({report.split_percentages})")
    print(f"Behaviors      : {report.behavior_counts_by_split}")
    print(f"Languages      : {report.language_counts_by_split}")
    print(f"Train [n] pos  : {report.train_single_citation_positions}")
    print(f"Max cross-split Jaccard similarity: {report.max_cross_split_question_jaccard}")

    if not report.passed:
        print(f"\nValidation FAILED with {len(report.errors)} error(s):", file=sys.stderr)
        for err in report.errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print("\nValidation PASSED: schema, grounding, position balance, and zero-leakage checks ok.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
