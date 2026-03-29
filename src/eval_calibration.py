"""
Confidence calibration analysis.

Reads eval/generation_results.jsonl and checks whether the confidence
label (High / Medium / Low) correlates with faithfulness scores.

Usage:
  uv run python -m src.eval_calibration
  uv run python -m src.eval_calibration --results eval/generation_results.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

RESULTS_PATH = Path("eval/generation_results.jsonl")


def calibrate(results_path: Path) -> None:
    records = []
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Filter to records that have both confidence and faithfulness
    valid = [
        r for r in records
        if r.get("confidence") and r.get("faithfulness") is not None
    ]

    if not valid:
        print("No valid records found.")
        return

    # Group by confidence label
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in valid:
        label = r["confidence"].get("label", "Unknown")
        groups[label].append(r)

    label_order = ["High", "Medium", "Low", "Unknown"]

    print(f"\nConfidence Calibration Report")
    print(f"Source: {results_path}  ({len(valid)} records)")
    print("=" * 70)
    print(f"{'Label':<10} {'Avg Conf':>9} {'Avg Faith':>10} {'Avg Rel':>8} {'Count':>6}")
    print("-" * 70)

    for label in label_order:
        if label not in groups:
            continue
        rows = groups[label]
        avg_conf  = sum(r["confidence"]["value"] for r in rows) / len(rows)
        avg_faith = sum(r["faithfulness"] for r in rows) / len(rows)
        avg_rel   = sum(r["answer_relevance"] for r in rows) / len(rows)
        bar = "█" * round(avg_faith * 4) + "░" * (20 - round(avg_faith * 4))
        print(f"{label:<10} {avg_conf:>8.0%} {avg_faith:>9.2f}  {avg_rel:>7.2f}  {len(rows):>5}  {bar}")

    print("=" * 70)

    # Per-question detail sorted by confidence value
    print("\nPer-question detail (sorted by confidence value desc):")
    print(f"{'ID':<6} {'Label':<8} {'Conf':>6} {'Faith':>6} {'Rel':>5}  Question")
    print("-" * 70)
    for r in sorted(valid, key=lambda x: x["confidence"]["value"], reverse=True):
        label = r["confidence"]["label"]
        conf  = r["confidence"]["value"]
        print(
            f"{r['id']:<6} {label:<8} {conf:>5.0%}  "
            f"{r['faithfulness']:>5}  {r['answer_relevance']:>4}  "
            f"{r['question'][:45]}"
        )

    # Calibration verdict
    print("\nCalibration verdict:")
    group_avgs = {}
    for label in ["High", "Medium", "Low"]:
        if label in groups:
            rows = groups[label]
            group_avgs[label] = sum(r["faithfulness"] for r in rows) / len(rows)

    labels_present = [l for l in ["High", "Medium", "Low"] if l in group_avgs]
    if len(labels_present) < 2:
        print("  Not enough label diversity to assess calibration.")
        print("  Tip: most questions score High — try adding harder questions to eval/questions.jsonl")
    else:
        monotonic = all(
            group_avgs[labels_present[i]] >= group_avgs[labels_present[i + 1]]
            for i in range(len(labels_present) - 1)
        )
        if monotonic:
            print("  WELL CALIBRATED — faithfulness decreases monotonically from High to Low.")
        else:
            print("  POORLY CALIBRATED — faithfulness does not decrease monotonically.")
            for l in labels_present:
                print(f"    {l}: {group_avgs[l]:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Confidence calibration analysis")
    parser.add_argument(
        "--results",
        default=str(RESULTS_PATH),
        help=f"Path to generation results JSONL (default: {RESULTS_PATH})",
    )
    args = parser.parse_args()
    calibrate(Path(args.results))


if __name__ == "__main__":
    main()
