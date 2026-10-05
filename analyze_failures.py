"""
analyze_failures.py
-------------------
Splits every ground-truth field comparison into WHY it succeeded or failed,
pooled across all model runs in results/. score_benchmark.py answers "how
often is the model right"; this answers "when it isn't right, what kind of
wrong is it" — which is the difference between a normalisation bug we fix in
post-processing and a genuine extraction miss.

Four mutually exclusive outcomes per (document, field):
  exact    string-identical to ground truth
  format   matches only after normalisation (date shape, thousands separators,
           accents, punctuation, casing). Counted CORRECT by score_benchmark —
           surfaced separately here because a large 'format' share means the
           ground truth, not the model, is what needs work.
  wrong    model returned a value and it does not match. The costly one in AP:
           a silent error no reviewer is prompted to look at.
  missing  model returned null/absent. Cheap by comparison — routes to review.

Fields with no ground-truth value for a document are skipped entirely (they
can't be extracted), as is DocumentCategory (see score_benchmark.py).

Usage:
  python3 analyze_failures.py                      # table to stdout
  python3 analyze_failures.py --out failures.json  # + JSON for the report
"""

import json
import argparse
from pathlib import Path

from score_benchmark import (FIELD_TYPES, EXCLUDE_FROM_OVERALL,
                             load_latest_records, values_match, is_filled)


def is_exact(gt_value, model_value) -> bool:
    """String-identical before any normalisation. Mirrors values_match's
    list handling so a multi-value answer isn't penalised for being a list."""
    if isinstance(model_value, list):
        return any(is_exact(gt_value, item)
                   for item in model_value if item is not None)
    return str(gt_value).strip() == str(model_value).strip()


def classify(results_dir: Path, ground_truth: dict) -> dict:
    agg = {f: {"exact": 0, "format": 0, "wrong": 0, "missing": 0, "present": 0}
           for f in FIELD_TYPES if f not in EXCLUDE_FROM_OVERALL}

    for result_file in sorted(results_dir.glob("*.jsonl")):
        for doc_id, rec in load_latest_records(result_file, ground_truth).items():
            extracted = rec.get("extracted")
            if extracted is None:
                continue
            gt = ground_truth[doc_id]
            for field, ftype in FIELD_TYPES.items():
                if field in EXCLUDE_FROM_OVERALL:
                    continue
                gt_val = gt.get(field)
                if not is_filled(gt_val):
                    continue
                bucket = agg[field]
                bucket["present"] += 1
                model_val = (extracted.get(field)
                             if isinstance(extracted, dict) else None)
                if not is_filled(model_val):
                    bucket["missing"] += 1
                elif values_match(gt_val, model_val, ftype):
                    bucket["exact" if is_exact(gt_val, model_val) else "format"] += 1
                else:
                    bucket["wrong"] += 1
    return agg


def main():
    parser = argparse.ArgumentParser(
        description="Classify extraction outcomes by failure type")
    parser.add_argument("--results-dir",  default="results")
    parser.add_argument("--ground-truth", default="ground_truth.json")
    parser.add_argument("--out", default=None,
                        help="Write percentages as JSON (used by the report page)")
    args = parser.parse_args()

    with open(args.ground_truth, encoding="utf-8") as f:
        ground_truth = json.load(f)

    agg = classify(Path(args.results_dir), ground_truth)

    shares = {}
    for field, b in agg.items():
        n = b["present"]
        shares[field] = {k: round(b[k] / n * 100, 1)
                         for k in ("exact", "format", "wrong", "missing")}
        shares[field]["present"] = n

    print(f"\n{'Field':<24}{'exact':>8}{'format':>8}{'wrong':>8}{'missing':>9}"
          f"{'match':>8}{'n':>10}")
    print("─" * 75)
    for field, s in sorted(shares.items(), key=lambda kv: -(kv[1]["exact"] + kv[1]["format"])):
        print(f"{field:<24}{s['exact']:7.1f}%{s['format']:7.1f}%{s['wrong']:7.1f}%"
              f"{s['missing']:8.1f}%{s['exact'] + s['format']:7.1f}%{s['present']:10,}")
    total = sum(s["present"] for s in shares.values())
    print(f"\n{total:,} field comparisons across "
          f"{len(list(Path(args.results_dir).glob('*.jsonl')))} model runs")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(shares, f)
        print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
