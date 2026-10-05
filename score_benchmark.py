"""
score_benchmark.py
------------------
Compares model extractions (results/<model>.jsonl) against ground_truth.json
and produces per-model, per-field benchmark metrics.

Metrics (per field, per model):
  coverage  = fields extracted (non-null) / fields present in ground truth
              -> "was the model able to extract values that were present?"
  accuracy  = correct extractions / extracted
              -> "out of extracted, were they correct?"
  e2e       = correct / present in ground truth  (end-to-end score)
  extra     = model returned a value where ground truth had none
              (possible hallucination OR a ground-truth gap — reported, not penalised)

Value matching is normalised per field type:
  number  : strips currency symbols, handles "1.234,56" and "1,234.56", tol 0.01
  date    : parses Spanish + English + numeric formats, compares calendar dates
  id      : alphanumeric-only, case-insensitive, containment allowed
  string  : accent/case/punctuation-insensitive, fuzzy >= 0.85 or containment

Usage:
  python score_benchmark.py
  python score_benchmark.py --results-dir results --out-csv benchmark_scores.csv
"""

import json
import csv
import re
import argparse
import unicodedata
from pathlib import Path
from difflib import SequenceMatcher
from collections import defaultdict

# ---------------------------------------------------------------------------
# Field type mapping for comparison
# ---------------------------------------------------------------------------
# DocumentCategory ground truth stores an internal ERP classification
# ("Expense", "Stockable") that has no relationship to anything printed on
# the invoice — every model tested scores 0.0% accuracy on it since there's
# nothing in the document to extract it from. Kept in FIELD_TYPES so its
# per-field row still prints (for visibility), but excluded from the
# OVERALL/leaderboard aggregate so it doesn't mask real signal.
EXCLUDE_FROM_OVERALL = {"DocumentCategory"}

FIELD_TYPES = {
    "VendorCompanyName":     "string",
    "VendorCompanyID":       "id",
    "InvoiceDate":           "date",
    "VendorCompanyAddress":  "string",
    "CustomerCompanyName":   "string",
    "TotalAmount":           "number",
    "InvoiceID":             "id",
    "CustomerCompanyID":     "id",
    "TotalwithoutTax":       "number",
    "TotalTax":              "number",
    "DocumentCategory":      "string",
    "CustomerCompanyAddress": "string",
    "PONumber":              "id",
    "DeliveryNote":          "id",
    "DueDate":               "date",
    "PaymentTerm":           "string",
}

SPANISH_MONTHS = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}
ENGLISH_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}
MONTHS = {**SPANISH_MONTHS, **ENGLISH_MONTHS}
# 3-letter abbreviations
for m, n in list(MONTHS.items()):
    MONTHS.setdefault(m[:3], n)


# ---------------------------------------------------------------------------
# Normalisers
# ---------------------------------------------------------------------------
def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))


def norm_string(s: str) -> str:
    s = strip_accents(str(s)).casefold()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def norm_id(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", strip_accents(str(s)).casefold())


def parse_number(s) -> float | None:
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    txt = str(s)
    txt = re.sub(r"[€$£]|EUR|USD|GBP", "", txt, flags=re.I).strip()
    txt = re.sub(r"[\s ]", "", txt)
    txt = txt.replace("−", "-")
    m = re.search(r"-?[\d.,]+", txt)
    if not m:
        return None
    num = m.group()
    if "," in num and "." in num:
        # last separator is the decimal one
        if num.rfind(",") > num.rfind("."):
            num = num.replace(".", "").replace(",", ".")
        else:
            num = num.replace(",", "")
    elif "," in num:
        parts = num.split(",")
        if len(parts[-1]) == 3 and len(parts) > 1 and len(parts[0]) <= 3:
            num = num.replace(",", "")     # 1,234 -> thousands
        else:
            num = num.replace(",", ".")    # 12,34 -> decimal
    elif "." in num:
        parts = num.split(".")
        if len(parts[-1]) == 3 and len(parts) > 1 and len(parts[0]) <= 3:
            num = num.replace(".", "")     # 1.234 -> thousands
    try:
        return float(num)
    except ValueError:
        return None


def parse_dates(s) -> set:
    """Return the set of plausible (y, m, d) tuples for a date string."""
    if s is None:
        return set()
    txt = strip_accents(str(s)).casefold().strip()
    results = set()

    # ISO: 2026-04-15
    for m in re.finditer(r"(\d{4})-(\d{1,2})-(\d{1,2})", txt):
        results.add((int(m[1]), int(m[2]), int(m[3])))

    # numeric d/m/y or m/d/y or y/m/d with / . -
    for m in re.finditer(r"(\d{1,4})[/.\-](\d{1,2})[/.\-](\d{1,4})", txt):
        a, b, c = int(m[1]), int(m[2]), int(m[3])
        if a > 1900:                      # y-m-d
            results.add((a, b, c))
        elif c > 1900 or c < 100:
            year = c if c > 1900 else 2000 + c
            if a <= 31 and b <= 12:
                results.add((year, b, a))  # d/m/y
            if a <= 12 and b <= 31:
                results.add((year, a, b))  # m/d/y

    # textual: 15 de abril de 2026 | april 15, 2026 | 15 abril 2026
    month_pat = "|".join(sorted(MONTHS, key=len, reverse=True))
    for m in re.finditer(
            rf"(\d{{1,2}})\s*(?:de\s+)?({month_pat})\w*\s*(?:de\s+|,\s*)?(\d{{4}})", txt):
        results.add((int(m[3]), MONTHS[m[2]], int(m[1])))
    for m in re.finditer(
            rf"({month_pat})\w*\s+(\d{{1,2}})\s*,?\s*(\d{{4}})", txt):
        results.add((int(m[3]), MONTHS[m[1]], int(m[2])))

    return results


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
# Below this length, substring containment is too weak a signal (e.g. GT="1"
# would "match" any model output containing the digit 1 anywhere) — require
# an exact match instead. Empirically: every confirmed false-positive found
# in the corpus came from a ground-truth ID value of 1-2 characters, while
# legitimate short IDs (delivery notes like "701", "4290") are 3+ characters.
MIN_CONTAINMENT_LEN = 3


def values_match(gt_value, model_value, field_type: str) -> bool:
    if model_value is None:
        return False

    # Models sometimes return a list when an invoice legitimately has multiple
    # values (e.g. several delivery notes) — count it correct if ANY item matches.
    if isinstance(model_value, list):
        return any(values_match(gt_value, item, field_type)
                  for item in model_value if item is not None)
    # Rare case: a structured object (e.g. address as {street, city, ...}).
    # Flatten to a string rather than comparing against Python's str(dict).
    if isinstance(model_value, dict):
        model_value = " ".join(str(v) for v in model_value.values() if v)

    if field_type == "number":
        a, b = parse_number(gt_value), parse_number(model_value)
        return a is not None and b is not None and abs(a - b) <= 0.011
    if field_type == "date":
        a, b = parse_dates(gt_value), parse_dates(model_value)
        return bool(a & b)
    if field_type == "id":
        a, b = norm_id(gt_value), norm_id(model_value)
        if not a or not b:
            return False
        if a == b:
            return True
        if min(len(a), len(b)) < MIN_CONTAINMENT_LEN:
            return False
        return a in b or b in a
    # string
    a, b = norm_string(gt_value), norm_string(model_value)
    if not a or not b:
        return False
    if a == b:
        return True
    if min(len(a), len(b)) >= MIN_CONTAINMENT_LEN and (a in b or b in a):
        return True
    return SequenceMatcher(None, a, b).ratio() >= 0.85


def is_filled(v) -> bool:
    if v is None:
        return False
    if isinstance(v, str):
        return v.strip().casefold() not in ("", "null", "none", "n/a", "-")
    if isinstance(v, (list, dict)):
        return len(v) > 0
    return True


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def load_latest_records(result_path: Path, ground_truth: dict) -> dict:
    """
    A doc_id can appear multiple times in a results file: run_benchmark.py
    retries failed docs on re-runs by APPENDING a new line, never overwriting
    the old one. The last line written for a doc_id is always the most recent
    attempt, so it must take precedence over any earlier failed attempt.
    """
    latest = {}
    with open(result_path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc_id = rec.get("doc_id")
            if doc_id is None or doc_id not in ground_truth:
                continue
            latest[doc_id] = rec  # later occurrence overwrites earlier one
    return latest


def score_model(result_path: Path, ground_truth: dict, restrict_to: set = None):
    """
    restrict_to: if given, only score docs whose doc_id is in this set. Used
    for a fair cross-model leaderboard when models have processed very
    different numbers of docs — comparing on each model's full (differently
    sized, differently sampled) set is misleading, so we score everyone on
    just the doc_ids every model in the comparison actually has data for.
    """
    per_field = defaultdict(lambda: {"present": 0, "extracted": 0,
                                     "correct": 0, "extra": 0})
    details = []          # per doc × field comparison rows
    n_docs = n_failed = 0
    latencies = []
    total_tokens = 0
    total_cost = 0.0

    latest_records = load_latest_records(result_path, ground_truth)
    if restrict_to is not None:
        latest_records = {d: r for d, r in latest_records.items() if d in restrict_to}
    for doc_id, rec in latest_records.items():
        extracted = rec.get("extracted")
        if extracted is None:
            n_failed += 1
            continue
        n_docs += 1
        if rec.get("latency_s"):
            latencies.append(rec["latency_s"])
        if rec.get("usage"):
            total_tokens += rec["usage"].get("total_tokens", 0)
            total_cost += rec["usage"].get("cost", 0) or 0

        gt = ground_truth[doc_id]
        for field, ftype in FIELD_TYPES.items():
            gt_val = gt.get(field)
            model_val = extracted.get(field) if isinstance(extracted, dict) else None
            gt_has = is_filled(gt_val)
            model_has = is_filled(model_val)

            if gt_has:
                per_field[field]["present"] += 1
                if model_has:
                    per_field[field]["extracted"] += 1
                    if values_match(gt_val, model_val, ftype):
                        per_field[field]["correct"] += 1
                        status = "correct"
                    else:
                        status = "wrong_value"
                else:
                    status = "missed"          # in GT, model returned null
            elif model_has:
                per_field[field]["extra"] += 1
                status = "extra"               # model value, GT empty
            else:
                continue                       # both empty — nothing to record

            details.append({
                "doc_id": doc_id,
                "field": field,
                "status": status,
                "ground_truth": gt_val,
                "model_value": model_val,
            })

    return per_field, details, n_docs, n_failed, latencies, total_tokens, total_cost


def pct(a, b):
    return f"{a / b * 100:.1f}%" if b else "—"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir",  default="results")
    parser.add_argument("--ground-truth", default="ground_truth.json")
    parser.add_argument("--out-csv",      default="benchmark_scores.csv",
                        help="Per-model, per-field breakdown (coverage/accuracy/e2e "
                             "for each extracted field)")
    parser.add_argument("--out-overall-csv", default="benchmark_scores_overall.csv",
                        help="One row per model: overall coverage/accuracy/e2e, "
                             "cost, and token usage")
    parser.add_argument("--out-json",     default="benchmark_scores.json")
    parser.add_argument("--show-errors",  type=int, default=10,
                        help="How many wrong/missed examples to print per model "
                             "(0 to hide, default 10)")
    parser.add_argument("--fair-subset",  action="store_true",
                        help="Models have processed wildly different numbers of "
                             "docs (a few hundred to several thousand) — scoring "
                             "each on its own full set isn't a fair comparison. "
                             "This restricts every model to the doc_ids that ALL "
                             "result files have in common, so the leaderboard "
                             "compares everyone on the exact same documents. Uses "
                             "only already-collected data, no extra API calls.")
    args = parser.parse_args()

    with open(args.ground_truth, encoding="utf-8") as f:
        ground_truth = json.load(f)

    result_files = sorted(Path(args.results_dir).glob("*.jsonl"))
    if not result_files:
        raise SystemExit(f"No .jsonl result files found in {args.results_dir}/")

    common_docs = None
    if args.fair_subset:
        docsets = [set(load_latest_records(rf, ground_truth).keys()) for rf in result_files]
        common_docs = set.intersection(*docsets) if docsets else set()
        print(f"--fair-subset: scoring all {len(result_files)} models on the "
              f"{len(common_docs):,} doc_ids every one of them has processed\n")

    all_rows = []
    model_summaries = []

    for rf in result_files:
        model = rf.stem
        per_field, details, n_docs, n_failed, latencies, tokens, cost = score_model(
            rf, ground_truth, restrict_to=common_docs)
        if n_docs == 0:
            print(f"\n{model}: no scored documents, skipping")
            continue

        tot = {"present": 0, "extracted": 0, "correct": 0, "extra": 0}
        for field, c in per_field.items():
            if field in EXCLUDE_FROM_OVERALL:
                continue
            for k in tot:
                tot[k] += c[k]

        avg_lat = sum(latencies) / len(latencies) if latencies else 0

        print(f"\n{'═'*78}")
        print(f"  MODEL: {model}")
        print(f"  docs scored: {n_docs:,} | failed: {n_failed:,} | "
              f"avg latency: {avg_lat:.1f}s | tokens: {tokens:,}")
        print(f"{'═'*78}")
        print(f"  {'Field':<24} {'GT has':>7} {'Extr.':>7} {'Corr.':>7} "
              f"{'Coverage':>9} {'Accuracy':>9} {'E2E':>7} {'Extra':>6}")
        print(f"  {'─'*24} {'─'*7} {'─'*7} {'─'*7} {'─'*9} {'─'*9} {'─'*7} {'─'*6}")

        for field in FIELD_TYPES:
            c = per_field[field]
            print(f"  {field:<24} {c['present']:>7,} {c['extracted']:>7,} "
                  f"{c['correct']:>7,} {pct(c['extracted'], c['present']):>9} "
                  f"{pct(c['correct'], c['extracted']):>9} "
                  f"{pct(c['correct'], c['present']):>7} {c['extra']:>6,}")
            all_rows.append({
                "model": model, "field": field,
                "gt_present": c["present"], "extracted": c["extracted"],
                "correct": c["correct"], "extra": c["extra"],
                "coverage": round(c["extracted"] / c["present"], 4) if c["present"] else None,
                "accuracy": round(c["correct"] / c["extracted"], 4) if c["extracted"] else None,
                "e2e": round(c["correct"] / c["present"], 4) if c["present"] else None,
            })

        print(f"  {'─'*24}")
        print(f"  {'OVERALL (excl. DocCat)':<24} {tot['present']:>7,} {tot['extracted']:>7,} "
              f"{tot['correct']:>7,} {pct(tot['extracted'], tot['present']):>9} "
              f"{pct(tot['correct'], tot['extracted']):>9} "
              f"{pct(tot['correct'], tot['present']):>7} {tot['extra']:>6,}")

        # Show mismatch examples in terminal
        wrong  = [d for d in details if d["status"] == "wrong_value"]
        missed = [d for d in details if d["status"] == "missed"]
        if wrong and args.show_errors:
            print(f"\n  WRONG VALUES ({len(wrong)} total, showing up to {args.show_errors}):")
            for d in wrong[:args.show_errors]:
                gt_s = str(d['ground_truth'])[:45]
                mv_s = str(d['model_value'])[:45]
                print(f"    [{d['doc_id']}] {d['field']}")
                print(f"       GT   : {gt_s}")
                print(f"       Model: {mv_s}")
        if missed and args.show_errors:
            print(f"\n  MISSED FIELDS ({len(missed)} total, showing up to {args.show_errors}):")
            for d in missed[:args.show_errors]:
                gt_s = str(d['ground_truth'])[:60]
                print(f"    [{d['doc_id']}] {d['field']} — GT had: {gt_s}")

        model_summaries.append({
            "model": model,
            "docs_scored": n_docs,
            "docs_failed": n_failed,
            "avg_latency_s": round(avg_lat, 2),
            "total_tokens": tokens,
            "total_cost": round(cost, 4),
            "avg_tokens_per_doc": round(tokens / n_docs, 1) if n_docs else 0,
            "avg_cost_per_doc": round(cost / n_docs, 6) if n_docs else 0,
            **{k: tot[k] for k in tot},
            "coverage": round(tot["extracted"] / tot["present"], 4) if tot["present"] else None,
            "accuracy": round(tot["correct"] / tot["extracted"], 4) if tot["extracted"] else None,
            "e2e": round(tot["correct"] / tot["present"], 4) if tot["present"] else None,
        })

    # Leaderboard: Coverage / Accuracy / E2E / cost / tokens per model
    if model_summaries:
        print(f"\n{'═'*100}")
        print("  MODEL COMPARISON — Coverage / Accuracy / E2E / Cost / Tokens")
        print(f"{'═'*100}")
        print(f"  {'Model':<38} {'Coverage':>9} {'Accuracy':>9} {'E2E':>7} "
              f"{'Total$':>9} {'Avg tok/doc':>12} {'Avg $/doc':>11} {'Total tok':>11}")
        print(f"  {'─'*38} {'─'*9} {'─'*9} {'─'*7} {'─'*9} {'─'*12} {'─'*11} {'─'*11}")
        for s in sorted(model_summaries, key=lambda x: x["e2e"] or 0, reverse=True):
            print(f"  {s['model']:<38} {pct(s['extracted'], s['present']):>9} "
                  f"{pct(s['correct'], s['extracted']):>9} "
                  f"{pct(s['correct'], s['present']):>7} "
                  f"${s['total_cost']:>8.4f} "
                  f"{s['avg_tokens_per_doc']:>12,.1f} "
                  f"${s['avg_cost_per_doc']:>10.6f} "
                  f"{s['total_tokens']:>11,}")

    # Save
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    overall_ranked = sorted(model_summaries, key=lambda x: x["e2e"] or 0, reverse=True)
    overall_fieldnames = [
        "model", "coverage", "accuracy", "e2e",
        "docs_scored", "docs_failed", "present", "extracted", "correct", "extra",
        "total_cost", "avg_cost_per_doc", "total_tokens", "avg_tokens_per_doc",
        "avg_latency_s",
    ]
    with open(args.out_overall_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=overall_fieldnames)
        writer.writeheader()
        writer.writerows({k: s[k] for k in overall_fieldnames} for s in overall_ranked)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"models": model_summaries, "fields": all_rows}, f,
                  ensure_ascii=False, indent=2)

    print(f"\n  Saved: {args.out_csv}, {args.out_overall_csv}, {args.out_json}\n")


if __name__ == "__main__":
    main()
