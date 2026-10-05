"""
rescue_unparseable.py
----------------------
One-off repair pass: re-parses raw_response for every record where
error == "unparseable_json" in results/*.jsonl, using the improved
parse_json_response() from run_benchmark.py. No new API calls — this
just fixes a bug in our own JSON parsing, not the model's output.

Rewrites each results/<model>.jsonl file in place (backup saved as .bak).

Usage:
  python rescue_unparseable.py
"""

import json
import shutil
from pathlib import Path
from run_benchmark import parse_json_response


def main():
    result_files = sorted(Path("results").glob("*.jsonl"))
    grand_total_recovered = 0
    grand_total_failed = 0

    for rf in result_files:
        records = []
        recovered = still_failed = 0
        with open(rf, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("error") == "unparseable_json" and rec.get("raw_response"):
                    fixed = parse_json_response(rec["raw_response"])
                    if fixed is not None:
                        rec["extracted"] = fixed
                        rec["error"] = None
                        rec["raw_response"] = None
                        recovered += 1
                    else:
                        still_failed += 1
                records.append(rec)

        if recovered == 0:
            continue

        shutil.copy(rf, rf.with_suffix(".jsonl.bak"))
        with open(rf, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"{rf.stem:<38} recovered={recovered:>4}  still_failed={still_failed:>3}")
        grand_total_recovered += recovered
        grand_total_failed += still_failed

    print(f"\nTotal recovered: {grand_total_recovered}")
    print(f"Total still unparseable: {grand_total_failed}")
    print("Backups saved as results/<model>.jsonl.bak")


if __name__ == "__main__":
    main()
