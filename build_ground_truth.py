"""
build_ground_truth.py
---------------------
Builds ground_truth.json from merged_documents.json using the fields
defined in invoice_schema.json. Only documents that have a PDF on disk
are included.

Output format:
  {
    "<doc_id>": {
      "VendorCompanyName": "ACME S.L.",
      "InvoiceDate": "15 de abril de 2026",
      ...only fields that are actually filled in the ground truth...
    },
    ...
  }

Usage:
  python build_ground_truth.py
"""

import json
import argparse
from pathlib import Path


def is_filled(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   default="merged_documents.json")
    parser.add_argument("--schema",  default="invoice_schema.json")
    parser.add_argument("--pdf-dir", default="pdfs")
    parser.add_argument("--output",  default="ground_truth.json")
    args = parser.parse_args()

    with open(args.schema, encoding="utf-8") as f:
        schema = json.load(f)
    schema_fields = {f["name"] for f in schema["fields"]}

    print(f"Loading {args.input} …")
    with open(args.input, encoding="utf-8") as f:
        merged = json.load(f)

    docs = merged.get("documents", merged)
    pdf_dir = Path(args.pdf_dir)

    ground_truth = {}
    skipped_no_pdf = 0
    skipped_empty = 0

    for doc_id, doc in docs.items():
        if doc_id == "_meta":
            continue
        if not (pdf_dir / f"{doc_id}.pdf").exists():
            skipped_no_pdf += 1
            continue

        inner = doc.get("Document", doc) if isinstance(doc, dict) else {}
        fields_list = inner.get("Fields", []) if isinstance(inner, dict) else []

        gt = {}
        for field in fields_list:
            if not isinstance(field, dict):
                continue
            name = field.get("Name")
            if name in schema_fields and is_filled(field.get("Value")):
                # First occurrence wins if a field name repeats
                gt.setdefault(name, str(field["Value"]).strip())

        if gt:
            ground_truth[doc_id] = gt
        else:
            skipped_empty += 1

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, ensure_ascii=False, indent=2)

    n_fields = sum(len(v) for v in ground_truth.values())
    print(f"""
Ground truth built:
  Documents included : {len(ground_truth):,}
  Skipped (no PDF)   : {skipped_no_pdf:,}
  Skipped (no fields): {skipped_empty:,}
  Total field values : {n_fields:,}
  Avg fields per doc : {n_fields / len(ground_truth):.1f}
  Saved to           : {args.output}
""")


if __name__ == "__main__":
    main()
