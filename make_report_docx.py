"""
make_report_docx.py
-------------------
Builds an editable Word version of the benchmark report (benchmark_report.docx).

Text, headings and tables are native Word objects so anyone can edit the wording
or re-sort a table. The four charts are embedded as PNG images rendered from the
HTML report — SVG has no faithful .docx equivalent, so they are pictures rather
than Word charts. Re-run this after re-scoring to refresh everything.

The numbers are read straight out of benchmark_report.html (the `const DATA` and
`const FAILURE` blobs the page already carries) so the Word file can never drift
from the published report.

Requires: pip3 install python-docx
Chart images: written by render_charts() via headless Chrome, or pass
--charts-dir to reuse ones already rendered.

Usage:
  python3 make_report_docx.py
  python3 make_report_docx.py --html benchmark_report.html --out benchmark_report.docx
"""

import re
import json
import shutil
import argparse
import tempfile
import subprocess
from pathlib import Path

from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

CHROME = ("/Applications/Google Chrome.app/Contents/MacOS/"
          "Google Chrome")

INK = RGBColor(0x0B, 0x0B, 0x0B)
ACCENT = RGBColor(0x10, 0x42, 0x81)      # dark end of the report's blue ramp
MUTED = RGBColor(0x52, 0x51, 0x4E)
HEADER_FILL = "E4EEFB"                   # palest ramp step, for table headers
ZEBRA_FILL = "F4F8FD"

FIELD_SHORT = {
    "VendorCompanyName": "Vendor name", "VendorCompanyID": "Vendor ID",
    "InvoiceDate": "Invoice date", "VendorCompanyAddress": "Vendor address",
    "CustomerCompanyName": "Customer name", "TotalAmount": "Total amount",
    "InvoiceID": "Invoice ID", "CustomerCompanyID": "Customer ID",
    "TotalwithoutTax": "Subtotal (excl. tax)", "TotalTax": "Total tax",
    "CustomerCompanyAddress": "Customer address", "PONumber": "PO number",
    "DeliveryNote": "Delivery note", "DueDate": "Due date",
    "PaymentTerm": "Payment terms",
}

# (section index in the HTML, heading, intro paragraph) — index drives the
# ?only=N export harness in the HTML page.
CHART_SECTIONS = [
    (1, "Leaderboard — end-to-end extraction score",
     "Bars show each model on its full run. Because models processed different "
     "document counts, diamonds re-score every model on the identical 215 "
     "documents all ten completed — the ranking holds, so the full-run numbers "
     "are trustworthy."),
    (2, "Cost vs. quality",
     "What one thousand documents cost against what you get back. Up and to the "
     "left is better; the cost axis is logarithmic. Gemini 3.6 Flash (GCP "
     "deployment) is omitted — it is billed through the GCP contract, so no "
     "per-call cost is reported."),
    (3, "Where extraction breaks down, field by field",
     "End-to-end score per field, darkest for the highest score. Header fields "
     "(invoice ID, dates, totals) are largely solved; the losses concentrate in "
     "PO number, delivery note and payment terms — mostly fields the models fail "
     "to return at all — and in vendor-name formatting mismatches."),
    (4, "What kind of failure is it?",
     "Every ground-truth field across all model runs combined — about 315,000 "
     "comparisons — split by outcome. The totals are almost never "
     "string-identical (ground truth stores raw numbers, models copy the printed "
     "format) but overwhelmingly correct after normalisation. Vendor name and the "
     "addresses are mostly a normalisation problem; PO number, delivery note and "
     "payment terms are a coverage problem — the model returns nothing at all."),
]

CAVEATS = [
    ("Setup.", "Invoice PDFs rendered locally to page images (up to 4 pages per "
     "document, 150 dpi) and sent to each model with the same extraction prompt "
     "and JSON schema, temperature 0, via OpenRouter — plus one Gemini 3.6 Flash "
     "deployment on GCP Vertex for comparison. Reasoning modes were capped at "
     "minimum effort so models are compared on extraction, not thinking-token "
     "budgets."),
    ("Ground truth.", "Model output is compared field-by-field against a reviewed "
     "ground-truth set, with tolerant matching for dates and number formats."),
    ("Document category is excluded", "from all scores. Every model scored about "
     "0% on it because the ground truth uses internal labels (e.g. “Expense”) "
     "while models return the document’s printed type — a taxonomy mismatch to "
     "fix in the ground truth, not a model failure."),
    ("Unequal run sizes.", "Runs range from 234 documents (Claude Opus 5) to "
     "4,973 (Ministral 14B) because of per-model cost caps. The fair-subset "
     "re-score on the 215 shared documents confirms the ordering, with one "
     "change: Opus 5 moves to first place there."),
    ("Vendor name scores understate the models.", "A large share of vendor-name "
     "misses are formatting variants of the correct name; tightening "
     "ground-truth normalisation would lift every model roughly equally."),
    ("Failure types.", "Exact = string-identical to ground truth; formatting "
     "mismatch = same value after normalisation (date formats, thousand "
     "separators, accents, punctuation) — these count as correct in all scores; "
     "wrong value = returned but doesn’t match; not returned = null or absent."),
    ("Costs", "are actual billed OpenRouter charges at list prices, not estimates."),
]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def extract_embedded_json(html: str, var_name: str):
    """Pull `const <var> = {...};` out of the report page."""
    m = re.search(rf"const {var_name} = (\{{.*?\}});\n", html, re.S)
    if not m:
        raise SystemExit(f"ERROR: could not find `const {var_name}` in the HTML")
    return json.loads(m.group(1))


def shoot(export: Path, section_idx: int, png: Path, zoom: int, timeout: int = 90):
    """Screenshot one chart, then stop the browser we started.

    Chrome given its own --user-data-dir writes the screenshot and then fails to
    exit, so waiting on the process just burns the timeout. Instead: wait for the
    PNG to appear and stop growing, then terminate the child we spawned — never a
    broad pkill, which would take down the user's own browser.
    """
    import time

    if png.exists():
        png.unlink()
    profile = tempfile.mkdtemp(prefix="chartshot-")
    proc = subprocess.Popen(
        [CHROME, "--headless", "--disable-gpu", "--hide-scrollbars",
         "--no-first-run", "--no-default-browser-check",
         f"--user-data-dir={profile}",
         f"--window-size={1010 * zoom + 40},{1500 * zoom}",
         f"--screenshot={png}",
         # Absolute path: a relative file:// URL makes Chrome read the first
         # path segment as a hostname and load nothing.
         f"file://{export.resolve()}?only={section_idx}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + timeout
        last = -1
        while time.time() < deadline:
            if png.exists():
                size = png.stat().st_size
                if size > 0 and size == last:
                    return
                last = size
            time.sleep(0.5)
        raise SystemExit(f"ERROR: chart {section_idx} did not render within "
                         f"{timeout}s")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(profile, ignore_errors=True)


def render_charts(html_path: Path, out_dir: Path) -> dict:
    """Screenshot each chart card on its own via the page's ?only=N harness.

    Resolution comes from CSS zoom on the export page, not
    --force-device-scale-factor: the flag is ignored when it lands on an
    already-running browser (silently giving 1x images that print blurry),
    whereas zoom is applied by the page itself and always takes effect. The
    charts are SVG, so zooming re-renders them sharply rather than upscaling.
    """
    from PIL import Image, ImageChops

    ZOOM = 2
    out_dir.mkdir(parents=True, exist_ok=True)
    export = out_dir / "export.html"
    body = html_path.read_text(encoding="utf-8")
    # The published file is a full document; the harness only needs its body.
    body = re.sub(r"^<!doctype html>.*?<body>", "", body, flags=re.S | re.I)
    body = re.sub(r"</body></html>\s*$", "", body, flags=re.I)

    harness = """
<script>
(function () {
  const n = new URLSearchParams(location.search).get("only");
  if (n === null) return;
  document.querySelectorAll("details").forEach(d => d.remove());
  const sections = [...document.querySelectorAll("section")];
  const keep = sections[+n];
  document.querySelector("header.report").remove();
  sections.forEach(s => { if (s !== keep) s.remove(); });
  keep.querySelectorAll("h2, .sub").forEach(e => e.remove());
  const card = keep.querySelector(".card");
  card.style.border = "none";
  card.style.borderRadius = "0";
  const wrap = document.querySelector(".wrap");
  wrap.style.padding = "0";
  wrap.style.maxWidth = "none";
  keep.style.margin = "0";
  document.body.style.background = "#ffffff";
  document.body.style.zoom = __ZOOM__;
})();
</script>
""".replace("__ZOOM__", str(ZOOM))
    export.write_text(
        '<!doctype html><html data-theme="light"><head><meta charset="utf-8">'
        "</head><body>" + body + harness + "</body></html>", encoding="utf-8")

    images = {}
    for idx, _heading, _intro in CHART_SECTIONS:
        png = out_dir / f"chart{idx}.png"
        # A UNIQUE profile per launch is load-bearing: with no --user-data-dir an
        # already-open Chrome services the request and ignores --window-size
        # (yielding a tiny image), while reusing one directory across launches
        # deadlocks on the profile's singleton lock.
        shoot(export, idx, png, ZOOM)
        im = Image.open(png).convert("RGB")
        bbox = ImageChops.difference(
            im, Image.new("RGB", im.size, (255, 255, 255))).getbbox()
        if bbox is None:
            raise SystemExit(f"ERROR: chart {idx} rendered blank — is the "
                             f"viewport too tall for this Chrome build?")
        pad = 8
        im = im.crop((max(bbox[0] - pad, 0), max(bbox[1] - pad, 0),
                      min(bbox[2] + pad, im.width), min(bbox[3] + pad, im.height)))
        if im.width < 1500:
            raise SystemExit(f"ERROR: chart {idx} came out {im.width}px wide — "
                             f"expected ~{940 * ZOOM}px. Zoom did not apply.")
        im.save(png)
        images[idx] = png
    return images


# ---------------------------------------------------------------------------
# Word helpers
# ---------------------------------------------------------------------------
def shade(cell, fill: str):
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(shd)


def set_cell(cell, text, bold=False, align=None, size=8.5, color=INK):
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    if align:
        p.alignment = align
    run = p.add_run(str(text))
    run.bold = bold
    run.font.size = Pt(size)
    run.font.color.rgb = color


def para(doc, text, size=10, color=INK, italic=False, space_after=8, bold_lead=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    if bold_lead:
        r = p.add_run(bold_lead + " ")
        r.bold = True
        r.font.size = Pt(size)
        r.font.color.rgb = INK
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.font.color.rgb = color
    r.italic = italic
    return p


def heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = ACCENT
        run.font.size = Pt(15 if level == 1 else 12)
    h.paragraph_format.space_before = Pt(18)
    h.paragraph_format.space_after = Pt(4)
    # Keep a heading with the paragraph and picture that follow it.
    h.paragraph_format.keep_with_next = True
    return h


def styled_table(doc, headers, rows, widths=None, numeric_cols=None):
    """numeric_cols: indices to right-align (digits line up). Anything else —
    labels and prose — stays left-aligned; right-aligned sentences read badly."""
    numeric = set(range(1, len(headers))) if numeric_cols is None else set(numeric_cols)

    def al(i):
        return WD_ALIGN_PARAGRAPH.RIGHT if i in numeric else WD_ALIGN_PARAGRAPH.LEFT

    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    # Without a fixed layout Word re-fits columns to their content and the
    # widths set below are ignored — which squashes the label columns.
    # (autofit = False emits the w:tblLayout element itself.)
    table.autofit = False
    if widths:
        tblW = table._tbl.tblPr.find(qn("w:tblW"))
        if tblW is None:
            tblW = OxmlElement("w:tblW")
            table._tbl.tblPr.append(tblW)
        tblW.set(qn("w:type"), "dxa")
        tblW.set(qn("w:w"), str(int(Cm(sum(widths)).twips)))
    for i, h in enumerate(headers):
        c = table.rows[0].cells[i]
        set_cell(c, h, bold=True, size=8, align=al(i))
        shade(c, HEADER_FILL)
    for r_i, row in enumerate(rows):
        cells = table.add_row().cells
        for i, v in enumerate(row):
            set_cell(cells[i], v, bold=(i == 0), align=al(i))
            if r_i % 2 == 1:
                shade(cells[i], ZEBRA_FILL)
    if widths:
        for i, w in enumerate(widths):
            table.columns[i].width = Cm(w)
            for row in table.rows:          # Word honours cell widths, not column
                row.cells[i].width = Cm(w)
    return table


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build(html_path: Path, out_path: Path, images: dict):
    html = html_path.read_text(encoding="utf-8")
    data = extract_embedded_json(html, "DATA")
    failure = extract_embedded_json(html, "FAILURE")
    models = data["models"]

    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    # Pin the heading faces too — left alone, Word's Heading styles pull in
    # Calibri Light, which renders noticeably lighter than the body text.
    for name in ("Heading 1", "Heading 2", "Title"):
        try:
            st = doc.styles[name]
        except KeyError:
            continue
        st.font.name = "Calibri"
        st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
        st.element.rPr.rFonts.set(qn("w:cs"), "Calibri")

    s = doc.sections[0]
    s.page_width, s.page_height = Cm(21), Cm(29.7)
    s.left_margin = s.right_margin = Cm(1.9)
    s.top_margin = s.bottom_margin = Cm(1.9)
    content_cm = 21 - 3.8

    # ---- title block ----
    t = doc.add_paragraph()
    t.paragraph_format.space_after = Pt(2)
    r = t.add_run("MODEL EVALUATION · DOCUMENT AI")
    r.font.size = Pt(8)
    r.bold = True
    r.font.color.rgb = MUTED

    t = doc.add_paragraph()
    t.paragraph_format.space_after = Pt(4)
    r = t.add_run("Invoice Field Extraction — Model Benchmark")
    r.font.size = Pt(22)
    r.bold = True
    r.font.color.rgb = INK

    para(doc, "August 6, 2026  ·  10 model configurations  ·  23,343 documents "
              "scored  ·  15 fields per invoice  ·  $89.78 total API spend",
         size=9, color=MUTED, space_after=14)

    # ---- key results ----
    heading(doc, "Key results")
    para(doc, "Each model reads invoice PDFs as page images and returns 15 "
              "structured fields (vendor, customer, totals, dates, references). "
              "The headline metric is end-to-end score: the share of all fields "
              "present in a document that the model extracted correctly.")
    styled_table(
        doc,
        ["", "Result", "Detail"],
        [["Top score", "88.9%", "Gemini 3.6 Flash · $10.04 / 1k docs"],
         ["Best value", "87.3%", "Qwen 3.6 27B · $2.20 / 1k docs, open-weight"],
         ["Fastest response", "6.9 s", "GPT-5.6 Terra · 86.5% E2E · $8.27 / 1k docs"],
         ["Premium ceiling", "92.2%", "Claude Opus 5 accuracy — at $44.19 / 1k docs"]],
        widths=[4.2, 2.4, 10.6], numeric_cols=[1])

    para(doc, "", space_after=6)
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    r = p.add_run("Bottom line: ")
    r.bold = True
    for text, bold in [
            ("Gemini 3.6 Flash", True),
            (" remains the best accuracy-for-price choice for production, and ", False),
            ("Qwen 3.6 27B", True),
            (" is the strongest open-weight alternative at roughly a fifth of the "
             "cost with only 1.6 points less accuracy. ", False),
            ("GPT-5.6 Terra", True),
            (" lands mid-pack: it does not beat Gemini on accuracy or Qwen on "
             "price, but it is the fastest model tested — relevant if we ever need "
             "real-time extraction in the product UI. ", False),
            ("Claude Opus 5", True),
            (" gives the most accurate individual answers and wins the like-for-like "
             "subset, but at 4–20× the unit cost it only makes sense for high-value "
             "documents.", False)]:
        r = p.add_run(text)
        r.bold = bold

    # ---- chart sections ----
    for idx, head, intro in CHART_SECTIONS:
        heading(doc, head)
        intro_p = para(doc, intro, space_after=8)
        intro_p.paragraph_format.keep_with_next = True
        doc.add_picture(str(images[idx]), width=Cm(content_cm))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

    # ---- full comparison ----
    heading(doc, "Full comparison")
    rows = []
    for m in models:
        cost = "n/a" if m["costPerDoc"] == 0 else f"${m['costPerDoc']*1000:.2f}"
        lat = (f"{m['latency']:.0f} s*"
               if m["slug"] == "anthropic_claude-haiku-4.5" else f"{m['latency']:.1f} s")
        rows.append([m["name"], f"{m['e2e']:.1f}%", f"{m['fairE2e']:.1f}%",
                     f"{m['cov']:.1f}%", f"{m['acc']:.1f}%", cost,
                     f"{round(m['tokPerDoc']):,}", lat, f"{m['docs']:,}"])
    styled_table(
        doc,
        ["Model", "End-to-end", "E2E (215 docs)", "Coverage", "Accuracy",
         "Cost / 1k docs", "Tokens / doc", "Latency", "Docs run"],
        rows,
        widths=[3.6, 1.7, 1.9, 1.5, 1.5, 1.9, 1.6, 1.3, 1.2])
    para(doc, "Coverage = share of fields present in the document that the model "
              "returned. Accuracy = share of returned values that match ground "
              "truth. End-to-end = correct ÷ present. * Claude Haiku 4.5 latency is "
              "inflated by rate-limit retries during its run and is not comparable.",
         size=8, color=MUTED, space_after=4)

    # ---- failure-type appendix ----
    heading(doc, "Outcome split by field")
    para(doc, "The numbers behind “What kind of failure is it?”, pooled across all "
              "ten model runs.")
    order = sorted(failure, key=lambda f: -(failure[f]["exact"] + failure[f]["format"]))
    rows = [[FIELD_SHORT[f], f"{failure[f]['exact']:.1f}%",
             f"{failure[f]['format']:.1f}%", f"{failure[f]['wrong']:.1f}%",
             f"{failure[f]['missing']:.1f}%",
             f"{failure[f]['exact'] + failure[f]['format']:.1f}%",
             f"{failure[f]['present']:,}"] for f in order]
    styled_table(
        doc,
        ["Field", "Exact", "Formatting", "Wrong value", "Not returned",
         "Correct total", "Comparisons"],
        rows,
        widths=[4.0, 1.9, 2.1, 2.1, 2.2, 2.2, 2.2])

    # ---- method ----
    heading(doc, "Method & caveats")
    for lead, text in CAVEATS:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(5)
        r = p.add_run(lead + " ")
        r.bold = True
        r.font.size = Pt(9.5)
        r = p.add_run(text)
        r.font.size = Pt(9.5)
        r.font.color.rgb = MUTED

    doc.save(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Build the editable Word report")
    ap.add_argument("--html", default="benchmark_report.html")
    ap.add_argument("--out", default="benchmark_report.docx")
    ap.add_argument("--charts-dir", default=None,
                    help="Reuse chart PNGs from this directory instead of "
                         "re-rendering them with headless Chrome")
    args = ap.parse_args()

    html_path = Path(args.html)
    if not html_path.exists():
        raise SystemExit(f"ERROR: {html_path} not found — publish the HTML report first")

    if args.charts_dir:
        images = {idx: Path(args.charts_dir) / f"chart{idx}.png"
                  for idx, _h, _i in CHART_SECTIONS}
        missing = [str(p) for p in images.values() if not p.exists()]
        if missing:
            raise SystemExit("ERROR: missing chart images: " + ", ".join(missing))
    else:
        print("Rendering charts with headless Chrome…")
        images = render_charts(html_path, Path(".report_charts"))

    out = build(html_path, Path(args.out), images)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
