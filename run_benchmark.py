"""
run_benchmark.py
----------------
Sends each PDF to one or more models via the OpenRouter API and asks them to
extract the invoice fields defined in invoice_schema.json.

Results are appended to results/<model_slug>.jsonl — one line per document —
so the run is fully RESUMABLE: re-running skips documents already processed.

Requirements:
  export OPENROUTER_API_KEY=sk-or-...

Usage:
  python run_benchmark.py                                  # default models, all docs
  python run_benchmark.py --limit 100                      # first 100 docs
  python run_benchmark.py --models qwen/qwen2.5-vl-72b-instruct
  python run_benchmark.py --concurrency 4

Input modes:
  --input-mode images (default)  Render PDF pages locally with PyMuPDF and send
                                 them as images. True vision-model benchmark.
                                 Requires: pip3 install pymupdf
  --input-mode pdf               Send the PDF via OpenRouter's file-parser.
                                 Use --pdf-engine mistral-ocr for scanned docs
                                 ($2/1000 pages); pdf-text is free but fails on
                                 scanned/SAP-generated PDFs.
  --input-mode ocr-text          For TEXT-ONLY models (e.g. mistral-nemo): first
                                 OCRs each page with --ocr-model (a vision model),
                                 then sends the extracted text to the target model.
                                 OCR results are cached in --ocr-cache so re-running
                                 or testing another text-only model doesn't re-OCR.
"""

import os
import re
import ssl
import json
import time
import base64
import argparse
import threading
import urllib.request
import urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# HTTP codes that mean "this will never succeed by retrying" (bad request,
# auth, model/endpoint not found, etc.) — as opposed to 429/5xx which are
# transient and worth retrying both within a call and across re-runs.
# 402 (insufficient credits) is deliberately NOT here — it's an account-balance
# problem, not a problem with this request, so it gets its own exception below
# and stays eligible for retry once the account is topped up.
NON_RETRYABLE_HTTP_CODES = frozenset({400, 401, 403, 404, 422})


class PermanentError(RuntimeError):
    """Raised for API errors that will not go away on retry (e.g. a 404 for
    a model with no active OpenRouter endpoint). Callers should not keep
    retrying documents that fail with this across separate runs."""


class InsufficientCreditsError(RuntimeError):
    """Raised on HTTP 402. Distinct from PermanentError: the *account* ran
    out of credits, not this particular request being unfixable. Once the
    account is topped up, these documents should be retried, so callers must
    NOT mark them permanently done."""


def make_ssl_context() -> ssl.SSLContext:
    """macOS Python often lacks CA certs — prefer certifi's bundle if available."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


SSL_CONTEXT = make_ssl_context()

# Default open-source models (override with --models)
DEFAULT_MODELS = [
    "qwen/qwen2.5-vl-72b-instruct",
    "meta-llama/llama-4-maverick",
    "mistralai/mistral-small-3.1-24b-instruct",
]


def load_schema(path: str):
    with open(path, encoding="utf-8") as f:
        schema = json.load(f)
    return schema["fields"]


def build_prompt(schema_fields) -> str:
    lines = [
        "You are an invoice data-extraction system.",
        "Extract the following fields from the attached invoice document.",
        "",
        "Fields to extract:",
    ]
    for f in schema_fields:
        lines.append(f'- "{f["name"]}" ({f["type"]}): {f["description"]}')
    lines += [
        "",
        "Rules:",
        "1. Return ONLY a valid JSON object — no markdown, no explanation.",
        "2. Use exactly the field names listed above as JSON keys.",
        "3. Copy values VERBATIM as they appear in the document.",
        "4. If a field is not present in the document, set it to null.",
        "5. Do not invent or guess values.",
    ]
    return "\n".join(lines)


def encode_pdf(pdf_path: Path) -> str:
    with open(pdf_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def render_pdf_pages(pdf_path: Path, max_pages: int, dpi: int = 150) -> list:
    """Render PDF pages to base64 PNGs using PyMuPDF (pip install pymupdf)."""
    import fitz  # PyMuPDF
    images = []
    with fitz.open(pdf_path) as doc:
        for page in doc[:max_pages]:
            pix = page.get_pixmap(dpi=dpi)
            images.append(base64.b64encode(pix.tobytes("png")).decode())
    return images


def build_ocr_prompt() -> str:
    return (
        "Extract ALL text from this document image exactly as it appears.\n"
        "Preserve reading order, line breaks, and table structure as plain text.\n"
        "Return ONLY the raw extracted text — no commentary, no markdown fences."
    )


def load_ocr_cache(cache_path: Path) -> dict:
    cache = {}
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    cache[rec["doc_id"]] = rec["text"]
                except json.JSONDecodeError:
                    continue
    return cache


def repair_number_literals(text: str) -> str:
    """
    Models frequently emit European-formatted numbers as bare (unquoted) JSON
    literals — e.g. "TotalAmount": 2.410,10 or "TotalTax": 219,10 — which is
    invalid JSON syntax (a raw number can't contain a comma) and breaks parsing
    of the entire response. Quote these so the rest of the object still parses.
    """
    # comma-decimal: 2.410,10 | 219,10 | 10.109,440
    text = re.sub(r'(:\s*)(-?\d{1,3}(?:\.\d{3})*,\d{1,3})(\s*[,}])',
                 lambda m: f'{m.group(1)}"{m.group(2)}"{m.group(3)}', text)
    # dot-decimal with dot thousands-separator too: 11.004.95
    text = re.sub(r'(:\s*)(-?\d{1,3}(?:\.\d{3}){2,})(\s*[,}])',
                 lambda m: f'{m.group(1)}"{m.group(2)}"{m.group(3)}', text)
    return text


def repair_unquoted_scalars(text: str) -> str:
    """Wrap any other bare (non-numeric, non-null/bool) scalar value in quotes."""
    def fix(m):
        key, val = m.group(1), m.group(2).strip()
        if val in ("null", "true", "false") or re.fullmatch(r"-?\d+(\.\d+)?", val):
            return f'"{key}": {val}'
        if val.startswith('"'):
            return m.group(0)
        return f'"{key}": "{val}"'
    return re.sub(r'"([^"]+)":\s*([^",\{\[\n][^,\}\n]*)', fix, text)


def parse_json_response(text: str):
    """Extract a JSON object from a model response, tolerating markdown fences
    and common malformed-number patterns (see repair_number_literals)."""
    if text is None:
        # Reasoning models can spend their entire output budget on the hidden
        # reasoning phase and return empty/None visible content. That's a
        # legitimate "no answer" outcome, not a bug — don't crash on it (a
        # crash here used to escape into process_doc's generic except-clause
        # and silently drop the already-billed `usage` for the call).
        return None
    text = text.strip()
    if text.startswith("```"):
        # strip ```json ... ``` fences
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    start = text.find("{")
    end = text.rfind("}")
    candidate = text[start:end + 1] if start != -1 and end > start else text

    for strict in (True, False):
        try:
            return json.loads(candidate, strict=strict)
        except json.JSONDecodeError:
            pass
    for repair_fn in (repair_number_literals,
                      lambda t: repair_unquoted_scalars(repair_number_literals(t))):
        try:
            return json.loads(repair_fn(candidate), strict=False)
        except json.JSONDecodeError:
            continue
    return None


def call_openrouter(model: str, prompt: str, content_parts: list, api_key: str,
                    plugins: list = None, max_retries: int = 5, timeout: int = 300,
                    max_tokens: int = 4000, allow_reasoning: bool = False,
                    reasoning_effort: str = None):
    reasoning = {"enabled": allow_reasoning}
    if allow_reasoning and reasoning_effort:
        # Some endpoints (e.g. google/gemini-3.6-flash) make reasoning
        # mandatory and reject enabled:false outright, but still respect an
        # effort cap — this keeps the hidden "thinking" phase short instead
        # of burning thousands of tokens per doc on a task that's plain
        # extraction, not reasoning.
        reasoning["effort"] = reasoning_effort
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}] + content_parts,
            }
        ],
        "temperature": 0,
        # Hard cap on output tokens: a structured-extraction task never needs
        # more than a few hundred tokens of JSON. Without this, a reasoning
        # model can run its hidden "thinking" phase for thousands of tokens
        # per document with no ceiling — this is what drove one benchmark run
        # to $20+ on a single model. Applies to every model, reasoning or not.
        "max_tokens": max_tokens,
        # Disable extended "thinking"/reasoning by default. This is a data-
        # extraction task, not a reasoning task — chain-of-thought buys
        # nothing here but tokens. Pass allow_reasoning=True to explicitly
        # benchmark a model's reasoning mode instead.
        "reasoning": reasoning,
    }
    if plugins:
        payload["plugins"] = plugins
    body = json.dumps(payload).encode()

    last_err = None
    for attempt in range(max_retries):
        req = urllib.request.Request(
            OPENROUTER_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=SSL_CONTEXT) as resp:
                data = json.loads(resp.read().decode())
            if "error" in data:
                raise RuntimeError(data["error"].get("message", str(data["error"])))
            choice = data["choices"][0]
            content = choice["message"]["content"]
            usage = data.get("usage", {})
            return content, usage
        except urllib.error.HTTPError as e:
            body_text = e.read().decode(errors="replace")[:500]
            last_err = f"HTTP {e.code}: {body_text}"
            if e.code == 402:
                # Don't retry within this call (more attempts won't add
                # credits) and don't mark the doc permanently failed either —
                # see InsufficientCreditsError docstring.
                raise InsufficientCreditsError(last_err)
            if e.code in (429, 500, 502, 503, 520, 524):
                time.sleep(min(2 ** attempt * 2, 60))
                continue
            if e.code in NON_RETRYABLE_HTTP_CODES:
                raise PermanentError(last_err)
            break
        except Exception as e:  # timeouts, connection resets, malformed responses
            last_err = str(e)
            time.sleep(min(2 ** attempt * 2, 60))
    raise RuntimeError(last_err or "unknown error")


def model_slug(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def load_done_ids(result_path: Path) -> set:
    done = set()
    if result_path.exists():
        with open(result_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    # Only skip docs that succeeded or hit a genuinely
                    # unfixable error. credit_exhausted is deliberately
                    # excluded — it's an account-balance problem, not a
                    # problem with the document, so it must stay retryable
                    # once the account is topped up (see run_model, which
                    # also skips re-attempting these within the same run
                    # once the budget cap trips).
                    if rec.get("extracted") is not None or rec.get("permanent_error"):
                        done.add(rec["doc_id"])
                except json.JSONDecodeError:
                    continue
    return done


def get_ocr_text(doc_id: str, pdf_path: Path, ocr_model: str, api_key: str,
                 max_pages: int, ocr_cache: dict, ocr_lock: threading.Lock,
                 ocr_cache_path: Path) -> str:
    with ocr_lock:
        cached = ocr_cache.get(doc_id)
    if cached is not None:
        return cached

    pages = render_pdf_pages(pdf_path, max_pages)
    content_parts = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{p}"}}
        for p in pages
    ]
    # OCR dumps full page text (not a bounded JSON object), so it gets a
    # higher cap than the extraction calls; reasoning stays off — OCR is a
    # transcription task, not a reasoning one.
    text, _usage = call_openrouter(ocr_model, build_ocr_prompt(), content_parts,
                                   api_key, max_tokens=8000, allow_reasoning=False)

    with ocr_lock:
        ocr_cache[doc_id] = text
        with open(ocr_cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"doc_id": doc_id, "text": text}, ensure_ascii=False) + "\n")
    return text


def process_doc(doc_id: str, pdf_dir: Path, model: str, prompt: str,
                api_key: str, input_mode: str, pdf_engine: str, max_pages: int,
                ocr_model: str = None, ocr_cache: dict = None,
                ocr_lock: threading.Lock = None, ocr_cache_path: Path = None,
                max_tokens: int = 4000, allow_reasoning: bool = False,
                reasoning_effort: str = None):
    pdf_path = pdf_dir / f"{doc_id}.pdf"
    t0 = time.time()
    try:
        if input_mode == "images":
            # Render pages locally and send as images (true VLM benchmark)
            pages = render_pdf_pages(pdf_path, max_pages)
            content_parts = [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{p}"}}
                for p in pages
            ]
            plugins = None
            final_prompt = prompt
        elif input_mode == "ocr-text":
            # Text-only target model: OCR with a vision model first (cached),
            # then send the extracted text instead of images.
            ocr_text = get_ocr_text(doc_id, pdf_path, ocr_model, api_key,
                                    max_pages, ocr_cache, ocr_lock, ocr_cache_path)
            content_parts = []
            plugins = None
            final_prompt = prompt + "\n\n--- DOCUMENT TEXT (OCR) ---\n" + ocr_text
        else:
            pdf_b64 = encode_pdf(pdf_path)
            content_parts = [{
                "type": "file",
                "file": {
                    "filename": "invoice.pdf",
                    "file_data": f"data:application/pdf;base64,{pdf_b64}",
                },
            }]
            plugins = [{"id": "file-parser", "pdf": {"engine": pdf_engine}}]
            final_prompt = prompt

        content, usage = call_openrouter(model, final_prompt, content_parts,
                                         api_key, plugins,
                                         max_tokens=max_tokens,
                                         allow_reasoning=allow_reasoning,
                                         reasoning_effort=reasoning_effort)
        extracted = parse_json_response(content)
        return {
            "doc_id": doc_id,
            "model": model,
            "extracted": extracted,
            "raw_response": content if extracted is None else None,
            "usage": usage,
            "latency_s": round(time.time() - t0, 2),
            "error": None if extracted is not None else "unparseable_json",
        }
    except InsufficientCreditsError as e:
        return {
            "doc_id": doc_id,
            "model": model,
            "extracted": None,
            "usage": None,
            "latency_s": round(time.time() - t0, 2),
            "error": str(e)[:500],
            "credit_exhausted": True,
        }
    except PermanentError as e:
        return {
            "doc_id": doc_id,
            "model": model,
            "extracted": None,
            "usage": None,
            "latency_s": round(time.time() - t0, 2),
            "error": str(e)[:500],
            "permanent_error": True,
        }
    except Exception as e:
        return {
            "doc_id": doc_id,
            "model": model,
            "extracted": None,
            "usage": None,
            "latency_s": round(time.time() - t0, 2),
            "error": str(e)[:500],
        }


def run_model(model: str, doc_ids: list, pdf_dir: Path, prompt: str,
              api_key: str, out_dir: Path, concurrency: int,
              input_mode: str, pdf_engine: str, max_pages: int,
              ocr_model: str = None, ocr_cache: dict = None,
              ocr_lock: threading.Lock = None, ocr_cache_path: Path = None,
              max_tokens: int = 4000, allow_reasoning: bool = False,
              max_cost: float = 10.0, reasoning_effort: str = None):
    result_path = out_dir / f"{model_slug(model)}.jsonl"
    done = load_done_ids(result_path)
    todo = [d for d in doc_ids if d not in done]

    print(f"\n━━━ {model} ━━━")
    print(f"  already done: {len(done):,} | remaining: {len(todo):,}")
    if max_cost is not None:
        print(f"  cost budget: ${max_cost:.2f}")
    if not todo:
        return

    lock = threading.Lock()
    n_ok = n_err = 0
    total_tokens = 0
    total_cost = 0.0
    budget_stopped = False

    with open(result_path, "a", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(process_doc, d, pdf_dir, model, prompt, api_key,
                            input_mode, pdf_engine, max_pages,
                            ocr_model, ocr_cache, ocr_lock, ocr_cache_path,
                            max_tokens, allow_reasoning, reasoning_effort): d
                for d in todo
            }
            for i, fut in enumerate(as_completed(futures), 1):
                rec = fut.result()
                with lock:
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_f.flush()
                if rec["error"] is None:
                    n_ok += 1
                else:
                    n_err += 1
                # Track cost/tokens for ANY call that got a billed response,
                # not just successful extractions — an errored-but-billed
                # call (e.g. unparseable_json) still spent real money and
                # must count toward the budget.
                if rec.get("usage"):
                    total_tokens += rec["usage"].get("total_tokens", 0)
                    total_cost += rec["usage"].get("cost", 0) or 0
                if i % 25 == 0 or i == len(todo):
                    print(f"  [{i}/{len(todo)}] ok={n_ok} err={n_err} "
                          f"tokens={total_tokens:,} cost=${total_cost:.4f}")
                if (max_cost is not None and total_cost >= max_cost
                        and not budget_stopped):
                    budget_stopped = True
                    print(f"  !! cost budget of ${max_cost:.2f} reached "
                          f"(${total_cost:.4f} spent) — stopping remaining "
                          f"{len(todo) - i:,} document(s) for this model")
                    pool.shutdown(wait=False, cancel_futures=True)
                    break

    status = "budget cap hit" if budget_stopped else "finished"
    print(f"  {status}: ok={n_ok} err={n_err} cost=${total_cost:.4f}")


def main():
    parser = argparse.ArgumentParser(description="OpenRouter invoice extraction benchmark")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS),
                        help="Comma-separated OpenRouter model IDs")
    parser.add_argument("--schema",       default="invoice_schema.json")
    parser.add_argument("--ground-truth", default="ground_truth.json")
    parser.add_argument("--pdf-dir",      default="pdfs")
    parser.add_argument("--out-dir",      default="results")
    parser.add_argument("--limit",        type=int, default=None,
                        help="Max documents (default: all)")
    parser.add_argument("--concurrency",  type=int, default=8)
    parser.add_argument("--input-mode",   choices=["images", "pdf", "ocr-text"],
                        default="images",
                        help="'images': render pages locally, send to vision models "
                             "(needs: pip install pymupdf). "
                             "'pdf': send PDF with OpenRouter file-parser plugin. "
                             "'ocr-text': OCR with --ocr-model first, then send text "
                             "to a TEXT-ONLY target model (e.g. mistral-nemo).")
    parser.add_argument("--pdf-engine",   choices=["mistral-ocr", "pdf-text", "native"],
                        default="mistral-ocr",
                        help="file-parser engine for --input-mode pdf. "
                             "mistral-ocr: $2/1000 pages, works on scanned docs. "
                             "pdf-text: free but FAILS on scanned/SAP PDFs.")
    parser.add_argument("--max-pages",    type=int, default=4,
                        help="Max pages per doc in images/ocr-text mode (default 4)")
    parser.add_argument("--ocr-model",    default="qwen/qwen2.5-vl-72b-instruct",
                        help="Vision model used to OCR pages in --input-mode ocr-text")
    parser.add_argument("--ocr-cache",    default="ocr_cache.jsonl",
                        help="Resumable cache of OCR'd page text, keyed by doc_id")
    parser.add_argument("--max-tokens",   type=int, default=4000,
                        help="Output token cap per call (default 4000). Applies "
                             "to every model — the main guard against a single "
                             "runaway generation blowing the budget.")
    parser.add_argument("--allow-reasoning", action="store_true",
                        help="Let reasoning/'thinking' models use extended "
                             "chain-of-thought. OFF by default: for structured "
                             "extraction it just burns tokens (a past run spent "
                             "$20+ on one reasoning model this way) without "
                             "improving accuracy. Pass this only to deliberately "
                             "benchmark a model's reasoning mode.")
    parser.add_argument("--max-cost",     type=float, default=10.0,
                        help="Stop a model once its cumulative spend this run "
                             "hits this many dollars (default $10). Pass 0 or "
                             "a negative number to disable the cap.")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"],
                        default=None,
                        help="Cap reasoning depth for endpoints where reasoning "
                             "is mandatory and --allow-reasoning alone can't turn "
                             "it off (e.g. google/gemini-3.6-flash). 'low' cuts "
                             "hidden thinking-phase tokens substantially — this "
                             "is a plain extraction task, not a reasoning one, so "
                             "less thinking budget shouldn't cost accuracy. "
                             "No effect unless --allow-reasoning is also passed.")
    args = parser.parse_args()
    max_cost = args.max_cost if args.max_cost and args.max_cost > 0 else None

    if args.input_mode in ("images", "ocr-text"):
        try:
            import fitz  # noqa: F401
        except ImportError:
            raise SystemExit(
                "ERROR: this mode needs PyMuPDF — run:  pip3 install pymupdf\n"
                "  (or use:  --input-mode pdf  to use OpenRouter's OCR instead)")

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: set OPENROUTER_API_KEY environment variable first.")

    schema_fields = load_schema(args.schema)
    prompt = build_prompt(schema_fields)

    with open(args.ground_truth, encoding="utf-8") as f:
        ground_truth = json.load(f)

    doc_ids = sorted(ground_truth.keys())
    if args.limit:
        doc_ids = doc_ids[:args.limit]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    print(f"Benchmarking {len(models)} model(s) on {len(doc_ids):,} documents")
    print(f"Models: {models}")
    if args.input_mode == "pdf":
        mode_desc = f" (engine: {args.pdf_engine})"
    elif args.input_mode == "ocr-text":
        mode_desc = f" (ocr-model: {args.ocr_model}, max {args.max_pages} pages/doc)"
    else:
        mode_desc = f" (max {args.max_pages} pages/doc)"
    print(f"Input mode: {args.input_mode}{mode_desc}")

    ocr_cache = ocr_lock = ocr_cache_path = None
    if args.input_mode == "ocr-text":
        ocr_cache_path = Path(args.ocr_cache)
        ocr_cache = load_ocr_cache(ocr_cache_path)
        ocr_lock = threading.Lock()
        print(f"OCR cache: {len(ocr_cache):,} documents already OCR'd "
              f"({ocr_cache_path})")

    print(f"Max tokens/call: {args.max_tokens} | reasoning: "
          f"{'allowed' if args.allow_reasoning else 'disabled'}"
          f"{f' (effort={args.reasoning_effort})' if args.allow_reasoning and args.reasoning_effort else ''} | "
          f"cost cap/model: {'none' if max_cost is None else f'${max_cost:.2f}'}")

    pdf_dir = Path(args.pdf_dir)
    for model in models:
        run_model(model, doc_ids, pdf_dir, prompt, api_key, out_dir,
                  args.concurrency, args.input_mode, args.pdf_engine,
                  args.max_pages, args.ocr_model, ocr_cache, ocr_lock,
                  ocr_cache_path, args.max_tokens, args.allow_reasoning,
                  max_cost, args.reasoning_effort)

    print("\nAll done. Score with:  python score_benchmark.py")


if __name__ == "__main__":
    main()
