"""
run_benchmark_gemini_gcp.py
----------------------------
Same invoice-extraction benchmark as run_benchmark.py, but calls Gemini
directly via the Google Generative Language API (generativelanguage.
googleapis.com) using an API key, instead of routing through OpenRouter.

This exists to A/B a model already benchmarked through OpenRouter
(results/google_gemini-3.6-flash.jsonl) against calling Google's endpoint
directly — same prompt, same PDFs, same scoring pipeline — so any gap can be
attributed to OpenRouter's routing/pricing rather than the model itself.

Requirements:
  export GEMINI_API_KEY=...        (or pass --api-key-env to use another var)
  pip3 install pymupdf             (only needed for --input-mode images)

Usage:
  python run_benchmark_gemini_gcp.py --limit 10                # small test batch
  python run_benchmark_gemini_gcp.py                            # full run
  python run_benchmark_gemini_gcp.py --model gemini-2.5-flash   # different model id

Output:
  results/<out-name>.jsonl (default: google_gemini-3.6-flash_gcp.jsonl) — same
  record schema as run_benchmark.py, so score_benchmark.py picks it up as just
  another model with zero changes.
"""

import os
import json
import time
import argparse
import threading
import urllib.request
import urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from run_benchmark import (
    load_schema, build_prompt, render_pdf_pages, parse_json_response,
    load_done_ids, SSL_CONTEXT, PermanentError, InsufficientCreditsError,
    NON_RETRYABLE_HTTP_CODES,
)

GEMINI_URL_TMPL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def call_gemini(model: str, prompt: str, image_b64_list: list, api_key: str,
                max_retries: int = 5, timeout: int = 300, max_tokens: int = 4000,
                thinking_budget: int = None):
    parts = [{"text": prompt}] + [
        {"inline_data": {"mime_type": "image/png", "data": b64}}
        for b64 in image_b64_list
    ]
    generation_config = {
        "temperature": 0,
        "maxOutputTokens": max_tokens,
    }
    if thinking_budget is not None:
        # Mirrors run_benchmark.py's --reasoning-effort handling for
        # OpenRouter: caps the hidden "thinking" phase instead of letting a
        # plain extraction task burn an unbounded number of tokens on it.
        generation_config["thinkingConfig"] = {"thinkingBudget": thinking_budget}

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": generation_config,
    }
    body = json.dumps(payload).encode()
    url = GEMINI_URL_TMPL.format(model=model)

    last_err = None
    for attempt in range(max_retries):
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "x-goog-api-key": api_key,
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=SSL_CONTEXT) as resp:
                data = json.loads(resp.read().decode())
            if "error" in data:
                raise RuntimeError(data["error"].get("message", str(data["error"])))
            candidates = data.get("candidates") or []
            if not candidates:
                # Blocked by safety filters or genuinely empty response —
                # a legitimate "no answer", not a bug (see run_benchmark.py's
                # parse_json_response docstring for the analogous case).
                content = None
            else:
                cand = candidates[0]
                text_parts = [
                    p.get("text", "") for p in cand.get("content", {}).get("parts", [])
                    if not p.get("thought")
                ]
                content = "".join(text_parts) if text_parts else None
            usage_meta = data.get("usageMetadata", {})
            usage = {
                "prompt_tokens": usage_meta.get("promptTokenCount", 0),
                "completion_tokens": usage_meta.get("candidatesTokenCount", 0),
                "total_tokens": usage_meta.get("totalTokenCount", 0),
                "reasoning_tokens": usage_meta.get("thoughtsTokenCount", 0),
                # The Generative Language API (API-key auth) doesn't return
                # a cost figure the way OpenRouter does — leave unset rather
                # than fabricate a number.
                "cost": None,
            }
            return content, usage
        except urllib.error.HTTPError as e:
            body_text = e.read().decode(errors="replace")[:500]
            last_err = f"HTTP {e.code}: {body_text}"
            if e.code == 429:
                time.sleep(min(2 ** attempt * 2, 60))
                continue
            if e.code in (500, 502, 503, 520, 524):
                time.sleep(min(2 ** attempt * 2, 60))
                continue
            if e.code in NON_RETRYABLE_HTTP_CODES:
                raise PermanentError(last_err)
            break
        except Exception as e:  # timeouts, connection resets, malformed responses
            last_err = str(e)
            time.sleep(min(2 ** attempt * 2, 60))
    raise RuntimeError(last_err or "unknown error")


def process_doc(doc_id: str, pdf_dir: Path, model: str, prompt: str,
                api_key: str, max_pages: int, max_tokens: int,
                thinking_budget: int):
    pdf_path = pdf_dir / f"{doc_id}.pdf"
    t0 = time.time()
    try:
        images = render_pdf_pages(pdf_path, max_pages)
        content, usage = call_gemini(model, prompt, images, api_key,
                                     max_tokens=max_tokens,
                                     thinking_budget=thinking_budget)
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
    except PermanentError as e:
        return {
            "doc_id": doc_id, "model": model, "extracted": None, "usage": None,
            "latency_s": round(time.time() - t0, 2),
            "error": str(e)[:500], "permanent_error": True,
        }
    except Exception as e:
        return {
            "doc_id": doc_id, "model": model, "extracted": None, "usage": None,
            "latency_s": round(time.time() - t0, 2),
            "error": str(e)[:500],
        }


def run_model(model: str, doc_ids: list, pdf_dir: Path, prompt: str,
              api_key: str, result_path: Path, concurrency: int,
              max_pages: int, max_tokens: int, thinking_budget: int):
    done = load_done_ids(result_path)
    todo = [d for d in doc_ids if d not in done]

    print(f"\n--- {model} (direct Gemini API) -> {result_path.name} ---")
    print(f"  already done: {len(done):,} | remaining: {len(todo):,}")
    if not todo:
        return

    lock = threading.Lock()
    n_ok = n_err = 0
    total_tokens = 0

    with open(result_path, "a", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(process_doc, d, pdf_dir, model, prompt, api_key,
                            max_pages, max_tokens, thinking_budget): d
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
                if rec.get("usage"):
                    total_tokens += rec["usage"].get("total_tokens", 0)
                if i % 10 == 0 or i == len(todo):
                    print(f"  [{i}/{len(todo)}] ok={n_ok} err={n_err} tokens={total_tokens:,}")

    print(f"  finished: ok={n_ok} err={n_err}")


def main():
    parser = argparse.ArgumentParser(
        description="Direct Gemini API (GCP) invoice extraction benchmark")
    parser.add_argument("--model", default="gemini-3.6-flash",
                        help="Gemini model id as used by the Generative "
                             "Language API, e.g. gemini-3.6-flash")
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY",
                        help="Env var holding the API key (default: GEMINI_API_KEY, "
                             "falls back to GOOGLE_API_KEY if unset)")
    parser.add_argument("--schema",       default="invoice_schema.json")
    parser.add_argument("--ground-truth", default="ground_truth.json")
    parser.add_argument("--pdf-dir",      default="pdfs")
    parser.add_argument("--out-dir",      default="results")
    parser.add_argument("--out-name",     default=None,
                        help="Result filename (default: google_<model>_gcp.jsonl)")
    parser.add_argument("--limit",        type=int, default=None,
                        help="Max documents (default: all). Use a small number "
                             "for a first test batch, e.g. --limit 10")
    parser.add_argument("--concurrency",  type=int, default=8)
    parser.add_argument("--max-pages",    type=int, default=4)
    parser.add_argument("--max-tokens",   type=int, default=4000)
    parser.add_argument("--thinking-budget", type=int, default=0,
                        help="Caps the model's hidden 'thinking' phase, in "
                             "tokens (default 0 = minimal). Set higher, or "
                             "pass -1 for unbounded, if the endpoint requires "
                             "a nonzero thinking budget.")
    args = parser.parse_args()

    try:
        import fitz  # noqa: F401
    except ImportError:
        raise SystemExit("ERROR: this script needs PyMuPDF — run: pip3 install pymupdf")

    api_key = os.environ.get(args.api_key_env) or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit(
            f"ERROR: set {args.api_key_env} (or GOOGLE_API_KEY) to your Gemini API key.\n"
            "Get one at https://aistudio.google.com/apikey")

    schema_fields = load_schema(args.schema)
    prompt = build_prompt(schema_fields)

    with open(args.ground_truth, encoding="utf-8") as f:
        ground_truth = json.load(f)

    doc_ids = sorted(ground_truth.keys())
    if args.limit:
        doc_ids = doc_ids[:args.limit]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    out_name = args.out_name or f"google_{args.model.replace('/', '_')}_gcp.jsonl"
    result_path = out_dir / out_name

    thinking_budget = None if args.thinking_budget < 0 else args.thinking_budget

    print(f"Benchmarking {args.model} (direct Gemini API) on {len(doc_ids):,} documents")
    print(f"Max pages/doc: {args.max_pages} | max tokens/call: {args.max_tokens} | "
          f"thinking budget: {'unbounded' if thinking_budget is None else thinking_budget}")

    run_model(args.model, doc_ids, Path(args.pdf_dir), prompt, api_key,
              result_path, args.concurrency, args.max_pages, args.max_tokens,
              thinking_budget)

    print(f"\nDone. Score with:  python score_benchmark.py")


if __name__ == "__main__":
    main()
