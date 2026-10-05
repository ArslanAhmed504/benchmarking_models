"""
run_benchmark_claude.py
------------------------
Same invoice-extraction benchmark as run_benchmark.py, but calls Claude
directly via the official Anthropic API (client.messages.create) instead of
routing through OpenRouter. Lets you A/B a Claude model against the
OpenRouter-routed models already in results/ — same prompt, same PDFs, same
scoring pipeline.

Requirements:
  export ANTHROPIC_API_KEY=sk-ant-...   (or use `ant auth login` — the SDK
                                          picks up an active profile with no
                                          env var needed)
  pip3 install anthropic pymupdf

Usage:
  python run_benchmark_claude.py --limit 10        # small test batch first
  python run_benchmark_claude.py                    # full run
  python run_benchmark_claude.py --model claude-sonnet-5

Output:
  results/<out-name>.jsonl (default: anthropic_claude-opus-5.jsonl) — same
  record schema as run_benchmark.py, so score_benchmark.py picks it up as
  just another model with zero changes.
"""

import os
import json
import time
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import anthropic
except ImportError:
    raise SystemExit("ERROR: this script needs the Anthropic SDK — run: pip3 install anthropic")

from run_benchmark import load_schema, build_prompt, render_pdf_pages, parse_json_response, load_done_ids

# Claude API list pricing, $ per token (see the claude-api skill's model table).
# Used only to estimate spend for the --max-cost budget cap below — the
# Messages API returns token counts, not a dollar figure, unlike OpenRouter.
PRICING = {
    "claude-opus-5":    (5.00 / 1_000_000, 25.00 / 1_000_000),
    "claude-sonnet-5":  (3.00 / 1_000_000, 15.00 / 1_000_000),
    "claude-haiku-4-5": (1.00 / 1_000_000, 5.00 / 1_000_000),
}


class PermanentError(RuntimeError):
    """Errors that won't go away on retry: bad request, auth, unknown model,
    or a safety-classifier refusal (deterministic for a given document)."""


def call_claude(client, model, prompt, image_b64_list, max_tokens, effort,
                max_retries=5):
    content = [{"type": "text", "text": prompt}] + [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}}
        for b64 in image_b64_list
    ]
    last_err = None
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                # Plain structured extraction, not a reasoning task — thinking
                # only adds cost and shares the max_tokens budget for no
                # accuracy gain here. Disabling requires effort <= "high" on
                # Claude Opus 5; default effort "low" keeps cost down further.
                thinking={"type": "disabled"},
                output_config={"effort": effort},
                messages=[{"role": "user", "content": content}],
            )
            if response.stop_reason == "refusal":
                category = response.stop_details.category if response.stop_details else None
                raise PermanentError(f"refusal (category={category})")
            text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                "cache_read_input_tokens": response.usage.cache_read_input_tokens,
                "cache_creation_input_tokens": response.usage.cache_creation_input_tokens,
            }
            input_price, output_price = PRICING.get(model, (0.0, 0.0))
            usage["cost"] = (usage["prompt_tokens"] * input_price
                             + usage["completion_tokens"] * output_price)
            return text, usage
        except (anthropic.RateLimitError, anthropic.InternalServerError,
                anthropic.APIConnectionError) as e:
            last_err = str(e)
            time.sleep(min(2 ** attempt * 2, 60))
        except (anthropic.BadRequestError, anthropic.AuthenticationError,
                anthropic.PermissionDeniedError, anthropic.NotFoundError) as e:
            raise PermanentError(str(e)[:500])
    raise RuntimeError(last_err or "unknown error")


def process_doc(client, doc_id, pdf_dir, model, prompt, max_pages, max_tokens, effort):
    pdf_path = pdf_dir / f"{doc_id}.pdf"
    t0 = time.time()
    try:
        images = render_pdf_pages(pdf_path, max_pages)
        content, usage = call_claude(client, model, prompt, images, max_tokens, effort)
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


def run_model(client, model, doc_ids, pdf_dir, prompt, result_path, concurrency,
              max_pages, max_tokens, effort, max_cost):
    done = load_done_ids(result_path)
    todo = [d for d in doc_ids if d not in done]

    print(f"\n--- {model} (direct Claude API) -> {result_path.name} ---")
    print(f"  already done: {len(done):,} | remaining: {len(todo):,}")
    if max_cost is not None:
        print(f"  cost budget: ${max_cost:.2f} (estimated from list pricing)")
    if not todo:
        return

    lock = threading.Lock()
    n_ok = n_err = 0
    total_tokens = 0
    total_cost = 0.0

    with open(result_path, "a", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(process_doc, client, d, pdf_dir, model, prompt,
                            max_pages, max_tokens, effort): d
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
                    total_cost += rec["usage"].get("cost", 0) or 0
                if i % 10 == 0 or i == len(todo):
                    print(f"  [{i}/{len(todo)}] ok={n_ok} err={n_err} "
                          f"tokens={total_tokens:,} cost=${total_cost:.4f}")
                if max_cost is not None and total_cost >= max_cost:
                    print(f"  !! cost budget of ${max_cost:.2f} reached "
                          f"(${total_cost:.4f} spent) — stopping remaining "
                          f"{len(todo) - i:,} document(s)")
                    pool.shutdown(wait=False, cancel_futures=True)
                    break

    print(f"  finished: ok={n_ok} err={n_err} cost=${total_cost:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Direct Claude API invoice extraction benchmark")
    parser.add_argument("--model", default="claude-opus-5")
    parser.add_argument("--schema",       default="invoice_schema.json")
    parser.add_argument("--ground-truth", default="ground_truth.json")
    parser.add_argument("--pdf-dir",      default="pdfs")
    parser.add_argument("--out-dir",      default="results")
    parser.add_argument("--out-name",     default=None,
                        help="Result filename (default: anthropic_<model>.jsonl)")
    parser.add_argument("--limit",        type=int, default=None,
                        help="Max documents (default: all). Use a small number "
                             "for a first test batch, e.g. --limit 10")
    parser.add_argument("--concurrency",  type=int, default=8)
    parser.add_argument("--max-pages",    type=int, default=4)
    parser.add_argument("--max-tokens",   type=int, default=4000)
    parser.add_argument("--effort",       choices=["low", "medium", "high"],
                        default="low",
                        help="Reasoning effort with thinking disabled (default "
                             "low — cheapest, this is plain extraction not "
                             "reasoning). xhigh/max are not usable with "
                             "thinking disabled on Claude Opus 5.")
    parser.add_argument("--max-cost",     type=float, default=10.0,
                        help="Stop once cumulative estimated spend this run "
                             "hits this many dollars (default $10). Pass 0 or "
                             "negative to disable.")
    args = parser.parse_args()
    max_cost = args.max_cost if args.max_cost and args.max_cost > 0 else None

    try:
        import fitz  # noqa: F401
    except ImportError:
        raise SystemExit("ERROR: this script needs PyMuPDF — run: pip3 install pymupdf")

    client = anthropic.Anthropic()

    schema_fields = load_schema(args.schema)
    prompt = build_prompt(schema_fields)

    with open(args.ground_truth, encoding="utf-8") as f:
        ground_truth = json.load(f)

    doc_ids = sorted(ground_truth.keys())
    if args.limit:
        doc_ids = doc_ids[:args.limit]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    out_name = args.out_name or f"anthropic_{args.model.replace('/', '_')}.jsonl"
    result_path = out_dir / out_name

    print(f"Benchmarking {args.model} (direct Claude API) on {len(doc_ids):,} documents")
    print(f"Max pages/doc: {args.max_pages} | max tokens/call: {args.max_tokens} | "
          f"effort: {args.effort} | thinking: disabled")

    run_model(client, args.model, doc_ids, Path(args.pdf_dir), prompt,
              result_path, args.concurrency, args.max_pages, args.max_tokens,
              args.effort, max_cost)

    print(f"\nDone. Score with:  python score_benchmark.py")


if __name__ == "__main__":
    main()
