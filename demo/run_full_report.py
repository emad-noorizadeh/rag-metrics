#!/usr/bin/env python3
"""
Generate a full context-utilization report from a JSON input file.

Input JSON schema (single example):
{
  "question": "string (optional)",
  "answer": "string (required)",
  "context": ["string", "string", ...],   # list of snippets
  "metrics_config": { ... }               # optional dict of feature toggles (see metric_utils)
}

Usage:
  python demo/run_full_report.py --input demo/example.json
  python demo/run_full_report.py --input demo/example.json --output demo/report.json

Optional:
  --pretty            Pretty-print the JSON to stdout or the --output file
  --no-embed          Ensure embedding-based fields are skipped (default: skipped)
  --bm25/--no-bm25    Toggle BM25 best-context selection (default: on)

Notes:
- This uses metric_utils.context_utilization_report_with_entities exactly as-is.
- You can pass quick-win toggles in the input JSON under "metrics_config".
"""

import argparse
import json
import sys
import os
from pathlib import Path

# Ensure the repo root (which contains metric_utils.py) is importable when running from demo/
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from metric_utils import context_utilization_report_with_entities

def _read_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Failed to read JSON from {path}: {e}", file=sys.stderr)
        sys.exit(1)

def _validate_payload(payload: dict) -> None:
    if not isinstance(payload, dict):
        print("❌ Input JSON must be an object.", file=sys.stderr)
        sys.exit(1)
    if "answer" not in payload or not isinstance(payload["answer"], str):
        print("❌ Input must include a string field 'answer'.", file=sys.stderr)
        sys.exit(1)
    if "context" in payload and not isinstance(payload["context"], list):
        print("❌ 'context' must be a list of strings.", file=sys.stderr)
        sys.exit(1)

def main():
    ap = argparse.ArgumentParser(description="Full context-utilization report generator")
    ap.add_argument("--input", "-i", required=True, help="Path to input JSON with question/answer/context")
    ap.add_argument("--output", "-o", default=None, help="Optional path to write the report JSON")
    ap.add_argument("--pretty", action="store_true", help="Pretty print JSON output")
    ap.add_argument("--bm25", dest="use_bm25", action="store_true", default=True, help="Use BM25 for best-context selection (default)")
    ap.add_argument("--no-bm25", dest="use_bm25", action="store_false", help="Disable BM25 for best-context selection")
    ap.add_argument("--no-embed", dest="use_embed", action="store_false", default=False,
                    help="Ensure embedding-based fields are skipped (default: skipped)")
    args = ap.parse_args()

    inp = Path(args.input)
    payload = _read_json(inp)
    _validate_payload(payload)

    question = payload.get("question", "")
    answer   = payload["answer"]
    context  = payload.get("context", [])
    metrics_config = payload.get("metrics_config", None)

    report = context_utilization_report_with_entities(
        question=question,
        answer=answer,
        retrieved_contexts=context,
        use_bm25_for_best=args.use_bm25,
        use_embed_alignment=args.use_embed,
        metrics_config=metrics_config,
    )

    out_text = json.dumps(report, indent=2 if args.pretty or args.output else None, ensure_ascii=False, sort_keys=False)

    if args.output:
        outp = Path(args.output)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(out_text + ("\n" if not out_text.endswith("\n") else ""), encoding="utf-8")
        print(f"✅ Wrote report → {outp}")
    else:
        print(out_text)

if __name__ == "__main__":
    main()