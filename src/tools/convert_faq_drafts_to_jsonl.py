#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
convert_faq_drafts_to_jsonl.py

Convert faq_drafts.json (generated from logs) into your production FAQ jsonl format:
  {"qid":"w-faq-001","question":"...","answer":"...","aliases":[...],"pdf_url":"...","page":""}

- Avoid qid collisions by reading existing faq_waste.jsonl
- Generate new qid sequentially (w-faq-001, w-faq-002, ...)
- Output:
  - faq_waste_append.jsonl (new entries only)
  - faq_waste_full_merged.jsonl (existing + new)  [optional]

Usage:
  python tools/convert_faq_drafts_to_jsonl.py \
    --drafts out/faq_drafts.json \
    --existing faq_waste.jsonl \
    --outdir out \
    --qid-prefix w-faq \
    --start-from auto \
    --skip-empty-answer true
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple


QID_RE = re.compile(r"^(?P<prefix>[a-zA-Z0-9\-]+)-(?P<num>\d+)$")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except Exception:
                # ignore broken lines
                continue
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_drafts_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("drafts json must be a list")
    return data


def parse_qid_num(qid: str, expected_prefix: str) -> Optional[int]:
    """
    Expect qid like: w-faq-001
    We'll parse trailing number after last hyphen group if it matches expected_prefix.
    """
    if not qid or not isinstance(qid, str):
        return None

    # For "w-faq-001", prefix part is "w-faq" and num "001"
    m = QID_RE.match(qid)
    if not m:
        return None
    if m.group("prefix") != expected_prefix:
        return None
    try:
        return int(m.group("num"))
    except Exception:
        return None


def next_qid(prefix: str, n: int, width: int = 3) -> str:
    return f"{prefix}-{n:0{width}d}"


def uniq_preserve(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        if not x:
            continue
        s = str(x).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def coalesce_pdf_url(d: Dict[str, Any]) -> str:
    # drafts meta may store pdf_url under meta.pdf_url
    meta = d.get("meta") if isinstance(d.get("meta"), dict) else {}
    pdf_url = (d.get("pdf_url") or meta.get("pdf_url") or "").strip()
    # also accept "url" style
    if not pdf_url:
        pdf_url = (meta.get("url") or "").strip()
    return pdf_url


def coalesce_page(d: Dict[str, Any]) -> str:
    meta = d.get("meta") if isinstance(d.get("meta"), dict) else {}
    page = d.get("page", None)
    if page is None:
        page = meta.get("page", None)
    # You currently store page as "" (string). We'll output "" if unknown.
    if page is None or page == "":
        return ""
    # int -> str
    try:
        return str(int(page))
    except Exception:
        return ""


def build_entry_from_draft(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert one draft item to production format (without qid yet).
    Draft expected keys:
      question, answer, aliases, meta(pdf_url/page)
    """
    q = (d.get("question") or "").strip()
    a = (d.get("answer") or "").strip()
    aliases = d.get("aliases") or []
    if not isinstance(aliases, list):
        aliases = []
    aliases = uniq_preserve([str(x) for x in aliases])

    pdf_url = coalesce_pdf_url(d)
    page = coalesce_page(d)

    return {
        "question": q,
        "answer": a,
        "aliases": aliases,
        "pdf_url": pdf_url,
        "page": page,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafts", required=True, help="path to faq_drafts.json")
    ap.add_argument("--existing", required=True, help="path to existing faq_waste.jsonl (production)")
    ap.add_argument("--outdir", default="./out", help="output dir")
    ap.add_argument("--qid-prefix", default="w-faq", help="qid prefix like w-faq")
    ap.add_argument("--start-from", default="auto", help="auto or a number (e.g., 7)")
    ap.add_argument("--skip-empty-answer", default="true", help="true/false. If true, drop drafts with empty answer.")
    ap.add_argument("--width", type=int, default=3, help="qid number width (001 -> 3)")
    args = ap.parse_args()

    skip_empty = str(args.skip_empty_answer).lower() in ("1", "true", "yes", "y")

    existing_rows = read_jsonl(args.existing)
    drafts = load_drafts_json(args.drafts)

    # Determine next id
    nums = []
    for r in existing_rows:
        n = parse_qid_num(r.get("qid", ""), args.qid_prefix)
        if n is not None:
            nums.append(n)
    max_existing = max(nums) if nums else 0

    if args.start_from != "auto":
        try:
            start_num = int(args.start_from)
        except Exception:
            raise ValueError("--start-from must be 'auto' or an integer")
        next_num = max(start_num, max_existing + 1)
    else:
        next_num = max_existing + 1

    # Build a quick duplicate guard: do not add if same question already exists
    existing_questions = set()
    for r in existing_rows:
        q = (r.get("question") or "").strip()
        if q:
            existing_questions.add(q)

    append_rows = []
    for d in drafts:
        entry = build_entry_from_draft(d)

        if not entry["question"]:
            continue
        if entry["question"] in existing_questions:
            # Already exists in production -> skip
            continue
        if skip_empty and not entry["answer"]:
            # Safe default: don't add unanswered FAQ cards
            continue

        qid = next_qid(args.qid_prefix, next_num, width=args.width)
        next_num += 1

        out = {
            "qid": qid,
            **entry,
        }
        append_rows.append(out)
        existing_questions.add(entry["question"])

    out_append = os.path.join(args.outdir, "faq_waste_append.jsonl")
    out_merged = os.path.join(args.outdir, "faq_waste_full_merged.jsonl")

    write_jsonl(out_append, append_rows)
    write_jsonl(out_merged, existing_rows + append_rows)

    print(f"Existing rows: {len(existing_rows)}")
    print(f"Draft rows: {len(drafts)}")
    print(f"Appended new rows: {len(append_rows)}")
    print(f"Wrote: {out_append}")
    print(f"Wrote: {out_merged}")
    if skip_empty:
        print("Note: --skip-empty-answer=true -> empty answers were skipped (safe default).")


if __name__ == "__main__":
    main()
