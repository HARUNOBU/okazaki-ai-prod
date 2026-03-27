#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional

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
                continue
    return rows

def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_qid_num(qid: str, expected_prefix: str) -> Optional[int]:
    if not qid or not isinstance(qid, str):
        return None
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

def uniq_preserve(seq: List[Any]) -> List[str]:
    seen = set()
    out = []
    for x in seq or []:
        if not isinstance(x, str):
            continue
        s = x.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out

def coalesce_pdf_url(d: Dict[str, Any]) -> str:
    meta = d.get("meta") if isinstance(d.get("meta"), dict) else {}
    return (d.get("pdf_url") or meta.get("pdf_url") or "").strip()

def coalesce_page(d: Dict[str, Any]) -> Optional[int]:
    meta = d.get("meta") if isinstance(d.get("meta"), dict) else {}
    page = d.get("page", None)
    if page in ("", None):
        page = meta.get("page", None)

    if page in ("", None):
        return None
    try:
        return int(page)
    except Exception:
        return None

def coalesce_category(d: Dict[str, Any], default: str = "ごみ") -> str:
    meta = d.get("meta") if isinstance(d.get("meta"), dict) else {}
    return str(d.get("category") or meta.get("category") or default)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafts", required=True, help="out/faq_drafts.json")
    ap.add_argument("--existing", required=True, help="faq_waste.jsonl")
    ap.add_argument("--outdir", default="./out")
    ap.add_argument("--qid-prefix", default="w-faq")
    ap.add_argument("--width", type=int, default=3)
    ap.add_argument("--allow-empty-answer", default="false", help="true/false")
    args = ap.parse_args()

    allow_empty = str(args.allow_empty_answer).lower() in ("1","true","yes","y")

    existing = read_jsonl(args.existing)
    drafts = load_json(args.drafts)
    if not isinstance(drafts, list):
        raise ValueError("drafts must be a list")

    nums = []
    for r in existing:
        n = parse_qid_num(r.get("qid",""), args.qid_prefix)
        if n is not None:
            nums.append(n)
    next_num = (max(nums) if nums else 0) + 1

    existing_questions = set((r.get("question") or "").strip() for r in existing if (r.get("question") or "").strip())

    append_rows: List[Dict[str, Any]] = []
    for d in drafts:
        q = (d.get("question") or "").strip()
        a = (d.get("answer") or "").strip()
        if not q:
            continue
        if q in existing_questions:
            continue
        if (not allow_empty) and (not a):
            # 安全：回答未確定のカードは入れない
            continue

        aliases = uniq_preserve(d.get("aliases") or [])
        pdf_url = coalesce_pdf_url(d)
        page = coalesce_page(d)
        category = coalesce_category(d, default="ごみ")

        qid = next_qid(args.qid_prefix, next_num, width=args.width)
        next_num += 1

        append_rows.append({
            "qid": qid,
            "question": q,
            "answer": a,
            "aliases": aliases,
            "pdf_url": pdf_url,
            "page": page,          # NoneならJSONでは null
            "category": category,  # build_waste_faq.py が参照
        })
        existing_questions.add(q)

    out_append = os.path.join(args.outdir, "faq_waste_append.jsonl")
    out_merged = os.path.join(args.outdir, "faq_waste_full_merged.jsonl")
    write_jsonl(out_append, append_rows)
    write_jsonl(out_merged, existing + append_rows)

    print(f"Existing: {len(existing)}")
    print(f"Drafts: {len(drafts)}")
    print(f"Append: {len(append_rows)} -> {out_append}")
    print(f"Merged: {len(existing)+len(append_rows)} -> {out_merged}")
    if not allow_empty:
        print("NOTE: allow-empty-answer=false -> empty answers were skipped (safe).")

if __name__ == "__main__":
    main()
