#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="out/faq_drafts.csv")
    ap.add_argument("--out", required=True, help="out/faq_drafts.json")
    args = ap.parse_args()

    src = Path(args.csv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with src.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # answer_draft が空のものは残しても良いが、後段で skip される
            rows.append({
                "id": (r.get("id") or "").strip(),
                "qid": (r.get("qid") or "").strip(),  # あれば
                "category": (r.get("category") or "").strip(),
                "question": (r.get("question") or "").strip(),
                "answer": (r.get("answer_draft") or "").strip(),  # ★ここを answer として出す
                "aliases": [s.strip() for s in (r.get("aliases") or "").split("|") if s.strip()],
                "pdf_url": (r.get("pdf_url") or "").strip(),
                "page": (r.get("page") or "").strip(),
            })

    with out.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"written: {out} (rows={len(rows)})")

if __name__ == "__main__":
    main()
