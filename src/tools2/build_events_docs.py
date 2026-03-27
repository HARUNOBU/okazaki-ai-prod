from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from config import DATA_DIR

DEFAULT_INPUT = DATA_DIR / "events_2026.csv"
DEFAULT_OUTPUT = DATA_DIR / "events_docs_2026.json"


def row_to_doc(row: dict, idx: int) -> dict:
    title = (row.get("title") or row.get("name") or row.get("event_name") or "").strip()
    if not title:
        title = f"イベント{idx+1}"
    fields = []
    for key in ["summary", "description", "place", "venue", "start", "end", "date", "fee", "target", "contact", "url"]:
        val = str(row.get(key, "")).strip()
        if val:
            fields.append(f"{key}: {val}")
    body = "    ".join(fields)
    text = f
    "{title}    {body}".strip()
    meta = {k: ("" if v is None else str(v)) for k, v in row.items()}
    meta.setdefault("title", title)
    meta.setdefault("source", meta.get("url", ""))
    meta.setdefault("category", "観光")
    return {"id": str(row.get("id") or f"events_2026_{idx:04d}"), "text": text, "meta": meta}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    docs = [row_to_doc(r, i) for i, r in enumerate(rows)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")
    print("done")
    print("docs:", len(docs))
    print("output:", output_path)


if __name__ == "__main__":
    main()
