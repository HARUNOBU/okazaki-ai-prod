from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from openai import OpenAI


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue  # 空行は無視
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL parse error: {path} line {lineno}: {e}")
    return rows


def normalize_row(row: dict[str, Any], idx: int) -> tuple[str, str, dict[str, Any]]:
    """
    2形式に対応:
    1) {"question":"...", "answer":"..."}
    2) {"id":"...", "title":"...", "text":"...", "meta":{...}}
    """
    if "question" in row and "answer" in row:
        q = str(row["question"]).strip()
        a = str(row["answer"]).strip()
        if not q or not a:
            raise ValueError(f"question/answer empty at row {idx}")
        doc_id = str(row.get("id") or f"fixedfaq_{idx}")
        text = f"Q: {q}\nA: {a}"
        meta = {k: v for k, v in row.items() if k not in ("question", "answer")}
        meta["question"] = q
        meta["answer"] = a
        return doc_id, text, meta

    if "text" in row:
        doc_id = str(row.get("id") or f"fixedfaq_{idx}")
        text = str(row["text"]).strip()
        if not text:
            raise ValueError(f"text empty at row {idx}")
        meta = dict(row.get("meta") or {})
        if "title" in row:
            meta["title"] = row["title"]
        return doc_id, text, meta

    raise ValueError(f"unsupported row format at row {idx}: keys={list(row.keys())}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, help="入力JSONL")
    parser.add_argument("--reset", action="store_true", help="既存コレクション削除後に作り直す")
    parser.add_argument("--collection", default="okazaki_events_fixed_faq")
    args = parser.parse_args()

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY が見つかりません。")

    base_waste = Path(__file__).resolve().parents[1]     # .../okazaki_waste_rag/src
    portal_root = base_waste.parents[1]                  # .../okazaki-ai-portal
    db_dir = portal_root / "okazaki_rag" / "chroma_db"

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.is_absolute():
        jsonl_path = base_waste / jsonl_path

    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

    rows = load_jsonl(jsonl_path)

    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict[str, Any]] = []

    for i, row in enumerate(rows):
        doc_id, text, meta = normalize_row(row, i)
        ids.append(doc_id)
        docs.append(text)
        metas.append(meta)

    client = chromadb.PersistentClient(
        path=str(db_dir),
        settings=Settings(allow_reset=True)
    )

    if args.reset:
        try:
            client.delete_collection(args.collection)
            print(f"deleted collection: {args.collection}")
        except Exception:
            pass

    col = client.get_or_create_collection(args.collection)

    oa = OpenAI(api_key=api_key)
    emb = oa.embeddings.create(
        model="text-embedding-3-small",
        input=docs
    )
    vecs = [e.embedding for e in emb.data]

    col.upsert(
        ids=ids,
        documents=docs,
        embeddings=vecs,
        metadatas=metas
    )

    print("DONE")
    print("collection:", args.collection)
    print("count:", col.count())


if __name__ == "__main__":
    main()