from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import chromadb
from chromadb.config import Settings
from openai import OpenAI

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import KANKO_DB_DIR, OPENAI_API_KEY, DATA_DIR

EMBED_MODEL = "text-embedding-3-small"
DEFAULT_COLLECTION = "okazaki_events_fixed_faq"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows


def sanitize_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (meta or {}).items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = json.dumps(v, ensure_ascii=False)
    return out


def normalize_item(x: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    _id = str(x.get("id") or x.get("qid") or "").strip()
    if not _id:
        raise ValueError("jsonl item missing id/qid")

    title = str(x.get("title") or "").strip()
    text = str(x.get("text") or "").strip()
    if not text:
        q = str(x.get("question") or "").strip()
        a = str(x.get("answer") or "").strip()
        if q:
            text = f"Q: {q}\nA: {a}".strip()
    if not text:
        raise ValueError(f"jsonl item missing text: id={_id}")

    meta = dict(x.get("meta") or {})
    if "question" in x and "question" not in meta:
        meta["question"] = x.get("question")
    if "answer" in x and "answer" not in meta:
        meta["answer"] = x.get("answer")
    if "pdf_url" in x and "pdf_url" not in meta:
        meta["pdf_url"] = x.get("pdf_url")
    if "page" in x and "page" not in meta:
        meta["page"] = x.get("page")
    if "url" not in meta and meta.get("pdf_url"):
        meta["url"] = meta["pdf_url"]
    if title:
        meta["title"] = title
    meta.setdefault("source", "fixed_faq")
    meta.setdefault("category", "観光")

    return _id, text, sanitize_meta(meta)


def embed_texts(oa: OpenAI, texts: List[str], batch_size: int = 64, sleep_sec: float = 0.2) -> List[List[float]]:
    out: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        resp = oa.embeddings.create(model=EMBED_MODEL, input=chunk)
        out.extend([d.embedding for d in resp.data])
        time.sleep(sleep_sec)
    return out


def resolve_jsonl_path(arg_path: str | None) -> Path:
    if arg_path:
        return Path(arg_path)
    candidates = [
        DATA_DIR / "tourism_fixed_faq.jsonl",
        DATA_DIR / "events" / "tourism_fixed_faq.jsonl",
        DATA_DIR / "tourism" / "tourism_fixed_faq.jsonl",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("tourism_fixed_faq.jsonl not found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="", help="tourism_fixed_faq.jsonl のパス")
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--db", default=str(KANKO_DB_DIR))
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    api_key = OPENAI_API_KEY
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found")

    jsonl_path = resolve_jsonl_path(args.jsonl)
    rows = read_jsonl(jsonl_path)

    ids: List[str] = []
    texts: List[str] = []
    metas: List[Dict[str, Any]] = []

    for x in rows:
        _id, text, meta = normalize_item(x)
        ids.append(_id)
        texts.append(text)
        metas.append(meta)

    if len(ids) != len(set(ids)):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"Duplicate ids found: {dup[:10]} total={len(dup)}")

    print(f"[OK] jsonl loaded: {len(ids)} items")
    print(f"      jsonl: {jsonl_path}")
    print(f"      db_dir: {args.db}")
    print(f"      collection: {args.collection}")

    if args.dry_run:
        print("[DRY-RUN] skip embeddings / skip upsert")
        return

    oa = OpenAI(api_key=api_key)
    embs = embed_texts(oa, texts)

    client = chromadb.PersistentClient(
        path=str(args.db),
        settings=Settings(anonymized_telemetry=False)
    )

    if args.reset:
        try:
            client.delete_collection(args.collection)
            print("[INFO] deleted existing collection:", args.collection)
        except Exception:
            pass

    col = client.get_or_create_collection(args.collection)
    col.upsert(ids=ids, documents=texts, metadatas=metas, embeddings=embs)

    print("[DONE] upserted:", len(ids))
    print("[DONE] collection_count:", col.count())


if __name__ == "__main__":
    main()
