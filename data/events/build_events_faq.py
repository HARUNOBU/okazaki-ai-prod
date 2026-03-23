# src/build_events_faq.py
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import chromadb
from chromadb.config import Settings
from openai import OpenAI
from dotenv import load_dotenv

# app_portal_v2.py と同じDB位置の思想に合わせる
BASE_WASTE = Path(__file__).resolve().parents[1]
PORTAL_ROOT = BASE_WASTE.parents[0]
KANKO_DB_DIR = str(PORTAL_ROOT / "okazaki_rag" / "chroma_db")

EMBED_MODEL = "text-embedding-3-small"
COL_NAME = "okazaki_events_faq"

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows

def chunks(lst: List[Any], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def main():
    if len(sys.argv) < 2:
        print("Usage: python src/build_events_faq.py data/events/okazaki_events_faq.jsonl")
        sys.exit(1)

    in_path = Path(sys.argv[1])
    rows = load_jsonl(in_path)

    load_dotenv()
    oa = OpenAI()

    client = chromadb.PersistentClient(
        path=KANKO_DB_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    col = client.get_or_create_collection(name=COL_NAME)

    ids = []
    docs = []
    metas = []

    for r in rows:
        qid = str(r.get("qid") or "").strip()
        q = (r.get("question") or "").strip()
        a = (r.get("answer") or "").strip()

        if not qid or not q:
            continue

        ids.append(qid)
        docs.append(f"Q: {q}\nA: {a}")
        metas.append({
            "qid": qid,
            "question": q,
            "pdf_url": r.get("pdf_url", "") or "",
            "page": r.get("page", "") or "",
            "source": "faq",
            "category": "観光",
        })

    # upsert（IDが同じなら更新）
    BATCH = 64
    for idxs, dcs, mts in zip(chunks(ids, BATCH), chunks(docs, BATCH), chunks(metas, BATCH)):
        embs = oa.embeddings.create(model=EMBED_MODEL, input=dcs).data
        emb_vectors = [e.embedding for e in embs]
        col.upsert(ids=idxs, documents=dcs, metadatas=mts, embeddings=emb_vectors)

    print("OK")
    print("collection:", COL_NAME)
    print("count:", col.count())

if __name__ == "__main__":
    main()
