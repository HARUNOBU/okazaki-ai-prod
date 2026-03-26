from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List
import os

import chromadb
from chromadb.config import Settings
from openai import OpenAI

# src/config.py を読めるようにする
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import KANKO_DB_DIR, OPENAI_API_KEY, DATA_DIR

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


def sanitize_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = json.dumps(v, ensure_ascii=False)
    return out


def resolve_input_path() -> Path:
    if len(sys.argv) >= 2 and not sys.argv[1].startswith("--"):
        return Path(sys.argv[1])

    candidates = [
        DATA_DIR / "events" / "okazaki_events_faq.jsonl",
        DATA_DIR / "okazaki_events_faq.jsonl",
        DATA_DIR / "events_faq.jsonl",
    ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "events FAQ jsonl not found. "
        "指定例: python src/tools/build_events_faq.py data/events/okazaki_events_faq.jsonl"
    )


def main():
    in_path = resolve_input_path()
    rows = load_jsonl(in_path)

    api_key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found")

    oa = OpenAI(api_key=api_key)

    client = chromadb.PersistentClient(
        path=str(KANKO_DB_DIR),
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
        metas.append(sanitize_meta({
            "qid": qid,
            "question": q,
            "pdf_url": r.get("pdf_url", "") or "",
            "page": r.get("page", "") or "",
            "source": "faq",
            "category": "観光",
        }))

    if not docs:
        print("no faq rows")
        return

    BATCH = 64
    for idxs, dcs, mts in zip(chunks(ids, BATCH), chunks(docs, BATCH), chunks(metas, BATCH)):
        embs = oa.embeddings.create(model=EMBED_MODEL, input=dcs).data
        emb_vectors = [e.embedding for e in embs]
        col.upsert(ids=idxs, documents=dcs, metadatas=mts, embeddings=emb_vectors)

    print("OK")
    print("input:", in_path)
    print("collection:", COL_NAME)
    print("count:", col.count())


if __name__ == "__main__":
    main()