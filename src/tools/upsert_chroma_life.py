from __future__ import annotations

import json
from pathlib import Path
import sys

import chromadb
from chromadb.config import Settings
from openai import OpenAI

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import WASTE_LIFE_DB_DIR, OPENAI_API_KEY, DATA_DIR

COLLECTION_NAME = "okazaki_life"
DOCS_JSON = DATA_DIR / "docs" / "docs_okazaki_life.json"
EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 96


def batched(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def normalize_meta(m: dict) -> dict:
    m = dict(m or {})

    if "pdf_url" not in m or not m.get("pdf_url"):
        m["pdf_url"] = (
            m.get("file_url")
            or m.get("file_url_pdf")
            or m.get("apply_url")
            or m.get("source_url")
            or m.get("source_page")
            or m.get("source")
            or ""
        )

    if "source" not in m or not m.get("source"):
        m["source"] = m.get("source_url") or m.get("source_page") or ""

    m.setdefault("page", None)
    m.setdefault("chunk_index", 0)
    return m


def main():
    api_key = OPENAI_API_KEY
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found")

    if not DOCS_JSON.exists():
        raise FileNotFoundError(DOCS_JSON)

    oa = OpenAI(api_key=api_key)
    docs = json.loads(DOCS_JSON.read_text(encoding="utf-8"))

    ids = [d["id"] for d in docs]
    texts = [d["text"] for d in docs]
    metas = [normalize_meta(d.get("meta", {})) for d in docs]

    client = chromadb.PersistentClient(
        path=str(WASTE_LIFE_DB_DIR),
        settings=Settings(anonymized_telemetry=False)
    )
    col = client.get_or_create_collection(COLLECTION_NAME)

    existing = set()
    try:
        got = col.get(include=[])
        existing = set(got["ids"])
    except Exception:
        pass

    new_ids, new_texts, new_metas = [], [], []
    for i, t, m in zip(ids, texts, metas):
        if i in existing:
            continue
        new_ids.append(i)
        new_texts.append(t)
        new_metas.append(m)

    print(f"docs: {len(docs)} / new: {len(new_ids)}")
    if not new_ids:
        print("nothing to add.")
        return

    for id_b, text_b, meta_b in zip(batched(new_ids, BATCH_SIZE), batched(new_texts, BATCH_SIZE), batched(new_metas, BATCH_SIZE)):
        emb = oa.embeddings.create(model=EMBED_MODEL, input=text_b)
        vecs = [e.embedding for e in emb.data]
        if hasattr(col, "upsert"):
            col.upsert(ids=id_b, embeddings=vecs, documents=text_b, metadatas=meta_b)
        else:
            col.add(ids=id_b, embeddings=vecs, documents=text_b, metadatas=meta_b)
        print("added:", len(id_b))

    print("DONE. collection:", COLLECTION_NAME)
    print("count:", col.count())


if __name__ == "__main__":
    main()
