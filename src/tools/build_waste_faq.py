from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List
import os

import chromadb
from chromadb.config import Settings
from openai import OpenAI

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import WASTE_LIFE_DB_DIR, OPENAI_API_KEY, DATA_DIR

EMBED_MODEL = "text-embedding-3-small"
COL_NAME = "okazaki_waste_faq"


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
    for k, v in (meta or {}).items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = json.dumps(v, ensure_ascii=False)
    return out


def make_doc(r: Dict[str, Any]) -> str:
    q = str(r.get("question") or "").strip()
    a = str(r.get("answer") or "").strip()
    aliases = [str(x).strip() for x in (r.get("aliases") or []) if str(x).strip()]
    alias_str = " / ".join(aliases)
    pdf_url = str(r.get("pdf_url") or "").strip()
    page = str(r.get("page") or "").strip()
    return (
        f"【Q】{q}\n"
        f"【言い換え】{alias_str}\n"
        f"【A要旨】{a}\n"
        f"【出典】{pdf_url} p.{page}\n"
    ).strip()


def make_embed_text(r: Dict[str, Any]) -> str:
    q = str(r.get("question") or "").strip()
    a = str(r.get("answer") or "").strip()
    aliases = [str(x).strip() for x in (r.get("aliases") or []) if str(x).strip()]
    return f"Q: {q}\nA: {a}\nALIASES: {' / '.join(aliases)}"


def make_unique_ids(rows: List[Dict[str, Any]]) -> List[str]:
    used: Dict[str, int] = {}
    out: List[str] = []
    for i, r in enumerate(rows, start=1):
        base = str(r.get("qid") or r.get("id") or "").strip() or f"faq_auto_{i:04d}"
        used[base] = used.get(base, 0) + 1
        out.append(base if used[base] == 1 else f"{base}__{used[base]:02d}")
    return out


def resolve_input_path() -> Path:
    if len(sys.argv) >= 2 and not sys.argv[1].startswith("--"):
        return Path(sys.argv[1])

    candidates = [
        DATA_DIR / "waste" / "faq_waste.jsonl",
        DATA_DIR / "faq_waste.jsonl",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("faq_waste.jsonl not found")


def main():
    in_path = resolve_input_path()
    rows = load_jsonl(in_path)

    api_key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found")

    oa = OpenAI(api_key=api_key)

    client = chromadb.PersistentClient(
        path=str(WASTE_LIFE_DB_DIR),
        settings=Settings(anonymized_telemetry=False)
    )
    col = client.get_or_create_collection(COL_NAME)

    ids = make_unique_ids(rows)
    docs = [make_doc(r) for r in rows]
    metas = [
        sanitize_meta({
            "source": "waste_faq",
            "qid": r.get("qid") or r.get("id"),
            "question": r.get("question", ""),
            "pdf_url": r.get("pdf_url", ""),
            "page": r.get("page", ""),
            "aliases": r.get("aliases", []),
            "category": r.get("category", "ごみ"),
        })
        for r in rows
    ]
    texts_for_embed = [make_embed_text(r) for r in rows]

    if not docs:
        print("no faq rows")
        return

    BATCH = 64
    for id_b, dcs, mts, emb_src in zip(chunks(ids, BATCH), chunks(docs, BATCH), chunks(metas, BATCH), chunks(texts_for_embed, BATCH)):
        embs = oa.embeddings.create(model=EMBED_MODEL, input=emb_src).data
        emb_vectors = [e.embedding for e in embs]
        col.upsert(ids=id_b, documents=dcs, metadatas=mts, embeddings=emb_vectors)

    print("OK")
    print("input:", in_path)
    print("collection:", COL_NAME)
    print("count:", col.count())


if __name__ == "__main__":
    main()
