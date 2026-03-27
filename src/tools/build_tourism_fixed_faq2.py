# tools/build_tourism_fixed_faq.py
from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

import chromadb
from chromadb.config import Settings
from openai import OpenAI


# =========================
# パス解決（app_portal_v3.py と同じ思想）
# =========================
BASE_WASTE = Path(__file__).resolve().parents[1]          # .../okazaki_waste_rag/src
PORTAL_ROOT = BASE_WASTE.parents[0]                      # .../okazaki-ai-portal
KANKO_DB_DIR = str(PORTAL_ROOT / "chroma_db")

# =========================
# 設定
# =========================
EMBED_MODEL = "text-embedding-3-small"
DEFAULT_COLLECTION = "okazaki_events_fixed_faq"  # ★固定情報専用（新設）


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return rows


def normalize_item(x: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    """
    returns: (id, text, meta)
    - id: str 必須
    - text: str 必須
    - meta: dict（url/source を保持）
    """
    _id = str(x.get("id") or "").strip()
    if not _id:
        raise ValueError("jsonl item missing id")

    title = (x.get("title") or "").strip()
    text = (x.get("text") or "").strip()
    if not text:
        raise ValueError(f"jsonl item missing text: id={_id}")

    meta = dict(x.get("meta") or {})
    # よく使うキーを整形
    if "url" not in meta and "pdf_url" in meta:
        meta["url"] = meta["pdf_url"]
    meta["title"] = title

    # Chromaのメタは基本primitive推奨
    for k, v in list(meta.items()):
        if isinstance(v, (dict, list)):
            meta[k] = json.dumps(v, ensure_ascii=False)

    return _id, text, meta


def embed_texts(oa: OpenAI, texts: List[str], batch_size: int = 64, sleep_sec: float = 0.2) -> List[List[float]]:
    """
    OpenAI embeddings をバッチで取得
    """
    out: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        resp = oa.embeddings.create(model=EMBED_MODEL, input=chunk)
        # resp.data は入力順で返る
        out.extend([d.embedding for d in resp.data])
        time.sleep(sleep_sec)
    return out


def get_client(db_dir: str) -> chromadb.PersistentClient:
    return chromadb.PersistentClient(
        path=db_dir,
        settings=Settings(anonymized_telemetry=False)
    )


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="tourism_fixed_faq.jsonl のパス")
    ap.add_argument("--collection", default=DEFAULT_COLLECTION, help="作成/更新するコレクション名")
    ap.add_argument("--db", default=KANKO_DB_DIR, help="ChromaDBディレクトリ（観光DB）")
    ap.add_argument("--reset", action="store_true", help="コレクションを削除して作り直す（全入れ替え）")
    ap.add_argument("--dry-run", action="store_true", help="DB更新せず検査だけ")
    args = ap.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY が環境変数または .env にありません。")

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        raise FileNotFoundError(jsonl_path)

    rows = read_jsonl(jsonl_path)

    ids: List[str] = []
    texts: List[str] = []
    metas: List[Dict[str, Any]] = []

    for x in rows:
        _id, text, meta = normalize_item(x)
        ids.append(_id)
        texts.append(text)
        metas.append(meta)

    # 事前検査：重複ID
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise ValueError(f"Duplicate ids found: {sorted(list(dup))[:10]} ... total={len(dup)}")

    print(f"[OK] jsonl loaded: {len(ids)} items")
    print(f"      db_dir: {args.db}")
    print(f"      collection: {args.collection}")

    if args.dry_run:
        print("[DRY-RUN] skip embeddings / skip upsert")
        return

    oa = OpenAI(api_key=api_key)
    embs = embed_texts(oa, texts)

    client = get_client(args.db)

    if args.reset:
        try:
            client.delete_collection(args.collection)
            print("[INFO] deleted existing collection:", args.collection)
        except Exception:
            pass

    col = client.get_or_create_collection(args.collection)

    # upsert（上書き可能）
    col.upsert(
        ids=ids,
        documents=texts,
        metadatas=metas,
        embeddings=embs
    )

    print("[DONE] upserted:", len(ids))
    print("[DONE] collection_count:", col.count())


if __name__ == "__main__":
    main()