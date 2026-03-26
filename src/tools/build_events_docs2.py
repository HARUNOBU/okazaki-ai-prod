from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import chromadb
from chromadb.config import Settings
from openai import OpenAI

# src/config.py を読めるようにする
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, KANKO_DB_DIR, OPENAI_API_KEY

# =========================
# パス設定
# =========================
INPUT_CSV = DATA_DIR / "events_2026.csv"
OUTPUT_JSON = DATA_DIR / "events_docs_2026.json"

DATA_PATH = OUTPUT_JSON
CHROMA_DB_DIR = KANKO_DB_DIR

COLLECTION_NAME = "okazaki_events"
EMBED_MODEL = "text-embedding-3-small"


# =========================
# utility
# =========================
def s(v) -> str:
    if v is None:
        return ""
    return str(v).strip()


def pick_url(row: dict) -> str:
    return (
        s(row.get("URL"))
        or s(row.get("コンテンツURL"))
        or s(row.get("Web開催URL"))
    )


def make_text(row: dict) -> str:
    title = s(row.get("イベント名"))
    start_date = s(row.get("開始日"))
    end_date = s(row.get("終了日"))
    start_time = s(row.get("開始時間"))
    end_time = s(row.get("終了時間"))
    place = s(row.get("場所名称"))
    address = s(row.get("所在地_連結表記"))
    summary = s(row.get("概要"))
    desc = s(row.get("説明"))
    organizer = s(row.get("主催者"))
    contact_name = s(row.get("連絡先名称"))
    contact_tel = s(row.get("連絡先電話番号"))
    fee_basic = s(row.get("料金(基本)"))
    fee_detail = s(row.get("料金(詳細)"))
    url = pick_url(row)

    lines = []

    if title:
        lines.append(f"イベント名：{title}")

    if start_date and end_date:
        if start_date == end_date:
            lines.append(f"開催日：{start_date}")
        else:
            lines.append(f"開催日：{start_date}〜{end_date}")
    elif start_date:
        lines.append(f"開催日：{start_date}")

    if start_time and end_time:
        lines.append(f"時間：{start_time}〜{end_time}")
    elif start_time:
        lines.append(f"時間：{start_time}")

    if place:
        lines.append(f"場所：{place}")

    if address:
        lines.append(f"住所：{address}")

    if summary:
        lines.append(f"概要：{summary}")

    if desc and desc != summary:
        lines.append(f"説明：{desc}")

    if organizer:
        lines.append(f"主催者：{organizer}")

    if contact_name:
        lines.append(f"連絡先：{contact_name}")

    if contact_tel:
        lines.append(f"電話：{contact_tel}")

    if fee_basic:
        lines.append(f"料金：{fee_basic}")

    if fee_detail:
        lines.append(f"料金詳細：{fee_detail}")

    if url:
        lines.append(f"詳細URL：{url}")

    return "\n".join(lines).strip()


def row_to_doc(row: dict) -> dict | None:
    event_id = s(row.get("ID"))
    title = s(row.get("イベント名"))
    start_date = s(row.get("開始日"))
    end_date = s(row.get("終了日"))
    start_time = s(row.get("開始時間"))
    end_time = s(row.get("終了時間"))
    place_name = s(row.get("場所名称"))
    address = s(row.get("所在地_連結表記"))
    url = pick_url(row)
    summary = s(row.get("概要"))
    desc = s(row.get("説明"))

    if not title:
        return None

    text = make_text(row)
    if not text:
        return None

    year = None
    if start_date and len(start_date) >= 4 and start_date[:4].isdigit():
        year = int(start_date[:4])

    doc_id = event_id or f"event_{title}_{start_date}_{place_name}".replace(" ", "_")

    meta = {
        "category": "観光",
        "title": title,
        "start_date": start_date,
        "end_date": end_date,
        "start_time": start_time,
        "end_time": end_time,
        "place_name": place_name,
        "address": address,
        "url": url,
        "source": url,
        "summary": summary,
        "description": desc,
        "year": year,
    }

    meta = sanitize_meta({k: v for k, v in meta.items() if v not in (None, "")})

    return {
        "id": doc_id,
        "text": text,
        "meta": meta,
    }


# =========================
# metadata安全化
# =========================
def sanitize_meta(meta: dict):
    out = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = json.dumps(v, ensure_ascii=False)
    return out


# =========================
# docs 生成
# =========================
def build_docs_json():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"input csv not found: {INPUT_CSV}")

    docs = []

    with INPUT_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc = row_to_doc(row)
            if doc is not None:
                docs.append(doc)

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(docs, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("done")
    print("input :", INPUT_CSV)
    print("docs  :", len(docs))
    print("output:", OUTPUT_JSON)

    if docs:
        print("sample id   :", docs[0]["id"])
        print("sample title:", docs[0]["meta"].get("title", ""))


# =========================
# chroma upsert
# =========================
def upsert_chroma():
    api_key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found")

    oa = OpenAI(api_key=api_key)

    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    docs = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    print("docs:", len(docs))

    ids = []
    texts = []
    metas = []

    for d in docs:
        ids.append(d["id"])
        texts.append(d["text"])
        metas.append(sanitize_meta(d.get("meta", {})))

    if not texts:
        print("no docs to upsert")
        return

    emb = oa.embeddings.create(
        model=EMBED_MODEL,
        input=texts
    )

    vectors = [e.embedding for e in emb.data]

    client = chromadb.PersistentClient(
        path=str(CHROMA_DB_DIR),
        settings=Settings(anonymized_telemetry=False)
    )

    col = client.get_or_create_collection(COLLECTION_NAME)

    col.upsert(
        ids=ids,
        documents=texts,
        metadatas=metas,
        embeddings=vectors
    )

    print("collection:", COLLECTION_NAME)
    print("count:", col.count())


def main():
    build_docs_json()
    upsert_chroma()


if __name__ == "__main__":
    main()