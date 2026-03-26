import csv
import json
import hashlib
from datetime import datetime
from pathlib import Path
import sys



# src/config.py を読めるようにする
sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import DATA_DIR

INPUT_CSV = DATA_DIR / "events_2026.csv"
OUTPUT_JSON = DATA_DIR / "events_docs_2026.json"

CITY_CODE = "okazaki"
DATA_TYPE = "event"


def norm_date(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    s = s.replace("/", "-")
    return s


def make_stable_id(source: str, title: str, start_date: str, place_name: str) -> str:
    base = f"{source}|{title}|{start_date}|{place_name}".encode("utf-8")
    h = hashlib.sha1(base).hexdigest()
    return f"{CITY_CODE}_evt_{h}"


def build_text(row: dict) -> str:
    title = row.get("title", "").strip()
    start_date = row.get("start_date", "").strip()
    end_date = row.get("end_date", "").strip()
    place = row.get("place_name", "").strip()
    desc = row.get("description", "").strip()
    apply_url = row.get("apply_url", "").strip()

    lines = []
    if title:
        lines.append(f"イベント名：{title}")
    if start_date or end_date:
        if start_date and end_date:
            lines.append(f"開催日：{start_date}〜{end_date}")
        elif start_date:
            lines.append(f"開催日：{start_date}")
        else:
            lines.append(f"開催日：{end_date}")
    if place:
        lines.append(f"場所：{place}")
    if desc:
        lines.append(f"概要：{desc}")
    if apply_url:
        lines.append(f"詳細：{apply_url}")
    return "\n".join(lines).strip()


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"入力CSVが見つかりません: {INPUT_CSV}")

    docs = []

    with INPUT_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            title = (r.get("title") or "").strip()
            start_date = norm_date(r.get("start_date") or "")
            end_date = norm_date(r.get("end_date") or "")
            place_name = (r.get("place_name") or "").strip()
            apply_url = (r.get("apply_url") or "").strip()
            source = (r.get("source") or apply_url or "").strip()

            if not title or not start_date:
                continue

            try:
                year = int(start_date[:4])
            except Exception:
                year = None

            doc_id = make_stable_id(source, title, start_date, place_name)
            text = build_text({
                "title": title,
                "start_date": start_date,
                "end_date": end_date,
                "place_name": place_name,
                "description": (r.get("text") or r.get("description") or "").strip(),
                "apply_url": apply_url,
            })

            meta = {
                "title": title,
                "start_date": start_date,
                "end_date": end_date,
                "place_name": place_name,
                "apply_url": apply_url,
                "source": source,
                "year": year,
                "city": CITY_CODE,
                "data_type": DATA_TYPE,
                "updated_at": datetime.now().strftime("%Y-%m-%d"),
            }

            docs.append({
                "id": doc_id,
                "text": text,
                "meta": meta,
            })

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(docs, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"OK: {len(docs)} docs -> {OUTPUT_JSON}")


if __name__ == "__main__":
    main()