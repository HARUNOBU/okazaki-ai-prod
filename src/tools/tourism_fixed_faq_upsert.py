import csv
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import chromadb
from chromadb.config import Settings
from openai import OpenAI


# ====== 使い方 ======
# python okazaki_rag/tools/tourism_fixed_faq_upsert.py --csv in/tourism_fixed_faq.csv --collection okazaki_events_fixed_faq --dbdir okazaki_rag/chroma_db
#
# CSV列（必須）:
#   id, question, answer, title, pdf_url, page
# 推奨列（任意）:
#   source, tags
#
# ルール:
# - id は一意（upsertキー）
# - pdf_url は一次ソースURL（市/協会PDF等）
# - page は数字（不明なら空欄OK）


EMBED_MODEL = "text-embedding-3-small"


def eprint(*args):
    print(*args, file=sys.stderr)


def parse_args(argv: List[str]) -> Dict[str, str]:
    args = {"--csv": "", "--collection": "okazaki_events_fixed_faq", "--dbdir": "", "--dry-run": "0"}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in args:
            if i + 1 >= len(argv):
                raise SystemExit(f"missing value for {a}")
            args[a] = argv[i + 1]
            i += 2
        else:
            i += 1
    if not args["--csv"]:
        raise SystemExit("required: --csv <path>")
    if not args["--dbdir"]:
        raise SystemExit("required: --dbdir <path> (ex: okazaki_rag/chroma_db)")
    return args


@dataclass
class Row:
    id: str
    question: str
    answer: str
    title: str
    pdf_url: str
    page: str
    source: str = ""
    tags: str = ""

    def to_text(self) -> str:
        # retrieval 用：Q/A を連結（短文でも情報密度を確保）
        q = self.question.strip()
        a = self.answer.strip()
        return f"Q: {q}\nA: {a}"

    def meta(self) -> Dict[str, Any]:
        m: Dict[str, Any] = {
            "title": (self.title or "").strip(),
            "pdf_url": (self.pdf_url or "").strip(),
            "page": (self.page or "").strip(),
            "source": (self.source or "").strip(),
            "tags": (self.tags or "").strip(),
            "kind": "fixed_faq",
        }
        return m


RE_BAD_TODAY = re.compile(r"(今日|本日|いま|今|現在|混雑|満開|開花|見頃|空いて|空き|臨時|休業|運休|中止|延期)")


def validate_row(r: Row) -> List[str]:
    errs: List[str] = []
    if not r.id.strip():
        errs.append("id is empty")
    if not r.question.strip():
        errs.append("question is empty")
    if not r.answer.strip():
        errs.append("answer is empty")
    if not r.pdf_url.strip():
        errs.append("pdf_url is empty (primary source required)")
    # 固定FAQに「当日変動」語が入っていたら事故るので強制エラー
    if RE_BAD_TODAY.search(r.question) or RE_BAD_TODAY.search(r.answer):
        errs.append("contains 'today/volatile' keyword (NOT allowed in fixed_faq)")
    # pageは任意だが、入ってるなら数字で
    p = r.page.strip()
    if p and (not p.isdigit()):
        errs.append("page is not numeric")
    return errs


def load_csv(path: Path) -> List[Row]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = ["id", "question", "answer", "title", "pdf_url", "page"]
        for k in required:
            if k not in (reader.fieldnames or []):
                raise SystemExit(f"CSV missing required column: {k}")
        rows: List[Row] = []
        for i, d in enumerate(reader, start=2):
            rr = Row(
                id=(d.get("id") or "").strip(),
                question=(d.get("question") or "").strip(),
                answer=(d.get("answer") or "").strip(),
                title=(d.get("title") or "").strip(),
                pdf_url=(d.get("pdf_url") or "").strip(),
                page=(d.get("page") or "").strip(),
                source=(d.get("source") or "").strip(),
                tags=(d.get("tags") or "").strip(),
            )
            rows.append(rr)
    return rows


def get_collection(dbdir: str, name: str):
    client = chromadb.PersistentClient(
        path=dbdir,
        settings=Settings(anonymized_telemetry=False),
    )
    # get_or_create で運用簡略化（存在していればそのまま）
    col = client.get_or_create_collection(name)
    return client, col


def embed_texts(oa: OpenAI, texts: List[str]) -> List[List[float]]:
    # OpenAI Embeddings: バッチで投げる
    resp = oa.embeddings.create(model=EMBED_MODEL, input=texts)
    return [x.embedding for x in resp.data]


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    csv_path = Path(args["--csv"]).resolve()
    col_name = args["--collection"]
    dbdir = str(Path(args["--dbdir"]).resolve())
    dry_run = args["--dry-run"] == "1"

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        eprint("ERROR: OPENAI_API_KEY is not set (.env / env)")
        return 2
    oa = OpenAI(api_key=api_key)

    rows = load_csv(csv_path)
    if not rows:
        eprint("ERROR: CSV has no rows")
        return 2

    # 1) validate
    seen = set()
    bad: List[Tuple[str, List[str]]] = []
    for r in rows:
        if r.id in seen:
            bad.append((r.id, ["duplicate id"]))
        seen.add(r.id)
        errs = validate_row(r)
        if errs:
            bad.append((r.id or "(empty)", errs))

    if bad:
        eprint("VALIDATION FAILED. Fix CSV then rerun.")
        for rid, errs in bad[:50]:
            eprint(f"- id={rid}: {', '.join(errs)}")
        eprint(f"bad_rows={len(bad)} / total={len(rows)}")
        return 3

    # 2) build docs  metas
    ids = [r.id for r in rows]
    docs = [r.to_text() for r in rows]
    metas = [r.meta() for r in rows]

    # 3) embed
    embs = embed_texts(oa, docs)

    # 4) upsert
    client, col = get_collection(dbdir=dbdir, name=col_name)
    before = col.count()
    if dry_run:
        print(f"DRY_RUN=1: would upsert rows={len(ids)} into collection={col_name}")
        print(f"dbdir={dbdir}")
        print(f"count_before={before}")
        return 0

    col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
    after = col.count()

    print("OK: upsert done")
    print(f"collection={col_name}")
    print(f"dbdir={dbdir}")
    print(f"rows={len(ids)}")
    print(f"count_before={before}")
    print(f"count_after={after}")

    # 5) quick spot-check: query 3 samples
    sample_q = [
        rows[0].question,
        rows[min(1, len(rows)-1)].question,
        rows[min(2, len(rows)-1)].question,
    ]
    for q in sample_q:
        q_emb = embed_texts(oa, [q])[0]
        res = col.query(query_embeddings=[q_emb], n_results=3, include=["distances", "metadatas"])
        d0 = res.get("distances", [[]])[0]
        m0 = res.get("metadatas", [[]])[0]
        best = d0[0] if d0 else None
        best_title = (m0[0] or {}).get("title") if m0 else None
        print(f"spotcheck: q={q}")
        print(f"  best_dist={best} best_title={best_title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))