import csv
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import chromadb
from chromadb.config import Settings


# ===== 使い方 =====
# python okazaki_rag/tools/tourism_fixed_faq_audit.py --collection okazaki_events_fixed_faq --dbdir okazaki_rag/chroma_db --out out/fixed_faq_audit.csv


def eprint(*args):
    print(*args, file=sys.stderr)


def parse_args(argv: List[str]) -> Dict[str, str]:
    args = {"--collection": "okazaki_events_fixed_faq", "--dbdir": "", "--out": "out/fixed_faq_audit.csv", "--limit": "0"}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in args:
            if i + 1 >= len(argv):
                raise SystemExit(f"missing value for {a}")
            args[a] = argv[i +1]
            i = 2
        else:
            i = 1
    if not args["--dbdir"]:
        raise SystemExit("required: --dbdir <path> (ex: okazaki_rag/chroma_db)")
    return args


RE_BAD_TODAY = re.compile(r"(今日|本日|いま|今|現在|混雑|満開|開花|見頃|空いて|空き|臨時|休業|運休|中止|延期)")


def get_collection(dbdir: str, name: str):
    client = chromadb.PersistentClient(
        path=dbdir,
        settings=Settings(anonymized_telemetry=False),
    )
    col = client.get_collection(name)
    return col


def audit_one(doc: str, meta: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    m = meta or {}

    pdf_url = (m.get("pdf_url") or "").strip()
    page = (m.get("page") or "").strip()
    title = (m.get("title") or "").strip()

    if not title:
        issues.append("missing:title")
    if not pdf_url:
        issues.append("missing:pdf_url")
    if page and (not str(page).isdigit()):
        issues.append("bad:page_not_numeric")

    # 固定FAQに揺れる情報の混入を検出（強制排除対象）
    if RE_BAD_TODAY.search(doc or ""):
        issues.append("bad:volatile_keyword")

    # 文量が短すぎると retrieval が不安定になる
    if len((doc or "").strip()) < 25:
        issues.append("warn:doc_too_short")

    return issues


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    col_name = args["--collection"]
    dbdir = str(Path(args["--dbdir"]).resolve())
    out_path = Path(args["--out"]).resolve()
    limit = int(args["--limit"])

    col = get_collection(dbdir=dbdir, name=col_name)
    n = col.count()
    if limit > 0:
        n = min(n, limit)

    # Chroma は export API が弱いので get() で分割取得
    BATCH = 500
    rows: List[Tuple[str, str, Dict[str, Any], List[str]]] = []
    for offset in range(0, n, BATCH):
        got = col.get(
            include=["documents", "metadatas"],
            limit=min(BATCH, n - offset),
            offset=offset,
        )
        ids = got.get("ids", [])
        docs = got.get("documents", [])
        metas = got.get("metadatas", [])
        for _id, doc, meta in zip(ids, docs, metas):
            issues = audit_one(doc or "", meta or {})
            rows.append((_id, doc or "", meta or {}, issues))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "title", "pdf_url", "page", "issues", "doc_len"])
        for _id, doc, meta, issues in rows:
            m = meta or {}
            w.writerow([
                _id,
                (m.get("title") or ""),
                (m.get("pdf_url") or ""),
                (m.get("page") or ""),
                "|".join(issues),
                len((doc or "").strip()),
            ])

    bad = sum(1 for _, _, _, issues in rows if any(x.startswith("bad:") or x.startswith("missing:") for x in issues))
    warn = sum(1 for _, _, _, issues in rows if any(x.startswith("warn:") for x in issues))
    print("OK: audit done")
    print(f"collection={col_name}")
    print(f"dbdir={dbdir}")
    print(f"count={col.count()}")
    print(f"audited={len(rows)}")
    print(f"bad_or_missing={bad}")
    print(f"warn={warn}")
    print(f"out={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))