import csv
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional


"""
使い方（例）:
python okazaki_rag/tools/beta_log_judge.py ^
  --log logs/qa_log.jsonl ^
  --fixed okazaki_rag/tests/beta_fixed.csv ^
  --volatile okazaki_rag/tests/beta_volatile.csv ^
  --general okazaki_rag/tests/beta_general.csv ^
  --allow domains_allowlist.txt ^
  --out okazaki_rag/out/beta_judge_report.csv

判定ロジック（β条件を機械判定）:
1) fixed:   status=ok かつ answered_by=faq_fixed
2) volatile: status=abstain
3) general: status=ok（abstain禁止）
4) 引用一次ソース: log_obj["top_sources"][].url のドメインが allowlist のみ

※ top_sources は portal 側で log に保存されています :contentReference[oaicite:2]{index=2}
"""


def eprint(*args):
    print(*args, file=sys.stderr)


def parse_args(argv: List[str]) -> Dict[str, str]:
    args = {
        "--log": "",
        "--fixed": "",
        "--volatile": "",
        "--general": "",
        "--allow": "",
        "--out": "okazaki_rag/out/beta_judge_report.csv",
        "--require_sources": "1",   # 1=generalはtop_sources必須
    }
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
    if not args["--log"]:
        raise SystemExit("required: --log <path>")
    return args


def load_allowlist(path: str) -> List[str]:
    if not path:
        # 最低限：岡崎市公式（必要ならファイルで追加）
        return ["city.okazaki.lg.jp"]
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"allowlist file not found: {path}")
    out: List[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def domain_of_url(url: str) -> str:
    s = (url or "").strip()
    if not s:
        return ""
    s = s.replace("http://", "").replace("https://", "")
    # / 以降は削る
    if "/" in s:
        s = s.split("/", 1)[0]
    return s.lower()


def is_allowed(url: str, allow_domains: List[str]) -> bool:
    d = domain_of_url(url)
    if not d:
        return False
    for ad in allow_domains:
        ad = ad.lower().strip()
        if not ad:
            continue
        # サブドメインも許可（*.example.com）
        if d == ad or d.endswith("."  +ad):
            return True
    return False


def load_suite_csv(path: str) -> List[Dict[str, str]]:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"suite csv not found: {path}")
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        req = ["id", "suite", "query", "expect_status", "expect_answered_by"]
        for k in req:
            if k not in (r.fieldnames or []):
                raise SystemExit(f"CSV missing column {k}: {path}")
        rows = []
        for row in r:
            rows.append({k: (row.get(k) or "").strip() for k in (r.fieldnames or [])})
        return rows


def load_logs(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"log not found: {path}")
    out: List[Dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def index_latest_by_suite_query(logs: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    (suite, query) の最新ログを使う（同一クエリを複数回投げたときに最後を採用）
    """
    idx: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for rec in logs:
        suite = (rec.get("suite") or "").strip()
        q = (rec.get("query") or "").strip()
        if not suite or not q:
            continue
        idx[(suite, q)] = rec
    return idx


def judge_case(
    case: Dict[str, str],
    rec: Optional[Dict[str, Any]],
    allow_domains: List[str],
    require_sources: bool,
) -> Tuple[bool, str]:
    """
    Returns: (pass, reason)
    """
    if rec is None:
        return False, "missing_log"

    exp_status = case["expect_status"]
    exp_by = case["expect_answered_by"]

    status = (rec.get("status") or "").strip()
    by = (rec.get("answered_by") or "").strip()

    if exp_status and status != exp_status:
        return False, f"status_mismatch got={status} exp={exp_status}"

    if exp_by and by != exp_by:
        return False, f"answered_by_mismatch got={by} exp={exp_by}"

    # 一次ソース縛り：top_sources の url が allowlist のみ
    top_sources = rec.get("top_sources") or []
    if require_sources and exp_status == "ok":
        if not top_sources:
            return False, "missing_top_sources"

    # top_sources があるなら全部チェック
    for s in top_sources:
        url = (s or {}).get("url") or ""
        if url and (not is_allowed(url, allow_domains)):
            return False, f"non_primary_source url={url}"

    return True, "ok"


def run_suite(
    suite_rows: List[Dict[str, str]],
    idx: Dict[Tuple[str, str], Dict[str, Any]],
    allow_domains: List[str],
    require_sources: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    passed = 0
    for case in suite_rows:
        suite = case["suite"]
        q = case["query"]
        rec = idx.get((suite, q))
        ok, reason = judge_case(case, rec, allow_domains, require_sources=require_sources)
        if ok:
            passed = 1
        results.append({
            "id": case["id"],
            "suite": suite,
            "query": q,
            "expect_status": case["expect_status"],
            "expect_answered_by": case["expect_answered_by"],
            "got_status": (rec.get("status") if rec else ""),
            "got_answered_by": (rec.get("answered_by") if rec else ""),
            "got_category": (rec.get("category") if rec else ""),
            "got_collection": (rec.get("collection_used") if rec else ""),
            "pass": "1" if ok else "0",
            "fail_reason": "" if ok else reason,
            "top_sources": json.dumps((rec.get("top_sources") if rec else []), ensure_ascii=False),
        })
    summary = {
        "total": len(suite_rows),
        "passed": passed,
        "pass_rate": (passed / len(suite_rows) if suite_rows else 0.0),
    }
    return results, summary


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    log_path = args["--log"]
    fixed_csv = args["--fixed"]
    volatile_csv = args["--volatile"]
    general_csv = args["--general"]
    out_path = Path(args["--out"]).resolve()
    allow_domains = load_allowlist(args["--allow"])
    require_sources = args["--require_sources"] == "1"

    logs = load_logs(log_path)
    idx = index_latest_by_suite_query(logs)

    # suite CSV は必須ではない（指定されたものだけ判定）
    all_results: List[Dict[str, Any]] = []
    summaries: Dict[str, Any] = {}

    if fixed_csv:
        rows = load_suite_csv(fixed_csv)
        res, sm = run_suite(rows, idx, allow_domains, require_sources=require_sources)
        all_results.extend(res)
        summaries["fixed"] = sm

    if volatile_csv:
        rows = load_suite_csv(volatile_csv)
        res, sm = run_suite(rows, idx, allow_domains, require_sources=False)  # abstain は sources不要
        all_results.extend(res)
        summaries["volatile"] = sm

    if general_csv:
        rows = load_suite_csv(general_csv)
        res, sm = run_suite(rows, idx, allow_domains, require_sources=require_sources)
        all_results.extend(res)
        summaries["general"] = sm

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "id","suite","query",
                "expect_status","expect_answered_by",
                "got_status","got_answered_by","got_category","got_collection",
                "pass","fail_reason","top_sources",
            ],
        )
        w.writeheader()
        for r in all_results:
            w.writerow(r)

    # β合格判定（機械判定）
    ok_fixed = (summaries.get("fixed", {}).get("passed") == summaries.get("fixed", {}).get("total")) if "fixed" in summaries else False
    ok_volatile = (summaries.get("volatile", {}).get("passed") == summaries.get("volatile", {}).get("total")) if "volatile" in summaries else False
    ok_general = (summaries.get("general", {}).get("passed") == summaries.get("general", {}).get("total")) if "general" in summaries else False

    print("=== beta judge summary ===")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f"report_csv={out_path}")
    print(f"allow_domains={allow_domains}")
    print("=== beta exit criteria (machine) ===")
    print(f"fixed_all_pass={ok_fixed}")
    print(f"volatile_all_pass={ok_volatile}")
    print(f"general_all_pass={ok_general}")
    overall = ok_fixed and ok_volatile and ok_general
    print(f"BETA_MACHINE_PASS={overall}")

    # 終了コード：合格=0 / 不合格=10
    return 0 if overall else 10


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))