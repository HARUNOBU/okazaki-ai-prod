# tools/beta_log_judge.py
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def norm_text(s: Any) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\u3000", " ")   # 全角空白→半角
    s = s.strip()
    s = re.sub(r"[ \t]+", " ", s)  # 連続空白圧縮
    return s


def read_csv_rows(csv_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            row2 = dict(row)
            row2["_rowno"] = i
            row2["_query_norm"] = norm_text(row.get("query", ""))
            rows.append(row2)
    return rows


def read_jsonl(log_path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                obj["_lineno"] = lineno
                obj["_query_norm"] = norm_text(obj.get("query", ""))
                out.append(obj)
            except Exception as e:
                out.append({
                    "_lineno": lineno,
                    "_json_error": str(e),
                    "_raw": line,
                    "_query_norm": "",
                })
    return out


def build_query_index(logs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    idx: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for log in logs:
        q = log.get("_query_norm", "")
        if q:
            idx[q].append(log)
    return idx


def infer_expected_mode_from_suite(suite: str) -> str:
    """
    beta_fixed    -> ok
    beta_general  -> ok
    beta_volatile -> abstain
    """
    s = norm_text(suite).lower()
    if s == "beta_volatile":
        return "abstain"
    return "ok"


def pick_best_log_for_case(
    candidate_logs: List[Dict[str, Any]],
    suite: str,
    used_request_ids: set[str],
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    優先順位:
    1. query一致 かつ suite一致 かつ 未使用
    2. query一致 かつ suite一致
    3. query一致 かつ 未使用
    4. query一致のみ
    """

    target_suite = norm_text(suite)

    same_suite_unused = [
        x for x in candidate_logs
        if norm_text(x.get("suite", "")) == target_suite
        and str(x.get("request_id", "")) not in used_request_ids
    ]
    if same_suite_unused:
        return same_suite_unused[-1], None

    same_suite_any = [
        x for x in candidate_logs
        if norm_text(x.get("suite", "")) == target_suite
    ]
    if same_suite_any:
        return same_suite_any[-1], "duplicate_log_used"

    any_unused = [
        x for x in candidate_logs
        if str(x.get("request_id", "")) not in used_request_ids
    ]
    if any_unused:
        return any_unused[-1], "suite_mismatch"

    if candidate_logs:
        return candidate_logs[-1], "duplicate_log_used"

    return None, "query_not_found"


def judge_one_case(
    row: Dict[str, Any],
    query_index: Dict[str, List[Dict[str, Any]]],
    suite: str,
    used_request_ids: set[str],
) -> Dict[str, Any]:
    query = row.get("query", "")
    query_norm = row.get("_query_norm", "")
    rowno = row.get("_rowno", "")

    result: Dict[str, Any] = {
        "rowno": rowno,
        "query": query,
        "suite_expected": suite,
        "expected_mode": infer_expected_mode_from_suite(suite),
        "judge": "fail",
        "fail_reason": "",
        "matched_suite": "",
        "matched_status": "",
        "matched_answered_by": "",
        "request_id": "",
        "log_lineno": "",
    }

    if not query_norm:
        result["fail_reason"] = "empty_query"
        return result

    candidate_logs = query_index.get(query_norm, [])
    if not candidate_logs:
        result["fail_reason"] = "missing_log"
        return result

    matched_log, pre_reason = pick_best_log_for_case(candidate_logs, suite, used_request_ids)
    if not matched_log:
        result["fail_reason"] = pre_reason or "query_not_found"
        return result

    request_id = str(matched_log.get("request_id", ""))
    if request_id:
        used_request_ids.add(request_id)

    result["matched_suite"] = str(matched_log.get("suite", ""))
    result["matched_status"] = str(matched_log.get("status", ""))
    result["matched_answered_by"] = str(matched_log.get("answered_by", ""))
    result["request_id"] = request_id
    result["log_lineno"] = matched_log.get("_lineno", "")

    expected_mode = result["expected_mode"]
    actual_status = norm_text(matched_log.get("status", "")).lower()

    # まず suite不一致を優先的に見せる
    if pre_reason == "suite_mismatch":
        result["fail_reason"] = "suite_mismatch"
        return result

    if expected_mode == "ok":
        if actual_status == "ok":
            result["judge"] = "ok"
            result["fail_reason"] = ""
            return result
        else:
            result["fail_reason"] = "status_not_ok"
            return result

    if expected_mode == "abstain":
        if actual_status == "abstain":
            result["judge"] = "ok"
            result["fail_reason"] = ""
            return result
        else:
            result["fail_reason"] = "status_not_abstain"
            return result

    result["fail_reason"] = "unexpected_error"
    return result


def write_detail_csv(out_path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "rowno",
        "judge",
        "fail_reason",
        "suite_expected",
        "expected_mode",
        "matched_suite",
        "matched_status",
        "matched_answered_by",
        "request_id",
        "log_lineno",
        "query",
    ]
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def print_summary(
    suite: str,
    csv_rows: List[Dict[str, Any]],
    judged_rows: List[Dict[str, Any]],
    logs: List[Dict[str, Any]],
) -> None:
    total = len(csv_rows)
    ok_count = sum(1 for r in judged_rows if r["judge"] == "ok")
    fail_count = total - ok_count

    suite_norm = norm_text(suite)
    suite_logs = [x for x in logs if norm_text(x.get("suite", "")) == suite_norm]
    suite_match_count = len(suite_logs)

    missing_log_count = sum(1 for r in judged_rows if r["fail_reason"] == "missing_log")
    fail_counter = Counter(r["fail_reason"] for r in judged_rows if r["fail_reason"])

    print("=== beta judge summary ===")
    print(f"suite            : {suite}")
    print(f"csv_total         : {total}")
    print(f"log_total         : {len(logs)}")
    print(f"suite_match_count : {suite_match_count}")
    print(f"ok_count          : {ok_count}")
    print(f"fail_count        : {fail_count}")
    print(f"missing_log_count : {missing_log_count}")
    print("fail_reason_count :")
    if fail_counter:
        for k, v in fail_counter.most_common():
            print(f"  - {k}: {v}")
    else:
        print("  - none")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="qa_log.jsonl path")
    parser.add_argument("--csv", required=True, help="beta target csv path")
    parser.add_argument("--suite", required=True, help="beta_fixed / beta_volatile / beta_general")
    parser.add_argument("--out", default="", help="detail result csv path")
    args = parser.parse_args()

    log_path = Path(args.log)
    csv_path = Path(args.csv)
    suite = args.suite
    out_path = Path(args.out) if args.out else csv_path.with_name(f"{csv_path.stem}_judge_result.csv")

    if not log_path.exists():
        print(f"[ERROR] log not found: {log_path}")
        return 2

    if not csv_path.exists():
        print(f"[ERROR] csv not found: {csv_path}")
        return 2

    csv_rows = read_csv_rows(csv_path)
    logs = read_jsonl(log_path)
    query_index = build_query_index(logs)

    judged_rows: List[Dict[str, Any]] = []
    used_request_ids: set[str] = set()

    for row in csv_rows:
        judged = judge_one_case(
            row=row,
            query_index=query_index,
            suite=suite,
            used_request_ids=used_request_ids,
        )
        judged_rows.append(judged)

    write_detail_csv(out_path, judged_rows)
    print_summary(
        suite=suite,
        csv_rows=csv_rows,
        judged_rows=judged_rows,
        logs=logs,
    )
    print(f"detail_csv        : {out_path}")

    total = len(judged_rows)
    ok_count = sum(1 for r in judged_rows if r["judge"] == "ok")

    # 全件OKなら 0、それ以外は 1
    return 0 if total > 0 and ok_count == total else 1


if __name__ == "__main__":
    sys.exit(main())