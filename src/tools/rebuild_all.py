from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "src"))
from config import WASTE_LIFE_DB_DIR, LOG_DIR, DATA_DIR


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src" / "tools"
CHROMA_DB = Path(WASTE_LIFE_DB_DIR)

LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"rebuild_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_cmd(cmd: list[str], cwd: Path, step_name: str, required: bool = True, env: dict | None = None) -> bool:
    log("=" * 70)
    log(f"START: {step_name}")
    log(f"CWD  : {cwd}")
    log(f"CMD  : {' '.join(cmd)}")

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            print(line)
            with LOG_FILE.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

        rc = proc.wait()
        if rc != 0:
            log(f"ERROR: {step_name} failed with exit code {rc}")
            if required:
                return False
            log(f"SKIP-CONTINUE: {step_name} failed but continue because required=False")
            return True

        log(f"DONE : {step_name}")
        return True

    except FileNotFoundError as e:
        log(f"ERROR: command not found in {step_name}: {e}")
        return not required
    except Exception as e:
        log(f"ERROR: unexpected error in {step_name}: {e}")
        return not required


def require_file(path: Path, label: str) -> bool:
    if path.exists():
        log(f"FOUND: {label} -> {path}")
        return True
    log(f"MISSING: {label} -> {path}")
    return False


def safe_rmtree(path: Path, label: str) -> None:
    if path.exists():
        log(f"DELETE: {label} -> {path}")
        shutil.rmtree(path)
    else:
        log(f"SKIP DELETE: {label} not found -> {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="岡崎市AI総合案内: 単一 chroma_db 構成用 全コレクション再構築スクリプト")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--reset-db", action="store_true")
    parser.add_argument("--skip-waste", action="store_true")
    parser.add_argument("--skip-life", action="store_true")
    parser.add_argument("--skip-waste-faq", action="store_true")
    parser.add_argument("--skip-events", action="store_true")
    parser.add_argument("--skip-fixed-faq", action="store_true")
    args = parser.parse_args()

    py = args.python
    env = dict(**__import__("os").environ)

    log("REBUILD ALL START")
    log(f"PROJECT_ROOT = {PROJECT_ROOT}")
    log(f"SRC_DIR      = {SRC_DIR}")
    log(f"CHROMA_DB    = {CHROMA_DB}")
    log(f"LOG_FILE     = {LOG_FILE}")
    log(f"PYTHON       = {py}")

    required_paths = [
        (SRC_DIR / "upsert_chroma_waste.py", "src/tools/upsert_chroma_waste.py"),
        (SRC_DIR / "upsert_chroma_life.py", "src/tools/upsert_chroma_life.py"),
        (SRC_DIR / "build_waste_faq.py", "src/tools/build_waste_faq.py"),
        (SRC_DIR / "build_events_docs2.py", "src/tools/build_events_docs2.py"),
        (SRC_DIR / "upsert_chroma_events.py", "src/tools/upsert_chroma_events.py"),
        (SRC_DIR / "build_events_faq.py", "src/tools/build_events_faq.py"),
        (SRC_DIR / "build_tourism_fixed_faq2.py", "src/tools/build_tourism_fixed_faq2.py"),
    ]

    ok = True
    for path, label in required_paths:
        if not require_file(path, label):
            ok = False
    if not ok:
        log("ABORT: 必須ファイル不足")
        return 1

    if args.reset_db:
        log("OPTION: --reset-db enabled")
        safe_rmtree(CHROMA_DB, "single chroma_db")
    CHROMA_DB.mkdir(parents=True, exist_ok=True)

    if not args.skip_waste:
        if not run_cmd([py, "upsert_chroma_waste.py"], cwd=SRC_DIR, step_name="waste main rebuild", env=env):
            return 1

    if not args.skip_life:
        if not run_cmd([py, "upsert_chroma_life.py"], cwd=SRC_DIR, step_name="life main rebuild", env=env):
            return 1

    if not args.skip_waste_faq:
        if not run_cmd([py, "build_waste_faq.py"], cwd=SRC_DIR, step_name="waste FAQ rebuild", env=env):
            return 1

    if not args.skip_events:
        events_csv = DATA_DIR / "events_2026.csv"
        events_docs_json = DATA_DIR / "events_docs_2026.json"
        events_faq_jsonl = DATA_DIR / "events" / "okazaki_events_faq.jsonl"

        if not require_file(events_csv, "data/events_2026.csv"):
            log("ABORT: events CSV が無いため docs build できません")
            return 1

        if not run_cmd([py, "build_events_docs2.py"], cwd=SRC_DIR, step_name="events docs build", env=env):
            return 1

        if not require_file(events_docs_json, "data/events_docs_2026.json"):
            log("ABORT: events_docs_2026.json が生成されていません")
            return 1

        if not run_cmd([py, "upsert_chroma_events.py"], cwd=SRC_DIR, step_name="events main rebuild", env=env):
            return 1

        if not require_file(events_faq_jsonl, "data/events/okazaki_events_faq.jsonl"):
            log("ABORT: events FAQ JSONL が無いため FAQ rebuild できません")
            return 1

        if not run_cmd([py, "build_events_faq.py", str(events_faq_jsonl)], cwd=SRC_DIR, step_name="events FAQ rebuild", env=env):
            return 1

        if not args.skip_fixed_faq:
            fixed_faq_jsonl = DATA_DIR / "tourism_fixed_faq.jsonl"
            if not require_file(fixed_faq_jsonl, "data/tourism_fixed_faq.jsonl"):
                log("SKIP: tourism_fixed_faq.jsonl が無いため fixed FAQ rebuild は実行しません")
            else:
                if not run_cmd([py, "build_tourism_fixed_faq2.py", "--jsonl", str(fixed_faq_jsonl), "--reset"], cwd=SRC_DIR, step_name="tourism fixed FAQ rebuild", env=env):
                    return 1

    log("=" * 70)
    log("REBUILD ALL DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
