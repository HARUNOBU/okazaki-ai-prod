import json
from pathlib import Path

p = Path("logs/qa_log.jsonl")
with p.open("r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        print("-----")
        print(line)

