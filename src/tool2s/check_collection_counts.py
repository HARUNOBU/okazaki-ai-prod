from pathlib import Path
import sys
import chromadb
from chromadb.config import Settings

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import WASTE_LIFE_DB_DIR, KANKO_DB_DIR

def print_counts(dbdir, label: str):
    client = chromadb.PersistentClient(
        path=str(dbdir),
        settings=Settings(anonymized_telemetry=False)
    )
    print(f"=== {label}: {dbdir} ===")
    for c in client.list_collections():
        name = c.name if hasattr(c, "name") else str(c)
        col = client.get_collection(name)
        print(f"{name}: {col.count()}")

def main():
    print_counts(WASTE_LIFE_DB_DIR, "WASTE_LIFE_DB")
    if str(KANKO_DB_DIR) != str(WASTE_LIFE_DB_DIR):
        print_counts(KANKO_DB_DIR, "KANKO_DB")

if __name__ == "__main__":
    main()
