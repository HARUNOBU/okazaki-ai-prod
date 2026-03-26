from pathlib import Path
import sys
import chromadb
from chromadb.config import Settings

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import WASTE_LIFE_DB_DIR, KANKO_DB_DIR

TARGETS = [
    WASTE_LIFE_DB_DIR,
    KANKO_DB_DIR,
]

def main():
    seen = set()
    for dbdir in TARGETS:
        dbdir = Path(dbdir)
        key = str(dbdir.resolve()) if dbdir.exists() else str(dbdir)
        if key in seen:
            continue
        seen.add(key)

        print("=" * 80)
        print("DB:", dbdir)
        if not dbdir.exists():
            print("  not found")
            continue

        try:
            client = chromadb.PersistentClient(
                path=str(dbdir),
                settings=Settings(anonymized_telemetry=False)
            )
            cols = client.list_collections()
            if not cols:
                print("  no collections")
                continue

            for c in cols:
                name = c.name if hasattr(c, "name") else str(c)
                try:
                    col = client.get_collection(name)
                    print(f"  {name}: {col.count()}")
                except Exception as e:
                    print(f"  {name}: ERROR {e}")
        except Exception as e:
            print("  open error:", e)

if __name__ == "__main__":
    main()
