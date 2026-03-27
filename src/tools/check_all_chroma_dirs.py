from pathlib import Path
import chromadb
from chromadb.config import Settings

PORTAL_ROOT = Path(__file__).resolve().parents[1]

TARGETS = [
    PORTAL_ROOT / "chroma_db",
    PORTAL_ROOT / "okazaki_rag" / "chroma_db",
    PORTAL_ROOT / "okazaki_waste_rag" / "chroma_db",
    PORTAL_ROOT / "okazaki_waste_rag" / "src" / "chroma_db",
]

def main():
    for dbdir in TARGETS:
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