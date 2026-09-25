from pathlib import Path
import sqlite3
import csv
import time


PROJECT_ROOT = Path(__file__).resolve().parents[4]

NORMALIZED_DIR = PROJECT_ROOT / "data" / "normalized" / "train"
DB_DIR = PROJECT_ROOT / "data" / "lookup"

CHUNK_SIZE = 50_000


FILES = {
    "s2": NORMALIZED_DIR / "train_source2_normalized.tsv",
    "s3": NORMALIZED_DIR / "train_source3_normalized.tsv",
}


def build_database(source_name: str, input_path: Path):
    db_path = DB_DIR / f"{source_name}.db"

    if db_path.exists():
        print(f"\n{source_name.upper()} database already exists:")
        print(db_path)
        print("Skipping.")
        return

    print(f"\n{'=' * 70}")
    print(f"BUILDING {source_name.upper()} LOOKUP")
    print(f"{'=' * 70}")
    print(f"Input : {input_path}")
    print(f"Output: {db_path}")

    start = time.time()

    conn = sqlite3.connect(db_path)

    # Faster bulk-loading settings.
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA temp_store = MEMORY")

    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        columns = reader.fieldnames

        if "entity_id" not in columns:
            raise ValueError("entity_id column not found")

        column_defs = []

        for col in columns:
            if col == "entity_id":
                column_defs.append('"entity_id" TEXT PRIMARY KEY')
            else:
                column_defs.append(f'"{col}" TEXT')

        conn.execute(
            f"""
            CREATE TABLE entities (
                {", ".join(column_defs)}
            )
            """
        )

        placeholders = ",".join(["?"] * len(columns))

        insert_sql = f"""
            INSERT INTO entities ({",".join(f'"{c}"' for c in columns)})
            VALUES ({placeholders})
        """

        batch = []
        total = 0

        for row in reader:
            batch.append([row.get(c, "") for c in columns])

            if len(batch) >= CHUNK_SIZE:
                conn.executemany(insert_sql, batch)
                conn.commit()

                total += len(batch)
                batch.clear()

                if total % 500_000 == 0:
                    elapsed = time.time() - start
                    print(
                        f"  loaded {total:,} rows "
                        f"({elapsed / 60:.1f} min)"
                    )

        if batch:
            conn.executemany(insert_sql, batch)
            conn.commit()
            total += len(batch)

    elapsed = time.time() - start

    print(f"\nLoaded: {total:,} rows")
    print(f"Time:   {elapsed / 60:.2f} minutes")

    conn.execute("PRAGMA optimize")
    conn.close()

    print(f"Database created: {db_path}")


def main():
    DB_DIR.mkdir(parents=True, exist_ok=True)

    for source_name, input_path in FILES.items():
        build_database(source_name, input_path)


if __name__ == "__main__":
    main()