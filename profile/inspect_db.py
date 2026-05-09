"""Inspect rocpd SQLite DB structure and dump counter data."""
import sqlite3
import sys


def main():
    db = sys.argv[1]
    conn = sqlite3.connect(db)
    cur = conn.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    print(f"Tables: {tables}")

    for t in tables:
        cur.execute(f"SELECT * FROM {t} LIMIT 1")
        cols = [d[0] for d in cur.description]
        print(f"\n{t} columns ({len(cols)}): {cols}")
        row = cur.fetchone()
        if row:
            for c, v in zip(cols, row):
                val_str = str(v)[:80] if v is not None else "NULL"
                print(f"  {c}: {val_str}")

    # Try to find counter data in kernel_dispatch
    if "rocpd_kernel_dispatch" in tables:
        cur.execute("SELECT * FROM rocpd_kernel_dispatch LIMIT 1")
        cols = [d[0] for d in cur.description]
        sq_cols = [c for c in cols if "SQ" in c.upper() or "PMC" in c.upper() or "COUNTER" in c.upper()]
        print(f"\n\nPMC-like columns in kernel_dispatch: {sq_cols}")
        if sq_cols:
            cur.execute(f"SELECT {','.join(sq_cols)} FROM rocpd_kernel_dispatch LIMIT 5")
            for row in cur.fetchall():
                print(dict(zip(sq_cols, row)))

    conn.close()


if __name__ == "__main__":
    main()
