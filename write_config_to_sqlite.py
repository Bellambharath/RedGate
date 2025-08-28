# write_config_to_sqlite.py
import sqlite3
from pathlib import Path

# Edit values below (or leave empty strings for MSSQL_USER/MSSQL_PASS if using Trusted Connection)
config_values = {
    "DB_MODE": "mssql",
    "MSSQL_SERVER": "6R312S3",
    "MSSQL_DB": "master",
    "MSSQL_USER": "",   # supply if using SQL auth
    "MSSQL_PASS": "",   # supply if using SQL auth
    "GEMINI_API_KEY": "AIzaSyAWl-QwAzS5toyAUmfaNqdFtEs7OKPQW8A",
    "LOG_DB": "your_log_database_name",
    "HISTORY_DB_NAME": "master"
}

DB_PATH = Path(__file__).parent / "sample.db"

def ensure_table(conn):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS config (
      key   TEXT PRIMARY KEY,
      value TEXT NOT NULL
    );
    """)
    conn.commit()

def upsert_config(conn, key, value):
    conn.execute("""
    INSERT INTO config (key, value) VALUES (?, ?)
    ON CONFLICT(key) DO UPDATE SET value = excluded.value;
    """, (key, value))
    conn.commit()

def main():
    if not DB_PATH.exists():
        print(f"Warning: {DB_PATH} does not exist. Script will create the SQLite file and config table.")
    conn = sqlite3.connect(DB_PATH)
    try:
        ensure_table(conn)
        for k, v in config_values.items():
            if v is None:
                v = ""
            upsert_config(conn, k, str(v))
            print(f"Wrote {k} = {v!r}")
    finally:
        conn.close()
    print("Done. Your app.py will now read these config values from sample.db.")

if __name__ == "__main__":
    import sqlite3
    main()
