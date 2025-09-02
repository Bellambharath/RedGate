"""
update_config.py
Run this from the project directory (same folder as sample.db and app.py).

This script updates the config table in sample.db with DB_MODE and MSSQL settings.
"""

import sqlite3
import os

BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "sample.db")

# === EDIT THESE VALUES BELOW ===
NEW_DB_MODE = "mssql"                 # "sqlite" or "mssql"
MSSQL_SERVER = "your_sql_server_host" # e.g. "MYHOST\\SQLEXPRESS" or "sql.example.com"
MSSQL_USER = None                     # set to username if using SQL auth, otherwise None
MSSQL_PASS = None                     # set to password if using SQL auth, otherwise None
# ===============================

def upsert_config(key: str, value: str, conn):
    cur = conn.cursor()
    # Use INSERT ... ON CONFLICT to insert or update existing value
    cur.execute("""
      INSERT INTO config(key, value)
      VALUES (?, ?)
      ON CONFLICT(key) DO UPDATE SET value=excluded.value;
    """, (key, value))

def main():
    if not os.path.exists(DB_PATH):
        raise SystemExit(f"Database file not found: {DB_PATH} — run init_sample_db.py first")

    conn = sqlite3.connect(DB_PATH)
    try:
        upsert_config("DB_MODE", NEW_DB_MODE, conn)
        upsert_config("MSSQL_SERVER", MSSQL_SERVER, conn)
        if MSSQL_USER is not None:
            upsert_config("MSSQL_USER", MSSQL_USER, conn)
        if MSSQL_PASS is not None:
            upsert_config("MSSQL_PASS", MSSQL_PASS, conn)
        conn.commit()
        print("Updated config table successfully. Current values:")
        for row in conn.execute("SELECT key, value FROM config ORDER BY key"):
            print(row)
    finally:
        conn.close()

if __name__ == "__main__":
    main()
