# init_sample_db.py

import sqlite3
import os

BASE_DIR = os.path.dirname(__file__)
DB_PATH  = os.path.join(BASE_DIR, "sample.db")

def initialize():
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    # 1) config table (key/value)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS config (
      key   TEXT PRIMARY KEY,
      value TEXT NOT NULL
    );
    """)
    cur.executemany("""
    INSERT INTO config(key, value)
    VALUES (?, ?)
    ON CONFLICT(key) DO NOTHING;
    """, [
      ("MSSQL_SERVER",    "6R312S3"),
      ("LOG_DB",          "your_log_database_name"),
      ("HISTORY_DB_NAME", "master"),
    ])

    # 2) history log table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS ETL_RowCount_Log (
      LogDate        TEXT     DEFAULT CURRENT_TIMESTAMP,
      SourceDatabase TEXT,
      SourceTable    TEXT,
      TargetDatabase TEXT,
      TargetTable    TEXT,
      SourceRowCount INTEGER,
      TargetRowCount INTEGER,
      WHEREClause    TEXT,
      IsAnomaly      INTEGER
    );
    """)

    # 3) departments demo table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS departments (
      id   INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT    UNIQUE NOT NULL
    );
    """)
    cur.executemany("""
    INSERT INTO departments(name) VALUES (?)
    ON CONFLICT(name) DO NOTHING;
    """, [
      ("Sales",),
      ("Engineering",),
      ("HR",),
      ("Marketing",),
    ])

    # 4) employees demo table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS employees (
      id            INTEGER PRIMARY KEY AUTOINCREMENT,
      first_name    TEXT NOT NULL,
      last_name     TEXT NOT NULL,
      department_id INTEGER NOT NULL REFERENCES departments(id),
      hire_date     TEXT    NOT NULL,
      salary        REAL    NOT NULL
    );
    """)
    cur.executemany("""
    INSERT INTO employees(
      first_name, last_name, department_id, hire_date, salary
    ) VALUES (?, ?, ?, ?, ?)
    ON CONFLICT(id) DO NOTHING;
    """, [
      ("Alice", "Wong",    1, "2022-01-15", 65000.0),
      ("Bob",   "Patel",   2, "2021-03-23", 85000.0),
      ("Carol", "García",  2, "2020-11-05", 90000.0),
      ("David", "Schmidt", 3, "2023-06-01", 55000.0),
      ("Eva",   "Kumar",   4, "2022-08-17", 72000.0),
      ("Frank", "O’Connor",1, "2019-12-10", 68000.0),
    ])

    conn.commit()
    conn.close()
    print(f"Initialized SQLite at {DB_PATH}")

if __name__ == "__main__":
    initialize()
