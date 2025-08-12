# init_multiple_dbs.py
"""
Create three sample sqlite DB files with properly-matched seed rows:
 - analytics.db  -> events, metrics
 - reporting.db   -> sales, customers
 - logging.db     -> app_log
Run once from the project directory.
"""

import sqlite3
import os

BASE_DIR = os.path.dirname(__file__)
DB_DIR = BASE_DIR

DB_SPECS = {
    "analytics.db": [
        # create SQL, insert SQL, rows
        ("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT,
            created_at TEXT
        );
        """,
        "INSERT INTO events (event_type, created_at) VALUES (?, ?)",
        [
            ("click", "2024-07-01T10:00:00"),
            ("view",  "2024-07-01T10:05:00"),
            ("click", "2024-07-02T09:00:00"),
        ]),

        ("""
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_name TEXT,
            value REAL
        );
        """,
        "INSERT INTO metrics (metric_name, value) VALUES (?, ?)",
        [
            ("latency", 120.5),
            ("throughput", 44.2),
        ]),
    ],

    "reporting.db": [
        ("""
        CREATE TABLE IF NOT EXISTS sales (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product TEXT,
            amount REAL,
            sale_date TEXT
        );
        """,
        "INSERT INTO sales (product, amount, sale_date) VALUES (?, ?, ?)",
        [
            ("Widget", 120.0, "2024-06-21"),
            ("Gadget", 250.0, "2024-06-22"),
        ]),

        ("""
        CREATE TABLE IF NOT EXISTS customers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            country TEXT
        );
        """,
        "INSERT INTO customers (name, country) VALUES (?, ?)",
        [
            ("Alice", "US"),
            ("Bob",   "UK"),
        ]),
    ],

    "logging.db": [
        ("""
        CREATE TABLE IF NOT EXISTS app_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            level TEXT,
            message TEXT,
            ts TEXT
        );
        """,
        "INSERT INTO app_log (level, message, ts) VALUES (?, ?, ?)",
        [
            ("INFO",  "startup", "2024-01-01T00:00:00"),
            ("ERROR", "boom",    "2024-01-02T01:02:03"),
        ]),
    ]
}


def create_db_file(db_filename, spec_list):
    path = os.path.join(DB_DIR, db_filename)
    if os.path.exists(path):
        print(f"{db_filename} exists — skipping creation")
        return
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    try:
        for create_sql, insert_sql, rows in spec_list:
            cur.execute(create_sql)
            if rows:
                try:
                    cur.executemany(insert_sql, rows)
                except Exception as e:
                    print(f"Could not insert sample rows into {insert_sql.split()[2]}: {e}")
        conn.commit()
        print(f"Created {db_filename}")
    finally:
        conn.close()


def main():
    for fname, spec in DB_SPECS.items():
        create_db_file(fname, spec)
    print("All done. Created DBs:", ", ".join(DB_SPECS.keys()))


if __name__ == "__main__":
    main()
