# app.py
import os
import sqlite3
import certifi
import warnings
import logging
import traceback
import pandas as pd
from urllib.parse import quote_plus
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from sqlalchemy import create_engine, inspect, text
from urllib3.exceptions import InsecureRequestWarning
import openai
import json
import re 
from dotenv import load_dotenv

# ─── CONFIG & LOGGING ─────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
BASE_DIR   = os.path.dirname(__file__)
CONFIG_DB  = os.path.join(BASE_DIR, "sample.db")  # only used if the file already exists

# Fallback MSSQL server from your SSMS screenshot. Code prefers env/config first.
# DEFAULT_MSSQL_SERVER = r"2S3Q7R3\MSSQLSERVER1"

DEFAULT_MSSQL_SERVER = r"DESKTOP-GSJB5GJ"


os.environ["SSL_CERT_FILE"] = certifi.where()
warnings.filterwarnings("ignore", category=InsecureRequestWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load env (optional)
load_dotenv("ATT92742.env")

# Note: startup no longer creates/initializes any local DB files or config entries.
# Configuration is read from environment variables first; if sample.db exists, we may read from it (read-only).

def get_config_value(key: str) -> str | None:
    """
    Read configuration value from environment first.
    If not present and sample.db exists, read from its config table (read-only).
    Do NOT create or modify sample.db here.
    """
    # 1) environment variable (highest priority)
    val = os.getenv(key)
    if val is not None and val != "":
        return val

    # 2) sample.db (only if file exists) - read-only
    if os.path.exists(CONFIG_DB):
        try:
            conn = sqlite3.connect(CONFIG_DB)
            cur = conn.cursor()
            cur.execute("SELECT value FROM config WHERE key = ?", (key,))
            row = cur.fetchone()
            conn.close()
            return row[0] if row else None
        except Exception as e:
            logger.debug(f"Failed to read {key} from sample.db: {e}")
            return None

    # 3) not found
    return None


def get_db_mode() -> str:
    # priority: env -> sample.db (if present) -> default mssql
    return (get_config_value("DB_MODE") or "mssql").lower()


# ─── FLASK SETUP ───────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)


# ─── SAFE IDENTIFIERS (basic check to avoid injection in identifiers) ──────────
def _safe_identifier(name: str) -> str:
    """
    Allow only basic alphanumeric + underscore characters for table/database names.
    Raises ValueError for anything unsafe.
    """
    # if not name or not re.match(r'^[A-Za-z0-9_]+$', name):
    #     raise ValueError("Invalid identifier. Only A-Z, a-z, 0-9 and underscore allowed.")
    return name


def _build_mssql_conn_str(db_name: str | None = None) -> str:
    """
    Build a SQLAlchemy pyodbc connection string for MSSQL.
    Prefers environment variables; if missing falls back to DEFAULT_MSSQL_SERVER.
    """
    server = get_config_value("MSSQL_SERVER") or DEFAULT_MSSQL_SERVER
    default_db = db_name or (get_config_value("MSSQL_DB") or "master")
    user = get_config_value("MSSQL_USER") or None
    pwd  = get_config_value("MSSQL_PASS") or None

    # Optional driver override
    driver_name = get_config_value("MSSQL_DRIVER") or "ODBC Driver 17 for SQL Server"

    if user and pwd:
        odbc_str = (
            f"DRIVER={{{driver_name}}};"
            f"SERVER={server};DATABASE={default_db};UID={user};PWD={pwd};"
        )
    else:
        odbc_str = (
            f"DRIVER={{{driver_name}}};"
            f"SERVER={server};DATABASE={default_db};Trusted_Connection=yes;"
        )

    return "mssql+pyodbc:///?odbc_connect=" + quote_plus(odbc_str)


def connect_to_server(db_name: str | None = None):
    """
    Returns a SQLAlchemy engine:
      - mssql mode -> engine connected to MSSQL server 'master' (or configured default DB)
      - sqlite fallback -> engine pointed at existing CONFIG_DB (only if present)
    No database creation or modification is performed here.
    """
    mode = get_db_mode()
    if mode == "mssql":      
        
        conn_str = _build_mssql_conn_str(db_name)
        
        engine = create_engine(conn_str, pool_timeout=30, pool_recycle=3600)
        # basic smoke test
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception as e:
            logger.error(f"MS SQL connect test failed: {e}")
            raise
        logger.info(f"Connected to MSSQL server { (get_config_value('MSSQL_SERVER') or DEFAULT_MSSQL_SERVER) } (db=master)")
        return engine

    # sqlite fallback only if file exists
    if os.path.exists(CONFIG_DB):
        uri = f"sqlite:///{CONFIG_DB}"
        engine = create_engine(uri, connect_args={"check_same_thread": False})
        return engine

    raise RuntimeError("No valid database configuration found (DB_MODE not 'mssql' and sample.db missing)")


def connect_to_database(db_name: str):
    """
    Return engine to a specific database name:
      - mssql: returns engine targetting the requested database name on the server
      - sqlite: returns engine if CONFIG_DB exists
    """
    mode = get_db_mode()
    if mode == "mssql":
        _safe_identifier(db_name)
        conn_str = _build_mssql_conn_str(db_name)
        engine = create_engine(conn_str, pool_timeout=30, pool_recycle=3600)
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception as e:
            logger.error(f"MS SQL connect to DB '{db_name}' failed: {e}")
            raise
        logger.info(f"Connected to MSSQL database: {db_name}")
        return engine

    # sqlite fallback only if file exists
    if os.path.exists(CONFIG_DB):
        return connect_to_server()

    raise RuntimeError("SQLite DB not found and DB_MODE is not mssql.")
 
# ─── ROUTES ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    try:
        return render_template("index.html")
    except Exception:
        return "Row count validation API is running."


@app.route("/api/databases", methods=["GET"])
def get_databases_endpoint():
    """
    For mssql mode: return server databases.
    For sqlite fallback: return filename as logical DB.
    """
    try:
        mode = get_db_mode()
        if mode == "mssql":
            engine = connect_to_server()
            query = "SELECT name FROM sys.databases WHERE database_id > 4 AND state = 0"
            with engine.connect() as conn:
                df = pd.read_sql(query, conn)
            return jsonify(df['name'].tolist())
        else:
            return jsonify([os.path.basename(CONFIG_DB)])
    except Exception as e:
        logger.error(f"Error fetching databases: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/tables", methods=["POST"])
def get_tables_endpoint():
    """
    For mssql: given a database name (posted), return its table names.
    For sqlite fallback: list tables from sqlite_master (only if file exists).
    """
    try:
        data = request.get_json()
        if not data or "database" not in data:
            return jsonify({"error": "Database name is required"}), 400

        mode = get_db_mode()
        posted_db = data["database"]

        if mode == "mssql":
            engine = connect_to_database(posted_db)
            inspector = inspect(engine)
            # Get all schemas
            schemas = inspector.get_schema_names()
            # Fetch tables for each schema
            all_tables = []
            for schema in schemas:
                for table in inspector.get_table_names(schema=schema):
                    all_tables.append(f"{schema}.{table}")

            return jsonify(all_tables)
        else:
            if not os.path.exists(CONFIG_DB):
                return jsonify({"error": "SQLite DB not available"}), 500
            engine = connect_to_server()
            query = """
              SELECT name
                FROM sqlite_master
               WHERE type = 'table'
                 AND name NOT LIKE 'sqlite_%'
               ORDER BY name
            """
            df = pd.read_sql(query, engine)
            return jsonify(df["name"].tolist())

    except Exception as e:
        logger.error(f"Error fetching tables: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/columns", methods=["POST"])
def get_columns_endpoint():
    """
    Given {"table": "<table_name>"} return column list for that table.
    """
    try:
        data = request.get_json()
        table = data.get("table")
        if not table:
            return jsonify({"error": "table is required"}), 400
        engine = connect_to_server()
        inspector = inspect(engine)
        cols = inspector.get_columns(table)
        return jsonify([c["name"] for c in cols])
    except Exception as e:
        logger.error(f"Error fetching columns: {e}")
        return jsonify({"error": str(e)}), 500

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# ─── UTILITIES ─────────────────────────────────────────────────────

def extract_table_roles(prompt):
    text = " ".join(prompt.splitlines())

    src = re.search(r"source\s+table\s+([a-zA-Z_]+\.[a-zA-Z_]+)", text, re.IGNORECASE)
    tgt = re.search(r"target\s+table\s+([a-zA-Z_]+\.[a-zA-Z_]+)", text, re.IGNORECASE)

    return (
        src.group(1) if src else None,
        tgt.group(1) if tgt else None
    )

def split_prompt_into_jobs(prompt):
    parts = re.split(r"\n\s*\d+\)", prompt)
    return [p.strip() for p in parts if p.strip()]

SQL_KEYWORDS = {
    "and","or","not","is","null","like","in","between",
    "exists","select","where","from","on","join","inner","left","right"
}

def extract_columns_from_sql(expr):
    tokens = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', expr)
    return [t for t in tokens if t.lower() not in SQL_KEYWORDS]

def validate_columns_exist(rule, columns, table):
    colset = {c["name"].lower() for c in columns}

    for c in rule["checks"]:
        if c.get("column"):
            if c["column"].lower() not in colset:
                raise ValueError(f"❌ Column '{c['column']}' does not exist in {table}")

        for col in c.get("columns", []):
            if col.lower() not in colset:
                raise ValueError(f"❌ Column '{col}' does not exist in {table}")

        if c["type"] == "custom":
            for col in extract_columns_from_sql(c["sql"]):
                if col.lower() not in colset:
                    raise ValueError(f"❌ Column '{col}' does not exist in {table}")

def split_rules_by_table(prompt, tables):
    rules = {t: [] for t in tables}
    parts = re.split(r"[.;\n]", prompt)

    for part in parts:
        for t in tables:
            if t.lower() in part.lower():
                rules[t].append(part.strip())

    return {k: " ".join(v) for k, v in rules.items() if v}

def get_foreign_keys(engine, schema, table):
    q = f"""
    SELECT
        OBJECT_NAME(fkc.parent_object_id) AS parent_table,
        COL_NAME(fkc.parent_object_id, fkc.parent_column_id) AS parent_column,
        OBJECT_NAME(fkc.referenced_object_id) AS referenced_table,
        COL_NAME(fkc.referenced_object_id, fkc.referenced_column_id) AS referenced_column
    FROM sys.foreign_key_columns fkc
    WHERE OBJECT_SCHEMA_NAME(fkc.parent_object_id) = '{schema}'
      AND OBJECT_NAME(fkc.parent_object_id) = '{table}';
    """
    with engine.connect() as c:
        return pd.read_sql(q, c).to_dict("records")

def extract_joins(prompt, foreign_keys):
    joins = []
    m = re.search(r"join\s+with\s+([a-zA-Z_]+\.[a-zA-Z_]+)", prompt, re.IGNORECASE)
    if not m:
        return joins

    join_table = m.group(1)
    join_name = join_table.split(".")[1].lower()

    for fk in foreign_keys:
        if fk["referenced_table"].lower() == join_name:
            joins.append({
                "table": join_table,
                "on": f"t.{fk['parent_column']} = j.{fk['referenced_column']}",
                "type": "INNER"
            })

    return joins

# ─── GPT → DQ RULE ─────────────────────────────────────────────────

def parse_dq_rule(prompt, columns, foreign_keys, table):
    col_list = ", ".join([c["name"] for c in columns])

    fk_text = "\n".join(
        f"{f['parent_table']}.{f['parent_column']} = {f['referenced_table']}.{f['referenced_column']}"
        for f in foreign_keys
    ) or "None"

    gpt_prompt = f"""
You are a Data Quality rule parser.

Table: {table}

Known relationships:
{fk_text}

Available columns:
{col_list}

User rule:
{prompt}

Convert to JSON:

{{
 "checks":[
  {{
   "type":"row_count|null_check|unique|compound_unique|range|default|format|validity|custom",
   "row_count" supports optional filter on column,
   "column":"column or null",
   "columns":["c1","c2"],
   "operator":"=|between|in|like|regex",
   "value":null|string|number|[v1,v2],
  }}
 ],
 "joins":[
  {{
   "table":"schema.table",
   "on":"t.col = j.col",
   "type":"INNER|LEFT"
  }}
 ]
}}

Rules:
- Use ONLY available columns
- format → value must be LIKE pattern (example: '____-__-__')
- SQL must detect invalid rows
- If user says "is", "equals", "=", treat as operator "="
- Do NOT include SELECT or WHERE in custom.sql

Return ONLY JSON
"""

    r = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": gpt_prompt}],
        temperature=0
    )

    raw = r.choices[0].message.content
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    return json.loads(match.group())

# ─── SQL BUILDER ──────────────────────────────────────────────────

def get_non_datetime_columns(columns):
    return [
        c["name"]
        for c in columns
        if "date" not in str(c["type"]).lower()
    ]

def extract_columns_from_sql(expr):
    tokens = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', expr)
    keywords = {"and","or","not","is","null","like","in","between"}
    return [t for t in tokens if t.lower() not in keywords]

def prefix_columns(expr):
    return re.sub(
        r'(?<![a-zA-Z0-9_\.])([a-zA-Z_][a-zA-Z0-9_]*)(?!\s*\.)',
        r't.\1',
        expr
    )

def build_base(db, table, joins):
    sql = f" FROM {db}.{table} t "
    for j in joins:
        sql += f" {j['type']} JOIN {db}.{j['table']} j ON {j['on']} "
    return sql

def build_dq_sql(db, table, rule):
    base = build_base(db, table, rule.get("joins", []))
    sqls = []

    for c in rule["checks"]:

        if c["type"] == "row_count":
            if c.get("column"):
                sqls.append(
                    f"SELECT COUNT(*) {base} WHERE t.{c['column']} {c.get('operator', '=')} {c['value']}"
                )
            else:
                sqls.append(f"SELECT COUNT(*) {base}")

        elif c["type"] == "null_check":
            sqls.append(f"SELECT COUNT(*) {base} WHERE t.{c['column']} IS NULL")

        elif c["type"] == "range":
            sqls.append(f"SELECT COUNT(*) {base} WHERE t.{c['column']} BETWEEN {c['value'][0]} AND {c['value'][1]}")

        elif c["type"] == "default":
            sqls.append(f"SELECT COUNT(*) {base} WHERE t.{c['column']} = {c['value']}")

        elif c["type"] == "format":
            op = c.get("operator", "LIKE")
            sqls.append(
                f"SELECT COUNT(*) {base} WHERE t.{c['column']} {op} '{c['value']}'"
            )

        elif c["type"] == "validity":
            vals = ",".join(f"'{v}'" for v in c["value"])
            op = c.get("operator", "IN").upper()
            if op == "IN":
                sqls.append(f"SELECT COUNT(*) {base} WHERE t.{c['column']} NOT IN ({vals})")
            else:
                sqls.append(f"SELECT COUNT(*) {base} WHERE t.{c['column']} IN ({vals})")

        elif c["type"] == "row_count" and c.get("column"):
            sqls.append(f"SELECT COUNT(*) {base} WHERE t.{c['column']} = {c['value']}")       

        elif c["type"] == "unique":
            sqls.append(f"""
            SELECT COUNT(DISTINCT t.{c['column']})
            FROM {db}.{table} t
            """)

        elif c["type"] == "compound_unique":
            cols = ",".join(f"t.{col}" for col in c["columns"])
            sqls.append(f"""
            SELECT COUNT(*) FROM (
                SELECT {cols}, COUNT(*) c
                FROM {db}.{table} t
                GROUP BY {cols}
                HAVING COUNT(*) > 1
            ) x
            """)

        elif c["type"] == "custom":
            raw = prefix_columns(c["sql"])
            sqls.append(f"SELECT COUNT(*) FROM {db}.{table} t WHERE {raw}")

    return sqls

def run_sql(engine, sqls):
    with engine.connect() as c:
        return [int(c.execute(text(q)).scalar() or 0) for q in sqls]

# ─── ROUTES ───────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/databases", methods=["GET"])
def get_databases():
    engine = connect_to_database("master")
    df = pd.read_sql("SELECT name FROM sys.databases WHERE database_id > 4 AND state = 0", engine)
    return jsonify(df["name"].tolist())

@app.route("/api/tables", methods=["POST"])
def get_tables():
    db = request.get_json()["database"]
    engine = connect_to_database(db)
    insp = inspect(engine)

    tables = []
    for schema in insp.get_schema_names():
        for t in insp.get_table_names(schema=schema):
            tables.append(f"{schema}.{t}")
    return jsonify(tables)

@app.route("/api/validate", methods=["POST"])
def validate():
    try:
        data = request.get_json()
        source_db = data["source_db"]
        target_db = data["target_db"]
        prompt = data["prompt"]

        src_engine = connect_to_database(source_db)
        tgt_engine = connect_to_database(target_db)

        jobs = split_prompt_into_jobs(prompt)

        results = []

        for job in jobs:
            try:
                source_table, target_table = extract_table_roles(job)
                source_table, target_table = extract_table_roles(job)
                if not source_table or not target_table:
                    results.append({"error": f"Missing source/target in rule: {job}"})
                    continue

                s_schema, s_name = source_table.split(".", 1)
                t_schema, t_name = target_table.split(".", 1)

                # SOURCE
                src_cols = inspect(src_engine).get_columns(s_name, schema=s_schema)
                src_fk = get_foreign_keys(src_engine, s_schema, s_name)
                src_rule = parse_dq_rule(job, src_cols, src_fk, source_table)
                src_rule["joins"] = extract_joins(job, src_fk)
                validate_columns_exist(src_rule, src_cols, source_table)

                src_sql = build_dq_sql(source_db, source_table, src_rule)
                src_vals = run_sql(src_engine, src_sql)

                # TARGET
                tgt_cols = inspect(tgt_engine).get_columns(t_name, schema=t_schema)
                tgt_fk = get_foreign_keys(tgt_engine, t_schema, t_name)
                tgt_rule = parse_dq_rule(job, tgt_cols, tgt_fk, target_table)
                tgt_rule["joins"] = extract_joins(job, tgt_fk)
                validate_columns_exist(tgt_rule, tgt_cols, target_table)

                tgt_sql = build_dq_sql(target_db, target_table, tgt_rule)
                tgt_vals = run_sql(tgt_engine, tgt_sql)

                results.append({
                    "source_table": source_table,
                    "target_table": target_table,
                    "source_count": src_vals[0] if src_vals else 0,
                    "target_count": tgt_vals[0] if tgt_vals else 0,
                    "summary": f"Source: {sum(1 for x in src_vals if x>0)}, Target: {sum(1 for x in tgt_vals if x>0)}",
                    "source_query": ";\n".join(src_sql),
                    "target_query": ";\n".join(tgt_sql)
                })
            except Exception as e:
                results.append({
                    "error": str(e),
                    "rule": job
                })

        return jsonify({"results": results})

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 400

# ─── RUN ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)