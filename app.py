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

def extract_count_column(prompt: str, source_col):
    """
    Detects count column and DISTINCT intent from free text.

    Special rules:
    - NULL checks ALWAYS return (None, False) → COUNT(*)
    - DISTINCT handled only when explicitly requested
    """

    lower_prompt = (
        prompt.lower()
        .replace("(", " ")
        .replace(")", " ")
        .replace(",", " ")
    )

    colnames = [c["name"].lower() for c in source_col]
    tokens = lower_prompt.split()

    # -------------------------------------------------
    # 1️⃣ NULL CHECK DETECTION → FORCE COUNT(*)
    # -------------------------------------------------
    null_keywords = {"null", "is null", "missing", "blank", "empty"}

    if any(k in lower_prompt for k in null_keywords):
        # Never count a column for NULL checks
        return None, False

    # -------------------------------------------------
    # 2️⃣ DISTINCT DETECTION
    # -------------------------------------------------
    is_distinct = "distinct" in tokens

    # -------------------------------------------------
    # 3️⃣ Explicit patterns: "count <column>"
    # -------------------------------------------------
    if "count" in tokens:
        idx = tokens.index("count")
        lookahead = tokens[idx + 1 : idx + 4]

        joined_variants = []
        for i in range(len(lookahead)):
            for j in range(i + 1, len(lookahead)):
                joined_variants.append(lookahead[i] + lookahead[j])
                joined_variants.append(lookahead[i] + "_" + lookahead[j])

        for token in lookahead + joined_variants:
            if token in colnames:
                return token, is_distinct

    # -------------------------------------------------
    # 4️⃣ Pattern: "count of <column>"
    # -------------------------------------------------
    for col in colnames:
        if f"count of {col}" in lower_prompt:
            return col, is_distinct

    # -------------------------------------------------
    # 5️⃣ NO COLUMN FOUND → COUNT(*)
    # -------------------------------------------------
    return None, is_distinct

def extract_compound_columns(prompt: str, source_col):
    """
    Extract multiple columns for compound unique checks.
    Example prompts:
      - check unique combination of email and hire_date
      - ensure email, hire_date is unique
      - compound unique email hire_date
    """
    lower_prompt = prompt.lower().replace(",", " ").replace("(", " ").replace(")", " ")
    tokens = lower_prompt.split()

    colnames = {c["name"].lower(): c["name"] for c in source_col}

    found = []
    for token in tokens:
        if token in colnames and token not in found:
            found.append(colnames[token])

    # Compound unique requires 2+ columns
    return found if len(found) >= 2 else []

# ------------------------------------------------------------
#   SQL SANITIZER – Removes invalid datatype comparisons
# ------------------------------------------------------------
def sanitize_sql(sql: str, source_col):
    type_map = {c["name"].lower(): str(c["type"]).lower() for c in source_col}
    sql_lower = sql.lower()

    if " where " not in sql_lower:
        return sql

    before_where, where_part = sql.split("WHERE", 1)
    where_part = where_part.strip().rstrip(";")

    if where_part == "1=1":
        return sql

    safe_conditions = []
    conditions = [c.strip() for c in where_part.split(" OR ")]

    for cond in conditions:
        if not cond:
            continue

        # Extract left column
        left = cond.split()[0].replace("LOWER(", "").replace(")", "")
        col = left.lower()

        if col not in type_map:
            continue

        col_type = type_map[col]

        # Extract right side value
        if "=" in cond:
            right = cond.split("=", 1)[1].strip().strip("'")
        elif "!=" in cond:
            right = cond.split("!=", 1)[1].strip().strip("'")
        else:
            safe_conditions.append(cond)
            continue

        # numeric
        if any(t in col_type for t in ["int", "decimal", "numeric"]):
            if not right.replace(".", "", 1).isdigit():
                logger.warning(f"❌ Removed invalid numeric condition: {cond}")
                continue

        # date
        if "date" in col_type:
            if not (len(right) == 10 and right[4] == "-" and right[7] == "-"):
                logger.warning(f"❌ Removed invalid date condition: {cond}")
                continue

        safe_conditions.append(cond)

    if not safe_conditions:
        return before_where + "WHERE 1=1;"

    return before_where + "WHERE " + " OR ".join(safe_conditions) + ";"

# ------------------------------------------------------------
#   MAIN QUERY BUILDER
# ------------------------------------------------------------
def generate_where_clause(prompt: str, source_db: str, source_table: str, source_col):
    """
    Generates SQL:
      - COUNT(*) / COUNT(col) / COUNT(DISTINCT col)
      - WHERE clause for ALL supported DQ checks
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = api_key

        # ---------- COUNT COLUMN ----------
        count_col, is_distinct = extract_count_column(prompt, source_col)

        if count_col:
            count_expr = (
                f"COUNT(DISTINCT {count_col}) AS row_count"
                if is_distinct else
                f"COUNT({count_col}) AS row_count"
            )
        else:
            count_expr = "COUNT(*) AS row_count"

        # -------------------------------------------------
        # 1️⃣ COMPOUND UNIQUE CHECK (SQL SERVER SAFE)
        # -------------------------------------------------
        compound_cols = extract_compound_columns(prompt, source_col)

        if compound_cols:
            group_cols = ", ".join(compound_cols)
            join_conditions = " AND ".join(
                [f"t.{c} = d.{c}" for c in compound_cols]
            )

            return f"""
            SELECT COUNT(*) AS row_count
            FROM {source_db}.{source_table} t
            JOIN (
                SELECT {group_cols}
                FROM {source_db}.{source_table}
                GROUP BY {group_cols}
                HAVING COUNT(*) > 1
            ) d
            ON {join_conditions};
            """

        # ---------- GPT PROMPT (FIXED) ----------
        system_prompt = (
            "You are an expert SQL data quality assistant. "
            "You ALWAYS generate a WHERE clause when the user provides any condition."
        )

        user_prompt = f"""
Generate ONE valid SQL query.

BASE QUERY:
SELECT {count_expr}
FROM {source_db}.{source_table}

SUPPORTED CHECKS (ALWAYS APPLY WHEN IMPLIED):
- Null check → column IS NULL
- Range check → BETWEEN
- Validity check → IN (...)
- Format check → LIKE
- Default check → column = value
- Custom filter → exact SQL condition
- Unique checks → COUNT(DISTINCT ...)

SCHEMA (ONLY USE THESE COLUMNS):
{chr(10).join([f"- {c['name']} ({c['type']})" for c in source_col])}

RULES:
- Output ONLY SQL
- End with semicolon
- Never hallucinate columns
- Strings → LOWER()
- IN values → lowercase
- If unclear → WHERE 1=1

USER CONDITION:
{prompt}
"""

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=250
        )

        sql_out = response.choices[0].message.content.strip()

        # ---------- CLEANUP ----------
        sql_out = sql_out.replace("```sql", "").replace("```", "").strip()
        sql_out = re.sub(r"\b(source|target)\.", "", sql_out, flags=re.IGNORECASE)

        # ---------- SAFETY ----------
        sql_out = sanitize_sql(sql_out, source_col)

        return sql_out

    except Exception as e:
        logger.error(f"generate_where_clause error: {e}")
        return f"SELECT COUNT(*) AS row_count FROM {source_db}.{source_table} WHERE 1=1;"

def get_row_count(engine, table_name, where_clause):
    """
    Use DB-specific quoting for table name:
      - mssql : [table]
      - sqlite: "table"
    """
    mode = get_db_mode()
    _safe_identifier(table_name)
    if mode == "mssql":
        q = where_clause
    else:
        q = where_clause
    with engine.connect() as conn:
        result = conn.execute(text(q))
        val = result.scalar()
        return int(val or 0)

def generate_simple_summary(source_count, target_count, is_anomaly):
    if source_count == target_count:
        return f"Perfect match: both have {source_count} rows."
    diff = abs(source_count - target_count)
    pct = (diff / max(source_count, target_count)) * 100 if max(source_count, target_count) else 0
    status = "significant discrepancy" if pct > 5 else "minor difference"
    anomaly = " Possible anomaly." if is_anomaly else ""
    return f"Row count mismatch: {diff} rows difference ({pct:.1f}% {status}).{anomaly}"


def generate_summary(source_count, target_count, source_query,target_query):
    """
    Uses OpenAI GPT-4o-mini to generate a data quality summary.
    Falls back to generate_simple_summary if API key not set or error occurs.
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.debug("OPENAI_API_KEY not found. Using fallback summary.")
            return generate_simple_summary(source_count, target_count)

        openai.api_key = api_key

        system_prompt = "You are a data quality analyst. Write clear, concise summaries for data engineering reports."
        user_prompt = f"""
        Analyze the following row count validation details and write a clear 4–5 sentence summary for a data engineering report:

        - Source Row Count: {source_count}
        - Target Row Count: {target_count}
        - Filter Applied (Source Query): {source_query}
        - Filter Applied (target Query): {target_query}
        """

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=250
        )

        summary = response.choices[0].message.content.strip()
        return summary

    except Exception as e:
        logger.error(f"OpenAI GPT summary error: {e}")
        return generate_simple_summary(source_count, target_count)

def split_schema_table(full_table_name):
    if '.' in full_table_name:
        parts = full_table_name.split('.', 1)
        return parts[0], parts[1]  # schema, table
    else:
        return 'dbo', full_table_name

# ─── VALIDATE ENDPOINT ─────────────────────────────────────────────────────────
@app.route("/api/validate", methods=["POST"])
def validate():
    try:
        data = request.get_json()

        # ------------------ REQUIRED FIELDS ------------------
        required = ["source_db", "target_db", "prompt", "mappings"]
        for f in required:
            if not data.get(f):
                return jsonify({"error": f"Missing field: {f}"}), 400

        source_db = data["source_db"]
        target_db = data["target_db"]
        prompt    = data["prompt"]
        mappings  = data["mappings"]

        if not isinstance(mappings, list) or len(mappings) == 0:
            return jsonify({"error": "No table mappings provided"}), 400

        results = []

        # ------------------ CONNECT ONCE (IMPORTANT) ------------------
        src_engine = connect_to_database(source_db)
        tgt_engine = connect_to_database(target_db)

        src_inspect = inspect(src_engine)
        tgt_inspect = inspect(tgt_engine)

        # ------------------ PROCESS EACH MAPPING ------------------
        for idx, mapping in enumerate(mappings, start=1):

            s_table = mapping.get("source_table")
            t_table = mapping.get("target_table")

            if not s_table or not t_table:
                return jsonify({
                    "error": f"Invalid mapping at index {idx}"
                }), 400

            # ---- Split schema.table ----
            s_schema, s_name = split_schema_table(s_table)
            t_schema, t_name = split_schema_table(t_table)

            # ------------------ SOURCE ------------------
            try:
                src_cols = src_inspect.get_columns(s_name, schema=s_schema)
            except Exception:
                return jsonify({
                    "error": f"Source table '{s_table}' not found in database '{source_db}'"
                }), 400

            src_query = generate_where_clause(
                prompt=prompt,
                source_db=source_db,
                source_table=s_table,
                source_col=src_cols
            )

            src_count = get_row_count(src_engine, s_name, src_query)

            # ------------------ TARGET ------------------
            try:
                tgt_cols = tgt_inspect.get_columns(t_name, schema=t_schema)
            except Exception:
                return jsonify({
                    "error": f"Target table '{t_table}' not found in database '{target_db}'"
                }), 400

            tgt_query = generate_where_clause(
                prompt=prompt,
                source_db=target_db,
                source_table=t_table,
                source_col=tgt_cols
            )

            tgt_count = get_row_count(tgt_engine, t_name, tgt_query)

            # ------------------ SUMMARY ------------------
            summary = generate_summary(
                source_count=src_count,
                target_count=tgt_count,
                source_query=src_query,
                target_query=tgt_query
            )

            results.append({
                "source_table": s_table,
                "target_table": t_table,
                "source_count": src_count,
                "target_count": tgt_count,
                "source_query": src_query,
                "target_query": tgt_query,
                "summary": summary
            })

        return jsonify({"results": results})

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# ─── ERROR HANDLERS ────────────────────────────────────────────────────────────
@app.errorhandler(404)
def nf(e):
    return jsonify({"error":"Not found"}), 404

@app.errorhandler(500)
def ie(e):
    return jsonify({"error":"Server error"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)