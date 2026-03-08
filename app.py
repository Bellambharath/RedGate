import os
import logging
import traceback
import re
from urllib.parse import quote_plus

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from sqlalchemy import create_engine, inspect, text
from dotenv import load_dotenv
from openai import OpenAI

DEFAULT_MSSQL_SERVER = "localhost"
DEFAULT_db = "AdventureWorks2022"
DEFAULT_DRIVER = "ODBC Driver 17 for SQL Server"
DEFAULT_MODEL = "gpt-4o-mini"

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)


def cfg(key: str) -> str | None:
    v = os.getenv(key)
    return v if v not in (None, "") else None


def clean_server(server: str) -> str:
    s = server.strip()
    if s.endswith("/"):
        s = s[:-1]
    return s.replace("/", "\\")


def tbl_ref(db: str, schema: str, table: str) -> str:
    return f"[{db}].[{schema}].[{table}]"


def llm_client():
    key = cfg("OPENAI_API_KEY")
    if not key:
        return None
    return OpenAI(api_key=key)


def mk_conn_str(db_name: str | None = None) -> str:
    server = clean_server(cfg("MSSQL_SERVER") or DEFAULT_MSSQL_SERVER)
    dbName = db_name or (cfg("MSSQL_DB") or DEFAULT_db)
    driver = cfg("MSSQL_DRIVER") or DEFAULT_DRIVER
    user = cfg("MSSQL_USER")
    pwd = cfg("MSSQL_PASS")

    extra = cfg("MSSQL_EXTRA_PARAMS")
    if extra:
        extra = extra.strip().strip(";") + ";"
    else:
        extra = "TrustServerCertificate=yes;"

    if user and pwd:
        odbc = (
            f"DRIVER={{{driver}}};"
            f"SERVER={server};DATABASE={dbName};"
            f"UID={user};PWD={pwd};{extra}"
        )
    else:
        odbc = (
            f"DRIVER={{{driver}}};"
            f"SERVER={server};DATABASE={dbName};"
            f"Trusted_Connection=yes;{extra}"
        )

    return "mssql+pyodbc:///?odbc_connect=" + quote_plus(odbc)


def db_engine(dbName: str):
    connStr = mk_conn_str(dbName)
    eng = create_engine(connStr, pool_timeout=30, pool_recycle=3600)
    try:
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception as ex:
        log.error("SQL Server connection failed: %s", ex)
        raise
    return eng


def split_table(full_name: str):
    if "." in full_name:
        schema, table = full_name.split(".", 1)
        return schema, table
    return "dbo", full_name


def find_count_col(prompt: str, cols):
    txt = (
        prompt.lower()
        .replace("(", " ")
        .replace(")", " ")
        .replace(",", " ")
    )

    colnames = [c["name"].lower() for c in cols]
    tokens = txt.split()

    null_words = {"null", "is null", "missing", "blank", "empty"}
    if any(k in txt for k in null_words):
        return None, False

    isDistinct = "distinct" in tokens

    if "count" in tokens:
        idx = tokens.index("count")
        lookahead = tokens[idx + 1 : idx + 4]

        joined = []
        for i in range(len(lookahead)):
            for j in range(i + 1, len(lookahead)):
                joined.append(lookahead[i] + lookahead[j])
                joined.append(lookahead[i] + "_" + lookahead[j])

        for token in lookahead + joined:
            if token in colnames:
                return token, isDistinct

    for col in colnames:
        if f"count of {col}" in txt:
            return col, isDistinct

    return None, isDistinct


def scrub_sql(sql: str, cols):
    typeMap = {c["name"].lower(): str(c["type"]).lower() for c in cols}
    match = re.search(r"\bwhere\b", sql, re.IGNORECASE)
    if not match:
        return sql

    before_where = sql[:match.end()]
    where_part = sql[match.end():].strip().rstrip(";")

    if where_part.lower() == "1=1":
        return sql

    parts = re.split(r"\s+OR\s+", where_part, flags=re.IGNORECASE)
    good = []

    for cond in parts:
        cond = cond.strip()
        if not cond:
            continue

        left = cond.split()[0]
        left = left.replace("LOWER(", "").replace("UPPER(", "").replace(")", "")
        left = left.split(".")[-1]
        left = left.strip("[]")
        col = left.lower()

        if col not in typeMap:
            continue

        colType = typeMap[col]

        if re.search(r"\bbetween\b|\blike\b|\bin\b", cond, re.IGNORECASE):
            good.append(cond)
            continue

        if "!=" in cond:
            right = cond.split("!=", 1)[1].strip().strip("'")
        elif "=" in cond:
            right = cond.split("=", 1)[1].strip().strip("'")
        else:
            good.append(cond)
            continue

        if any(t in colType for t in ["int", "decimal", "numeric", "float", "real", "money", "bigint", "smallint", "tinyint"]):
            if not right.replace(".", "", 1).isdigit():
                log.warning("Removed invalid numeric condition: %s", cond)
                continue

        if "date" in colType or "time" in colType:
            if not re.match(r"^\d{4}-\d{2}-\d{2}", right):
                log.warning("Removed invalid date condition: %s", cond)
                continue

        good.append(cond)

    if not good:
        return before_where + " 1=1;"

    return before_where + " " + " OR ".join(good) + ";"


def pick_where(sql: str) -> str | None:
    m = re.search(r"\bwhere\b", sql, re.IGNORECASE)
    if not m:
        return None
    clause = sql[m.start():].strip().rstrip(";")
    if not clause.lower().startswith("where"):
        clause = "WHERE " + clause
    return clause


def build_count_sql(prompt: str, db: str, schema: str, table: str, cols):
    count_col, isDistinct = find_count_col(prompt, cols)
    if count_col:
        col_ref = f"[{count_col}]"
        count_expr = f"COUNT(DISTINCT {col_ref})" if isDistinct else f"COUNT({col_ref})"
    else:
        count_expr = "COUNT(*)"

    base_tbl = tbl_ref(db, schema, table)
    base = f"SELECT {count_expr} AS row_count FROM {base_tbl}"

    client = llm_client()
    if not client:
        return base + " WHERE 1=1;"

    col_lines = "\n".join([f"- {c['name']} ({c['type']})" for c in cols])
    sys_msg = "You are an expert SQL Server data quality assistant."
    user_msg = f"""Create ONE SQL Server query based on the user prompt.

Base query:
{base}

Rules:
- Return ONLY SQL, end with semicolon.
- Use ONLY the columns listed.
- Do not change the base table or database.
- If no filter is implied, use WHERE 1=1.
- Strings must be single-quoted.
- For case-insensitive comparisons, use LOWER(column) and lowercase values.

Columns:
{col_lines}

User prompt:
{prompt}
"""

    try:
        resp = client.chat.completions.create(
            model=cfg("OPENAI_MODEL") or DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0,
            max_tokens=300
        )
        sql_out = resp.choices[0].message.content.strip()
        sql_out = sql_out.replace("```sql", "").replace("```", "").strip()
        sql_out = re.sub(r"\b(source|target)\.", "", sql_out, flags=re.IGNORECASE)
        where_clause = pick_where(sql_out) or "WHERE 1=1"
        full_sql = f"{base} {where_clause};"
        return scrub_sql(full_sql, cols)
    except Exception as ex:
        log.error("Query generation error: %s", ex)
        return base + " WHERE 1=1;"


def run_count(eng, sql: str) -> int:
    with eng.connect() as c:
        val = c.execute(text(sql)).scalar()
        return int(val or 0)


def basic_summary(src_cnt: int, tgt_cnt: int) -> str:
    if src_cnt == tgt_cnt:
        return f"Perfect match: both have {src_cnt} rows."
    diff = abs(src_cnt - tgt_cnt)
    pct = (diff / max(src_cnt, tgt_cnt)) * 100 if max(src_cnt, tgt_cnt) else 0
    status = "significant discrepancy" if pct > 5 else "minor difference"
    return f"Row count mismatch: {diff} rows difference ({pct:.1f}% {status})."


def llm_summary(prompt: str, src_cnt: int, tgt_cnt: int, src_sql: str, tgt_sql: str) -> str:
    client = llm_client()
    if not client:
        return basic_summary(src_cnt, tgt_cnt)

    diff = abs(src_cnt - tgt_cnt)
    pct = (diff / max(src_cnt, tgt_cnt)) * 100 if max(src_cnt, tgt_cnt) else 0

    sys_msg = "You are a data quality analyst. Write concise, actionable summaries."
    user_msg = f"""Write a 4-5 sentence validation report.

User prompt:
{prompt}

Source row count: {src_cnt}
Target row count: {tgt_cnt}
Difference: {diff}
Percent difference: {pct:.2f}%

Source query:
{src_sql}

Target query:
{tgt_sql}
"""

    try:
        resp = client.chat.completions.create(
            model=cfg("OPENAI_MODEL") or DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0,
            max_tokens=250
        )
        return resp.choices[0].message.content.strip()
    except Exception as ex:
        log.error("Summary generation error: %s", ex)
        return basic_summary(src_cnt, tgt_cnt)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/databases", methods=["GET"])
def get_dbs():
    try:
        eng = db_engine("master")
        query = "SELECT name FROM sys.databases WHERE database_id > 4 AND state = 0 ORDER BY name"
        with eng.connect() as c:
            rows = c.execute(text(query)).fetchall()
        return jsonify([r[0] for r in rows])
    except Exception as ex:
        log.error("Error fetching databases: %s", ex)
        return jsonify({"error": str(ex)}), 500


@app.route("/api/tables", methods=["POST"])
def tables_for_db():
    try:
        data = request.get_json() or {}
        dbName = data.get("database")
        if not dbName:
            return jsonify({"error": "database is required"}), 400

        eng = db_engine(dbName)
        insp = inspect(eng)
        tables = []
        for schema in insp.get_schema_names():
            for table in insp.get_table_names(schema=schema):
                tables.append(f"{schema}.{table}")
        tables.sort()
        return jsonify(tables)
    except Exception as ex:
        log.error("Error fetching tables: %s", ex)
        return jsonify({"error": str(ex)}), 500


@app.route("/api/validate", methods=["POST"])
def validate():
    try:
        data = request.get_json() or {}
        required = ["source_db", "target_db", "source_table", "target_table", "prompt"]
        for f in required:
            if not data.get(f):
                return jsonify({"error": f"Missing field: {f}"}), 400

        srcDb = data["source_db"]
        tgtDb = data["target_db"]
        srcTable = data["source_table"]
        tgtTable = data["target_table"]
        prompt = data["prompt"]

        src_eng = db_engine(srcDb)
        tgt_eng = db_engine(tgtDb)

        src_schema, src_name = split_table(srcTable)
        tgt_schema, tgt_name = split_table(tgtTable)

        src_cols = inspect(src_eng).get_columns(src_name, schema=src_schema)
        if not src_cols:
            return jsonify({"error": f"Source table '{srcTable}' not found in '{srcDb}'"}), 400

        tgt_cols = inspect(tgt_eng).get_columns(tgt_name, schema=tgt_schema)
        if not tgt_cols:
            return jsonify({"error": f"Target table '{tgtTable}' not found in '{tgtDb}'"}), 400

        src_sql = build_count_sql(prompt, srcDb, src_schema, src_name, src_cols)
        tgt_sql = build_count_sql(prompt, tgtDb, tgt_schema, tgt_name, tgt_cols)

        src_cnt = run_count(src_eng, src_sql)
        tgt_cnt = run_count(tgt_eng, tgt_sql)

        diff = abs(src_cnt - tgt_cnt)
        pct = (diff / max(src_cnt, tgt_cnt)) * 100 if max(src_cnt, tgt_cnt) else 0.0
        is_match = src_cnt == tgt_cnt

        summary = llm_summary(prompt, src_cnt, tgt_cnt, src_sql, tgt_sql)

        return jsonify({
            "result": {
                "source_db": srcDb,
                "target_db": tgtDb,
                "source_table": f"{src_schema}.{src_name}",
                "target_table": f"{tgt_schema}.{tgt_name}",
                "source_count": src_cnt,
                "target_count": tgt_cnt,
                "diff": diff,
                "pct_diff": round(pct, 2),
                "is_match": is_match,
                "source_query": src_sql,
                "target_query": tgt_sql,
                "summary": summary
            }
        })
    except Exception:
        log.error(traceback.format_exc())
        return jsonify({"error": "Unexpected server error"}), 500


@app.errorhandler(404)
def nf(e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def ie(e):
    return jsonify({"error": "Server error"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
