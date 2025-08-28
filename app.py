# app.py
import os
import sqlite3
import certifi
import warnings
import logging
import traceback
import json
import requests
import pandas as pd
import numpy as np
import re

from datetime import datetime
from urllib.parse import quote_plus
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from sqlalchemy import create_engine, inspect, text
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from urllib3.exceptions import InsecureRequestWarning
from dotenv import load_dotenv
from urllib.parse import quote_plus

# ─── CONFIG & LOGGING ─────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
CONFIG_DB  = os.path.join(BASE_DIR, "sample.db")
os.environ["SSL_CERT_FILE"] = certifi.where()
warnings.filterwarnings("ignore", category=InsecureRequestWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load env (you used ATT92742.env previously)
load_dotenv("ATT92742.env")

# ─── SQLITE/CONFIG BOOTSTRAP ──────────────────────────────────────────────────
def ensure_config_db(db_path: str = CONFIG_DB):
    """
    Create config table and seed default keys if missing.
    Safe to run every startup.
    """
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS config (
      key   TEXT PRIMARY KEY,
      value TEXT NOT NULL
    );
    """)
    # seed defaults (won't overwrite existing)
    cur.executemany("""
    INSERT INTO config(key, value)
    VALUES (?, ?)
    ON CONFLICT(key) DO NOTHING;
    """, [
      ("MSSQL_SERVER",    "localhost"),
      ("MSSQL_USER",      ""),
      ("MSSQL_PASS",      ""),
      ("MSSQL_DB",        "master"),
      ("LOG_DB",          "your_log_database_name"),
      ("HISTORY_DB_NAME", "master"),
      ("DB_MODE",         "sqlite"),   # 'sqlite' or 'mssql'
    ])
    conn.commit()
    conn.close()

def get_config_value(key: str, db_path: str = CONFIG_DB) -> str | None:
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute("SELECT value FROM config WHERE key = ?", (key,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None

# initialize config if missing
ensure_config_db()

def get_db_mode() -> str:
    # priority: config table -> env var -> default sqlite
    return (get_config_value("DB_MODE") or os.getenv("DB_MODE") or "sqlite").lower()

# ─── FLASK SETUP ───────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# ─── SAFE IDENTIFIERS (basic check to avoid injection in identifiers) ──────────
def _safe_identifier(name: str) -> str:
    """
    Allow only basic alphanumeric + underscore characters for table/database names.
    Raises ValueError for anything unsafe.
    """
    if not name or not re.match(r'^[A-Za-z0-9_]+$', name):
        raise ValueError("Invalid identifier. Only A-Z, a-z, 0-9 and underscore allowed.")
    return name



def _build_mssql_conn_str(db_name: str | None = None) -> str:
    # read server/creds from config or env (same as before)
    server = get_config_value("MSSQL_SERVER") or os.getenv("MSSQL_SERVER")
    if not server:
        raise ValueError("MSSQL_SERVER not set in config or environment for mssql mode")
    default_db = db_name or (get_config_value("MSSQL_DB") or os.getenv("MSSQL_DB") or "master")
    user = get_config_value("MSSQL_USER") or os.getenv("MSSQL_USER")
    pwd  = get_config_value("MSSQL_PASS") or os.getenv("MSSQL_PASS")

    # If you want to override driver name from env/config, you can add MSSQL_DRIVER key.
    driver_name = get_config_value("MSSQL_DRIVER") or os.getenv("MSSQL_DRIVER") or "ODBC Driver 17 for SQL Server"

    # Build the exact ODBC string (braces kept around driver)
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

    # URL-encode the whole ODBC string and return SQLAlchemy URI using odbc_connect
    return "mssql+pyodbc:///?odbc_connect=" + quote_plus(odbc_str)

def connect_to_server():
    """
    Returns a SQLAlchemy engine:
      - sqlite mode -> engine pointed at CONFIG_DB file
      - mssql  mode -> engine connected to MSSQL server 'master' (or configured default DB)
    """
    mode = get_db_mode()
    if mode == "mssql":
        conn_str = _build_mssql_conn_str("master")
        engine = create_engine(conn_str, pool_timeout=30, pool_recycle=3600)
        # basic smoke test
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception as e:
            logger.error(f"MS SQL connect test failed: {e}")
            raise
        logger.info(f"Connected to MSSQL server { (get_config_value('MSSQL_SERVER') or os.getenv('MSSQL_SERVER')) } (db=master)")
        return engine

    # default sqlite
    uri = f"sqlite:///{CONFIG_DB}"
    engine = create_engine(uri, connect_args={"check_same_thread": False})
    return engine

def connect_to_database(db_name: str):
    """
    Returns engine to a specific database:
      - sqlite: returns the same engine (we only use one file)
      - mssql: returns engine targetting the requested database name on the server
    """
    mode = get_db_mode()
    if mode == "mssql":
        # validate db_name
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

    # sqlite mode: single file
    return connect_to_server()

# ─── ROUTES ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    # If you have an index template this will render; otherwise you can return simple text
    try:
        return render_template("index.html")
    except Exception:
        return "Row count validation API is running."

@app.route("/api/databases", methods=["GET"])
def get_databases_endpoint():
    """
    For sqlite mode: return single logical DB name (filename).
    For mssql mode: return server databases.
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
            # sqlite -> return filename as logical DB
            return jsonify([os.path.basename(CONFIG_DB)])
    except Exception as e:
        logger.error(f"Error fetching databases: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/tables", methods=["POST"])
def get_tables_endpoint():
    """
    For sqlite: return list of tables in the single file.
    For mssql: given a database name (posted), return its table names.
    """
    try:
        data = request.get_json()
        if not data or "database" not in data:
            return jsonify({"error": "Database name is required"}), 400

        mode = get_db_mode()
        posted_db = data["database"]

        if mode == "mssql":
            # posted_db is expected to be the database name on the server
            engine = connect_to_database(posted_db)
            inspector = inspect(engine)
            return jsonify(inspector.get_table_names())
        else:
            # sqlite: list tables from sqlite_master
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
    Useful for UI to show column selection separately.
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

# ─── SQL HELPERS ─────────────────────────────────────────────────────────────
def generate_where_clause(prompt: str) -> str:
    """
    Uses Gemini (if API key present) to generate a WHERE clause. Fallback to WHERE 1=1.
    """
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return "WHERE 1=1"
        url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={api_key}"
        gemini_prompt = f"""
        You are an expert SQL assistant.
        Generate ONLY a valid SQL WHERE clause for this user condition:
        "{prompt}"
        MUST start with 'WHERE'. If unclear, return WHERE 1=1.
        """
        payload = {"contents": [{"parts": [{"text": gemini_prompt}]}]}
        resp = requests.post(url, headers={"Content-Type": "application/json"},
                             data=json.dumps(payload), timeout=10, verify=False)
        resp.raise_for_status()
        wc = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        return wc if wc.lower().startswith("where") else f"WHERE {wc}"
    except Exception as e:
        logger.debug(f"Gemini error: {e}")
        return "WHERE 1=1"


def get_row_count(engine, table_name, where_clause):
    """
    Use DB-specific quoting for table name:
      - sqlite: "table"
      - mssql : [table]
    """
    mode = get_db_mode()
    # validate table_name
    _safe_identifier(table_name)
    if mode == "mssql":
        q = f"SELECT COUNT(*) as count FROM [{table_name}] {where_clause}"
    else:
        q = f'SELECT COUNT(*) as count FROM "{table_name}" {where_clause}'
    with engine.connect() as conn:
        result = conn.execute(text(q))
        val = result.scalar()
        return int(val or 0)


def get_historical_data(source_db, source_table, target_db, target_table, where_clause, history_db='master'):
    """
    Fetch historical data. SQL differs across backends:
    - MS SQL: use TOP 5
    - SQLite: use LIMIT 5
    """
    try:
        engine = connect_to_database(history_db)
        mode = get_db_mode()
        if mode == "mssql":
            query = text(""" 
                SELECT TOP 5 SourceRowCount, TargetRowCount
                FROM ETL_RowCount_Log
                WHERE SourceDatabase = :source_db
                  AND SourceTable = :source_table
                  AND TargetDatabase = :target_db
                  AND TargetTable = :target_table
                  AND WHEREClause = :where_clause
                ORDER BY LogDate DESC;
            """)
        else:
            query = text("""
                SELECT SourceRowCount, TargetRowCount
                FROM ETL_RowCount_Log
                WHERE SourceDatabase = :source_db
                  AND SourceTable = :source_table
                  AND TargetDatabase = :target_db
                  AND TargetTable = :target_table
                  AND WHEREClause = :where_clause
                ORDER BY LogDate DESC
                LIMIT 5;
            """)

        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={
                'source_db': source_db,
                'source_table': source_table,
                'target_db': target_db,
                'target_table': target_table,
                'where_clause': where_clause
            })

        if not df.empty:
            logger.info(f"Fetched {len(df)} historical records for {source_table} -> {target_table}")
            return df['SourceRowCount'].tolist(), df['TargetRowCount'].tolist()
        else:
            return [], []

    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        return [], []


def log_validation_result(source_db, source_table, target_db, target_table, source_count, target_count, where_clause, is_anomaly, history_db='master'):
    try:
        engine = connect_to_database(history_db)
        query = text("""
            INSERT INTO ETL_RowCount_Log (SourceDatabase, SourceTable, TargetDatabase, TargetTable,
                SourceRowCount, TargetRowCount, WHEREClause, IsAnomaly)
            VALUES (:source_db, :source_table, :target_db, :target_table,
                    :source_count, :target_count, :where_clause, :is_anomaly)
        """)
        with engine.connect() as conn:
            conn.execute(query, {
                'source_db': source_db,
                'source_table': source_table,
                'target_db': target_db,
                'target_table': target_table,
                'source_count': source_count,
                'target_count': target_count,
                'where_clause': where_clause,
                'is_anomaly': int(bool(is_anomaly))
            })
            # commit for some dialects
            try:
                conn.commit()
            except Exception:
                pass
    except Exception as e:
        logger.error(f"Error logging validation result: {e}")


# ─── PREDICTION HELPERS & SUMMARY ──────────────────────────────────────────────
def simple_lstm_prediction(historical_data):
    if len(historical_data) < 5:
        return 0
    recent = np.array(historical_data[-5:])
    if np.std(recent) == 0:
        return int(recent[-1])
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(recent.reshape(-1, 1))
    trend = scaled[-1, 0] - scaled[-2, 0]
    pred_scaled = np.clip(scaled[-1, 0] + trend, 0, 1)
    return max(0, int(scaler.inverse_transform([[pred_scaled]])[0][0]))


def simple_arima_prediction(historical_data):
    try:
        if len(historical_data) < 5:
            return 0
        series = np.array(historical_data)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(series, order=(1, 1, 1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=1)[0]
            return max(0, int(forecast))
    except Exception as e:
        logger.error(f"ARIMA failed: {e}")
        return 0


def generate_simple_summary(source_count, target_count, is_anomaly):
    if source_count == target_count:
        return f"Perfect match: both have {source_count} rows."
    diff = abs(source_count - target_count)
    pct = (diff / max(source_count, target_count)) * 100 if max(source_count, target_count) else 0
    status = "significant discrepancy" if pct > 5 else "minor difference"
    anomaly = " Possible anomaly." if is_anomaly else ""
    return f"Row count mismatch: {diff} rows difference ({pct:.1f}% {status}).{anomaly}"


def generate_summary(source_count, target_count, where_clause, lstm_pred, arima_pred, is_anomaly):
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return generate_simple_summary(source_count, target_count, is_anomaly)

        url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={api_key}"
        prompt = f"""
        You are a data quality analyst. Analyze the following row count validation details and write a clear 4–5 sentence summary for a data engineering report:

        - Source Row Count: {source_count}
        - Target Row Count: {target_count}
        - Filter Applied (WHERE clause): {where_clause}
        - LSTM Predicted Target: {lstm_pred}
        - ARIMA Predicted Target: {arima_pred}
        - Anomaly Flag: {is_anomaly}
        """
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        response = requests.post(url, headers={"Content-Type": "application/json"},
                                 data=json.dumps(payload), timeout=10, verify=False)
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        logger.error(f"REST summary error: {e}")
        return generate_simple_summary(source_count, target_count, is_anomaly)


# ─── VALIDATE ENDPOINT ─────────────────────────────────────────────────────────
@app.route("/api/validate", methods=["POST"])
def validate():
    try:
        data = request.get_json()
        for field in ['source_db', 'target_db', 'source_table', 'target_table', 'prompt']:
            if not data.get(field):
                return jsonify({"error": f"Missing field: {field}"}), 400

        source_db = data['source_db']
        target_db = data['target_db']
        source_table = data['source_table']
        target_table = data['target_table']
        prompt = data['prompt']
        history_db = get_config_value("HISTORY_DB_NAME") or "master"

        where_clause = generate_where_clause(prompt)
        src_engine = connect_to_database(source_db)
        tgt_engine = connect_to_database(target_db)

        src_count = get_row_count(src_engine, source_table, where_clause)
        tgt_count = get_row_count(tgt_engine, target_table, where_clause)

        hist_src, hist_tgt = get_historical_data(source_db, source_table, target_db, target_table, where_clause, history_db)

        lstm_pred = simple_lstm_prediction(hist_tgt)
        arima_pred = simple_arima_prediction(hist_tgt)

        if len(hist_tgt) < 10:
            is_anomaly = abs(src_count - tgt_count) > 0.1 * max(src_count, tgt_count, 1)
        else:
            is_anomaly = IsolationForest(contamination=0.05).fit(np.array(hist_tgt).reshape(-1, 1)).predict([[tgt_count]])[0] == -1

        summary = generate_summary(src_count, tgt_count, where_clause, lstm_pred, arima_pred, is_anomaly)
        log_validation_result(source_db, source_table, target_db, target_table, src_count, tgt_count, where_clause, is_anomaly, history_db)

        return jsonify({
            "source_count": src_count,
            "target_count": tgt_count,
            "where_clause": where_clause,
            "lstm_prediction": lstm_pred,
            "arima_prediction": arima_pred,
            "is_anomaly": is_anomaly,
            "summary": summary
        })

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Validation failed: {str(e)}"}), 500


# ─── ERROR HANDLERS & MAIN ─────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    # quick preflight: ensure DB_MODE exists in config
    logger.info(f"Starting app in DB_MODE={get_db_mode()}")
    if get_db_mode() == "mssql" and not (get_config_value("MSSQL_SERVER") or os.getenv("MSSQL_SERVER")):
        print("Please set MSSQL_SERVER in config table or as environment variable for mssql mode.")
    else:
        app.run(debug=True, host='0.0.0.0', port=5000)
