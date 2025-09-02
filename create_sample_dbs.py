# create_sample_dbs.py
"""
One-time script to create SampleDB1 and SampleDB2 on a SQL Server,
create Employees and Orders tables, and insert a few sample rows.

Usage:
    pip install python-dotenv pyodbc
    (ensure an ODBC Driver for SQL Server is installed, e.g. "ODBC Driver 17 for SQL Server")

    # Option A: use ATT92742.env (or .env)
    python create_sample_dbs.py --env ATT92742.env

    # Option B: use environment variables
    export MSSQL_SERVER='localhost\\SQLEXPRESS'
    export MSSQL_USER='sa'            # omit or leave empty to use Trusted Connection
    export MSSQL_PASS='YourPass'
    python create_sample_dbs.py

You can also pass custom db names:
    python create_sample_dbs.py --dbs MyTestDB1 MyTestDB2
"""
import os
import argparse
import pyodbc
from dotenv import load_dotenv
import sys
import textwrap

DEFAULT_DRIVER = "ODBC Driver 17 for SQL Server"

CREATE_DB_SQL_TPL = """
IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = N'{db}')
BEGIN
    CREATE DATABASE [{db}];
END
"""

CREATE_EMP_SQL = """
IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[Employees]') AND type in (N'U'))
BEGIN
    CREATE TABLE dbo.Employees (
        Id INT IDENTITY(1,1) PRIMARY KEY,
        Name NVARCHAR(100) NOT NULL,
        Department NVARCHAR(100) NULL,
        CreatedAt DATETIME2 DEFAULT SYSUTCDATETIME()
    );
END
"""

CREATE_ORDERS_SQL = """
IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[Orders]') AND type in (N'U'))
BEGIN
    CREATE TABLE dbo.Orders (
        Id INT IDENTITY(1,1) PRIMARY KEY,
        EmployeeId INT NOT NULL,
        Amount DECIMAL(10,2) NOT NULL,
        OrderDate DATETIME2 DEFAULT SYSUTCDATETIME(),
        CONSTRAINT FK_Orders_Employees FOREIGN KEY(EmployeeId) REFERENCES dbo.Employees(Id)
    );
END
"""

SEED_EMP_SQL = """
IF NOT EXISTS (SELECT 1 FROM dbo.Employees)
BEGIN
    INSERT INTO dbo.Employees (Name, Department) VALUES
    (N'John Doe', N'Engineering'),
    (N'Jane Smith', N'Finance'),
    (N'Alice Kumar', N'Analytics');
END
"""

SEED_ORDERS_SQL = """
IF NOT EXISTS (SELECT 1 FROM dbo.Orders)
BEGIN
    INSERT INTO dbo.Orders (EmployeeId, Amount) VALUES
    (1, 123.45),
    (1, 67.89),
    (2, 250.00);
END
"""

def build_pyodbc_conn_str(server: str, driver: str, user: str | None, pwd: str | None, database: str = "master") -> str:
    # Use Trusted_Connection if user/pwd not provided
    drv = driver or DEFAULT_DRIVER
    if user and pwd:
        return f"DRIVER={{{drv}}};SERVER={server};DATABASE={database};UID={user};PWD={pwd}"
    else:
        return f"DRIVER={{{drv}}};SERVER={server};DATABASE={database};Trusted_Connection=yes"

def create_database(master_conn, db_name: str):
    sql = CREATE_DB_SQL_TPL.format(db=db_name)
    master_conn.execute(sql)

def create_tables_and_seed(db_conn):
    db_conn.execute(CREATE_EMP_SQL)
    db_conn.execute(CREATE_ORDERS_SQL)
    db_conn.execute(SEED_EMP_SQL)
    db_conn.execute(SEED_ORDERS_SQL)

def main():
    parser = argparse.ArgumentParser(description="Create sample SQL Server DBs and seed data.")
    parser.add_argument("--env", help="Path to .env file (optional). If provided, will load it.", default=None)
    parser.add_argument("--dbs", nargs=2, metavar=("DB1", "DB2"), help="Two database names to create", default=["SampleDB1", "SampleDB2"])
    parser.add_argument("--driver", help=f"ODBC driver name (default: {DEFAULT_DRIVER})", default=DEFAULT_DRIVER)
    args = parser.parse_args()

    if args.env:
        if not os.path.exists(args.env):
            print(f"Env file {args.env} not found.", file=sys.stderr)
            sys.exit(1)
        load_dotenv(args.env)
    else:
        # try loading default ATT92742.env if present (keeps UX smooth)
        if os.path.exists("ATT92742.env"):
            load_dotenv("ATT92742.env")

    server = os.getenv("MSSQL_SERVER")
    user = os.getenv("MSSQL_USER")
    pwd  = os.getenv("MSSQL_PASS")
    driver = args.driver

    if not server:
        print("MSSQL_SERVER is not set. Provide via --env or environment variable MSSQL_SERVER.", file=sys.stderr)
        sys.exit(1)

    # Basic validation for DB names (alphanumeric + underscore)
    def safe_name(n: str) -> str:
        if not n.isidentifier():
            raise ValueError(f"Invalid database name '{n}'. Use only letters, digits, and underscore, and don't start with digit.")
        return n

    db1, db2 = args.dbs
    try:
        db1 = safe_name(db1)
        db2 = safe_name(db2)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    # Connect to master to create DBs
    master_conn_str = build_pyodbc_conn_str(server, driver, user, pwd, database="master")
    print("Connecting to server (master) using driver:", driver)
    try:
        master_cn = pyodbc.connect(master_conn_str, autocommit=True)
    except Exception as e:
        print("Failed to connect to SQL Server. Error:", e, file=sys.stderr)
        # Helpful hint:
        print(textwrap.dedent("""
            - Check ODBC drivers with:
                python -c "import pyodbc; print(pyodbc.drivers())"
            - If driver not present, install Microsoft ODBC Driver for SQL Server.
            - If using a named instance, use server like MACHINE\\SQLEXPRESS or host,port
            - For Windows Integrated Auth (Trusted_Connection), omit MSSQL_USER/MSSQL_PASS
        """), file=sys.stderr)
        sys.exit(1)

    try:
        print(f"Ensuring database '{db1}' exists...")
        create_database(master_cn, db1)
        print(f"Ensuring database '{db2}' exists...")
        create_database(master_cn, db2)
    except Exception as e:
        print("Error creating databases:", e, file=sys.stderr)
        master_cn.close()
        sys.exit(1)

    # Connect to each DB and create tables + seed
    for db in (db1, db2):
        print(f"Connecting to database {db} to create tables and seed data...")
        conn_str = build_pyodbc_conn_str(server, driver, user, pwd, database=db)
        try:
            cn = pyodbc.connect(conn_str, autocommit=True)
            create_tables_and_seed(cn)
            cn.close()
            print(f"Created tables and seeded data in {db}")
        except Exception as e:
            print(f"Error setting up {db}: {e}", file=sys.stderr)

    master_cn.close()
    print("All done. The databases were created/seeded. You can now connect from your app.py using the usual mssql+pyodbc connection string.")

if __name__ == "__main__":
    main()
