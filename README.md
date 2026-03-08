# RedGate Row Count Validator

A Flask app that compares row counts between two SQL Server tables using a natural language prompt. It uses an LLM to generate SQL filters and to produce a short validation report.

Quick start
1. Set environment variables: `OPENAI_API_KEY`, `MSSQL_SERVER` (example: `localhost/SQLExpress`), optional `MSSQL_DRIVER`, `MSSQL_USER`, `MSSQL_PASS`, `MSSQL_EXTRA_PARAMS`, `OPENAI_MODEL`.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the app: `python app.py`.
4. Open `http://localhost:5000` in your browser.

Notes
- Windows authentication is used when `MSSQL_USER` and `MSSQL_PASS` are not set.
- Default database is `AdventureWorks2022`.
