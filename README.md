
# Flask + MSSQL App

This project is a Flask web application that connects to a Microsoft SQL Server database.  
It uses `pyodbc` and SQLAlchemy to fetch data from SQL Server.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv venv
# Windows PowerShell
venv\Scripts\Activate.ps1
# Command Prompt
venv\Scripts\activate.bat
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the project root:
```env
DB_MODE=mssql
SQL_SERVER=DESKTOP-GSJB5GJ
SQL_DATABASE=master
SQL_USERNAME=
SQL_PASSWORD=
FLASK_ENV=development
```

If using **Trusted Connection**, leave `SQL_USERNAME` and `SQL_PASSWORD` blank.

---

## 📂 Sample Databases

This app is designed to work with sample SQL Server databases like:
1. **AdventureWorks**
   - [Download AdventureWorks](https://learn.microsoft.com/en-us/sql/samples/adventureworks-install-configure?view=sql-server-ver17&tabs=ssms)
2. **Wide World Importers**
   - [Download Wide World Importers](https://github.com/microsoft/sql-server-samples/releases/tag/wide-world-importers-v1.0?utm_source=chatgpt.com)

### 🔹 Restore Databases

After downloading the `.bak` files, follow these steps:

1. Open **SQL Server Management Studio (SSMS)**.
2. Connect to your SQL Server instance.
3. Right-click **Databases > Restore Database...**
4. Select **Device**, click **Add**, and choose the `.bak` file from your **Downloads** folder.
5. Click **OK** to restore.

Once restored, you’ll have AdventureWorks and WideWorldImporters available.

---

## ▶️ Run the App

```bash
python app.py
```

You should see output like:
```
 * Running on http://127.0.0.1:5000
```

Visit the URL in your browser.

---

## 🛑 Notes
- This is a **development server**. For production, use a WSGI server like Gunicorn or uWSGI.
- Make sure `ODBC Driver 17 for SQL Server` is installed.
- Keep your `.env` file private and never commit it to Git.

---

## 🗑️ .gitignore Suggestions
```
venv/
__pycache__/
*.pyc
*.pyo
.env
*.db
*.bak
instance/
```
