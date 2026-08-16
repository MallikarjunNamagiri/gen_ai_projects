import sqlite3
conn = sqlite3.connect("db/workflow.db")
print([r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()])