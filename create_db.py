# create_db.py
import sqlite3

conn = sqlite3.connect("university.db")
cur = conn.cursor()

cur.execute("CREATE TABLE IF NOT EXISTS departments (id INTEGER PRIMARY KEY, name TEXT)")
cur.execute("CREATE TABLE IF NOT EXISTS students (id INTEGER PRIMARY KEY, name TEXT, gpa REAL, department_id INTEGER)")

cur.execute("INSERT INTO departments (name) VALUES ('Computer Science'), ('Biology'), ('Mathematics')")
cur.execute("INSERT INTO students (name, gpa, department_id) VALUES ('Alice', 3.9, 1), ('Bob', 3.5, 2), ('Charlie', 3.7, 1), ('Diana', 3.8, 3), ('Eve', 3.6, 2)")

conn.commit()
conn.close()