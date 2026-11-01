import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).parent / "sensor_data.db"


def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP,
            IR REAL, RED REAL, HR REAL, SpO2 REAL,
            status TEXT, is_abnormal INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reading_id INTEGER, model TEXT,
            IR REAL, RED REAL, HR REAL, SpO2 REAL,
            status TEXT, ts DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reading_id INTEGER, message TEXT,
            sent INTEGER DEFAULT 0, ts DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            username TEXT NOT NULL UNIQUE COLLATE NOCASE,
            email TEXT NOT NULL UNIQUE COLLATE NOCASE,
            password_hash TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_login DATETIME
        );
        CREATE TABLE IF NOT EXISTS explanations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reading_id INTEGER, model_used TEXT, narrative TEXT,
            contributions TEXT, causes TEXT, remedies TEXT,
            severity TEXT, ts DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """)
        # migration: per-user language (en / kn)
        cols = [r[1] for r in con.execute("PRAGMA table_info(users)")]
        if "lang" not in cols:
            con.execute("ALTER TABLE users ADD COLUMN lang TEXT DEFAULT 'en'")


@contextmanager
def db():
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def insert_reading(IR, RED, HR, SpO2, status, is_abnormal):
    with db() as con:
        cur = con.execute(
            "INSERT INTO readings (IR,RED,HR,SpO2,status,is_abnormal) VALUES (?,?,?,?,?,?)",
            (IR, RED, HR, SpO2, status, int(is_abnormal)))
        return cur.lastrowid


def insert_predictions(reading_id, rows):
    with db() as con:
        con.executemany(
            "INSERT INTO predictions (reading_id,model,IR,RED,HR,SpO2,status) "
            "VALUES (?,?,?,?,?,?,?)",
            [(reading_id, r["Model"], r["IR"], r["RED"], r["HR"], r["SpO2"], r["Status"])
             for r in rows])


def insert_alert(reading_id, message, sent=1):
    with db() as con:
        con.execute(
            "INSERT INTO alerts (reading_id,message,sent) VALUES (?,?,?)",
            (reading_id, message, sent))


def insert_explanation(reading_id, xai: dict):
    import json
    with db() as con:
        con.execute(
            """INSERT INTO explanations
               (reading_id, model_used, narrative, contributions, causes, remedies, severity)
               VALUES (?,?,?,?,?,?,?)""",
            (reading_id, xai.get("model_used"), xai.get("narrative"),
             json.dumps(xai.get("contributions", [])),
             json.dumps(xai.get("causes", [])),
             json.dumps(xai.get("remedies", [])),
             xai.get("severity", "info")))


def recent_readings(limit=200):
    with db() as con:
        return [dict(r) for r in con.execute(
            "SELECT * FROM readings ORDER BY id DESC LIMIT ?", (limit,))][::-1]


def recent_predictions(limit=200):
    with db() as con:
        return [dict(r) for r in con.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,))][::-1]


def recent_alerts(limit=50):
    with db() as con:
        return [dict(r) for r in con.execute(
            "SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,))]


def recent_explanations(limit=100):
    with db() as con:
        return [dict(r) for r in con.execute(
            "SELECT * FROM explanations ORDER BY id DESC LIMIT ?", (limit,))]


def model_performance():
    with db() as con:
        rows = con.execute("""
        SELECT p.model,
               AVG(ABS(p.IR-r.IR))     AS mae_ir,
               AVG(ABS(p.RED-r.RED))   AS mae_red,
               AVG(ABS(p.HR-r.HR))     AS mae_hr,
               AVG(ABS(p.SpO2-r.SpO2)) AS mae_spo2,
               COUNT(*) as n
        FROM predictions p JOIN readings r ON p.reading_id=r.id
        GROUP BY p.model
        """).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Users (login / register)
# ---------------------------------------------------------------------------
def create_user(full_name, username, email, password_hash, lang="en"):
    """Returns new user id, or raises sqlite3.IntegrityError on duplicate."""
    with db() as con:
        cur = con.execute(
            "INSERT INTO users (full_name, username, email, password_hash, lang) VALUES (?,?,?,?,?)",
            (full_name, username, email, password_hash, lang))
        return cur.lastrowid


def set_user_lang(user_id, lang):
    with db() as con:
        con.execute("UPDATE users SET lang = ? WHERE id = ?", (lang, user_id))


def latest_active_user():
    """The most recently logged-in user - receives alerts raised by the sensor stream."""
    with db() as con:
        row = con.execute(
            "SELECT * FROM users WHERE last_login IS NOT NULL "
            "ORDER BY last_login DESC, id DESC LIMIT 1").fetchone()
        return dict(row) if row else None


def get_user_by_login(identifier):
    """Look up by username OR email (case-insensitive)."""
    with db() as con:
        row = con.execute(
            "SELECT * FROM users WHERE username = ? OR email = ?",
            (identifier, identifier)).fetchone()
        return dict(row) if row else None


def get_user(user_id):
    with db() as con:
        row = con.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None


def touch_login(user_id):
    with db() as con:
        con.execute("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?", (user_id,))


def stats_summary():
    """Small numbers for the home page / monitor header."""
    with db() as con:
        r = con.execute("SELECT COUNT(*) FROM readings").fetchone()[0]
        a = con.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
        p = con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    return {"readings": r, "alerts": a, "predictions": p}
