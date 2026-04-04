"""
database.py — SQLite setup + table creation
Smart Traffic Detection System
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "traffic.db")

# ─────────────────────────────────────────────
#  CONNECTION
# ─────────────────────────────────────────────
def get_connection():
    """Get a SQLite connection with row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # allows dict-like access
    conn.execute("PRAGMA journal_mode=WAL")  # better concurrent access
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

# ─────────────────────────────────────────────
#  CREATE TABLES
# ─────────────────────────────────────────────
def init_db():
    """Create all tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()

    # ── Users table ──────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT    NOT NULL UNIQUE,
            email         TEXT    NOT NULL UNIQUE,
            password_hash TEXT    NOT NULL,
            role          TEXT    NOT NULL DEFAULT 'user',
            created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ── Detections table ─────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp      DATETIME DEFAULT CURRENT_TIMESTAMP,
            status         TEXT NOT NULL,
            vehicle_count  INTEGER NOT NULL DEFAULT 0,
            accident_class TEXT    NOT NULL DEFAULT 'none',
            confidence     REAL    NOT NULL DEFAULT 0.0
        )
    """)

    # ── Daily stats table ────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_stats (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            date              DATE    NOT NULL UNIQUE,
            total_detections  INTEGER DEFAULT 0,
            total_accidents   INTEGER DEFAULT 0,
            total_severe      INTEGER DEFAULT 0,
            total_moderate    INTEGER DEFAULT 0,
            total_congestion  INTEGER DEFAULT 0,
            avg_vehicle_count REAL    DEFAULT 0.0,
            updated_at        DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ── Indexes for faster queries ────────────
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_detections_timestamp
        ON detections (timestamp)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_detections_status
        ON detections (status)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_daily_stats_date
        ON daily_stats (date)
    """)

    conn.commit()
    conn.close()
    print("[DB] ✅ Database initialized successfully!")
    print(f"[DB] 📁 Database path: {os.path.abspath(DB_PATH)}")

# ─────────────────────────────────────────────
#  DETECTION QUERIES
# ─────────────────────────────────────────────
def insert_detection(status, vehicle_count, accident_class, confidence):
    """Insert a new detection record."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO detections (status, vehicle_count, accident_class, confidence)
        VALUES (?, ?, ?, ?)
    """, (status, vehicle_count, accident_class, round(confidence, 4)))
    detection_id = cursor.lastrowid
    conn.commit()
    conn.close()

    # Update daily stats after every insert
    update_daily_stats()
    return detection_id

def get_latest_detection():
    """Get the most recent detection."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM detections
        ORDER BY timestamp DESC
        LIMIT 1
    """)
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_detections(page=1, per_page=20, status_filter=None, date_filter=None):
    """Get paginated detections with optional filters."""
    conn = get_connection()
    cursor = conn.cursor()

    conditions = []
    params = []

    if status_filter and status_filter != "ALL":
        conditions.append("status LIKE ?")
        params.append(f"%{status_filter}%")

    if date_filter:
        conditions.append("DATE(timestamp) = ?")
        params.append(date_filter)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    # Total count
    cursor.execute(f"SELECT COUNT(*) FROM detections {where}", params)
    total = cursor.fetchone()[0]

    # Paginated results
    offset = (page - 1) * per_page
    params.extend([per_page, offset])
    cursor.execute(f"""
        SELECT * FROM detections {where}
        ORDER BY timestamp DESC
        LIMIT ? OFFSET ?
    """, params)

    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()

    return {
        "data": rows,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page
    }

def get_today_detections():
    """Get all detections from today."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM detections
        WHERE DATE(timestamp) = DATE('now')
        ORDER BY timestamp DESC
    """)
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows

def get_hourly_stats():
    """Get detection counts grouped by hour for today."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            strftime('%H', timestamp) AS hour,
            COUNT(*) AS total,
            SUM(CASE WHEN status LIKE '%ACCIDENT%' THEN 1 ELSE 0 END) AS accidents,
            SUM(CASE WHEN status = 'CONGESTION' THEN 1 ELSE 0 END) AS congestions,
            SUM(CASE WHEN status = 'NORMAL' THEN 1 ELSE 0 END) AS normals,
            AVG(vehicle_count) AS avg_vehicles
        FROM detections
        WHERE DATE(timestamp) = DATE('now')
        GROUP BY hour
        ORDER BY hour
    """)
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows

def get_daily_stats_range(days=7):
    """Get daily stats for the last N days."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM daily_stats
        WHERE date >= DATE('now', ?)
        ORDER BY date ASC
    """, (f"-{days} days",))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows

def get_summary_stats():
    """Get overall summary statistics."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            COUNT(*) AS total_detections,
            SUM(CASE WHEN status LIKE '%ACCIDENT%' THEN 1 ELSE 0 END) AS total_accidents,
            SUM(CASE WHEN status LIKE '%SEVERE%' THEN 1 ELSE 0 END) AS total_severe,
            SUM(CASE WHEN status LIKE '%MODERATE%' THEN 1 ELSE 0 END) AS total_moderate,
            SUM(CASE WHEN status = 'CONGESTION' THEN 1 ELSE 0 END) AS total_congestion,
            SUM(CASE WHEN status = 'NORMAL' THEN 1 ELSE 0 END) AS total_normal,
            AVG(vehicle_count) AS avg_vehicle_count,
            MAX(vehicle_count) AS max_vehicle_count
        FROM detections
    """)
    row = cursor.fetchone()

    # Today's count
    cursor.execute("""
        SELECT COUNT(*) AS today_total
        FROM detections
        WHERE DATE(timestamp) = DATE('now')
    """)
    today = cursor.fetchone()

    conn.close()

    result = dict(row) if row else {}
    result["today_total"] = today["today_total"] if today else 0
    return result

# ─────────────────────────────────────────────
#  DAILY STATS UPDATE
# ─────────────────────────────────────────────
def update_daily_stats():
    """Recalculate and upsert today's daily stats."""
    conn = get_connection()
    cursor = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d")

    cursor.execute("""
        SELECT
            COUNT(*) AS total_detections,
            SUM(CASE WHEN status LIKE '%ACCIDENT%' THEN 1 ELSE 0 END) AS total_accidents,
            SUM(CASE WHEN status LIKE '%SEVERE%' THEN 1 ELSE 0 END) AS total_severe,
            SUM(CASE WHEN status LIKE '%MODERATE%' THEN 1 ELSE 0 END) AS total_moderate,
            SUM(CASE WHEN status = 'CONGESTION' THEN 1 ELSE 0 END) AS total_congestion,
            AVG(vehicle_count) AS avg_vehicle_count
        FROM detections
        WHERE DATE(timestamp) = ?
    """, (today,))

    row = cursor.fetchone()
    if row:
        cursor.execute("""
            INSERT INTO daily_stats
                (date, total_detections, total_accidents, total_severe,
                 total_moderate, total_congestion, avg_vehicle_count, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(date) DO UPDATE SET
                total_detections  = excluded.total_detections,
                total_accidents   = excluded.total_accidents,
                total_severe      = excluded.total_severe,
                total_moderate    = excluded.total_moderate,
                total_congestion  = excluded.total_congestion,
                avg_vehicle_count = excluded.avg_vehicle_count,
                updated_at        = CURRENT_TIMESTAMP
        """, (
            today,
            row["total_detections"],
            row["total_accidents"],
            row["total_severe"],
            row["total_moderate"],
            row["total_congestion"],
            round(row["avg_vehicle_count"] or 0, 2)
        ))
        conn.commit()
    conn.close()

# ─────────────────────────────────────────────
#  USER QUERIES
# ─────────────────────────────────────────────
def create_user(username, email, password_hash, role="user"):
    """Insert a new user."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO users (username, email, password_hash, role)
            VALUES (?, ?, ?, ?)
        """, (username, email, password_hash, role))
        user_id = cursor.lastrowid
        conn.commit()
        return user_id
    except sqlite3.IntegrityError as e:
        raise ValueError(f"User already exists: {e}")
    finally:
        conn.close()

def get_user_by_email(email):
    """Get user by email."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_user_by_username(username):
    """Get user by username."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

# ─────────────────────────────────────────────
#  RUN DIRECTLY TO INITIALIZE
# ─────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
