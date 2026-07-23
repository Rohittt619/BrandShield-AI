import sqlite3
import os
from pathlib import Path

class BrandShieldDB:
    """
    SQLite Database Manager for BrandShield-AI.
    Persists counterfeit inspection logs, threat alerts, and brand analytics.
    """

    def __init__(self):
        db_dir = Path(__file__).resolve().parent
        os.makedirs(db_dir, exist_ok=True)
        db_path = db_dir / "brandshield.db"

        self.conn = sqlite3.connect(
            str(db_path),
            check_same_thread=False
        )

        self.cursor = self.conn.cursor()

        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS inspections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            brand TEXT,
            verdict TEXT,
            score REAL,
            threat_level TEXT,
            edge_density REAL,
            keypoints_count INTEGER,
            source_type TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        self.conn.commit()

    def save_inspection(self, brand: str, verdict: str, score: float, threat_level: str, edge_density: float, keypoints_count: int, source_type: str = "File Upload"):
        self.cursor.execute("""
        INSERT INTO inspections (brand, verdict, score, threat_level, edge_density, keypoints_count, source_type)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (brand, verdict, score, threat_level, edge_density, keypoints_count, source_type))
        self.conn.commit()

    def fetch_inspections(self, limit: int = 50):
        self.cursor.execute("""
        SELECT id, brand, verdict, score, threat_level, edge_density, keypoints_count, source_type, created_at
        FROM inspections
        ORDER BY id DESC
        LIMIT ?
        """, (limit,))
        return self.cursor.fetchall()

    def fetch_stats(self):
        self.cursor.execute("SELECT COUNT(*) FROM inspections")
        total = self.cursor.fetchone()[0]

        self.cursor.execute("SELECT COUNT(*) FROM inspections WHERE verdict LIKE '%COUNTERFEIT%' OR verdict LIKE '%ALTERED%'")
        counterfeits = self.cursor.fetchone()[0]

        self.cursor.execute("SELECT COUNT(*) FROM inspections WHERE verdict LIKE '%AUTHENTIC%'")
        authentics = self.cursor.fetchone()[0]

        return {
            "total_inspections": total,
            "counterfeits_blocked": counterfeits,
            "authentics_verified": authentics
        }
