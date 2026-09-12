import sqlite3
import json
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base.db")


def get_connection():
    """Veritabanı bağlantısı oluşturur."""
    return sqlite3.connect(DB_PATH)


def init_db(conn=None):
    """Tabloları hazırlar."""
    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True

    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            content TEXT,
            embedding TEXT
        )
    """)
    conn.commit()
    if should_close:
        conn.close()


def clear_db(conn=None):
    """Veritabanını temizler."""
    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True

    cursor = conn.cursor()
    cursor.execute("DELETE FROM chunks")
    conn.commit()
    if should_close:
        conn.close()


def save_chunk(conn, source: str, content: str, embedding: list):
    """Parçayı ve embedding vektörünü kaydeder."""
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chunks (source, content, embedding) VALUES (?, ?, ?)",
        (source, content, json.dumps(embedding))
    )
    conn.commit()


def count_chunks(conn) -> int:
    """Toplam parça sayısını döndürür."""
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM chunks")
    row = cursor.fetchone()
    return row[0] if row else 0


def fetch_all_chunks(conn) -> list[dict]:
    """Tüm parçaları ve embedding'leri çeker.

    retrieve.py dict-style erişim bekliyor (chunk["source"], chunk["embedding"]
    gibi) ve embedding'in JSON string değil, gerçek float listesi olmasını
    istiyor - o yüzden burada id de çekilip embedding decode ediliyor.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT id, source, content, embedding FROM chunks")
    rows = cursor.fetchall()
    return [
        {"id": row[0], "source": row[1], "content": row[2], "embedding": json.loads(row[3])}
        for row in rows
    ]
