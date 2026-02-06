"""
SQLite-based request logging for analytics and debugging.
"""
import sqlite3
import threading
import json
from typing import Optional, Dict, Any
from datetime import datetime
from config import config
import os


class RequestLogger:
    """Thread-safe request logger with SQLite backend."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(RequestLogger, cls).__new__(cls)
                    cls._instance._initialize_db()
        return cls._instance
    
    def _initialize_db(self):
        """Initialize the SQLite database."""
        db_dir = os.path.dirname(config.REQUEST_LOG_DB_PATH)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        
        self.db_path = config.REQUEST_LOG_DB_PATH
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                request_id TEXT,
                user_ip TEXT,
                query TEXT,
                intent TEXT,
                extracted_entities TEXT,
                retrieval_count INTEGER,
                latency_ms REAL,
                status TEXT,
                error_message TEXT,
                context_sources TEXT
            )
        """)
        
        # Create indices for common queries
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON requests(timestamp)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_intent ON requests(intent)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON requests(status)")
        
        self.conn.commit()
    
    def log_request(
        self,
        request_id: str,
        query: str,
        intent: Optional[str] = None,
        extracted_entities: Optional[Dict] = None,
        retrieval_count: int = 0,
        latency_ms: float = 0,
        status: str = "success",
        error_message: Optional[str] = None,
        context_sources: Optional[list] = None,
        user_ip: Optional[str] = None
    ):
        """Log a request to the database."""
        try:
            with self._lock:
                self.conn.execute("""
                    INSERT INTO requests (
                        timestamp, request_id, user_ip, query, intent,
                        extracted_entities, retrieval_count, latency_ms,
                        status, error_message, context_sources
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(),
                    request_id,
                    user_ip or 'unknown',
                    query[:500],  # Truncate long queries
                    intent,
                    json.dumps(extracted_entities) if extracted_entities else None,
                    retrieval_count,
                    latency_ms,
                    status,
                    error_message,
                    json.dumps(context_sources[:10]) if context_sources else None  # Limit to 10 sources
                ))
                self.conn.commit()
        except Exception as e:
            # Don't let logging errors break the application
            print(f"[RequestLogger] Failed to log request: {e}")
    
    def get_recent_requests(self, limit: int = 100) -> list:
        """Get recent requests."""
        try:
            with self._lock:
                cursor = self.conn.execute("""
                    SELECT timestamp, request_id, query, intent, latency_ms, status
                    FROM requests
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (limit,))
                
                return [
                    {
                        "timestamp": row[0],
                        "request_id": row[1],
                        "query": row[2],
                        "intent": row[3],
                        "latency_ms": row[4],
                        "status": row[5]
                    }
                    for row in cursor.fetchall()
                ]
        except Exception as e:
            print(f"[RequestLogger] Failed to fetch requests: {e}")
            return []
    
    def get_intent_distribution(self, hours: int = 24) -> Dict[str, int]:
        """Get intent distribution for the last N hours."""
        try:
            with self._lock:
                from datetime import timedelta
                cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
                
                cursor = self.conn.execute("""
                    SELECT intent, COUNT(*) as count
                    FROM requests
                    WHERE timestamp > ?
                    GROUP BY intent
                """, (cutoff,))
                
                return {row[0]: row[1] for row in cursor.fetchall()}
        except Exception as e:
            print(f"[RequestLogger] Failed to get intent distribution: {e}")
            return {}
    
    def get_error_rate(self, hours: int = 24) -> float:
        """Get error rate for the last N hours."""
        try:
            with self._lock:
                from datetime import timedelta
                cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
                
                cursor = self.conn.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as errors
                    FROM requests
                    WHERE timestamp > ?
                """, (cutoff,))
                
                row = cursor.fetchone()
                total, errors = row[0], row[1]
                
                return errors / max(1, total) if total > 0 else 0.0
        except Exception as e:
            print(f"[RequestLogger] Failed to get error rate: {e}")
            return 0.0
    
    def get_average_latency(self, intent: Optional[str] = None, hours: int = 24) -> float:
        """Get average latency, optionally filtered by intent."""
        try:
            with self._lock:
                from datetime import timedelta
                cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
                
                if intent:
                    cursor = self.conn.execute("""
                        SELECT AVG(latency_ms)
                        FROM requests
                        WHERE timestamp > ? AND intent = ?
                    """, (cutoff, intent))
                else:
                    cursor = self.conn.execute("""
                        SELECT AVG(latency_ms)
                        FROM requests
                        WHERE timestamp > ?
                    """, (cutoff,))
                
                result = cursor.fetchone()[0]
                return result if result is not None else 0.0
        except Exception as e:
            print(f"[RequestLogger] Failed to get average latency: {e}")
            return 0.0


# Global request logger instance
request_logger = RequestLogger()
