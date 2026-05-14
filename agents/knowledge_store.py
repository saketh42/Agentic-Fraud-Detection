"""
Knowledge Store - Knowledge phase of MAPE-K
SQLite-based persistent memory store for agentic learning.
"""
import sqlite3
import json
import os
from datetime import datetime
from typing import List, Optional


class KnowledgeStore:
    def __init__(self, db_path: str = "knowledge_store.db"):
        self.db_path = db_path
        self._conn = None
        self._init_database()

    def _get_connection(self):
        if self._conn is not None:
            return self._conn
        if self.db_path == ":memory:":
            self._conn = sqlite3.connect(self.db_path)
            return self._conn
        return sqlite3.connect(self.db_path)

    def _close_connection(self, conn):
        if self.db_path == ":memory:":
            return
        conn.close()

    def _init_database(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT UNIQUE,
                timestamp TEXT,
                is_fraud INTEGER,
                semantic_profile TEXT,
                label_count INTEGER,
                tactic_count INTEGER,
                raw_data TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT,
                predicted_fraud_prob REAL,
                predicted_label TEXT,
                model_confidence REAL,
                risk_level TEXT,
                timestamp TEXT,
                FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reasoning (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT,
                reasoning_summary TEXT,
                evidence TEXT,
                fraud_pattern TEXT,
                adversarial_risk TEXT,
                recommended_next_step TEXT,
                timestamp TEXT,
                FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_name TEXT UNIQUE,
                pattern_type TEXT,
                frequency INTEGER DEFAULT 1,
                first_seen TEXT,
                last_seen TEXT,
                is_emerging INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 0.0
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS adversarial_variants (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_transaction_id TEXT,
                variant_type TEXT,
                original_score REAL,
                variant_score REAL,
                is_robust INTEGER,
                timestamp TEXT,
                FOREIGN KEY (original_transaction_id) REFERENCES transactions(transaction_id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS execution_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT,
                action TEXT,
                outcome TEXT,
                details TEXT,
                timestamp TEXT,
                FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT,
                feedback_type TEXT,
                is_correct INTEGER,
                timestamp TEXT,
                FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id)
            )
        ''')
        conn.commit()
        self._close_connection(conn)

    def store_transaction(self, transaction_id: str, is_fraud: int,
                         semantic_profile: str, label_count: int,
                         tactic_count: int, raw_data: dict) -> bool:
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO transactions
                (transaction_id, timestamp, is_fraud, semantic_profile, label_count, tactic_count, raw_data)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (transaction_id, datetime.now().isoformat(), is_fraud, semantic_profile,
                  label_count, tactic_count, json.dumps(raw_data)))
            conn.commit()
            return True
        except Exception as e:
            print("Error storing transaction:", e)
            return False
        finally:
            self._close_connection(conn)

    def store_prediction(self, transaction_id: str, predicted_prob: float,
                        predicted_label: str, confidence: float,
                        risk_level: str) -> bool:
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO predictions
                (transaction_id, predicted_fraud_prob, predicted_label, model_confidence, risk_level, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (transaction_id, predicted_prob, predicted_label, confidence, risk_level,
                  datetime.now().isoformat()))
            conn.commit()
            return True
        except Exception as e:
            print("Error storing prediction:", e)
            return False
        finally:
            self._close_connection(conn)

    def store_reasoning(self, transaction_id: str, reasoning_summary: str,
                       evidence: list, fraud_pattern: str,
                       adversarial_risk: str, recommended_next_step: str) -> bool:
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO reasoning
                (transaction_id, reasoning_summary, evidence, fraud_pattern,
                 adversarial_risk, recommended_next_step, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (transaction_id, reasoning_summary, json.dumps(evidence),
                  fraud_pattern, adversarial_risk, recommended_next_step,
                  datetime.now().isoformat()))
            conn.commit()
            return True
        except Exception as e:
            print("Error storing reasoning:", e)
            return False
        finally:
            self._close_connection(conn)

    def store_pattern(self, pattern_name: str, pattern_type: str,
                     is_emerging: bool = False) -> bool:
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO patterns (pattern_name, pattern_type, first_seen, last_seen, is_emerging)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(pattern_name) DO UPDATE SET
                frequency = frequency + 1,
                last_seen = ?
            ''', (pattern_name, pattern_type, datetime.now().isoformat(),
                  datetime.now().isoformat(), 1 if is_emerging else 0,
                  datetime.now().isoformat()))
            conn.commit()
            return True
        except Exception as e:
            print("Error storing pattern:", e)
            return False
        finally:
            self._close_connection(conn)

    def store_adversarial_variant(self, original_id: str, variant_type: str,
                                 original_score: float, variant_score: float,
                                 is_robust: bool) -> bool:
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO adversarial_variants
                (original_transaction_id, variant_type, original_score, variant_score, is_robust, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (original_id, variant_type, original_score, variant_score,
                  1 if is_robust else 0, datetime.now().isoformat()))
            conn.commit()
            return True
        except Exception as e:
            print("Error storing adversarial variant:", e)
            return False
        finally:
            self._close_connection(conn)

    def store_execution(self, transaction_id: str, action: str,
                       outcome: str, details: str = "") -> bool:
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO execution_logs (transaction_id, action, outcome, details, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (transaction_id, action, outcome, details, datetime.now().isoformat()))
            conn.commit()
            return True
        except Exception as e:
            print("Error storing execution:", e)
            return False
        finally:
            self._close_connection(conn)

    def store_feedback(self, transaction_id: str, feedback_type: str,
                      is_correct: bool) -> bool:
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO feedback
                (transaction_id, feedback_type, is_correct, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (transaction_id, feedback_type, 1 if is_correct else 0,
                  datetime.now().isoformat()))
            conn.commit()
            return True
        except Exception as e:
            print("Error storing feedback:", e)
            return False
        finally:
            self._close_connection(conn)

    def get_recent_transactions(self, limit: int = 10) -> List[dict]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT transaction_id, timestamp, is_fraud, semantic_profile, label_count, tactic_count
            FROM transactions ORDER BY timestamp DESC LIMIT ?
        ''', (limit,))
        results = [{
            'transaction_id': r[0], 'timestamp': r[1], 'is_fraud': r[2],
            'semantic_profile': r[3], 'label_count': r[4], 'tactic_count': r[5]
        } for r in cursor.fetchall()]
        self._close_connection(conn)
        return results

    def get_pattern_frequency(self, pattern_name: str) -> int:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT frequency FROM patterns WHERE pattern_name = ?', (pattern_name,))
        result = cursor.fetchone()
        self._close_connection(conn)
        return result[0] if result else 0

    def get_pattern_success_rate(self, pattern_name: str) -> float:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT AVG(CASE WHEN f.is_correct = 1 THEN 1.0 ELSE 0.0 END)
            FROM feedback f
            JOIN reasoning r ON f.transaction_id = r.transaction_id
            WHERE r.fraud_pattern = ?
        ''', (pattern_name,))
        result = cursor.fetchone()
        self._close_connection(conn)
        return result[0] if result and result[0] else 0.5

    def get_similar_transactions(self, semantic_profile: str, limit: int = 5) -> List[dict]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT transaction_id, timestamp, is_fraud, label_count, tactic_count
            FROM transactions WHERE semantic_profile = ?
            ORDER BY timestamp DESC LIMIT ?
        ''', (semantic_profile, limit))
        results = [{
            'transaction_id': r[0], 'timestamp': r[1], 'is_fraud': r[2],
            'label_count': r[3], 'tactic_count': r[4]
        } for r in cursor.fetchall()]
        self._close_connection(conn)
        return results

    def get_label_frequencies(self) -> dict:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT semantic_profile, COUNT(*) FROM transactions GROUP BY semantic_profile')
        results = {r[0]: r[1] for r in cursor.fetchall()}
        self._close_connection(conn)
        return results

    def get_all_patterns(self) -> List[dict]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT pattern_name, pattern_type, frequency, is_emerging, success_rate
            FROM patterns ORDER BY frequency DESC
        ''')
        results = [{
            'name': r[0], 'type': r[1], 'frequency': r[2],
            'is_emerging': bool(r[3]), 'success_rate': r[4]
        } for r in cursor.fetchall()]
        self._close_connection(conn)
        return results

    def get_tactic_success_rate(self, tactic_name: str) -> float:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT AVG(CASE WHEN f.is_correct = 1 THEN 1.0 ELSE 0.0 END)
            FROM feedback f
            JOIN reasoning r ON f.transaction_id = r.transaction_id
            WHERE r.reasoning_summary LIKE ?
        ''', (f'%{tactic_name}%',))
        result = cursor.fetchone()
        self._close_connection(conn)
        return result[0] if result and result[0] else 0.5

    def get_impersonation_patterns(self) -> List[dict]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT pattern_name, frequency FROM patterns
            WHERE pattern_name LIKE '%AUTHORITY%' OR pattern_name LIKE '%SOCIAL%'
            ORDER BY frequency DESC
        ''')
        results = [{'name': r[0], 'frequency': r[1]} for r in cursor.fetchall()]
        self._close_connection(conn)
        return results

    def get_transaction(self, transaction_id: str) -> Optional[dict]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM transactions WHERE transaction_id = ?', (transaction_id,))
        row = cursor.fetchone()
        self._close_connection(conn)
        if row:
            return {
                'id': row[0], 'transaction_id': row[1], 'timestamp': row[2],
                'is_fraud': row[3], 'semantic_profile': row[4],
                'label_count': row[5], 'tactic_count': row[6], 'raw_data': row[7]
            }
        return None

    def get_adversarial_variants(self, transaction_id: str) -> List[dict]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM adversarial_variants WHERE original_transaction_id = ?
        ''', (transaction_id,))
        results = [{
            'id': r[0], 'original_transaction_id': r[1], 'variant_type': r[2],
            'original_score': r[3], 'variant_score': r[4], 'is_robust': bool(r[5]),
            'timestamp': r[6]
        } for r in cursor.fetchall()]
        self._close_connection(conn)
        return results

    def calculate_learning_improvement(self) -> float:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT AVG(CASE WHEN is_correct = 1 THEN 1.0 ELSE 0.0 END)
            FROM (SELECT is_correct FROM feedback ORDER BY timestamp DESC LIMIT 100)
        ''')
        result = cursor.fetchone()
        self._close_connection(conn)
        return result[0] if result and result[0] else 0.0

    def get_analytics(self) -> dict:
        conn = self._get_connection()
        cursor = conn.cursor()
        analytics = {}
        cursor.execute('SELECT COUNT(*) FROM transactions')
        analytics['total_transactions'] = cursor.fetchone()[0]
        cursor.execute('SELECT AVG(is_fraud) FROM transactions')
        analytics['fraud_rate'] = cursor.fetchone()[0] or 0.0
        cursor.execute('SELECT COUNT(*) FROM patterns')
        analytics['total_patterns'] = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM patterns WHERE is_emerging = 1')
        analytics['emerging_patterns'] = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM feedback')
        analytics['total_feedback'] = cursor.fetchone()[0]
        cursor.execute('SELECT AVG(CASE WHEN is_correct = 1 THEN 1.0 ELSE 0.0 END) FROM feedback')
        analytics['feedback_accuracy'] = cursor.fetchone()[0] or 0.0
        self._close_connection(conn)
        return analytics
