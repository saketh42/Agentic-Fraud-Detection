"""
Knowledge Store - Knowledge phase of MAPE-K
SQLite-based persistent memory store for agentic learning.
"""
import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional


class KnowledgeStore:
    """
    SQLite-based knowledge store for continuous learning.
    
    Tables:
    - transactions: Raw transaction data
    - predictions: Model predictions and decisions
    - reasoning: LLM reasoning traces
    - patterns: Learned fraud patterns
    - adversarial_variants: Adversarial test cases
    - execution_logs: Action execution history
    - feedback: Human feedback for learning
    - metrics: Agentic performance metrics
    """
    
    def __init__(self, db_path: str = "knowledge_store.db"):
        self.db_path = db_path
        self._init_database()
    
    def _get_connection(self):
        return sqlite3.connect(self.db_path)
    
    def _init_database(self):
        """Initialize database schema"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Transactions table
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
        
        # Predictions table
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
        
        # Reasoning table
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
        
        # Patterns table
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
        
        # Adversarial variants table
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
        
        # Execution logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS execution_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT,
                action TEXT,
                outcome TEXT,
                timestamp TEXT,
                FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id)
            )
        ''')
        
        # Feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT,
                feedback_type TEXT,
                feedback_text TEXT,
                is_correct INTEGER,
                timestamp TEXT,
                FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id)
            )
        ''')
        
        # Metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT,
                metric_value REAL,
                timestamp TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_transaction(self, transaction_id: str, is_fraud: int, 
                         semantic_profile: str, label_count: int, 
                         tactic_count: int, raw_data: dict) -> bool:
        """Store transaction in memory"""
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
            conn.close()
    
    def store_prediction(self, transaction_id: str, predicted_prob: float, 
                     predicted_label: str, confidence: float, 
                     risk_level: str) -> bool:
        """Store prediction"""
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
            conn.close()
    
    def store_reasoning(self, transaction_id: str, reasoning_summary: str,
                       evidence: List[str], fraud_pattern: str,
                       adversarial_risk: str, recommended_next_step: str) -> bool:
        """Store LLM reasoning"""
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
            conn.close()
    
    def store_pattern(self, pattern_name: str, pattern_type: str, 
                   is_emerging: bool = False) -> bool:
        """Store or update pattern"""
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
            conn.close()
    
    def store_adversarial_variant(self, original_id: str, variant_type: str,
                             original_score: float, variant_score: float,
                             is_robust: bool) -> bool:
        """Store adversarial variant test result"""
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
            conn.close()
    
    def store_execution(self, transaction_id: str, action: str, 
                      outcome: str) -> bool:
        """Store action execution"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO execution_logs (transaction_id, action, outcome, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (transaction_id, action, outcome, datetime.now().isoformat()))
            conn.commit()
            return True
        except Exception as e:
            print("Error storing execution:", e)
            return False
        finally:
            conn.close()
    
    def store_feedback(self, transaction_id: str, feedback_type: str,
                    feedback_text: str, is_correct: bool) -> bool:
        """Store human feedback"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO feedback 
                (transaction_id, feedback_type, feedback_text, is_correct, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (transaction_id, feedback_type, feedback_text,
                  1 if is_correct else 0, datetime.now().isoformat()))
            conn.commit()
            return True
        except Exception as e:
            print("Error storing feedback:", e)
            return False
        finally:
            conn.close()
    
    def get_recent_transactions(self, limit: int = 10) -> List[Dict]:
        """Get recent transactions for context"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT transaction_id, timestamp, is_fraud, semantic_profile, label_count, tactic_count
            FROM transactions ORDER BY timestamp DESC LIMIT ?
        ''', (limit,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'transaction_id': row[0],
                'timestamp': row[1],
                'is_fraud': row[2],
                'semantic_profile': row[3],
                'label_count': row[4],
                'tactic_count': row[5]
            })
        
        conn.close()
        return results
    
    def get_pattern_frequency(self, pattern_name: str) -> int:
        """Get how often a pattern has appeared"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT frequency FROM patterns WHERE pattern_name = ?', (pattern_name,))
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else 0
    
    def get_pattern_success_rate(self, pattern_name: str) -> float:
        """Get success rate for a pattern based on feedback"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT AVG(CASE WHEN f.is_correct = 1 THEN 1.0 ELSE 0.0 END)
            FROM feedback f
            JOIN predictions p ON f.transaction_id = p.transaction_id
            JOIN reasoning r ON f.transaction_id = r.transaction_id
            WHERE r.fraud_pattern = ?
        ''', (pattern_name,))
        
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result[0] else 0.5
    
    def get_similar_transactions(self, semantic_profile: str, limit: int = 5) -> List[Dict]:
        """Get transactions with similar profiles"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT transaction_id, timestamp, is_fraud, label_count, tactic_count
            FROM transactions 
            WHERE semantic_profile = ?
            ORDER BY timestamp DESC LIMIT ?
        ''', (semantic_profile, limit))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'transaction_id': row[0],
                'timestamp': row[1],
                'is_fraud': row[2],
                'label_count': row[3],
                'tactic_count': row[4]
            })
        
        conn.close()
        return results
    
    def get_label_frequencies(self) -> Dict[str, int]:
        """Get frequency of each label from history"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT semantic_profile, COUNT(*) as count
            FROM transactions
            GROUP BY semantic_profile
        ''')
        
        results = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()
        return results
    
    def get_all_patterns(self) -> List[Dict]:
        """Get all learned patterns"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT pattern_name, pattern_type, frequency, is_emerging, success_rate
            FROM patterns ORDER BY frequency DESC
        ''')
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'name': row[0],
                'type': row[1],
                'frequency': row[2],
                'is_emerging': bool(row[3]),
                'success_rate': row[4]
            })
        
        conn.close()
        return results
    
    def calculate_learning_improvement(self) -> float:
        """Calculate if feedback improves decisions over time"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Compare early vs late feedback correctness
        cursor.execute('''
            SELECT 
                AVG(CASE WHEN is_correct = 1 THEN 1.0 ELSE 0.0 END) as late_accuracy
            FROM (
                SELECT is_correct, timestamp
                FROM feedback
                ORDER BY timestamp DESC
                LIMIT 100
            )
        ''')
        
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result and result[0] else 0.0
    
    def get_analytics(self) -> Dict:
        """Get overall analytics"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        analytics = {}
        
        # Total transactions
        cursor.execute('SELECT COUNT(*) FROM transactions')
        analytics['total_transactions'] = cursor.fetchone()[0]
        
        # Fraud rate
        cursor.execute('SELECT AVG(is_fraud) FROM transactions')
        analytics['fraud_rate'] = cursor.fetchone()[0] or 0.0
        
        # Pattern count
        cursor.execute('SELECT COUNT(*) FROM patterns')
        analytics['total_patterns'] = cursor.fetchone()[0]
        
        # Emerging patterns
        cursor.execute('SELECT COUNT(*) FROM patterns WHERE is_emerging = 1')
        analytics['emerging_patterns'] = cursor.fetchone()[0]
        
        # Feedback count
        cursor.execute('SELECT COUNT(*) FROM feedback')
        analytics['total_feedback'] = cursor.fetchone()[0]
        
        # Feedback accuracy
        cursor.execute('SELECT AVG(CASE WHEN is_correct = 1 THEN 1.0 ELSE 0.0 END) FROM feedback')
        analytics['feedback_accuracy'] = cursor.fetchone()[0] or 0.0
        
        conn.close()
        return analytics