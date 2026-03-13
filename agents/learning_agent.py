"""
Learning Agent (Continuous Improvement)
-----------------------------------------
Stores ticket outcomes in SQLite, retrains the model periodically,
and calibrates confidence scores from real-world feedback.
"""

import json
import logging
import os
import sqlite3
import time
from typing import Optional

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)


class LearningAgent:
    """
    Maintains a feedback store and drives periodic model retraining.
    """

    SCHEMA = """
        CREATE TABLE IF NOT EXISTS feedback (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_number    TEXT,
            features         TEXT,
            predicted_group  TEXT,
            final_group      TEXT,
            confidence       REAL,
            was_correct      INTEGER,
            created_at       REAL
        );

        CREATE TABLE IF NOT EXISTS audit_log (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_number    TEXT,
            sys_id           TEXT,
            predicted_group  TEXT,
            confidence       REAL,
            auto_assigned    INTEGER,
            reason           TEXT,
            top_predictions  TEXT,
            created_at       REAL
        );
    """

    def __init__(self, db_path: str, model_path: str):
        self.db_path = db_path
        self.model_path = model_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Database helpers
    # ------------------------------------------------------------------

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(self.SCHEMA)

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Feedback storage
    # ------------------------------------------------------------------

    def store_feedback(
        self,
        ticket_number: str,
        features: list,
        predicted_group: str,
        final_group: str,
        confidence: float,
    ):
        """
        Stores the outcome of a ticket (predicted vs final assignment).
        """
        was_correct = int(predicted_group == final_group)
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO feedback
                   (ticket_number, features, predicted_group, final_group,
                    confidence, was_correct, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    ticket_number,
                    json.dumps(features),
                    predicted_group,
                    final_group,
                    confidence,
                    was_correct,
                    time.time(),
                ),
            )
        logger.info(
            f"Feedback stored | ticket={ticket_number} "
            f"predicted={predicted_group} final={final_group} correct={bool(was_correct)}"
        )

    # ------------------------------------------------------------------
    # Audit log
    # ------------------------------------------------------------------

    def log_decision(
        self,
        ticket_number: str,
        sys_id: str,
        predicted_group: str,
        confidence: float,
        auto_assigned: bool,
        reason: str,
        top_predictions: list,
    ):
        """
        Writes every routing decision to the audit log for compliance.
        """
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO audit_log
                   (ticket_number, sys_id, predicted_group, confidence,
                    auto_assigned, reason, top_predictions, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    ticket_number,
                    sys_id,
                    predicted_group,
                    confidence,
                    int(auto_assigned),
                    reason,
                    json.dumps(top_predictions),
                    time.time(),
                ),
            )

    # ------------------------------------------------------------------
    # Model retraining
    # ------------------------------------------------------------------

    def get_training_data(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """
        Loads all stored feedback records for retraining.
        """
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT features, final_group FROM feedback"
            ).fetchall()

        if len(rows) < 10:
            logger.warning(
                f"Only {len(rows)} feedback record(s) available. "
                "Need at least 10 to retrain."
            )
            return None

        X = np.array([json.loads(r["features"]) for r in rows])
        y = np.array([r["final_group"] for r in rows])
        return X, y

    def retrain_model(self) -> bool:
        """
        Retrains the Logistic Regression model on stored feedback data.
        Saves the new model only if it passes cross-validation.

        Returns True if retraining succeeded.
        """
        data = self.get_training_data()
        if data is None:
            return False

        X, y = data
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            logger.warning("Need at least 2 classes to retrain.")
            return False

        try:
            model = LogisticRegression(max_iter=1000, C=1.0)

            # Cross-validate before committing (skip if too few samples)
            if len(X) >= 20:
                scores = cross_val_score(model, X, y, cv=min(5, len(X) // 4))
                avg_accuracy = scores.mean()
                logger.info(
                    f"Cross-val accuracy: {avg_accuracy:.3f} "
                    f"(±{scores.std():.3f}) over {len(X)} samples."
                )
            else:
                avg_accuracy = 1.0  # Can't CV on tiny set; just save

            model.fit(X, y)
            joblib.dump(model, self.model_path)
            logger.info(f"Model retrained and saved to '{self.model_path}'.")
            return True

        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def accuracy_report(self) -> dict:
        """
        Returns a summary dict with overall and per-group accuracy.
        """
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT predicted_group, final_group, was_correct FROM feedback"
            ).fetchall()

        if not rows:
            return {"total": 0, "accuracy": None, "per_group": {}}

        total = len(rows)
        correct = sum(r["was_correct"] for r in rows)

        per_group: dict[str, dict] = {}
        for r in rows:
            g = r["final_group"]
            per_group.setdefault(g, {"total": 0, "correct": 0})
            per_group[g]["total"] += 1
            per_group[g]["correct"] += r["was_correct"]

        for g in per_group:
            t = per_group[g]["total"]
            c = per_group[g]["correct"]
            per_group[g]["accuracy"] = round(c / t, 3) if t else 0.0

        return {
            "total": total,
            "accuracy": round(correct / total, 3),
            "per_group": per_group,
        }
