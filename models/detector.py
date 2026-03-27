"""
models/detector.py
==================
Trains Random Forest + Isolation Forest ensemble, scores every user,
and generates human-readable fraud reasons.

All feature names come from schema, all thresholds from config.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

from config.settings import MODEL, RISK_BANDS, RANDOM_SEED
from utils.schema import UsersSchema as US, FeatSchema as FS, ML_FEATURES
from utils.logger import get_logger

log = get_logger("models.detector")


class FraudDetector:
    """
    Fits an ensemble model and scores every user.

    Usage
    -----
        detector = FraudDetector()
        scored_df = detector.fit_predict(feat_df)
        # scored_df has new columns: fraud_score, rf_probability,
        #   anomaly_score, risk_level, flagged_reasons
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.rf     = RandomForestClassifier(
            n_estimators = MODEL["rf_n_estimators"],
            max_depth    = MODEL["rf_max_depth"],
            random_state = RANDOM_SEED,
            n_jobs       = -1,
        )
        self.iso = IsolationForest(
            n_estimators  = MODEL["iso_n_estimators"],
            contamination = MODEL["iso_contamination"],
            random_state  = RANDOM_SEED,
            n_jobs        = -1,
        )
        self._feature_importances: dict = {}

    # ── Public API ──────────────────────────────────────────────────────────

    def fit_predict(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        """Train models on feat_df and return feat_df with score columns added."""
        # Verify all required ML features exist in the dataframe
        missing = set(ML_FEATURES) - set(feat_df.columns)
        if missing:
            raise ValueError(f"Feature matrix is missing columns: {sorted(missing)}")

        X = feat_df[ML_FEATURES].values
        y = feat_df[US.IS_FRAUD].values

        log.info("Training on %d users with %d features", len(feat_df), len(ML_FEATURES))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size    = MODEL["test_size"],
            random_state = RANDOM_SEED,
            stratify     = y,
        )

        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s  = self.scaler.transform(X_test)
        X_all_s   = self.scaler.transform(X)

        # ── Supervised: Random Forest ────────────────────────────────────
        self.rf.fit(X_train_s, y_train)
        self._feature_importances = dict(zip(ML_FEATURES, self.rf.feature_importances_))

        rf_proba_all  = self.rf.predict_proba(X_all_s)[:, 1]
        rf_proba_test = self.rf.predict_proba(X_test_s)[:, 1]
        rf_pred_test  = self.rf.predict(X_test_s)

        log.info("Random Forest trained")
        log.info("\n%s", classification_report(y_test, rf_pred_test, target_names=["Normal", "Fraud"]))
        log.info("ROC-AUC: %.4f", roc_auc_score(y_test, rf_proba_test))

        # ── Unsupervised: Isolation Forest ───────────────────────────────
        self.iso.fit(X_all_s)
        iso_raw   = -self.iso.score_samples(X_all_s)
        iso_norm  = (iso_raw - iso_raw.min()) / (iso_raw.max() - iso_raw.min() + 1e-9)

        log.info("Isolation Forest trained")

        # ── Ensemble score (0–100) ────────────────────────────────────────
        rf_w  = MODEL["rf_weight"]
        iso_w = MODEL["iso_weight"]
        ensemble = rf_w * rf_proba_all + iso_w * iso_norm
        fraud_score = np.round(ensemble * 100).astype(int).clip(0, 100)

        # ── Attach scores to dataframe ────────────────────────────────────
        result = feat_df.copy()
        result[FS.FRAUD_SCORE]    = fraud_score
        result[FS.RF_PROBABILITY] = np.round(rf_proba_all * 100, 1)
        result[FS.ANOMALY_SCORE]  = np.round(iso_norm * 100, 1)
        result[FS.RISK_LEVEL]     = result[FS.FRAUD_SCORE].apply(self._score_to_risk)
        result[FS.FLAGGED_REASONS]= result.apply(self._build_reasons, axis=1)

        self._log_risk_summary(result)
        return result

    @property
    def feature_importances(self) -> dict:
        return self._feature_importances

    # ── Internal helpers ────────────────────────────────────────────────────

    @staticmethod
    def _score_to_risk(score: int) -> str:
        for lo, hi, label in RISK_BANDS:
            if lo <= score <= hi:
                return label
        return "Unknown"

    def _build_reasons(self, row: pd.Series) -> str:
        """
        Generates a human-readable reason string from the top feature values.
        All column names and thresholds come from schema / config.
        """
        # Rank features by importance (highest first)
        ranked = sorted(
            self._feature_importances.items(),
            key=lambda x: -x[1],
        )

        # Rule map: feature_col → (label, condition_fn, format_fn)
        rules = {
            FS.RETURN_RATE:       ("Return rate",        lambda v: v > 0.4,  lambda v: f"{v:.0%}"),
            FS.HIGH_VALUE_RETURNS:("HV refunds",         lambda v: v >= 2,   lambda v: f"{v:.0f} refunds"),
            FS.AVG_DAYS_TO_RETURN:("Avg days to return", lambda v: v < 4,    lambda v: f"{v:.1f} days avg"),
            FS.ORDERS_PER_DAY:    ("Orders/day",         lambda v: v > 0.3,  lambda v: f"{v:.2f}"),
            US.NUM_DEVICES:       ("Devices used",       lambda v: v >= 3,   lambda v: f"{v:.0f} devices"),
            FS.NEW_ACCOUNT_FLAG:  ("New account",        lambda v: v == 1,   lambda _: "< 60 days old"),
            FS.REFUND_SPEND_RATIO:("Refund ratio",       lambda v: v > 0.4,  lambda v: f"{v:.0%} of spend"),
            FS.MULTI_CITY_FLAG:   ("Multi-city orders",  lambda v: v == 1,   lambda _: "3+ cities"),
            FS.QUICK_RETURN_FLAG: ("Quick returns",      lambda v: v == 1,   lambda _: "within 2 days"),
            FS.MULTI_DEVICE_FLAG: ("Multi-device",       lambda v: v == 1,   lambda _: "3+ devices"),
        }

        reasons = []
        for col, _ in ranked:
            if col not in rules or col not in row.index:
                continue
            label, condition, fmt = rules[col]
            val = row[col]
            if condition(val):
                reasons.append(f"{label}: {fmt(val)}")
            if len(reasons) >= 3:
                break

        return " | ".join(reasons) if reasons else "Pattern-based anomaly"

    @staticmethod
    def _log_risk_summary(df: pd.DataFrame):
        counts = df[FS.RISK_LEVEL].value_counts().to_dict()
        total  = len(df)
        for label in ["High", "Medium", "Low"]:
            n = counts.get(label, 0)
            log.info("  %-8s risk: %4d users (%.1f%%)", label, n, n / total * 100)
