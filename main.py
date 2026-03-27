"""
main.py
=======
Entry point for the fraud detection pipeline.

Run:
    python main.py

What it does:
    1. Generates synthetic data  (users, orders, returns)
    2. Saves raw CSVs            (output/)
    3. Builds feature matrix     (feature engineering)
    4. Trains + scores           (ML ensemble)
    5. Saves scored CSV          (output/user_fraud_scores.csv)
    6. Writes Excel report       (output/Ecommerce_Fraud_Detection_Report.xlsx)
    7. Writes dashboard JSON     (output/dashboard_data.json)
"""



import sys
from pathlib import Path

from data.fraud_injector import inject_fraud_patterns
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Make sure project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import OUTPUT_DIR, RANDOM_SEED
from data.generator import DataGenerator
from data.loader import save_csv, load_users, load_orders, load_returns
from features.engineering import FeatureEngineer
from models.detector import FraudDetector
from reports.excel_writer import ExcelReportWriter
from reports.json_writer import build_dashboard_json
from utils.logger import get_logger
from data.fraud_injector import inject_fraud_patterns


log = get_logger("main")


def run_pipeline():
    log.info("=" * 60)
    log.info("E-COMMERCE FRAUD DETECTION PIPELINE  (seed=%d)", RANDOM_SEED)
    log.info("=" * 60)

    # # ── Step 1: Generate data ────────────────────────────────────────────
    # generator = DataGenerator(seed=RANDOM_SEED)
    # users_df, orders_df, returns_df = generator.generate()

    # # ── Step 2: Save raw CSVs ────────────────────────────────────────────
    # save_csv(users_df,   OUTPUT_DIR / "users.csv",   "users")
    # save_csv(orders_df,  OUTPUT_DIR / "orders.csv",  "orders")
    # save_csv(returns_df, OUTPUT_DIR / "returns.csv", "returns")
    users_df   = load_users("users.csv")
    orders_df  = load_orders("orders.csv")
    returns_df = load_returns("returns.csv")
    
    # # ── Step 3 (optional): reload from disk to prove the loader works ────
    # users_df   = load_users  (OUTPUT_DIR / "users.csv")
    # orders_df  = load_orders (OUTPUT_DIR / "orders.csv")
    # returns_df = load_returns(OUTPUT_DIR / "returns.csv")
    

    users_df, orders_df, returns_df = inject_fraud_patterns(
        users_df, orders_df, returns_df
    )

    # ── Step 4: Feature engineering ──────────────────────────────────────
    fe       = FeatureEngineer()
    feat_df  = fe.build(users_df, orders_df, returns_df)

    # ── Step 5: Train + score ─────────────────────────────────────────────
    detector  = FraudDetector()
    scored_df = detector.fit_predict(feat_df)

    save_csv(scored_df, OUTPUT_DIR / "user_fraud_scores.csv", "scored users")

    # ── Step 6: Excel report ──────────────────────────────────────────────
    excel_writer = ExcelReportWriter()
    excel_writer.write(
        scored_df, orders_df, returns_df,
        OUTPUT_DIR / "Ecommerce_Fraud_Detection_Report.xlsx",
    )

    # ── Step 7: Dashboard JSON ────────────────────────────────────────────
    build_dashboard_json(
        scored_df, orders_df, returns_df,
        feature_importances=detector.feature_importances,
        output_path=OUTPUT_DIR / "dashboard_data.json",
    )

    # ── Done ──────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("PIPELINE COMPLETE — output/ folder contains:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        log.info("  %s", f.name)
    log.info("=" * 60)


if __name__ == "__main__":
    run_pipeline()
