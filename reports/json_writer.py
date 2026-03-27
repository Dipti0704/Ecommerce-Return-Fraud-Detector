"""
reports/json_writer.py
======================
Builds the dashboard_data.json used by the Streamlit app.
All column references go through schema — no hardcoded strings.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import json
from pathlib import Path

import pandas as pd

from utils.schema import (
    OrdersSchema  as OS,
    ReturnsSchema as RS,
    FeatSchema    as FS,
    UsersSchema   as US,
)
from utils.logger import get_logger

log = get_logger("reports.json_writer")

_MONTHS_TO_SHOW = 6


def build_dashboard_json(
    scored_df : pd.DataFrame,
    orders_df : pd.DataFrame,
    returns_df: pd.DataFrame,
    feature_importances: dict,
    output_path: str | Path,
):
    log.info("Building dashboard JSON")

    risk_dist   = scored_df[FS.RISK_LEVEL].value_counts().to_dict()
    top_flagged = _top_flagged(scored_df)
    cat_rates   = _category_return_rates(orders_df, returns_df)
    monthly     = _monthly_trend(orders_df, scored_df)
    summary     = _summary(scored_df, orders_df, returns_df, feature_importances)

    data = {
        "risk_dist"            : risk_dist,
        "top_flagged"          : top_flagged,
        "category_return_rates": cat_rates,
        "monthly_orders"       : monthly,
        "summary"              : summary,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    log.info("Dashboard JSON saved → %s", Path(output_path).name)
    return data


# ── Helpers ──────────────────────────────────────────────────────────────────

def _top_flagged(scored_df: pd.DataFrame, n: int = 10) -> list:
    export_cols = [
        US.USER_ID, US.CITY,
        FS.FRAUD_SCORE, FS.RETURN_RATE, FS.TOTAL_ORDERS,
        FS.TOTAL_REFUND_AMT, FS.RISK_LEVEL, FS.FLAGGED_REASONS,
    ]
    # Only export columns that exist
    available = [c for c in export_cols if c in scored_df.columns]

    top = (
        scored_df[scored_df[FS.RISK_LEVEL] == "High"]
        .sort_values(FS.FRAUD_SCORE, ascending=False)
        .head(n)[available]
        .copy()
    )
    if FS.RETURN_RATE in top.columns:
        top[FS.RETURN_RATE] = (top[FS.RETURN_RATE] * 100).round(1)
    if FS.TOTAL_REFUND_AMT in top.columns:
        top[FS.TOTAL_REFUND_AMT] = top[FS.TOTAL_REFUND_AMT].round(0).astype(int)

    return top.to_dict("records")


def _category_return_rates(orders_df: pd.DataFrame, returns_df: pd.DataFrame) -> dict:
    merged = orders_df.merge(
        returns_df[[RS.ORDER_ID, RS.REFUND_AMOUNT]],
        on=OS.ORDER_ID,
        how="left",
    )
    merged["returned"] = merged[RS.REFUND_AMOUNT].notna().astype(int)
    rates = (
        merged.groupby(OS.CATEGORY)["returned"]
        .mean()
        .round(3)
        .sort_values(ascending=False)
        .to_dict()
    )
    return rates


def _monthly_trend(orders_df: pd.DataFrame, scored_df: pd.DataFrame) -> dict:
    labeled = orders_df.merge(
        scored_df[[US.USER_ID, FS.RISK_LEVEL]],
        on=US.USER_ID,
        how="left",
    )
    labeled["order_month"] = pd.to_datetime(labeled[OS.ORDER_DATE]).dt.to_period("M")

    pivot = (
        labeled.groupby(["order_month", FS.RISK_LEVEL])
        .size()
        .unstack(fill_value=0)
    )
    pivot.index = pivot.index.astype(str)
    last_n = pivot.tail(_MONTHS_TO_SHOW)
    return last_n.to_dict()


def _summary(scored_df, orders_df, returns_df, feature_importances) -> dict:
    n_users   = len(scored_df)
    n_orders  = len(orders_df)
    n_returns = len(returns_df)
    high_df   = scored_df[scored_df[FS.RISK_LEVEL] == "High"]

    top_feats = dict(
        sorted(feature_importances.items(), key=lambda x: -x[1])[:8]
    )

    return {
        "total_users"          : n_users,
        "total_orders"         : n_orders,
        "total_returns"        : n_returns,
        "overall_return_rate"  : round(n_returns / n_orders, 3) if n_orders else 0,
        "high_risk_count"      : int(scored_df[FS.RISK_LEVEL].eq("High").sum()),
        "medium_risk_count"    : int(scored_df[FS.RISK_LEVEL].eq("Medium").sum()),
        "low_risk_count"       : int(scored_df[FS.RISK_LEVEL].eq("Low").sum()),
        "total_refund_exposure": round(float(high_df[FS.TOTAL_REFUND_AMT].sum()), 2),
        "avg_fraud_score_high" : round(float(high_df[FS.FRAUD_SCORE].mean()), 1),
        "feature_importances"  : {k: round(float(v), 4) for k, v in top_feats.items()},
    }
