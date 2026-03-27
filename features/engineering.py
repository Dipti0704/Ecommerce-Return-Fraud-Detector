"""
features/engineering.py
========================
Builds the user-level feature matrix from users, orders, returns.

Rules
-----
- Every column name comes from utils/schema.py — no string literals in logic.
- Every threshold comes from config/settings.py — no magic numbers.
- Adding a new feature = add it to FeaturesSchema + add logic here.
  Nothing else needs changing.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

from config.settings import FEATURE_THRESHOLDS as FT, FRAUD_BEHAVIOUR as FB
from utils.schema import (
    UsersSchema   as US,
    OrdersSchema  as OS,
    ReturnsSchema as RS,
    FeatSchema    as FS,
    ML_FEATURES,
)
from utils.logger import get_logger

log = get_logger("features.engineering")


class FeatureEngineer:
    """
    Merges users + orders + returns into a single feature-rich DataFrame.

    Usage
    -----
        fe      = FeatureEngineer()
        feat_df = fe.build(users_df, orders_df, returns_df)
        X       = feat_df[ML_FEATURES].values
    """

    def build(
        self,
        users_df  : pd.DataFrame,
        orders_df : pd.DataFrame,
        returns_df: pd.DataFrame,
    ) -> pd.DataFrame:

        log.info("Building feature matrix")

        order_agg  = self._aggregate_orders(orders_df)
        return_agg = self._aggregate_returns(returns_df)
        hv_ret_agg = self._high_value_returns(returns_df)

        feat = (
            users_df
            .merge(order_agg,  on=US.USER_ID, how="left")
            .merge(return_agg, on=US.USER_ID, how="left")
            .merge(hv_ret_agg, on=US.USER_ID, how="left")
            .fillna(0)
        )

        feat = self._derived_features(feat)
        log.info(
            "Feature matrix ready: %d users × %d features  |  ML features=%d",
            len(feat), len(feat.columns), len(ML_FEATURES),
        )
        return feat

    # ── Aggregation helpers ─────────────────────────────────────────────────

    def _aggregate_orders(self, orders_df: pd.DataFrame) -> pd.DataFrame:
        return (
            orders_df
            .groupby(OS.USER_ID)
            .agg(
                **{
                    FS.TOTAL_ORDERS      : (OS.ORDER_ID,     "count"),
                    FS.TOTAL_SPEND       : (OS.ORDER_VALUE,  "sum"),
                    FS.AVG_ORDER_VALUE   : (OS.ORDER_VALUE,  "mean"),
                    FS.MAX_ORDER_VALUE   : (OS.ORDER_VALUE,  "max"),
                    FS.UNIQUE_CATEGORIES : (OS.CATEGORY,     "nunique"),
                    FS.UNIQUE_CITIES     : (OS.DELIVERY_CITY,"nunique"),
                }
            )
            .reset_index()
        )

    def _aggregate_returns(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        return (
            returns_df
            .groupby(RS.USER_ID)
            .agg(
                **{
                    FS.TOTAL_RETURNS      : (RS.RETURN_ID,      "count"),
                    FS.TOTAL_REFUND_AMT   : (RS.REFUND_AMOUNT,  "sum"),
                    FS.AVG_REFUND_AMT     : (RS.REFUND_AMOUNT,  "mean"),
                    FS.AVG_DAYS_TO_RETURN : (RS.DAYS_TO_RETURN, "mean"),
                    FS.MIN_DAYS_TO_RETURN : (RS.DAYS_TO_RETURN, "min"),
                }
            )
            .reset_index()
            .rename(columns={RS.USER_ID: US.USER_ID})
        )

    def _high_value_returns(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        threshold = FB["high_value_threshold"]
        hv = (
            returns_df[returns_df[RS.REFUND_AMOUNT] > threshold]
            .groupby(RS.USER_ID)
            .size()
            .reset_index(name=FS.HIGH_VALUE_RETURNS)
            .rename(columns={RS.USER_ID: US.USER_ID})
        )
        return hv

    # ── Derived / flag features ─────────────────────────────────────────────

    def _derived_features(self, feat: pd.DataFrame) -> pd.DataFrame:
        f = feat.copy()

        # Ratio features (guard against zero division)
        safe_orders = f[FS.TOTAL_ORDERS].clip(lower=1)
        safe_spend  = f[FS.TOTAL_SPEND].clip(lower=1)
        safe_age    = f[US.ACCOUNT_AGE_DAYS].clip(lower=1)

        f[FS.RETURN_RATE]        = f[FS.TOTAL_RETURNS]    / safe_orders
        f[FS.REFUND_SPEND_RATIO] = f[FS.TOTAL_REFUND_AMT] / safe_spend
        f[FS.ORDERS_PER_DAY]     = f[FS.TOTAL_ORDERS]     / safe_age

        # Binary flag features — thresholds from config, no magic numbers
        f[FS.QUICK_RETURN_FLAG]  = (f[FS.MIN_DAYS_TO_RETURN] <= FT["quick_return_days"]).astype(int)
        f[FS.MULTI_CITY_FLAG]    = (f[FS.UNIQUE_CITIES]       >= FT["multi_city_min"]).astype(int)
        f[FS.MULTI_DEVICE_FLAG]  = (f[US.NUM_DEVICES]         >= FT["multi_device_min"]).astype(int)
        f[FS.NEW_ACCOUNT_FLAG]   = (f[US.ACCOUNT_AGE_DAYS]    <= FT["new_account_days"]).astype(int)
        f[FS.HV_RETURN_FLAG]     = (f[FS.HIGH_VALUE_RETURNS]  >= FT["high_value_ret_count"]).astype(int)

        return f
