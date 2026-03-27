"""
utils/schema.py
===============
Defines the expected schema for each generated CSV.
The pipeline reads column names FROM these definitions — nothing is
hardcoded in the business logic modules.

Usage
-----
    from utils.schema import UsersSchema, OrdersSchema, ReturnsSchema
    df[UsersSchema.USER_ID]   # always safe, never a typo
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from dataclasses import dataclass, fields


@dataclass(frozen=True)
class _UsersSchema:
    USER_ID         : str = "user_id"
    JOIN_DATE       : str = "join_date"
    CITY            : str = "city"
    COUNTRY         : str = "country"
    ACCOUNT_AGE_DAYS: str = "account_age_days"
    LOYALTY_TIER    : str = "loyalty_tier"
    NUM_DEVICES     : str = "num_devices_used"
    PRIMARY_DEVICE  : str = "primary_device"
    IS_FRAUD        : str = "is_fraud_ground_truth"

    def all_columns(self):
        return [getattr(self, f.name) for f in fields(self)]


@dataclass(frozen=True)
class _OrdersSchema:
    ORDER_ID       : str = "order_id"
    USER_ID        : str = "user_id"
    ORDER_DATE     : str = "order_date"
    CATEGORY       : str = "category"
    ORDER_VALUE    : str = "order_value"
    PAYMENT_METHOD : str = "payment_method"
    DELIVERY_CITY  : str = "delivery_city"

    def all_columns(self):
        return [getattr(self, f.name) for f in fields(self)]


@dataclass(frozen=True)
class _ReturnsSchema:
    RETURN_ID     : str = "return_id"
    ORDER_ID      : str = "order_id"
    USER_ID       : str = "user_id"
    RETURN_DATE   : str = "return_date"
    RETURN_REASON : str = "return_reason"
    REFUND_AMOUNT : str = "refund_amount"
    DAYS_TO_RETURN: str = "days_to_return"

    def all_columns(self):
        return [getattr(self, f.name) for f in fields(self)]


@dataclass(frozen=True)
class _FeaturesSchema:
    """
    Columns that feature engineering adds on top of the users schema.
    ML_FEATURES is derived dynamically from this — no manual list.
    """
    TOTAL_ORDERS        : str = "total_orders"
    TOTAL_SPEND         : str = "total_spend"
    AVG_ORDER_VALUE     : str = "avg_order_value"
    MAX_ORDER_VALUE     : str = "max_order_value"
    UNIQUE_CATEGORIES   : str = "unique_categories"
    UNIQUE_CITIES       : str = "unique_cities"
    TOTAL_RETURNS       : str = "total_returns"
    TOTAL_REFUND_AMT    : str = "total_refund_amt"
    AVG_REFUND_AMT      : str = "avg_refund_amt"
    AVG_DAYS_TO_RETURN  : str = "avg_days_to_return"
    MIN_DAYS_TO_RETURN  : str = "min_days_to_return"
    HIGH_VALUE_RETURNS  : str = "high_value_returns"
    RETURN_RATE         : str = "return_rate"
    REFUND_SPEND_RATIO  : str = "refund_to_spend_ratio"
    ORDERS_PER_DAY      : str = "orders_per_day"
    QUICK_RETURN_FLAG   : str = "quick_return_flag"
    MULTI_CITY_FLAG     : str = "multi_city_flag"
    MULTI_DEVICE_FLAG   : str = "multi_device_flag"
    NEW_ACCOUNT_FLAG    : str = "new_account_flag"
    HV_RETURN_FLAG      : str = "high_value_return_flag"

    # Model output columns
    FRAUD_SCORE         : str = "fraud_score"
    RF_PROBABILITY      : str = "rf_probability"
    ANOMALY_SCORE       : str = "anomaly_score"
    RISK_LEVEL          : str = "risk_level"
    FLAGGED_REASONS     : str = "flagged_reasons"

    def ml_feature_columns(self):
        """
        Returns the list of columns to use as ML input features.
        Excludes schema metadata, label, and output columns.
        """
        _exclude = {
            "FRAUD_SCORE", "RF_PROBABILITY", "ANOMALY_SCORE",
            "RISK_LEVEL", "FLAGGED_REASONS",
        }
        return [
            getattr(self, f.name)
            for f in fields(self)
            if f.name not in _exclude
        ]

    def all_columns(self):
        return [getattr(self, f.name) for f in fields(self)]


# Singleton instances — import these
UsersSchema   = _UsersSchema()
OrdersSchema  = _OrdersSchema()
ReturnsSchema = _ReturnsSchema()
FeatSchema    = _FeaturesSchema()

# Convenience: full ML feature list derived from schema (no hardcoding)
ML_FEATURES = FeatSchema.ml_feature_columns()
