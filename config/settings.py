"""
config/settings.py
==================
Single source of truth for every parameter in the pipeline.
Change values here — nothing else needs to be touched.
"""

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Random seed ──────────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Data generation ──────────────────────────────────────────────────────────
DATA_GEN = {
    "n_users"           : 5000,
    "fraud_rate"        : 0.18,      # fraction of users that are fraudulent
    "fraud_orders_mean" : 12,        # exponential mean for fraud user order count
    "normal_orders_mean": 7,         # exponential mean for normal user order count
    "lookback_days"     : 180,       # order history window
}

# ── Reference lists (used only in data generation) ───────────────────────────
GEO = {
    "indian_cities": [
        "Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai", "Kolkata",
        "Pune", "Jaipur", "Ahmedabad", "Surat", "Lucknow", "Kanpur",
        "Nagpur", "Indore", "Bhopal", "Patna", "Vadodara", "Ludhiana",
        "Agra", "Nashik",
    ],
    "global_cities": [
        "New York", "Los Angeles", "London", "Dubai", "Singapore", "Toronto",
        "Sydney", "Frankfurt", "Tokyo", "Jakarta", "São Paulo", "Lagos",
        "Cairo", "Bangkok", "Seoul",
    ],
    "global_countries": ["USA", "UK", "UAE", "Singapore", "Australia", "Canada", "Germany"],
}

CATEGORIES = [
    "Electronics", "Fashion", "Home & Kitchen", "Books", "Sports",
    "Beauty", "Grocery", "Toys", "Automotive", "Health",
]

RETURN_REASONS = [
    "Product damaged", "Wrong item delivered", "Size not fitting",
    "Quality not as described", "Changed mind", "Duplicate order",
    "Better price elsewhere", "Product not working", "Delayed delivery",
]

DEVICES = ["Android", "iOS", "Web-Chrome", "Web-Firefox", "Web-Safari", "Web-Edge"]

PAYMENT_METHODS = [
    "Credit Card", "Debit Card", "UPI", "Net Banking",
    "COD", "Wallet", "EMI", "PayPal", "Stripe",
]

LOYALTY_TIERS = {
    "fraud" : {"tiers": ["Bronze", "Silver", "Gold"],           "weights": [0.7, 0.2, 0.1]},
    "normal": {"tiers": ["Bronze", "Silver", "Gold", "Platinum"], "weights": [0.4, 0.3, 0.2, 0.1]},
}

# ── Fraud behaviour parameters ────────────────────────────────────────────────
FRAUD_BEHAVIOUR = {
    "return_prob"          : 0.65,
    "normal_return_prob"   : 0.12,
    "return_days_range"    : (1, 5),     # fraud: returns very quickly
    "normal_return_days"   : (2, 25),
    "refund_pct_range"     : (0.8, 1.0), # fraud: near-full refunds
    "normal_refund_pct"    : (0.5, 1.0),
    "high_value_threshold" : 2000,       # ₹ above which a refund is "high value"
    "account_age_exp_mean" : 60,         # fraud accounts are new
    "normal_age_mean"      : 800,
    "normal_age_std"       : 400,
    "max_devices_fraud"    : (2, 6),
    "max_devices_normal"   : (1, 2),
}

# ── Feature engineering ───────────────────────────────────────────────────────
# These are DERIVED features — column names in the final feature matrix.
# Thresholds for binary flag features:
FEATURE_THRESHOLDS = {
    "quick_return_days"    : 2,     # ≤ N days → quick_return_flag = 1
    "multi_city_min"       : 3,     # ≥ N cities → multi_city_flag = 1
    "multi_device_min"     : 3,     # ≥ N devices → multi_device_flag = 1
    "new_account_days"     : 60,    # ≤ N days old → new_account_flag = 1
    "high_value_ret_count" : 2,     # ≥ N high-value returns → flag = 1
    "high_value_order_amt" : 2000,  # order value threshold
}

# ── ML model ──────────────────────────────────────────────────────────────────
MODEL = {
    "test_size"           : 0.2,
    "rf_n_estimators"     : 200,
    "rf_max_depth"        : 10,
    "iso_n_estimators"    : 100,
    "iso_contamination"   : 0.18,   # matches fraud_rate
    "rf_weight"           : 0.65,   # ensemble weight for RF
    "iso_weight"          : 0.35,   # ensemble weight for IsoForest
}

# ── Risk scoring bands ────────────────────────────────────────────────────────
RISK_BANDS = [
    (0,  30,  "Low"),
    (31, 70,  "Medium"),
    (71, 100, "High"),
]

# ── Excel report styling ──────────────────────────────────────────────────────
EXCEL_STYLE = {
    "header_fill"   : "1A1A2E",
    "header_font"   : "FFFFFF",
    "high_row_fill" : "FFE5E5",
    "med_row_fill"  : "FFF8E1",
    "low_row_fill"  : "E8F5E9",
    "high_score_font": "C62828",
    "med_score_font" : "E65100",
    "low_score_font" : "2E7D32",
    "high_badge_fill": "FFCDD2",
    "med_badge_fill" : "FFE0B2",
    "low_badge_fill" : "C8E6C9",
    "high_badge_font": "B71C1C",
    "med_badge_font" : "BF360C",
    "low_badge_font" : "1B5E20",
    "section_fill"   : "263238",
    "section_font"   : "ECEFF1",
    "hr_header_fill" : "8B0000",
    "reasons_font"   : "444444",
}

# ── Column display config for Excel (internal_name → display_name, width) ────
# Width = None means auto-fit. Order here controls column order in the sheet.
EXCEL_COLUMNS = [
    ("user_id",           "User ID",          12),
    ("city",              "City",             14),
    ("country",           "Country",          12),
    ("loyalty_tier",      "Tier",              8),
    ("account_age_days",  "Acct Age (days)",  15),
    ("total_orders",      "Total Orders",     12),
    ("total_spend",       "Total Spend (₹)",  16),
    ("return_rate",       "Return Rate",      12),
    ("total_refund_amt",  "Total Refund (₹)", 16),
    ("high_value_returns","HV Refunds",       10),
    ("orders_per_day",    "Orders/Day",       11),
    ("num_devices_used",  "Devices",           8),
    ("fraud_score",       "Fraud Score",      12),
    ("risk_level",        "Risk Level",       10),
    ("flagged_reasons",   "Flagged Reasons",  52),
]

# ── Currency / locale ─────────────────────────────────────────────────────────
CURRENCY_SYMBOL = "₹"
