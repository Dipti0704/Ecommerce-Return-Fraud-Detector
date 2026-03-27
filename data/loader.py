"""
data/loader.py
==============
Loads CSV files and validates them against the schema.
If you swap in real data, just point the paths here —
no other module needs to change.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathlib import Path

import pandas as pd

from utils.schema import UsersSchema as US, OrdersSchema as OS, ReturnsSchema as RS
from utils.logger import get_logger

log = get_logger("data.loader")


def _validate_columns(df: pd.DataFrame, expected_cols: list, source: str):
    missing = set(expected_cols) - set(df.columns)
    if missing:
        raise ValueError(
            f"[{source}] Missing columns: {sorted(missing)}\n"
            f"  Found: {sorted(df.columns.tolist())}"
        )
    log.debug("[%s] Schema OK — %d columns, %d rows", source, len(df.columns), len(df))


def load_users(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    _validate_columns(df, US.all_columns(), "users")
    log.info("Loaded users: %s  rows=%d", Path(path).name, len(df))
    return df


def load_orders(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    _validate_columns(df, OS.all_columns(), "orders")
    log.info("Loaded orders: %s  rows=%d", Path(path).name, len(df))
    return df


def load_returns(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    _validate_columns(df, RS.all_columns(), "returns")
    log.info("Loaded returns: %s  rows=%d", Path(path).name, len(df))
    return df


def save_csv(df: pd.DataFrame, path: str | Path, name: str = ""):
    df.to_csv(path, index=False)
    log.info("Saved %s → %s  rows=%d", name or "dataframe", Path(path).name, len(df))
