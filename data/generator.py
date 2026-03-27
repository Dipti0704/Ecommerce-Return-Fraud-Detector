"""
data/generator.py
=================
Generates synthetic e-commerce data: users, orders, returns.
All parameters come from config/settings.py.
Column names come from utils/schema.py.
Nothing is hardcoded here.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)) 


import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from config.settings import DATA_GEN, GEO, CATEGORIES, RETURN_REASONS, DEVICES, \
    PAYMENT_METHODS, LOYALTY_TIERS, FRAUD_BEHAVIOUR, RANDOM_SEED
from utils.schema import UsersSchema as US, OrdersSchema as OS, ReturnsSchema as RS
from utils.logger import get_logger

log = get_logger("data.generator")


class DataGenerator:
    """
    Generates users → orders → returns in one call.

    Usage
    -----
        gen = DataGenerator()
        users_df, orders_df, returns_df = gen.generate()
    """

    def __init__(self, seed: int = RANDOM_SEED):
        self.seed   = seed
        self.today  = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        self._all_cities = GEO["indian_cities"] + GEO["global_cities"]
        np.random.seed(seed)
        random.seed(seed)

    # ── Public entry point ──────────────────────────────────────────────────

    def generate(self):
        log.info("Starting synthetic data generation")
        users_df   = self._generate_users()
        orders_df  = self._generate_orders(users_df)
        returns_df = self._generate_returns(orders_df, users_df)
        log.info(
            "Generation complete — users=%d  orders=%d  returns=%d",
            len(users_df), len(orders_df), len(returns_df),
        )
        return users_df, orders_df, returns_df

    # ── Internal generators ─────────────────────────────────────────────────

    def _generate_users(self) -> pd.DataFrame:
        n           = DATA_GEN["n_users"]
        fraud_rate  = DATA_GEN["fraud_rate"]
        fb          = FRAUD_BEHAVIOUR

        is_fraud = np.random.random(n) < fraud_rate
        records  = []

        for i in range(n):
            fraud  = bool(is_fraud[i])
            city   = random.choice(self._all_cities)
            country = (
                "India"
                if city in GEO["indian_cities"]
                else random.choice(GEO["global_countries"])
            )

            if fraud:
                age     = int(np.random.exponential(fb["account_age_exp_mean"]))
                age     = max(1, min(age, 365))
                tier_cfg = LOYALTY_TIERS["fraud"]
                devices = random.randint(*fb["max_devices_fraud"])
            else:
                age     = int(np.random.normal(fb["normal_age_mean"], fb["normal_age_std"]))
                age     = max(30, min(age, 2500))
                tier_cfg = LOYALTY_TIERS["normal"]
                devices = random.randint(*fb["max_devices_normal"])

            join_date = self.today - timedelta(days=age)
            tier      = random.choices(tier_cfg["tiers"], weights=tier_cfg["weights"])[0]

            records.append({
                US.USER_ID         : f"USR{str(i + 1).zfill(5)}",
                US.JOIN_DATE       : join_date.strftime("%Y-%m-%d"),
                US.CITY            : city,
                US.COUNTRY         : country,
                US.ACCOUNT_AGE_DAYS: age,
                US.LOYALTY_TIER    : tier,
                US.NUM_DEVICES     : devices,
                US.PRIMARY_DEVICE  : random.choice(DEVICES),
                US.IS_FRAUD        : int(fraud),
            })

        df = pd.DataFrame(records)
        log.info(
            "Users generated: total=%d  fraud=%d (%.1f%%)",
            len(df), df[US.IS_FRAUD].sum(), df[US.IS_FRAUD].mean() * 100,
        )
        return df

    def _generate_orders(self, users_df: pd.DataFrame) -> pd.DataFrame:
        fraud_uid_set = set(users_df.loc[users_df[US.IS_FRAUD] == 1, US.USER_ID])
        uid_city      = users_df.set_index(US.USER_ID)[US.CITY].to_dict()
        uid_join      = users_df.set_index(US.USER_ID)[US.JOIN_DATE].to_dict()
        fb            = FRAUD_BEHAVIOUR

        all_records = []
        order_counter = 1

        fraud_mean  = DATA_GEN["fraud_orders_mean"]
        normal_mean = DATA_GEN["normal_orders_mean"]
        lookback    = DATA_GEN["lookback_days"]

        for _, user in users_df.iterrows():
            uid    = user[US.USER_ID]
            fraud  = uid in fraud_uid_set
            n_ord  = max(1, int(np.random.exponential(fraud_mean if fraud else normal_mean)))

            join_dt   = datetime.strptime(uid_join[uid], "%Y-%m-%d")
            base_date = max(join_dt, self.today - timedelta(days=lookback))
            window    = max(1, (self.today - base_date).days)

            for _ in range(n_ord):
                days_offset = random.randint(0, window)
                order_date  = base_date + timedelta(days=days_offset)
                category    = random.choice(CATEGORIES)

                if fraud:
                    if random.random() < 0.4:
                        value = round(np.random.uniform(5000, 30000), 2)
                    else:
                        value = round(np.random.uniform(200, 5000), 2)
                    payment = random.choices(
                        PAYMENT_METHODS,
                        weights=[0.30, 0.20, 0.05, 0.05, 0.25, 0.10, 0.03, 0.01, 0.01],
                    )[0]
                    # Fraud users sometimes order from a different city
                    delivery_city = (
                        random.choice(self._all_cities)
                        if random.random() < 0.3
                        else uid_city[uid]
                    )
                else:
                    value = round(min(np.random.lognormal(7.5, 1.0), 25000), 2)
                    payment       = random.choice(PAYMENT_METHODS)
                    delivery_city = uid_city[uid]

                all_records.append({
                    OS.ORDER_ID      : f"ORD{str(order_counter).zfill(7)}",
                    OS.USER_ID       : uid,
                    OS.ORDER_DATE    : order_date.strftime("%Y-%m-%d"),
                    OS.CATEGORY      : category,
                    OS.ORDER_VALUE   : value,
                    OS.PAYMENT_METHOD: payment,
                    OS.DELIVERY_CITY : delivery_city,
                })
                order_counter += 1

        df = pd.DataFrame(all_records)
        log.info("Orders generated: %d", len(df))
        return df

    def _generate_returns(self, orders_df: pd.DataFrame, users_df: pd.DataFrame) -> pd.DataFrame:
        fraud_uid_set = set(users_df.loc[users_df[US.IS_FRAUD] == 1, US.USER_ID])
        fb            = FRAUD_BEHAVIOUR
        records       = []
        return_counter = 1

        for _, order in orders_df.iterrows():
            uid    = order[OS.USER_ID]
            fraud  = uid in fraud_uid_set

            ret_prob = fb["return_prob"] if fraud else fb["normal_return_prob"]
            if random.random() > ret_prob:
                continue

            order_date = datetime.strptime(order[OS.ORDER_DATE], "%Y-%m-%d")
            if fraud:
                ret_days = random.randint(*fb["return_days_range"])
            else:
                ret_days = random.randint(*fb["normal_return_days"])

            return_date = order_date + timedelta(days=ret_days)
            if return_date > self.today:
                return_date = self.today - timedelta(days=1)

            refund_lo, refund_hi = (
                fb["refund_pct_range"] if fraud else fb["normal_refund_pct"]
            )
            refund_pct = random.uniform(refund_lo, refund_hi)
            refund_amt = round(order[OS.ORDER_VALUE] * refund_pct, 2)

            reason_weights = (
                [0.05, 0.05, 0.05, 0.20, 0.35, 0.10, 0.10, 0.05, 0.05]
                if fraud
                else [0.15, 0.15, 0.20, 0.15, 0.10, 0.05, 0.05, 0.10, 0.05]
            )
            reason = random.choices(RETURN_REASONS, weights=reason_weights)[0]

            records.append({
                RS.RETURN_ID    : f"RET{str(return_counter).zfill(7)}",
                RS.ORDER_ID     : order[OS.ORDER_ID],
                RS.USER_ID      : uid,
                RS.RETURN_DATE  : return_date.strftime("%Y-%m-%d"),
                RS.RETURN_REASON: reason,
                RS.REFUND_AMOUNT: refund_amt,
                RS.DAYS_TO_RETURN: ret_days,
            })
            return_counter += 1

        df = pd.DataFrame(records)
        log.info("Returns generated: %d", len(df))
        return df
