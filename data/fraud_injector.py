import pandas as pd
import numpy as np


def inject_fraud_patterns(users_df, orders_df, returns_df):

    fraud_users = users_df[users_df["is_fraud_ground_truth"] == 1]["user_id"]
    normal_users = users_df[users_df["is_fraud_ground_truth"] == 0]["user_id"]

    # --- 1. Orders modification ---
    mask_orders = orders_df["user_id"].isin(fraud_users)

    # Increase order value (but not too much)
    orders_df.loc[mask_orders, "order_value"] *= np.random.uniform(1.1, 1.5)

    # --- 2. Add extra returns for some fraud users ---
    fraud_orders = orders_df[orders_df["user_id"].isin(fraud_users)]

    extra_returns = []
    sample_orders = fraud_orders.sample(frac=0.15, random_state=42)

   # 10% normal users behave like fraud sometimes
    noisy_users = users_df[users_df["is_fraud_ground_truth"] == 0].sample(frac=0.1)

    # give them high returns
    noisy_orders = orders_df[orders_df["user_id"].isin(noisy_users["user_id"])]

    extra_noise_returns = []
    for _, row in noisy_orders.sample(frac=0.2).iterrows():
        extra_noise_returns.append({
            "return_id": f"NOISE{np.random.randint(100000)}",
            "order_id": row["order_id"],
            "user_id": row["user_id"],
            "return_date": row["order_date"],
            "return_reason": "Changed mind",
            "refund_amount": row["order_value"],
            "days_to_return": np.random.randint(2, 10)
        })

    returns_df = pd.concat([returns_df, pd.DataFrame(extra_noise_returns)], ignore_index=True)

    # --- 3. Modify user behavior ---
    fraud_mask = users_df["user_id"].isin(fraud_users)

    # Some fraud users are new
    users_df.loc[fraud_mask, "account_age_days"] = np.random.randint(10, 120)

    # Increase devices
    users_df.loc[fraud_mask, "num_devices_used"] = np.random.randint(2, 5)

    # --- 4. Add noise: some normal users also behave suspicious ---
    noisy_users = users_df[users_df["user_id"].isin(normal_users)].sample(frac=0.05)

    orders_df.loc[orders_df["user_id"].isin(noisy_users["user_id"]), "order_value"] *= np.random.uniform(1.1, 1.5)

    return users_df, orders_df, returns_df