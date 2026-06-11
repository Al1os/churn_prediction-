import numpy as np


def create_features(df):

    # =========================
    # Ratio / Proportion Features
    # =========================

    df["discount_ratio"] = (
        (
            df["total_amount"]
            - df["discounted_total"]
        )
        /
        df["total_amount"]
    )

    df["avg_quantity_per_product"] = (
        df["total_quantity"]
        /
        df["total_products"]
    )

    # =========================
    # Time-Based Features
    # =========================

    np.random.seed(42)

    df["days_inactive"] = np.random.randint(
        1,
        365,
        len(df)
    )

    df["days_since_registration"] = (
        np.random.randint(
            30,
            2000,
            len(df)
        )
    )

    # =========================
    # Binary / Categorical Features
    # =========================

    df["large_cart"] = (
        df["total_products"]
        >
        df["total_products"].median()
    ).astype(int)

    df["high_spender"] = (
        df["total_amount"]
        >
        df["total_amount"].median()
    ).astype(int)

    # =========================
    # Churn Label
    # =========================

    threshold = (
        df["days_inactive"]
        .median()
    )

    df["churned"] = (
        df["days_inactive"]
        >
        threshold
    ).astype(int)

    return df