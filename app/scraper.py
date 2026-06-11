import requests
import pandas as pd
import random
import os


def load_data():

    users = requests.get(
        "https://dummyjson.com/users?limit=0"
    ).json()["users"]

    carts = requests.get(
        "https://dummyjson.com/carts?limit=0"
    ).json()["carts"]

    users_df = pd.DataFrame(users)

    records = []

    random.seed(42)

    for cart in carts:

        user = users_df[
            users_df["id"] == cart["userId"]
        ]

        if user.empty:
            continue

        user = user.iloc[0]

        # Generate multiple purchase sessions
        for _ in range(5):

            total_products = max(
                1,
                cart["totalProducts"] +
                random.randint(-1, 1)
            )

            total_quantity = max(
                1,
                cart["totalQuantity"] +
                random.randint(-2, 2)
            )

            total_amount = (
                cart["total"] *
                random.uniform(0.9, 1.1)
            )

            discounted_total = (
                cart["discountedTotal"] *
                random.uniform(0.9, 1.1)
            )

            records.append({

                "user_id": cart["userId"],

                # Binary/Categorical
                "gender": 1 if user["gender"] == "male" else 0,

                # Demographic (optional)
                "age": user["age"],

                # Aggregation
                "total_products": total_products,
                "total_quantity": total_quantity,
                "total_amount": total_amount,
                "discounted_total": discounted_total

            })

    df = pd.DataFrame(records)

    os.makedirs(
        "data/raw",
        exist_ok=True
    )

    df.to_csv(
        "data/raw/customer_data.csv",
        index=False
    )

    return df


if __name__ == "__main__":

    df = load_data()

    print(df.head())
    print("\nShape:", df.shape)
    print(
        "\nSaved to data/raw/customer_data.csv"
    )