from scraper import load_data
from features import create_features

from sklearn.model_selection import (
    train_test_split
)

from sklearn.ensemble import (
    RandomForestClassifier
)

import joblib

df = load_data()

df = create_features(df)

X = df.drop(
    columns=[
        "user_id",
        "churned"
    ]
)

y = df["churned"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

print(
    "Accuracy:",
    model.score(
        X_test,
        y_test
    )
)

joblib.dump(
    model,
    "app/model.pkl"
)