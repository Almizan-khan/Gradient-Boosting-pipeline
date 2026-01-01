import os
import sys
import numpy as np
import pandas as pd

try:
    from catboost import CatBoostRegressor
except Exception:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost"])
    from catboost import CatBoostRegressor

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_df = pd.read_csv("/kaggle/input/playground-series-s6e1/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s6e1/test.csv")

X = train_df.drop(columns=["id", "exam_score"])
y = train_df["exam_score"]

X_test = test_df.drop(columns=["id"])

categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

print("Categorical features:")
print(categorical_features)

N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\n--- Fold {fold+1} ---")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.03,
        depth=8,
        loss_function="RMSE",
        eval_metric="RMSE",
        cat_features=categorical_features,
        random_seed=42,
        verbose=200,
        early_stopping_rounds=200,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )

    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_test) / N_SPLITS

rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"\nFinal CV RMSE: {rmse:.5f}")

feature_importance = model.get_feature_importance(prettified=True)
print(feature_importance)

submission = pd.DataFrame({
    "id": test_df["id"],
    "exam_score": test_preds
})

submission.to_csv("submission.csv", index=False)
print("submission.csv created")