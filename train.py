"""
Training Pipeline — Real Estate Price Predictor
- Dataset : California Housing (sklearn) — 20,640 rows, 8 features
- Features : 8 raw + 4 engineered = 12 total
- Models   : Linear Regression (baseline) vs XGBoost vs Random Forest
- Tuning   : 5-fold GridSearchCV on XGBoost
- Serialise: joblib for model + metadata
"""

import joblib, os, warnings
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

MODEL_PATH   = os.path.join(os.path.dirname(__file__), "model.joblib")
META_PATH    = os.path.join(os.path.dirname(__file__), "meta.joblib")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset.joblib")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rooms_per_person"]  = df["AveRooms"]   / df["AveOccup"].clip(lower=0.1)
    df["bedrooms_ratio"]    = df["AveBedrms"]  / df["AveRooms"].clip(lower=0.1)
    df["income_per_room"]   = df["MedInc"]     / df["AveRooms"].clip(lower=0.1)
    df["pop_per_household"] = df["Population"] / df["AveOccup"].clip(lower=0.1)
    return df


def remove_outliers(X, y):
    q1, q3 = y.quantile(0.05), y.quantile(0.95)
    mask = (y >= q1) & (y <= q3)
    return X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)


def train_and_save():
    print("Loading California Housing dataset...")
    data  = fetch_california_housing(as_frame=True)
    X_raw = data.frame.drop(columns=["MedHouseVal"])
    y     = data.frame["MedHouseVal"]

    X = engineer_features(X_raw)
    X, y = remove_outliers(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = LinearRegression().fit(X_train, y_train)
    rmse_lr = np.sqrt(mean_squared_error(y_test, lr.predict(X_test)))
    r2_lr   = r2_score(y_test, lr.predict(X_test))

    param_grid = {
        "n_estimators": [300, 500], "max_depth": [4, 6],
        "learning_rate": [0.05, 0.1], "subsample": [0.8], "colsample_bytree": [0.8],
    }
    xgb = XGBRegressor(objective="reg:squarederror", random_state=42, verbosity=0, n_jobs=-1)
    gs  = GridSearchCV(xgb, param_grid, cv=5, scoring="r2", n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)
    best_xgb = gs.best_estimator_

    y_pred   = best_xgb.predict(X_test)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_xgb   = r2_score(y_test, y_pred)

    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1).fit(X_train, y_train)
    rmse_rf = np.sqrt(mean_squared_error(y_test, rf.predict(X_test)))
    r2_rf   = r2_score(y_test, rf.predict(X_test))

    meta = {
        "r2": round(r2_xgb, 3), "rmse": round(rmse_xgb, 4),
        "rmse_lr": round(rmse_lr, 4), "rmse_rf": round(rmse_rf, 4),
        "r2_lr": round(r2_lr, 3), "r2_rf": round(r2_rf, 3),
        "rmse_improvement_pct": round((rmse_lr - rmse_xgb) / rmse_lr * 100, 1),
        "train_rows": len(X_train), "n_features": X.shape[1],
        "best_params": gs.best_params_, "feature_names": list(X.columns),
    }

    joblib.dump(best_xgb, MODEL_PATH)
    joblib.dump(meta, META_PATH)
    # Save a sample of dataset for comparables
    sample = X.copy()
    sample["price"] = y.values * 100_000
    joblib.dump(sample.reset_index(drop=True), DATASET_PATH)

    print(f"R²: {r2_xgb:.3f} | RMSE improvement: {meta['rmse_improvement_pct']}%")
    return meta


if __name__ == "__main__":
    train_and_save()
