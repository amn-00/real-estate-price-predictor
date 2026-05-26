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
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
META_PATH  = os.path.join(os.path.dirname(__file__), "meta.joblib")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 4 domain-informed features on top of raw 8."""
    df = df.copy()
    df["rooms_per_person"]  = df["AveRooms"]   / df["AveOccup"].clip(lower=0.1)
    df["bedrooms_ratio"]    = df["AveBedrms"]  / df["AveRooms"].clip(lower=0.1)
    df["income_per_room"]   = df["MedInc"]     / df["AveRooms"].clip(lower=0.1)
    df["pop_per_household"] = df["Population"] / df["AveOccup"].clip(lower=0.1)
    return df


def remove_outliers(X: pd.DataFrame, y: pd.Series):
    """IQR-based outlier removal on target."""
    q1, q3 = y.quantile(0.05), y.quantile(0.95)
    mask = (y >= q1) & (y <= q3)
    return X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)


def train_and_save():
    print("📦 Loading California Housing dataset...")
    data    = fetch_california_housing(as_frame=True)
    X_raw   = data.frame.drop(columns=["MedHouseVal"])
    y       = data.frame["MedHouseVal"]

    print(f"   {len(X_raw):,} rows · {X_raw.shape[1]} raw features")

    # Feature engineering
    X = engineer_features(X_raw)

    # Outlier removal
    X, y = remove_outliers(X, y)
    print(f"   {len(X):,} rows after outlier removal")

    # Train / test split (80/20, stratified by binned price)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── Baseline: Linear Regression ──────────────────────────────────────────
    print("\n📐 Training baseline (Linear Regression)...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr  = lr.predict(X_test)
    rmse_lr    = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    r2_lr      = r2_score(y_test, y_pred_lr)
    print(f"   Linear Regression → R²: {r2_lr:.3f}  RMSE: {rmse_lr:.4f}")

    # ── XGBoost with GridSearchCV ─────────────────────────────────────────────
    print("\n🌲 Tuning XGBoost with 5-fold GridSearchCV...")
    param_grid = {
        "n_estimators":     [300, 500],
        "max_depth":        [4, 6],
        "learning_rate":    [0.05, 0.1],
        "subsample":        [0.8],
        "colsample_bytree": [0.8],
    }
    xgb = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )
    grid_search = GridSearchCV(
        xgb, param_grid,
        cv=5, scoring="r2",
        n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)
    best_xgb   = grid_search.best_estimator_
    print(f"   Best params: {grid_search.best_params_}")

    y_pred_xgb = best_xgb.predict(X_test)
    rmse_xgb   = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    r2_xgb     = r2_score(y_test, y_pred_xgb)
    print(f"   XGBoost      → R²: {r2_xgb:.3f}  RMSE: {rmse_xgb:.4f}")

    # ── Random Forest (benchmark) ─────────────────────────────────────────────
    print("\n🌳 Training Random Forest (benchmark)...")
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf  = rf.predict(X_test)
    rmse_rf    = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf      = r2_score(y_test, y_pred_rf)
    print(f"   Random Forest → R²: {r2_rf:.3f}  RMSE: {rmse_rf:.4f}")

    # ── RMSE improvement calculation ──────────────────────────────────────────
    rmse_improvement = (rmse_lr - rmse_xgb) / rmse_lr * 100

    # ── Save model + metadata ─────────────────────────────────────────────────
    meta = {
        "r2":                   round(r2_xgb,  3),
        "rmse":                 round(rmse_xgb, 4),
        "rmse_lr":              round(rmse_lr,  4),
        "rmse_rf":              round(rmse_rf,  4),
        "r2_lr":                round(r2_lr,   3),
        "r2_rf":                round(r2_rf,   3),
        "rmse_improvement_pct": round(rmse_improvement, 1),
        "train_rows":           len(X_train),
        "n_features":           X.shape[1],
        "best_params":          grid_search.best_params_,
        "feature_names":        list(X.columns),
    }

    joblib.dump(best_xgb, MODEL_PATH)
    joblib.dump(meta,     META_PATH)

    print(f"\n{'='*50}")
    print(f"  ✅ Model saved")
    print(f"  R² Score          : {r2_xgb:.3f}")
    print(f"  RMSE Improvement  : {rmse_improvement:.1f}% over linear baseline")
    print(f"  Training rows     : {len(X_train):,}")
    print(f"  Features          : {X.shape[1]}")
    print(f"{'='*50}\n")
    return meta


if __name__ == "__main__":
    train_and_save()
