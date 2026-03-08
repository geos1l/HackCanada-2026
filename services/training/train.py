"""
Phase 7 — XGBoost Model Training (Julie's half)

Trains an XGBoost Regressor on features.parquet to predict relative_lst_c
(how much hotter/cooler each cell is vs the city median LST).

Input:
  data/processed/features.parquet  <- output of Phase 6 (features.py)

Output:
  models/xgboost_heat_model.json   <- loaded by Georgio's evaluate.py
  models/train_test_split.json     <- cell_id lists for train/test sets (for Georgio)

Run:
  python services/training/train.py
"""

import sys
import json
import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT   = Path(__file__).resolve().parents[2]
FEATURES_PATH  = PROJECT_ROOT / "data/processed/features.parquet"
MODELS_DIR     = PROJECT_ROOT / "models"
MODEL_PATH     = MODELS_DIR / "xgboost_heat_model.json"
SPLIT_PATH     = MODELS_DIR / "train_test_split.json"

# ── Feature columns (all seg_, gis_, Landsat, and fusion features) ─────────────

FEATURE_COLS = [
    # Branch A — segmentation
    "seg_building_pct",
    "seg_road_pct",
    "seg_vegetation_pct",
    "seg_water_pct",
    "seg_land_pct",
    "seg_unlabeled_pct",
    # Branch B — Landsat
    "ndvi_mean",
    "brightness_mean",
    "nir_mean",
    # Branch C — GIS
    "gis_building_coverage",
    "gis_road_coverage",
    "gis_park_coverage",
    "water_distance_m",
    # Fusion
    "building_disagreement",
    "road_disagreement",
    "green_consensus",
]

TARGET_COL = "relative_lst_c"


# ── Load ───────────────────────────────────────────────────────────────────────

def load_features() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        print(f"ERROR: {FEATURES_PATH} not found.")
        print("Run Phase 6 (features.py) first.")
        sys.exit(1)

    df = pd.read_parquet(FEATURES_PATH)
    print(f"Loaded features.parquet: {len(df):,} rows, {len(df.columns)} columns")

    missing_cols = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing_cols:
        print(f"ERROR: missing columns: {missing_cols}")
        sys.exit(1)

    return df


# ── Split ──────────────────────────────────────────────────────────────────────

def split(df: pd.DataFrame) -> tuple:
    """
    80/20 random train/test split.
    (MVP uses stub Landsat labels so temporal split is not applicable.
    Post-demo: switch to temporal split by scene date to avoid leakage.)
    """
    from sklearn.model_selection import train_test_split

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, df["cell_id"],
        test_size=0.2,
        random_state=42,
    )

    print(f"Train: {len(X_train):,} cells  |  Test: {len(X_test):,} cells")
    return X_train, X_test, y_train, y_test, ids_train, ids_test


# ── Train ──────────────────────────────────────────────────────────────────────

def train(X_train, y_train):
    try:
        import xgboost as xgb
    except ImportError:
        print("ERROR: xgboost not installed. Run: pip install xgboost")
        sys.exit(1)

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    print("\nTraining XGBoost model...")
    model.fit(X_train, y_train)
    print("  Training complete.")
    return model


# ── Quick eval ─────────────────────────────────────────────────────────────────

def quick_eval(model, X_test, y_test) -> None:
    """
    Prints quick train-set metrics so Julie can sanity-check before handing off.
    Georgio runs the full evaluation in evaluate.py.
    """
    import numpy as np
    preds = model.predict(X_test)
    mae  = float(np.mean(np.abs(preds - y_test)))
    rmse = float(np.sqrt(np.mean((preds - y_test) ** 2)))
    ss_res = float(np.sum((preds - y_test) ** 2))
    ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
    r2   = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    print(f"\nQuick test-set metrics (Georgio will run full eval):")
    print(f"  MAE  : {mae:.4f} °C")
    print(f"  RMSE : {rmse:.4f} °C")
    print(f"  R²   : {r2:.4f}")


# ── Save ───────────────────────────────────────────────────────────────────────

def save(model, ids_train, ids_test) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model.save_model(MODEL_PATH)
    print(f"\nModel saved: {MODEL_PATH}")

    split_data = {
        "train_cell_ids": list(ids_train),
        "test_cell_ids":  list(ids_test),
    }
    with open(SPLIT_PATH, "w") as f:
        json.dump(split_data, f)
    print(f"Split saved : {SPLIT_PATH}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=== Phase 7: XGBoost Model Training ===\n")

    try:
        from sklearn.model_selection import train_test_split  # noqa: F401
    except ImportError:
        print("ERROR: scikit-learn not installed. Run: pip install scikit-learn")
        sys.exit(1)

    df = load_features()
    X_train, X_test, y_train, y_test, ids_train, ids_test = split(df)
    model = train(X_train, y_train)
    quick_eval(model, X_test, y_test)
    save(model, ids_train, ids_test)

    print("\nPhase 7 complete.")
    print("Tell Georgio: models/xgboost_heat_model.json + models/train_test_split.json are ready.")
    print("He runs evaluate.py → predictions.parquet.")


if __name__ == "__main__":
    main()
