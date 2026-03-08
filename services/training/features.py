"""
Phase 6 — Feature Engineering (Julie's half)

Merges all three branch parquets on cell_id, computes fusion features,
and writes the combined DataFrame for Georgio to validate and save as
data/processed/features.parquet.

Inputs (all required before running):
  data/processed/segmentation_cell_features.parquet  <- Georgio (Phase 2)
  data/processed/gis_cell_features.parquet           <- Georgio (Phase 3)
  data/processed/landsat_cell_features.parquet       <- Georgio (Phase 4 stub)

Output:
  data/processed/features.parquet
    Inner join on cell_id — only cells with data in ALL three branches survive.
    For MVP AOI (~400 cells); full 68k cells once real Landsat data is in.

Fusion features added:
  building_disagreement  = abs(seg_building_pct - gis_building_coverage)
  road_disagreement      = abs(seg_road_pct - gis_road_coverage)
  green_consensus        = (seg_vegetation_pct + gis_park_coverage) / 2

Run:
  python services/training/features.py
"""

import sys
import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data/processed"

SEG_PATH     = DATA / "segmentation_cell_features.parquet"
GIS_PATH     = DATA / "gis_cell_features.parquet"
LANDSAT_PATH = DATA / "landsat_cell_features.parquet"
OUTPUT_PATH  = DATA / "features.parquet"


# ── Load ───────────────────────────────────────────────────────────────────────

def load_inputs() -> tuple:
    missing = [p for p in [SEG_PATH, GIS_PATH, LANDSAT_PATH] if not p.exists()]
    if missing:
        for p in missing:
            print(f"ERROR: missing input: {p}")
        print("\nWait for Georgio to produce the missing parquet(s) before running.")
        sys.exit(1)

    seg     = pd.read_parquet(SEG_PATH)
    gis     = pd.read_parquet(GIS_PATH)
    landsat = pd.read_parquet(LANDSAT_PATH)

    print(f"segmentation_cell_features : {len(seg):,} rows  columns: {list(seg.columns)}")
    print(f"gis_cell_features          : {len(gis):,} rows  columns: {list(gis.columns)}")
    print(f"landsat_cell_features      : {len(landsat):,} rows  columns: {list(landsat.columns)}")

    return seg, gis, landsat


# ── Join ───────────────────────────────────────────────────────────────────────

def join_branches(seg: pd.DataFrame, gis: pd.DataFrame, landsat: pd.DataFrame) -> pd.DataFrame:
    """
    Inner join on cell_id — only cells with data in all three branches survive.
    Logs any cell_ids that are dropped so Georgio can inspect gaps.
    """
    all_ids = set(seg["cell_id"]) | set(gis["cell_id"]) | set(landsat["cell_id"])

    merged = (
        seg
        .merge(gis,     on="cell_id", how="inner")
        .merge(landsat, on="cell_id", how="inner")
    )

    dropped = len(all_ids) - len(merged)
    print(f"\nJoin result: {len(merged):,} cells (dropped {dropped:,} with incomplete data)")

    if dropped > 0:
        missing_from_seg     = set(gis["cell_id"]) - set(seg["cell_id"])
        missing_from_gis     = set(seg["cell_id"]) - set(gis["cell_id"])
        missing_from_landsat = set(seg["cell_id"]) - set(landsat["cell_id"])
        if missing_from_seg:
            print(f"  cell_ids in GIS but not SEG:     {len(missing_from_seg):,}")
        if missing_from_gis:
            print(f"  cell_ids in SEG but not GIS:     {len(missing_from_gis):,}")
        if missing_from_landsat:
            print(f"  cell_ids in SEG but not Landsat: {len(missing_from_landsat):,}")

    return merged


# ── Fusion features ────────────────────────────────────────────────────────────

def drop_incomplete_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops rows where any feature column is null.
    The branch parquets cover all 68k cells but only AOI cells have real values —
    non-AOI cells are NaN. This reduces the DataFrame to only cells with complete data.
    """
    before = len(df)
    df = df.dropna(subset=[c for c in df.columns if c != "cell_id"])
    after = len(df)
    print(f"  Dropped {before - after:,} rows with null values → {after:,} complete cells")
    if after == 0:
        raise RuntimeError("No cells with complete data across all three branches. Check inputs.")
    return df


def add_fusion_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes cross-branch features that capture agreement/disagreement
    between segmentation model outputs and GIS ground truth.
    """
    df = df.copy()
    df["building_disagreement"] = (
        df["seg_building_pct"] - df["gis_building_coverage"]
    ).abs()
    df["road_disagreement"] = (
        df["seg_road_pct"] - df["gis_road_coverage"]
    ).abs()
    df["green_consensus"] = (
        df["seg_vegetation_pct"] + df["gis_park_coverage"]
    ) / 2

    print(f"\nFusion features added:")
    print(f"  building_disagreement  mean={df['building_disagreement'].mean():.4f}")
    print(f"  road_disagreement      mean={df['road_disagreement'].mean():.4f}")
    print(f"  green_consensus        mean={df['green_consensus'].mean():.4f}")

    return df


# ── Validate ───────────────────────────────────────────────────────────────────

def validate(df: pd.DataFrame) -> None:
    """Checks schema matches 00_PROJECT_CORE.md cell schema exactly."""
    print("\nValidating schema...")

    required = {
        "cell_id",
        "seg_building_pct", "seg_road_pct", "seg_vegetation_pct",
        "seg_water_pct", "seg_land_pct", "seg_unlabeled_pct",
        "ndvi_mean", "brightness_mean", "nir_mean", "lst_c", "relative_lst_c",
        "gis_building_coverage", "gis_road_coverage", "gis_park_coverage",
        "water_distance_m",
        "building_disagreement", "road_disagreement", "green_consensus",
    }
    missing = required - set(df.columns)
    assert not missing, f"FAIL: missing columns: {missing}"
    print(f"  All required columns present")

    nulls = df[list(required)].isnull().sum()
    nulls = nulls[nulls > 0]
    if len(nulls):
        print(f"  WARNING: null values found:\n{nulls}")
    else:
        print(f"  No null values")

    dupes = df["cell_id"].duplicated().sum()
    assert dupes == 0, f"FAIL: {dupes} duplicate cell_ids"
    print(f"  No duplicate cell_ids")

    print(f"  All checks passed. {len(df):,} cells ready for training.\n")


# ── Save ───────────────────────────────────────────────────────────────────────

def save(df: pd.DataFrame) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved: {OUTPUT_PATH}")
    print(f"  Rows: {len(df):,}  |  Columns: {len(df.columns)}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=== Phase 6: Feature Engineering ===\n")

    seg, gis, landsat = load_inputs()
    merged = join_branches(seg, gis, landsat)
    merged = drop_incomplete_rows(merged)
    features = add_fusion_features(merged)
    validate(features)
    save(features)

    print("Phase 6 complete.")
    print("Hand off to Georgio: he validates features.parquet and confirms schema.")
    print("Then run: python services/training/train.py")


if __name__ == "__main__":
    main()
