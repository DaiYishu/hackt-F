"""
Resample 5-minute OHLCV bars to 15-minute bars
===============================================
Reads cleaned_data.parquet (≈53 M rows, 656 assets, 2010–2025)
and produces data_15min.parquet with proper per-asset resampling.

Strategy: floor every timestamp to its 15-min boundary, then groupby
(asset_id, floored_date).  This never creates artificial empty rows and
is far more memory-efficient than groupby().resample() on 53 M rows.
"""

import pandas as pd
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent / "data"
INPUT_PATH = DATA_DIR / "cleaned_data.parquet"
OUTPUT_PATH = DATA_DIR / "data_15min.parquet"

# ── 1. Load ──────────────────────────────────────────────────────────────
print("[1/6] Loading parquet …")
df = pd.read_parquet(INPUT_PATH, columns=["date", "open", "high", "low",
                                          "close", "volume", "asset_id"])
print(f"      Loaded {len(df):,} rows  |  {df['asset_id'].nunique()} assets")

# Ensure correct types
df["date"] = pd.to_datetime(df["date"])

# ── 2. Floor timestamps to 15-min boundaries ────────────────────────────
#    e.g. 09:30, 09:35, 09:40 → all map to 09:30
#    This defines the 15-min window each row belongs to.
print("[2/6] Flooring timestamps to 15-min windows …")
df["date_15"] = df["date"].dt.floor("15min")

# ── 3. Sort so that "first" / "last" inside each group are correct ──────
print("[3/6] Sorting by (asset_id, date) …")
df = df.sort_values(["asset_id", "date"])

# ── 4. Aggregate to 15-min bars per asset ────────────────────────────────
print("[4/6] Aggregating to 15-minute bars (per asset) …")

df_15 = (
    df
    .groupby(["asset_id", "date_15"], sort=False)
    .agg(
        open  =("open",   "first"),
        high  =("high",   "max"),
        low   =("low",    "min"),
        close =("close",  "last"),
        volume=("volume", "sum"),
    )
    .reset_index()
    .rename(columns={"date_15": "date"})
)

# ── 5. Drop rows where OHLC is missing ──────────────────────────────────
before = len(df_15)
df_15 = df_15.dropna(subset=["open", "high", "low", "close"])
dropped = before - len(df_15)
if dropped:
    print(f"      Dropped {dropped:,} rows with missing OHLC")

# Ensure final column order
df_15 = df_15[["asset_id", "date", "open", "high", "low", "close", "volume"]]

# ── 6. Sanity checks ────────────────────────────────────────────────────
print("[5/6] Sanity checks:")
print(f"      Rows          : {len(df_15):,}")
print(f"      Unique assets : {df_15['asset_id'].nunique()}")
print(f"      Date range    : {df_15['date'].min()}  →  {df_15['date'].max()}")
print()
print("      Preview (head):")
print(df_15.head(10).to_string(index=False))

# ── 7. Save ─────────────────────────────────────────────────────────────
print(f"\n[6/6] Saving to {OUTPUT_PATH} …")
df_15.to_parquet(OUTPUT_PATH, index=False)
print("      Done ✓")
