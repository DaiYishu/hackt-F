"""
Compute Technical Indicators: MACD (12,26,9) & Stochastic Oscillator (14,3)
===========================================================================
Reads  : data/data_15min.parquet   (15-min OHLCV, ~656 assets)
Writes : data/data_features.parquet
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent / "data"
INPUT_PATH = DATA_DIR / "data_15min.parquet"
OUTPUT_PATH = DATA_DIR / "data_features.parquet"

# =====================================================================
# STEP 1 — Load dataset
# =====================================================================
print("[1/6] Loading data …")
df = pd.read_parquet(INPUT_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["asset_id", "date"]).reset_index(drop=True)
print(f"      {len(df):,} rows  |  {df['asset_id'].nunique()} assets")

# =====================================================================
# STEP 2 — Compute MACD (12, 26, 9)
# =====================================================================
print("[2/6] Computing MACD (12, 26, 9) per asset …")

# EMA must be computed within each asset group.
# pandas ewm with adjust=False gives the recursive EMA formula.
df["ema12"] = (
    df.groupby("asset_id")["close"]
    .transform(lambda s: s.ewm(span=12, adjust=False).mean())
)
df["ema26"] = (
    df.groupby("asset_id")["close"]
    .transform(lambda s: s.ewm(span=26, adjust=False).mean())
)

df["macd"] = df["ema12"] - df["ema26"]

# Signal line = EMA(MACD, span=9)
df["macd_signal"] = (
    df.groupby("asset_id")["macd"]
    .transform(lambda s: s.ewm(span=9, adjust=False).mean())
)

df["macd_hist"] = df["macd"] - df["macd_signal"]

# Drop helper columns
df.drop(columns=["ema12", "ema26"], inplace=True)

print("      MACD done.")

# =====================================================================
# STEP 3 — Compute Stochastic Oscillator (14, 3)
# =====================================================================
print("[3/6] Computing Stochastic Oscillator (14, 3) per asset …")

# Rolling min/max of low/high over 14 periods, within each asset.
df["lowest_low_14"] = (
    df.groupby("asset_id")["low"]
    .transform(lambda s: s.rolling(14, min_periods=14).min())
)
df["highest_high_14"] = (
    df.groupby("asset_id")["high"]
    .transform(lambda s: s.rolling(14, min_periods=14).max())
)

# %K
denom = df["highest_high_14"] - df["lowest_low_14"]
df["stoch_k"] = np.where(denom != 0,
                         (df["close"] - df["lowest_low_14"]) / denom,
                         np.nan)

# %D = 3-period simple moving average of %K
df["stoch_d"] = (
    df.groupby("asset_id")["stoch_k"]
    .transform(lambda s: s.rolling(3, min_periods=3).mean())
)

# Drop helper columns
df.drop(columns=["lowest_low_14", "highest_high_14"], inplace=True)

print("      Stochastic done.")

# =====================================================================
# STEP 4 — Clean: drop rows with NaN indicators
# =====================================================================
print("[4/6] Dropping rows with missing indicators …")
indicator_cols = ["macd", "macd_signal", "macd_hist", "stoch_k", "stoch_d"]
before = len(df)
df = df.dropna(subset=indicator_cols).reset_index(drop=True)
print(f"      Dropped {before - len(df):,} warm-up rows  →  {len(df):,} remain")

# =====================================================================
# STEP 5 — Validation checks
# =====================================================================
print("[5/6] Validation checks …")

# 5-a  Summary statistics
print("\n── Summary statistics ──")
print(df[indicator_cols].describe().to_string())

# 5-b  Stochastic range check
out_of_range = ((df["stoch_k"] < 0) | (df["stoch_k"] > 1)).sum()
pct_oor = out_of_range / len(df) * 100
print(f"\n── Stochastic range check ──")
print(f"   stoch_k outside [0, 1]: {out_of_range:,} rows ({pct_oor:.4f}%)")

out_of_range_d = ((df["stoch_d"] < 0) | (df["stoch_d"] > 1)).sum()
pct_oor_d = out_of_range_d / len(df) * 100
print(f"   stoch_d outside [0, 1]: {out_of_range_d:,} rows ({pct_oor_d:.4f}%)")

# 5-c  Row / asset counts
n_assets = df["asset_id"].nunique()
print(f"\n── Final dataset ──")
print(f"   Rows   : {len(df):,}")
print(f"   Assets : {n_assets}")
print(f"   Date   : {df['date'].min()}  →  {df['date'].max()}")

# 5-d  Sample plots for 3 random assets
rng = np.random.default_rng(42)
sample_assets = rng.choice(df["asset_id"].unique(), size=min(3, n_assets), replace=False)

for asset in sample_assets:
    sub = df.loc[df["asset_id"] == asset].tail(200)  # last 200 bars
    if len(sub) < 20:
        continue

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Asset: {asset}  (last {len(sub)} bars)", fontsize=13)

    # Close price
    axes[0].plot(sub["date"], sub["close"], linewidth=0.8)
    axes[0].set_ylabel("Close")
    axes[0].grid(True, alpha=0.3)

    # MACD
    axes[1].plot(sub["date"], sub["macd"], label="MACD", linewidth=0.8)
    axes[1].plot(sub["date"], sub["macd_signal"], label="Signal", linewidth=0.8)
    axes[1].bar(sub["date"], sub["macd_hist"], width=0.005, alpha=0.4, label="Hist")
    axes[1].set_ylabel("MACD")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Stochastic
    axes[2].plot(sub["date"], sub["stoch_k"], label="%K", linewidth=0.8)
    axes[2].plot(sub["date"], sub["stoch_d"], label="%D", linewidth=0.8)
    axes[2].axhline(0.8, color="red",  linestyle="--", linewidth=0.5, alpha=0.6)
    axes[2].axhline(0.2, color="green", linestyle="--", linewidth=0.5, alpha=0.6)
    axes[2].set_ylabel("Stochastic")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = DATA_DIR / f"indicators_{asset.replace('.', '_')}.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"   Plot saved: {plot_path.name}")

# 5-e  Head preview
print("\n── Preview (head 10) ──")
print(df.head(10).to_string(index=False))

# =====================================================================
# STEP 6 — Save
# =====================================================================
final_cols = ["asset_id", "date", "open", "high", "low", "close", "volume",
              "macd", "macd_signal", "macd_hist", "stoch_k", "stoch_d"]
df = df[final_cols]

print(f"\n[6/6] Saving to {OUTPUT_PATH.name} …")
df.to_parquet(OUTPUT_PATH, index=False)
print("      Done ✓")
