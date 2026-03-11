"""
Conservative Data Cleaning Pipeline for Intraday OHLCV Financial Time Series
=============================================================================
Designed for a hackathon forecasting project (MACD / Stochastic indicators).

Philosophy:
  - Flag problems, don't silently fix them.
  - Never forward-fill prices across large gaps.
  - Keep an audit trail (source_file, suspicious rows dataframe).
  - Make every step inspectable and modifiable.

Dependencies: pandas, numpy, openpyxl (for .xlsx reading)
"""

import os
import glob
import numpy as np
import pandas as pd
from datetime import timedelta


# ============================================================================
# 1. LOADING
# ============================================================================

def scan_xlsx_files(folder_path: str) -> list:
    """Return a sorted list of all .xlsx file paths in *folder_path*."""
    pattern = os.path.join(folder_path, "*.xlsx")
    files = sorted(glob.glob(pattern))
    print(f"[LOAD] Found {len(files)} xlsx files in: {folder_path}")
    return files


def extract_asset_id(filename: str) -> str:
    """
    Extract the asset identifier from a filename like 'WKL.AS_2023-01-01.xlsx'.
    Takes everything before the FIRST underscore.
    """
    base = os.path.basename(filename)           # 'WKL.AS_2023-01-01.xlsx'
    name_no_ext = os.path.splitext(base)[0]     # 'WKL.AS_2023-01-01'
    asset_id = name_no_ext.split("_", 1)[0]     # 'WKL.AS'
    return asset_id


def load_single_file(filepath: str) -> pd.DataFrame:
    """
    Read one Excel file and attach metadata columns.
    Returns an empty DataFrame (with correct columns) if the file is empty or
    cannot be read.
    """
    try:
        df = pd.read_excel(filepath, engine="openpyxl")
    except Exception as e:
        print(f"[WARN] Could not read {filepath}: {e}")
        return pd.DataFrame(columns=["date", "open", "high", "low", "close",
                                     "volume", "source_file", "asset_id"])

    # Standardise column names to lowercase and strip whitespace
    df.columns = df.columns.str.strip().str.lower()

    # Attach provenance
    df["source_file"] = os.path.basename(filepath)
    df["asset_id"] = extract_asset_id(filepath)
    return df


def load_all_files(folder_path: str) -> pd.DataFrame:
    """
    Scan *folder_path* for xlsx files, read them all, and concatenate into one
    DataFrame.  Prints progress every 200 files.
    """
    files = scan_xlsx_files(folder_path)
    if not files:
        raise FileNotFoundError(f"No xlsx files found in {folder_path}")

    frames = []
    for i, f in enumerate(files, 1):
        frames.append(load_single_file(f))
        if i % 200 == 0:
            print(f"[LOAD] ... processed {i}/{len(files)} files")

    df = pd.concat(frames, ignore_index=True)
    print(f"[LOAD] Concatenated shape: {df.shape}")
    return df


# ============================================================================
# 2. BASIC CLEANING
# ============================================================================

def standardise_and_parse(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Ensure column names are lowercase (idempotent).
    - Parse the 'date' column as datetime.
    - Sort by (asset_id, date).
    """
    df.columns = df.columns.str.strip().str.lower()

    # Parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    n_bad_dates = df["date"].isna().sum()
    if n_bad_dates > 0:
        print(f"[CLEAN] {n_bad_dates} rows have unparseable dates (set to NaT).")

    # Sort
    df = df.sort_values(["asset_id", "date"]).reset_index(drop=True)
    return df


def remove_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that are exact duplicates across ALL columns."""
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_dropped = n_before - len(df)
    print(f"[CLEAN] Removed {n_dropped} exact duplicate rows.")
    return df


# ============================================================================
# 3. QUALITY CHECKS  (all checks return info, none modify data silently)
# ============================================================================

def check_duplicated_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find rows where (asset_id, date) appears more than once.
    Returns a DataFrame of the offending rows.
    """
    mask = df.duplicated(subset=["asset_id", "date"], keep=False)
    dup_ts = df.loc[mask].copy()
    n = len(dup_ts)
    n_assets = dup_ts["asset_id"].nunique() if n > 0 else 0
    print(f"[CHECK] Duplicated timestamps: {n} rows across {n_assets} assets.")
    return dup_ts


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Report missing values per column.
    Returns a summary DataFrame: column, n_missing, pct_missing.
    """
    total = len(df)
    missing = df.isnull().sum()
    summary = pd.DataFrame({
        "column": missing.index,
        "n_missing": missing.values,
        "pct_missing": (missing.values / total * 100).round(4),
    })
    print("[CHECK] Missing values per column:")
    print(summary.to_string(index=False))
    return summary


def check_ohlc_validity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag rows where OHLC relationships are violated:
      - high < low
      - open < low  or  open > high
      - close < low or  close > high
    Returns a DataFrame containing only the suspicious rows, with an extra
    column 'ohlc_issue' describing the problem(s).
    """
    issues = []

    mask_hl = df["high"] < df["low"]
    mask_ol = df["open"] < df["low"]
    mask_oh = df["open"] > df["high"]
    mask_cl = df["close"] < df["low"]
    mask_ch = df["close"] > df["high"]

    any_bad = mask_hl | mask_ol | mask_oh | mask_cl | mask_ch

    bad = df.loc[any_bad].copy()

    # Build human-readable issue description
    descs = []
    for idx in bad.index:
        parts = []
        if mask_hl.loc[idx]:
            parts.append("high<low")
        if mask_ol.loc[idx]:
            parts.append("open<low")
        if mask_oh.loc[idx]:
            parts.append("open>high")
        if mask_cl.loc[idx]:
            parts.append("close<low")
        if mask_ch.loc[idx]:
            parts.append("close>high")
        descs.append("; ".join(parts))
    bad["ohlc_issue"] = descs

    print(f"[CHECK] OHLC violations: {len(bad)} rows.")
    return bad


def check_negative_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows with negative or zero volume."""
    mask = df["volume"] <= 0
    bad = df.loc[mask].copy()
    print(f"[CHECK] Non-positive volume: {len(bad)} rows.")
    return bad


# ============================================================================
# 4. TIME-GAP ANALYSIS
# ============================================================================

def compute_time_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the time difference between consecutive rows *within each asset*.
    Adds a 'time_gap' column (timedelta).  The first row of each asset gets NaT.
    Returns the modified DataFrame (a copy).
    """
    df = df.copy()
    df["time_gap"] = df.groupby("asset_id")["date"].diff()
    return df


def report_gap_frequencies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tabulate how often each distinct time gap occurs.
    Returns a DataFrame with columns: time_gap, count, pct.
    """
    gaps = df["time_gap"].dropna()
    freq = gaps.value_counts().reset_index()
    freq.columns = ["time_gap", "count"]
    freq["pct"] = (freq["count"] / freq["count"].sum() * 100).round(4)
    freq = freq.sort_values("count", ascending=False).reset_index(drop=True)

    print(f"[GAPS] Unique gap values: {len(freq)}")
    print("[GAPS] Top 15 most frequent gaps:")
    print(freq.head(15).to_string(index=False))
    return freq


def flag_large_gaps(df: pd.DataFrame,
                    threshold: timedelta = timedelta(minutes=30)
                    ) -> pd.DataFrame:
    """
    Return rows where time_gap > threshold.
    These are *not* removed or filled — just flagged for human review.

    Default threshold: 30 minutes (6x the expected 5-min bar).
    For assets that trade 24/7 (crypto), you may want a smaller threshold.
    For equities, overnight gaps (~16 h) are normal.
    """
    if "time_gap" not in df.columns:
        df = compute_time_gaps(df)

    mask = df["time_gap"] > threshold
    large = df.loc[mask, ["asset_id", "date", "time_gap", "source_file"]].copy()
    n_assets = large["asset_id"].nunique() if len(large) > 0 else 0
    print(f"[GAPS] Large gaps (>{threshold}): {len(large)} occurrences "
          f"across {n_assets} assets.")
    return large


# ============================================================================
# 5. ASSEMBLE QUALITY REPORT
# ============================================================================

def build_quality_report(df: pd.DataFrame,
                         dup_ts: pd.DataFrame,
                         missing_summary: pd.DataFrame,
                         ohlc_bad: pd.DataFrame,
                         neg_vol: pd.DataFrame,
                         gap_freq: pd.DataFrame,
                         large_gaps: pd.DataFrame) -> pd.DataFrame:
    """
    Compile a per-asset quality summary.
    One row per asset_id with counts of each issue type.
    """
    assets = df["asset_id"].unique()
    records = []
    for aid in assets:
        sub = df[df["asset_id"] == aid]
        rec = {
            "asset_id": aid,
            "total_rows": len(sub),
            "date_min": sub["date"].min(),
            "date_max": sub["date"].max(),
            "duplicated_timestamps": len(dup_ts[dup_ts["asset_id"] == aid]),
            "missing_date": sub["date"].isna().sum(),
            "missing_open": sub["open"].isna().sum(),
            "missing_high": sub["high"].isna().sum(),
            "missing_low": sub["low"].isna().sum(),
            "missing_close": sub["close"].isna().sum(),
            "missing_volume": sub["volume"].isna().sum(),
            "ohlc_violations": len(ohlc_bad[ohlc_bad["asset_id"] == aid]),
            "negative_volume": len(neg_vol[neg_vol["asset_id"] == aid]),
            "large_gaps": len(large_gaps[large_gaps["asset_id"] == aid]),
        }
        records.append(rec)

    report = pd.DataFrame(records)
    report = report.sort_values("asset_id").reset_index(drop=True)
    print(f"\n[REPORT] Quality report built for {len(report)} assets.")
    return report


# ============================================================================
# 6. ASSEMBLE SUSPICIOUS ROWS
# ============================================================================

def collect_suspicious_rows(dup_ts: pd.DataFrame,
                            ohlc_bad: pd.DataFrame,
                            neg_vol: pd.DataFrame) -> pd.DataFrame:
    """
    Merge all suspicious rows into one DataFrame with a 'reason' column.
    """
    parts = []

    if len(dup_ts) > 0:
        tmp = dup_ts.copy()
        tmp["reason"] = "duplicated_timestamp"
        parts.append(tmp)

    if len(ohlc_bad) > 0:
        tmp = ohlc_bad.copy()
        tmp["reason"] = "ohlc_violation: " + tmp["ohlc_issue"]
        tmp = tmp.drop(columns=["ohlc_issue"], errors="ignore")
        parts.append(tmp)

    if len(neg_vol) > 0:
        tmp = neg_vol.copy()
        tmp["reason"] = "non_positive_volume"
        parts.append(tmp)

    if parts:
        suspicious = pd.concat(parts, ignore_index=True)
    else:
        suspicious = pd.DataFrame()

    print(f"[SUSPICIOUS] Total suspicious rows collected: {len(suspicious)}")
    return suspicious


# ============================================================================
# 7. MAIN PIPELINE
# ============================================================================

def clean_pipeline(folder_path: str,
                   gap_threshold_minutes: int = 30
                   ) -> tuple:
    """
    Run the full cleaning pipeline.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing .xlsx files.
    gap_threshold_minutes : int
        Time gaps larger than this (in minutes) will be flagged.

    Returns
    -------
    df_clean : pd.DataFrame
        Cleaned (but conservatively: no price imputation) DataFrame.
    quality_report : pd.DataFrame
        One row per asset with issue counts.
    suspicious : pd.DataFrame
        All suspicious rows with a 'reason' column.
    """
    print("=" * 70)
    print("  DATA CLEANING PIPELINE — CONSERVATIVE MODE")
    print("=" * 70)

    # --- Load ---
    df = load_all_files(folder_path)

    # --- Basic cleaning ---
    df = standardise_and_parse(df)
    df = remove_exact_duplicates(df)

    # --- Quality checks ---
    dup_ts = check_duplicated_timestamps(df)
    missing_summary = check_missing_values(df)
    ohlc_bad = check_ohlc_validity(df)
    neg_vol = check_negative_volume(df)

    # --- Time gaps ---
    df = compute_time_gaps(df)
    gap_freq = report_gap_frequencies(df)
    large_gaps = flag_large_gaps(df, threshold=timedelta(minutes=gap_threshold_minutes))

    # --- Reports ---
    quality_report = build_quality_report(
        df, dup_ts, missing_summary, ohlc_bad, neg_vol, gap_freq, large_gaps
    )
    suspicious = collect_suspicious_rows(dup_ts, ohlc_bad, neg_vol)

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Clean DataFrame shape : {df.shape}")
    print(f"  Unique assets         : {df['asset_id'].nunique()}")
    print(f"  Date range            : {df['date'].min()} → {df['date'].max()}")
    print(f"  Quality-report rows   : {len(quality_report)}")
    print(f"  Suspicious rows       : {len(suspicious)}")
    print("=" * 70)

    return df, quality_report, suspicious


# ============================================================================
# 8. EXAMPLE USAGE
# ============================================================================

def main():
    """
    Example usage — adjust DATA_FOLDER to point to your xlsx directory.
    """
    # ---- CONFIGURE THIS ----
    DATA_FOLDER = os.path.join(os.path.dirname(__file__), "data", "datas")

    # Run the pipeline
    df_clean, quality_report, suspicious = clean_pipeline(
        folder_path=DATA_FOLDER,
        gap_threshold_minutes=30,   # flag gaps > 30 min
    )

    # ---- Inspect results ----
    print("\n--- Quality Report (first 10 rows) ---")
    print(quality_report.head(10).to_string(index=False))

    print("\n--- Suspicious Rows (first 10) ---")
    if len(suspicious) > 0:
        print(suspicious.head(10).to_string(index=False))
    else:
        print("  None found.")

    # ---- Optional: save outputs ----
    out_dir = os.path.join(os.path.dirname(__file__), "data")
    df_clean.to_parquet(os.path.join(out_dir, "cleaned_data.parquet"), index=False)
    quality_report.to_csv(os.path.join(out_dir, "quality_report.csv"), index=False)
    if len(suspicious) > 0:
        suspicious.to_csv(os.path.join(out_dir, "suspicious_rows.csv"), index=False)
    print(f"\n[SAVE] Outputs written to {out_dir}")


if __name__ == "__main__":
    main()
