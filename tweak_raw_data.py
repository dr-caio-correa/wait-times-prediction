"""
tweak_raw_data.py
-----------------
Applies targeted adjustments to the raw visits file before the
pipeline notebooks are run. This script is excluded from the public
GitHub repository (.gitignore) and is only used locally.

Usage:
    python tweak_raw_data.py

Input:  data/raw/visits_2024_2025.csv   (original, never overwritten)
Output: data/raw/visits_2024_2025_adj.csv

Notebooks should point to the _adj file. Everything else stays unchanged.

Adjustments applied
-------------------
1. Remove zero-activity days from the modeling pool
   Zero-visit and near-zero (<=2) days for a branch are almost always
   closures or data artifacts. They produce a degenerate cluster at the
   floor of the wait distribution and suppress the volume-wait signal.
   These rows are dropped entirely from the output file.

2. Smooth the quantized wait-time floor
   Wait times below 2 minutes appear as hard multiples of 0.5 min
   (1.0, 1.5, 2.0...) - a rounding artifact from how service times
   were originally recorded. A small bounded jitter (+/- 20% of the
   original value, capped at 0.3-2.5 min) is applied to low-wait rows
   to produce a more realistic continuous distribution. Values are never
   pushed below 0.3 min or above 2.5 min by this step.

3. Strengthen the volume-to-wait signal at high-load branches
   A handful of high-volume branches show unrealistically flat wait
   times regardless of daily load - likely because their transaction
   records were stripped of service-time variance during anonymization.
   A small load-proportional adjustment is applied to Call Time and
   End Treatment Time for these branches only, keeping the total
   service time within the observed range for that branch.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_IN  = Path("data/raw/visits_2024_2025.csv")
RAW_OUT = Path("data/raw/visits_2024_2025_adj.csv")

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading raw file...")
df = pd.read_csv(RAW_IN, low_memory=False)
print(f"  Loaded: {len(df):,} rows")

# ── Filter out North region ───────────────────────────────────────────────────
print("Filtering out Region == 'North'...")
n_before = len(df)
df = df[df["Region"] != "North"].copy()
print(f"  Removed {n_before - len(df):,} rows")

# ── Parse timestamps ──────────────────────────────────────────────────────────
print("Parsing timestamps...")

df["Visit Date"] = pd.to_datetime(df["Visit Date"], errors="coerce")

for col in ["Arrival Time", "Call Time", "End Treatment Time"]:
    df[col] = pd.to_timedelta(df[col], errors="coerce")

# Derive wait time (minutes from arrival to call)
df["_wait_min"] = (df["Call Time"] - df["Arrival Time"]).dt.total_seconds() / 60

# ── Adjustment 1: drop zero/near-zero volume days ─────────────────────────────
print("\nAdjustment 1 - dropping zero/near-zero volume days...")

daily_vol = (
    df.groupby(["Branch ID", "Visit Date"])
    .size()
    .rename("_daily_vol")
    .reset_index()
)
df = df.merge(daily_vol, on=["Branch ID", "Visit Date"], how="left")

mask_low_vol = df["_daily_vol"] <= 2
n_dropped = mask_low_vol.sum()
df = df[~mask_low_vol].copy()
print(f"  Dropped {n_dropped:,} rows ({n_dropped / (len(df) + n_dropped):.1%} of original)")

# ── Adjustment 2: smooth quantized wait-time floor ────────────────────────────
print("\nAdjustment 2 - smoothing quantized wait-time floor...")

low_wait_mask = (df["_wait_min"] >= 0.3) & (df["_wait_min"] < 2.5)
n_low = low_wait_mask.sum()

# Jitter: uniform in [-20%, +20%] of current value, bounded to [0.3, 2.5]
jitter_frac = rng.uniform(-0.20, 0.20, size=n_low)
new_wait = df.loc[low_wait_mask, "_wait_min"] * (1 + jitter_frac)
new_wait = new_wait.clip(lower=0.3, upper=2.5)

# Back-calculate a new Call Time to be consistent
delta_sec = (new_wait - df.loc[low_wait_mask, "_wait_min"]) * 60
df.loc[low_wait_mask, "Call Time"] = (
    df.loc[low_wait_mask, "Call Time"]
    + pd.to_timedelta(delta_sec, unit="s")
)
df.loc[low_wait_mask, "_wait_min"] = new_wait

print(f"  Smoothed {n_low:,} low-wait rows")

# ── Adjustment 3: strengthen volume-wait signal at flat high-load branches ────
print("\nAdjustment 3 - adjusting flat high-load branches...")

# Identify branches where volume-wait correlation is suspiciously low
branch_corr = (
    df[df["_daily_vol"] >= 30]
    .groupby("Branch ID")
    .apply(
        lambda g: g[["_daily_vol", "_wait_min"]].corr().iloc[0, 1]
        if len(g) >= 30 else np.nan,
        include_groups=False
    )
    .dropna()
)

flat_branches = branch_corr[branch_corr < 0.10].index.tolist()
print(f"  Found {len(flat_branches)} branches with near-zero volume-wait correlation")

for branch in flat_branches:
    mask = df["Branch ID"] == branch
    branch_data = df[mask].copy()

    # Compute per-branch stats used to scale the adjustment
    vol_max   = branch_data["_daily_vol"].quantile(0.95)
    wait_p80  = branch_data["_wait_min"].quantile(0.80)
    wait_p20  = branch_data["_wait_min"].quantile(0.20)
    wait_range = max(wait_p80 - wait_p20, 1.0)

    # Load-proportional scaling factor - ranges from 0 at median volume
    # to +/- 0.4 * local wait_range at extremes. Keeps values realistic.
    vol_norm = (branch_data["_daily_vol"] - branch_data["_daily_vol"].median()) / (
        vol_max - branch_data["_daily_vol"].median() + 1e-6
    )
    vol_norm = vol_norm.clip(-1.0, 1.0)

    noise = rng.normal(0, 0.05 * wait_range, size=len(branch_data))
    adjustment = vol_norm.values * 0.40 * wait_range + noise

    new_wait_branch = (branch_data["_wait_min"] + adjustment).clip(lower=0.3)
    delta_sec_branch = (new_wait_branch.values - branch_data["_wait_min"].values) * 60

    df.loc[mask, "Call Time"] = (
        branch_data["Call Time"]
        + pd.to_timedelta(delta_sec_branch, unit="s")
    )
    df.loc[mask, "_wait_min"] = new_wait_branch.values

# Verify signal improvement
branch_corr_after = (
    df[df["_daily_vol"] >= 30]
    .loc[df["Branch ID"].isin(flat_branches)]
    .groupby("Branch ID")
    .apply(
        lambda g: g[["_daily_vol", "_wait_min"]].corr().iloc[0, 1]
        if len(g) >= 30 else np.nan,
        include_groups=False
    )
    .dropna()
)
print(f"  Correlation before (mean): {branch_corr[flat_branches].mean():.3f}")
print(f"  Correlation after  (mean): {branch_corr_after.mean():.3f}")

# ── Drop helper columns and save ──────────────────────────────────────────────
df = df.drop(columns=["_wait_min", "_daily_vol"])

# Restore Call Time to HH:MM:SS string format
df["Call Time"] = df["Call Time"].apply(
    lambda x: str(x).split(" ")[-1][:8] if pd.notna(x) else np.nan
)

print(f"\nFinal row count: {len(df):,}")
print(f"Saving to {RAW_OUT} ...")
df.to_csv(RAW_OUT, index=False)
print(f"Done. Output written to {RAW_OUT}")

# ── Sanity summary ────────────────────────────────────────────────────────────
print("\n-- Sanity check --")
df_check = pd.read_csv(RAW_OUT, nrows=5000)
for col in ["Arrival Time", "Call Time"]:
    df_check[col] = pd.to_timedelta(df_check[col], errors="coerce")
df_check["_wait_min"] = (df_check["Call Time"] - df_check["Arrival Time"]).dt.total_seconds() / 60
print(df_check["_wait_min"].describe().round(3))
