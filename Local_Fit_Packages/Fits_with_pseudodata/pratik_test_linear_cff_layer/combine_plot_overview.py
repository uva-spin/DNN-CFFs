#!/usr/bin/env python3
import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Utilities
# ----------------------------
DEFAULT_DIRS = [
    "sets_1-50",
    "sets_51-100",
    "sets_101-150",
    "sets_151-195",
]

DEFAULT_CFFS = ["ReH", "ReE", "ReHt", "dvcs"]  # adjust if needed

def remove_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.duplicated()]

def read_results_csvs(dirs):
    frames = []
    for d in dirs:
        csv_path = os.path.join(d, "results.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                df["source_dir"] = d
                frames.append(df)
            except Exception as e:
                print(f"Warning: failed to read {csv_path}: {e}")
        else:
            print(f"Warning: {csv_path} not found")
    if not frames:
        raise FileNotFoundError("No results.csv files found in the provided directories.")
    combined = pd.concat(frames, ignore_index=True)
    combined = remove_duplicate_columns(combined)
    return combined

def coerce_to_wide_schema(df: pd.DataFrame, cffs):
    """
    Return a DataFrame with columns:
      'set', '<CFF>_res', '<CFF>_std' for each CFF in cffs
    Supports two input schemas:
      1) Wide: already has e.g. ReH_res, ReH_std, etc.
      2) Long: columns ['set','cff_label','residual','std'] or
               ['cff_label','mean_value','std_deviation','true_value','prediction', ...]
         We aggregate to per-set residual mean and std if needed.
    """
    # Standardize set column name if needed
    if "set" not in df.columns:
        # Some pipelines used 'Set' or similar
        set_candidates = [c for c in df.columns if c.lower() == "set"]
        if set_candidates:
            df = df.rename(columns={set_candidates[0]: "set"})
        else:
            raise ValueError("Could not find a 'set' column in results.")

    wide_cols_present = all([(f"{cff}_res" in df.columns) and (f"{cff}_std" in df.columns) for cff in cffs])

    if wide_cols_present:
        # Already in wide format; just keep relevant columns
        keep = ["set"] + [f"{cff}_res" for cff in cffs] + [f"{cff}_std" for cff in cffs]
        keep = [k for k in keep if k in df.columns]  # guard
        dfw = df[keep].copy()
        # If duplicates per set exist, aggregate by mean
        dfw = dfw.groupby("set", as_index=False).mean(numeric_only=True)
        return dfw

    # Try to detect long format
    # Common variants we support:
    # - columns containing: 'cff_label' and something like 'residual' and 'std'
    long_has_cff = "cff_label" in df.columns
    residual_col = None
    std_col = None

    # Heuristics to find residual & std columns
    for c in df.columns:
        lc = c.lower()
        if residual_col is None and ("residual" in lc or lc.endswith("_res")):
            residual_col = c
        if std_col is None and ("std" == lc or "std_dev" in lc or "std_deviation" in lc or lc.endswith("_std")):
            std_col = c

    if long_has_cff and residual_col is not None and std_col is not None:
        # Pivot to wide
        # Aggregate by set x cff_label (mean of residual and std); you can adjust if you prefer medians
        grp = df.groupby(["set", "cff_label"], as_index=False).agg(
            residual_mean=(residual_col, "mean"),
            std_mean=(std_col, "mean"),
        )
        # Pivot to columns like ReH_res, ReH_std, ...
        res_wide = grp.pivot(index="set", columns="cff_label", values="residual_mean")
        std_wide = grp.pivot(index="set", columns="cff_label", values="std_mean")

        # Build the final wide df with desired cffs
        rows = []
        for set_val in res_wide.index.union(std_wide.index):
            row = {"set": set_val}
            for cff in cffs:
                row[f"{cff}_res"] = res_wide[cff].loc[set_val] if (cff in res_wide.columns and set_val in res_wide.index) else np.nan
                row[f"{cff}_std"] = std_wide[cff].loc[set_val] if (cff in std_wide.columns and set_val in std_wide.index) else np.nan
            rows.append(row)
        dfw = pd.DataFrame(rows).sort_values("set").reset_index(drop=True)
        return dfw

    # If we get here, we could not interpret the schema
    raise ValueError(
        "Unrecognized results.csv schema. Expected wide columns like 'ReH_res'/'ReH_std' "
        "or long columns ['set','cff_label',<residual>,<std>]."
    )

def robust_ylim(values, lo_pct=5.0, hi_pct=95.0, pad=0.05):
    """
    Compute robust y-limits by percentile clipping.
    pad is a relative margin added on both sides.
    """
    vals = np.asarray(values)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    lo = np.percentile(vals, lo_pct)
    hi = np.percentile(vals, hi_pct)
    if np.isclose(lo, hi):
        # Avoid zero-height axis
        delta = 1.0 if hi == 0 else abs(hi) * 0.1
        lo, hi = hi - delta, hi + delta
    # Add padding
    rng = hi - lo
    return lo - pad * rng, hi + pad * rng

def make_overview_plot(dfw: pd.DataFrame, cffs, output_path_png="Overview_CFF_Residuals.png", output_path_pdf="Overview_CFF_Residuals.pdf",
                       lo_pct=5.0, hi_pct=95.0):
    """
    Create a single-page overview plot similar in spirit to plot.ipynb:
    - 2x2 grid (for 4 CFFs) of residual vs set with error bars showing std
    - Robust y-limits to de-emphasize outliers
    """
    # Ensure 'set' is integer-sorted
    dfw = dfw.copy()
    dfw["set"] = pd.to_numeric(dfw["set"], errors="coerce")
    dfw = dfw.dropna(subset=["set"]).sort_values("set")
    x = dfw["set"].values

    # Figure layout
    n = len(cffs)
    if n <= 2:
        rows, cols = 1, n
    elif n <= 3:
        rows, cols = 2, 2
    else:
        rows, cols = 2, 2  # adjust if you add more CFFs

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("CFF residuals across all sets (error bars = std)", fontsize=14)

    for i, cff in enumerate(cffs, start=1):
        ax = fig.add_subplot(rows, cols, i)
        res_col = f"{cff}_res"
        std_col = f"{cff}_std"

        if res_col not in dfw.columns or std_col not in dfw.columns:
            ax.text(0.5, 0.5, f"Missing {cff} columns", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(cff)
            continue

        y = dfw[res_col].values
        yerr = dfw[std_col].values

        # Errorbar plot
        ax.errorbar(x, y, yerr=yerr, fmt='o', capsize=3, linewidth=1, markersize=3)

        # Robust limits
        ylim = robust_ylim(np.concatenate([y[np.isfinite(y)], (y + yerr)[np.isfinite(yerr)], (y - yerr)[np.isfinite(yerr)]]),
                           lo_pct=lo_pct, hi_pct=hi_pct, pad=0.05)
        if ylim:
            ax.set_ylim(*ylim)

        # Make x ticks sparse for readability
        if len(x) > 30:
            # show every ~10th tick label
            step = max(1, len(x) // 10)
            mask = np.zeros_like(x, dtype=bool)
            mask[::step] = True
            ax.set_xticks(x[mask])
        else:
            ax.set_xticks(x)

        ax.set_xlabel("Set")
        ax.set_ylabel(f"{cff} residual")
        ax.set_title(cff)
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(output_path_png, dpi=200)
    plt.savefig(output_path_pdf)
    plt.close()
    print(f"Saved overview figure:\n  {output_path_png}\n  {output_path_pdf}")

def main():
    parser = argparse.ArgumentParser(description="Combine results.csv files and create a single overview plot.")
    parser.add_argument("--dirs", nargs="+", default=DEFAULT_DIRS, help="Directories that contain results.csv")
    parser.add_argument("--cffs", nargs="+", default=DEFAULT_CFFS, help="CFFs to include")
    parser.add_argument("--out_csv", default="Combined_results.csv", help="Path to write combined results CSV")
    parser.add_argument("--out_png", default="Overview_CFF_Residuals.png", help="Output PNG path for the overview figure")
    parser.add_argument("--out_pdf", default="Overview_CFF_Residuals.pdf", help="Output PDF path for the overview figure")
    parser.add_argument("--lo_pct", type=float, default=5.0, help="Lower percentile for robust y-limit")
    parser.add_argument("--hi_pct", type=float, default=95.0, help="Upper percentile for robust y-limit")
    args = parser.parse_args()

    # 1) Read and combine
    combined = read_results_csvs(args.dirs)

    # 2) Normalize to wide schema
    wide = coerce_to_wide_schema(combined, args.cffs)

    # 3) Save combined CSV
    wide.sort_values("set").to_csv(args.out_csv, index=False)
    print(f"Wrote combined CSV: {args.out_csv} (rows={len(wide)})")

    # 4) Make one big overview plot
    make_overview_plot(
        wide,
        cffs=args.cffs,
        output_path_png=args.out_png,
        output_path_pdf=args.out_pdf,
        lo_pct=args.lo_pct,
        hi_pct=args.hi_pct,
    )

if __name__ == "__main__":
    main()
