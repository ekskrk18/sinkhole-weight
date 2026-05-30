# -*- coding: utf-8 -*-
r"""
SPI weight optimization for sinkhole precursor index (SPI)

This script is designed to extend the user's existing SPI workflow:
- Keeps the same data loading logic for GW / rainfall / SMAP
- Keeps the same SPI component definition (DI, RI, CI, SI, HI)
- Optimizes weight factors under the constraint sum(weights)=1, weights>=0
- Generates:
    1) Pareto front plot
    2) Sensitivity plot (Spearman rank-based importance)
    3) Bootstrap violin plot of optimal weights

Main idea
---------
For each sampled weight vector w = [w_DI, w_RI, w_CI, w_SI, w_HI], this script:
1. Computes SPI for each event and each accumulation window L in L_LIST
2. Extracts event-level metrics within the pre-event window
3. Selects the best L per event using maxSPI_pre14 (tie-break: smaller lead time)
4. Aggregates event-level performance into multi-objective scores
5. Finds Pareto-optimal weight vectors
6. Plots sensitivity and bootstrap stability results

Notes
-----
- This script intentionally follows the structure used in the uploaded files
  SPI_compare.py / SPI_compare2.py / SPI_rolling_timeseries.py / analysis.py.
- Only numpy, pandas, and matplotlib are used.
- If you want stricter optimization, increase N_WEIGHT_SAMPLES.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 0) USER SETTINGS
# =========================================================
BASE_DIR   = r"E:\20260411\00 KONKUK\02 Papers\01 SCIE\29th Sinkhole (Weight factor)\python"
GW_DIR     = os.path.join(BASE_DIR, "gims_groundwater_daily")
RAIN_DIR   = os.path.join(BASE_DIR, "kma_rn60m")
SMAP_DIR   = os.path.join(BASE_DIR, "SMAP_sinkhole_rootzone_30d")
EVENTS_CSV = os.path.join(BASE_DIR, "sinkhole_events.csv")

OUT_DIR    = os.path.join(BASE_DIR, "SPI_weight_optimization")
os.makedirs(OUT_DIR, exist_ok=True)

EVENT_ID_LIST = [1, 2, 3, 4, 5, 6, 7, 8]
L_LIST = [1, 3, 7, 10, 14]

EVAL_PRE_DAYS = 30
EVAL_POST_DAYS = 1
PEAK_SEARCH_WINDOW_DAYS = 14
ALPHA_RAIN = 0.01

# Weight optimization settings
N_WEIGHT_SAMPLES = 3000          # increase (e.g., 10000) for a denser search
DIRICHLET_ALPHA = np.array([2.0, 2.0, 2.0, 1.5, 1.5])
RANDOM_SEED = 42

# Pareto filtering / display
TOP_PARETO_TO_SAVE = 50

# Bootstrap settings
N_BOOTSTRAP = 300
BOOTSTRAP_RANDOM_SEED = 123

# Lead-time target (days before event).
# This can be tuned depending on whether you prefer very-near-event peaks or a little earlier warning.
TARGET_LEAD_DAYS = 2.0

# Composite score weights used only for choosing a single "best" solution from the Pareto set.
# The Pareto plot itself is still multi-objective.
LAMBDA_PEAK = 0.60
LAMBDA_LEAD = 0.40


# =========================================================
# 1) UTILITIES
# =========================================================
def _read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file:\n  {path}")
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path)


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def to_naive_date(dt_series: pd.Series) -> pd.Series:
    s = pd.to_datetime(dt_series, errors="coerce")
    if hasattr(s.dt, "tz") and s.dt.tz is not None:
        s = s.dt.tz_convert(None)
    return s.dt.floor("D")


def robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) + 1e-6
    return (x - med) / mad


def choose_best_shift(base_dates: pd.Series, df: pd.DataFrame, date_col: str = "date", shifts=(-1, 0, 1)):
    base_set = set(pd.to_datetime(base_dates).tolist())
    best_shift = 0
    best_overlap = -1
    best_df = df.copy()

    for s in shifts:
        tmp = df.copy()
        tmp[date_col] = tmp[date_col] + pd.Timedelta(days=s)
        overlap = len(set(tmp[date_col].tolist()) & base_set)
        if overlap > best_overlap:
            best_overlap = overlap
            best_shift = s
            best_df = tmp
    return best_df, best_shift, best_overlap


def rankdata_average(a: np.ndarray) -> np.ndarray:
    """Average-rank implementation without scipy."""
    a = np.asarray(a)
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty_like(sorter, dtype=float)
    arr = a[sorter]
    n = len(a)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and arr[j + 1] == arr[i]:
            j += 1
        rank = 0.5 * (i + j) + 1.0
        inv[sorter[i:j+1]] = rank
        i = j + 1
    return inv


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    rx = rankdata_average(np.asarray(x, dtype=float))
    ry = rankdata_average(np.asarray(y, dtype=float))
    sx = np.std(rx)
    sy = np.std(ry)
    if sx < 1e-12 or sy < 1e-12:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def dirichlet_weights(n: int, alpha: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.dirichlet(alpha, size=n)


# =========================================================
# 2) LOAD EVENTS / DAILY DATA (same spirit as existing scripts)
# =========================================================
def load_events_table(path: str) -> pd.DataFrame:
    ev = _norm_cols(_read_csv(path))

    if "event_id" not in ev.columns:
        for alt in ["id", "event", "case", "no"]:
            if alt in ev.columns:
                ev = ev.rename(columns={alt: "event_id"})
                break

    if "event_id" not in ev.columns or "gennum" not in ev.columns:
        raise ValueError(f"[EVENTS] need event_id & gennum. Columns={ev.columns.tolist()}")

    time_candidates = ["event_time_kst", "event_time", "event_time_utc", "time_kst", "time"]
    time_col = next((c for c in time_candidates if c in ev.columns), None)
    ev["event_time_dt"] = pd.to_datetime(ev[time_col], errors="coerce") if time_col else pd.NaT

    ev["event_id"] = pd.to_numeric(ev["event_id"], errors="coerce").astype("Int64")
    ev["gennum"] = pd.to_numeric(ev["gennum"], errors="coerce").astype("Int64")
    ev = ev.dropna(subset=["event_id", "gennum"]).copy()
    return ev.sort_values("event_id")


def load_gw_daily(gw_path: str) -> pd.DataFrame:
    gw = _norm_cols(_read_csv(gw_path))
    if "ymd" not in gw.columns:
        raise ValueError(f"[GW] ymd missing. Columns={gw.columns.tolist()}")

    dt = pd.to_datetime(gw["ymd"].astype(str), format="%Y%m%d", errors="coerce")
    gw["date"] = to_naive_date(pd.Series(dt))

    if "lev" not in gw.columns and "elev" in gw.columns:
        gw["lev"] = gw["elev"]

    for c in ["lev", "ec"]:
        if c not in gw.columns:
            raise ValueError(f"[GW] {c} missing. Columns={gw.columns.tolist()}")
        gw[c] = pd.to_numeric(gw[c], errors="coerce")

    return gw[["date", "lev", "ec"]].groupby("date", as_index=False).mean(numeric_only=True)


def load_smap_daily(sm_path: str) -> pd.DataFrame:
    sm = _norm_cols(_read_csv(sm_path))

    if "obs_time_kst" in sm.columns:
        tcol = "obs_time_kst"
    elif "obs_time_utc" in sm.columns:
        tcol = "obs_time_utc"
    else:
        raise ValueError(f"[SMAP] no obs_time. Columns={sm.columns.tolist()}")

    dt = pd.to_datetime(sm[tcol], errors="coerce")
    sm["date"] = to_naive_date(pd.Series(dt))

    if "sm_rootzone" not in sm.columns:
        raise ValueError(f"[SMAP] sm_rootzone missing. Columns={sm.columns.tolist()}")

    sm["sm_rootzone"] = pd.to_numeric(sm["sm_rootzone"], errors="coerce")
    return sm[["date", "sm_rootzone"]].groupby("date", as_index=False).mean()


def load_rain_daily(rain_path: str) -> pd.DataFrame:
    rn = _norm_cols(_read_csv(rain_path))

    if "rn_60m" not in rn.columns:
        for alt in ["rain", "rn", "precip", "prcp"]:
            if alt in rn.columns:
                rn = rn.rename(columns={alt: "rn_60m"})
                break
    if "rn_60m" not in rn.columns:
        raise ValueError(f"[RAIN] rn_60m missing. Columns={rn.columns.tolist()}")

    rn["rn_60m"] = pd.to_numeric(rn["rn_60m"], errors="coerce")

    if "date" in rn.columns:
        dt = pd.to_datetime(rn["date"], errors="coerce")
    else:
        dt = None
        for alt in ["datetime", "time", "tm", "timestamp", "obs_time", "obs_time_kst", "obs_time_utc"]:
            if alt in rn.columns:
                dt = pd.to_datetime(rn[alt], errors="coerce")
                break
        if dt is None:
            raise ValueError(f"[RAIN] no time column. Columns={rn.columns.tolist()}")

    rn["date"] = to_naive_date(pd.Series(dt))
    return rn[["date", "rn_60m"]].groupby("date", as_index=False).sum()


def build_backbone_window(event_date: pd.Timestamp) -> pd.DataFrame:
    start = (event_date - pd.Timedelta(days=EVAL_PRE_DAYS)).normalize()
    end = (event_date + pd.Timedelta(days=EVAL_POST_DAYS)).normalize()
    return pd.DataFrame({"date": pd.date_range(start, end, freq="D")})


def infer_event_date_from_gw(gw: pd.DataFrame) -> pd.Timestamp:
    return (gw["date"].max() - pd.Timedelta(days=1)).normalize()


def load_event_merged(event_id: int, gennum: int, event_time_dt) -> Tuple[pd.DataFrame, pd.Timestamp]:
    gw_path = os.path.join(GW_DIR, f"event_{event_id}_gennum_{int(gennum)}_daily.csv")
    rain_path = os.path.join(RAIN_DIR, f"event_{event_id}_rn60m.csv")
    sm_path = os.path.join(SMAP_DIR, f"event_{event_id}", f"event_{event_id}_rootzone_30d_to_1d.csv")

    gw = load_gw_daily(gw_path)
    sm = load_smap_daily(sm_path)
    rn = load_rain_daily(rain_path)

    if pd.notna(event_time_dt):
        event_date = to_naive_date(pd.Series([event_time_dt])).iloc[0]
    else:
        event_date = infer_event_date_from_gw(gw)

    backbone = build_backbone_window(event_date)
    sm2, _, _ = choose_best_shift(backbone["date"], sm, shifts=(-1, 0, 1))
    rn2, _, _ = choose_best_shift(backbone["date"], rn, shifts=(-1, 0, 1))

    df = backbone.merge(gw, on="date", how="left").merge(sm2, on="date", how="left").merge(rn2, on="date", how="left")
    df["rn_60m"] = df["rn_60m"].fillna(0.0)
    for col in ["lev", "ec", "sm_rootzone"]:
        df[col] = df[col].interpolate(limit_direction="both").ffill().bfill()

    return df, event_date


# =========================================================
# 3) SPI DEFINITION (same logic)
# =========================================================
def compute_spi(df: pd.DataFrame, L: int,
                w_DI: float, w_RI: float, w_CI: float, w_SI: float, w_HI: float,
                alpha_rain: float = ALPHA_RAIN) -> pd.DataFrame:
    d = df.copy()
    d["dlev"] = d["lev"] - d["lev"].shift(L)
    d["dec"] = d["ec"] - d["ec"].shift(L)
    d["dsm"] = d["sm_rootzone"] - d["sm_rootzone"].shift(L)
    d["rain_L"] = d["rn_60m"].rolling(L, min_periods=L).sum()

    rain_thr = d["rain_L"].quantile(0.25)
    d["NR"] = (d["rain_L"] <= rain_thr).astype(int)

    d["DI"] = np.maximum(0.0, -d["dlev"])
    d["RI"] = np.maximum(0.0, d["dlev"]) * d["NR"]
    d["CI"] = np.maximum(0.0, d["dec"]) * d["NR"]
    d["SI"] = np.maximum(0.0, d["dsm"]) + alpha_rain * d["rain_L"]
    d["HI"] = np.maximum(0.0, d["dsm"]) * d["NR"]

    for col in ["DI", "RI", "CI", "SI", "HI"]:
        d[col + "_z"] = robust_z(d[col].to_numpy())

    d["SPI"] = (
        w_DI * d["DI_z"] +
        w_RI * d["RI_z"] +
        w_CI * d["CI_z"] +
        w_SI * d["SI_z"] +
        w_HI * d["HI_z"]
    )
    return d


# =========================================================
# 4) METRICS
# =========================================================
@dataclass
class EventMetric:
    event_id: int
    best_L: int
    best_peak: float
    best_lead: float
    best_event_spi: float


def summarize_L_per_event(d_spi: pd.DataFrame, event_date: pd.Timestamp) -> Dict[str, float]:
    d = d_spi.copy()
    d["relday"] = (d["date"] - event_date).dt.days

    pre30 = d[(d["relday"] >= -EVAL_PRE_DAYS) & (d["relday"] <= 0)].dropna(subset=["SPI"])
    pre14 = d[(d["relday"] >= -PEAK_SEARCH_WINDOW_DAYS) & (d["relday"] <= 0)].dropna(subset=["SPI"])

    out = {
        "maxSPI_pre14": np.nan,
        "maxSPI_pre30": np.nan,
        "lead_peak14": np.nan,
        "spi_event": np.nan,
        "peak_to_mean_ratio": np.nan,
    }
    if len(pre30) == 0:
        return out

    out["maxSPI_pre30"] = float(pre30["SPI"].max())

    if len(pre14) > 0:
        idx = pre14["SPI"].idxmax()
        peak_day = pre14.loc[idx, "date"]
        out["maxSPI_pre14"] = float(pre14.loc[idx, "SPI"])
        out["lead_peak14"] = float((event_date - peak_day).days)
        mean_abs = float(np.abs(pre14["SPI"].mean()))
        out["peak_to_mean_ratio"] = float(pre14.loc[idx, "SPI"] / mean_abs) if mean_abs > 1e-12 else np.nan

    spi_event = pre30.loc[pre30["relday"] == 0, "SPI"]
    if len(spi_event) > 0:
        out["spi_event"] = float(spi_event.iloc[0])

    return out


def evaluate_single_weight_vector(events_cache: Dict[int, Tuple[pd.DataFrame, pd.Timestamp]],
                                  weights: np.ndarray,
                                  L_list: List[int]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    For a given weight vector, compute metrics across all events and L values,
    then choose best L per event using:
      - larger maxSPI_pre14 preferred
      - smaller lead_peak14 as tie-break
    """
    w_DI, w_RI, w_CI, w_SI, w_HI = weights.tolist()
    rows = []

    for event_id, (df, event_date) in events_cache.items():
        for L in L_list:
            d_spi = compute_spi(df, L, w_DI, w_RI, w_CI, w_SI, w_HI)
            m = summarize_L_per_event(d_spi, event_date)
            rows.append({
                "event_id": event_id,
                "L": L,
                **m
            })

    metrics_df = pd.DataFrame(rows).sort_values(["event_id", "L"]).reset_index(drop=True)

    # choose best L per event
    best_rows = []
    for eid, sub in metrics_df.groupby("event_id"):
        sub = sub.dropna(subset=["maxSPI_pre14"]).copy()
        if len(sub) == 0:
            continue
        sub["_lead_for_sort"] = sub["lead_peak14"].fillna(1e9)
        sub = sub.sort_values(["maxSPI_pre14", "_lead_for_sort"], ascending=[False, True])
        best = sub.iloc[0]
        best_rows.append({
            "event_id": int(eid),
            "best_L": int(best["L"]),
            "best_peak": float(best["maxSPI_pre14"]),
            "best_lead": float(best["lead_peak14"]),
            "best_event_spi": float(best["spi_event"]) if pd.notna(best["spi_event"]) else np.nan,
            "peak_to_mean_ratio": float(best["peak_to_mean_ratio"]) if pd.notna(best["peak_to_mean_ratio"]) else np.nan,
        })

    best_df = pd.DataFrame(best_rows).sort_values("event_id").reset_index(drop=True)
    if len(best_df) == 0:
        summary = {
            "mean_peak": np.nan,
            "mean_lead": np.nan,
            "lead_penalty": np.nan,
            "mean_event_spi": np.nan,
            "mean_peak_ratio": np.nan,
            "composite_score": np.nan,
            "n_events": 0,
        }
        return best_df, summary

    mean_peak = float(best_df["best_peak"].mean())
    mean_lead = float(best_df["best_lead"].mean())
    mean_event_spi = float(best_df["best_event_spi"].mean()) if best_df["best_event_spi"].notna().any() else np.nan
    mean_peak_ratio = float(best_df["peak_to_mean_ratio"].mean()) if best_df["peak_to_mean_ratio"].notna().any() else np.nan

    # smaller is better; zero means close to desired lead time
    lead_penalty = float(np.mean(np.abs(best_df["best_lead"].to_numpy(dtype=float) - TARGET_LEAD_DAYS)))

    # normalize-like composite using simple safe transforms for ranking one final solution
    composite_score = float(LAMBDA_PEAK * mean_peak - LAMBDA_LEAD * lead_penalty)

    summary = {
        "mean_peak": mean_peak,
        "mean_lead": mean_lead,
        "lead_penalty": lead_penalty,
        "mean_event_spi": mean_event_spi,
        "mean_peak_ratio": mean_peak_ratio,
        "composite_score": composite_score,
        "n_events": int(len(best_df)),
    }
    return best_df, summary


# =========================================================
# 5) PARETO / SENSITIVITY / BOOTSTRAP
# =========================================================
def pareto_mask(values_max_peak: np.ndarray, values_min_penalty: np.ndarray) -> np.ndarray:
    """
    Pareto condition for:
      objective 1 = maximize mean_peak
      objective 2 = minimize lead_penalty
    """
    n = len(values_max_peak)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        dominates_i = (
            (values_max_peak >= values_max_peak[i]) &
            (values_min_penalty <= values_min_penalty[i]) &
            ((values_max_peak > values_max_peak[i]) | (values_min_penalty < values_min_penalty[i]))
        )
        dominates_i[i] = False
        if np.any(dominates_i):
            is_pareto[i] = False
    return is_pareto


def plot_pareto_front(results_df: pd.DataFrame, pareto_df: pd.DataFrame, out_png: str):
    plt.figure(figsize=(7.5, 5.5))
    plt.scatter(results_df["lead_penalty"], results_df["mean_peak"], s=18, alpha=0.45, label="All sampled weights")
    plt.scatter(pareto_df["lead_penalty"], pareto_df["mean_peak"], s=36, alpha=0.95, label="Pareto-optimal weights")
    plt.xlabel(f"Lead-time penalty (target = {TARGET_LEAD_DAYS:.0f} days)")
    plt.ylabel("Mean best-event maxSPI_pre14")
    plt.title("Pareto front for SPI weight optimization")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def plot_sensitivity(results_df: pd.DataFrame, out_png: str):
    weight_cols = ["w_DI", "w_RI", "w_CI", "w_SI", "w_HI"]
    metric_cols = ["mean_peak", "lead_penalty", "composite_score"]

    corr_rows = []
    for metric in metric_cols:
        for wcol in weight_cols:
            corr_rows.append({
                "metric": metric,
                "weight": wcol,
                "spearman_r": spearman_corr(results_df[wcol].to_numpy(), results_df[metric].to_numpy())
            })
    corr_df = pd.DataFrame(corr_rows)

    # Plot composite score sensitivity as the main tornado-like panel
    sub = corr_df[corr_df["metric"] == "composite_score"].copy()
    sub["abs_r"] = sub["spearman_r"].abs()
    sub = sub.sort_values("abs_r", ascending=True)

    plt.figure(figsize=(7.2, 4.8))
    plt.barh(sub["weight"], sub["spearman_r"])
    plt.axvline(0.0, color="black", linewidth=1.0)
    plt.xlabel("Spearman rank correlation with composite score")
    plt.ylabel("Weight factor")
    plt.title("Sensitivity of optimized SPI performance to weight factors")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

    # Also save the table used for the plot
    corr_df.to_csv(os.path.join(os.path.dirname(out_png), "sensitivity_spearman_table.csv"), index=False, encoding="utf-8-sig")


def plot_bootstrap_violin(boot_df: pd.DataFrame, out_png: str):
    weight_cols = ["w_DI", "w_RI", "w_CI", "w_SI", "w_HI"]
    data = [boot_df[c].dropna().to_numpy() for c in weight_cols]

    plt.figure(figsize=(8.0, 4.8))
    parts = plt.violinplot(data, showmeans=False, showmedians=True, showextrema=True)
    plt.xticks(np.arange(1, len(weight_cols) + 1), weight_cols)
    plt.ylabel("Optimal weight")
    plt.title("Bootstrap distribution of optimal SPI weights")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def save_best_event_table(best_df: pd.DataFrame, out_csv: str):
    best_df.to_csv(out_csv, index=False, encoding="utf-8-sig")


# =========================================================
# 6) MAIN WORKFLOW
# =========================================================
def build_event_cache() -> Tuple[pd.DataFrame, Dict[int, Tuple[pd.DataFrame, pd.Timestamp]]]:
    events = load_events_table(EVENTS_CSV)
    events = events[events["event_id"].isin(EVENT_ID_LIST)].copy()
    events = events.dropna(subset=["event_id", "gennum"]).sort_values("event_id")

    cache: Dict[int, Tuple[pd.DataFrame, pd.Timestamp]] = {}
    for _, row in events.iterrows():
        eid = int(row["event_id"])
        gennum = int(row["gennum"])
        etime = row.get("event_time_dt", pd.NaT)
        df, event_date = load_event_merged(eid, gennum, etime)
        cache[eid] = (df, event_date)
        print(f"[Loaded] Event {eid}: n={len(df)} | event_date={event_date.date()}")
    return events, cache


def run_weight_search(events_cache: Dict[int, Tuple[pd.DataFrame, pd.Timestamp]]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weight_matrix = dirichlet_weights(N_WEIGHT_SAMPLES, DIRICHLET_ALPHA, RANDOM_SEED)

    rows = []
    best_tables = {}
    for i, w in enumerate(weight_matrix, start=1):
        best_df, summary = evaluate_single_weight_vector(events_cache, w, L_LIST)
        rows.append({
            "sample_id": i,
            "w_DI": w[0], "w_RI": w[1], "w_CI": w[2], "w_SI": w[3], "w_HI": w[4],
            **summary
        })
        best_tables[i] = best_df
        if i % 250 == 0 or i == len(weight_matrix):
            print(f"[Search] {i}/{len(weight_matrix)} completed")

    results_df = pd.DataFrame(rows)
    results_df = results_df.dropna(subset=["mean_peak", "lead_penalty"]).reset_index(drop=True)

    mask = pareto_mask(results_df["mean_peak"].to_numpy(), results_df["lead_penalty"].to_numpy())
    pareto_df = results_df.loc[mask].copy().sort_values(["lead_penalty", "mean_peak"], ascending=[True, False])
    pareto_df["pareto_rank_order"] = np.arange(1, len(pareto_df) + 1)

    # choose one final recommended solution from Pareto set using composite score
    final_best = pareto_df.sort_values("composite_score", ascending=False).head(1).copy()
    final_sample_id = int(final_best.iloc[0]["sample_id"])
    final_best_events = best_tables[final_sample_id].copy()

    results_df.to_csv(os.path.join(OUT_DIR, "weight_search_results.csv"), index=False, encoding="utf-8-sig")
    pareto_df.head(TOP_PARETO_TO_SAVE).to_csv(os.path.join(OUT_DIR, "pareto_optimal_weights_top.csv"), index=False, encoding="utf-8-sig")
    final_best.to_csv(os.path.join(OUT_DIR, "final_recommended_weight.csv"), index=False, encoding="utf-8-sig")
    save_best_event_table(final_best_events, os.path.join(OUT_DIR, "final_recommended_weight_bestL_by_event.csv"))

    return results_df, pareto_df, final_best_events


def run_bootstrap(events_cache: Dict[int, Tuple[pd.DataFrame, pd.Timestamp]],
                  events_df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_RANDOM_SEED)
    event_ids = events_df["event_id"].astype(int).tolist()

    # use a somewhat reduced search inside bootstrap for speed
    weight_matrix = dirichlet_weights(max(800, N_WEIGHT_SAMPLES // 4), DIRICHLET_ALPHA, BOOTSTRAP_RANDOM_SEED + 1)

    boot_rows = []
    for b in range(1, N_BOOTSTRAP + 1):
        sampled_ids = rng.choice(event_ids, size=len(event_ids), replace=True)
        sampled_cache = {eid: events_cache[eid] for eid in sampled_ids}

        local_rows = []
        for i, w in enumerate(weight_matrix, start=1):
            _, summary = evaluate_single_weight_vector(sampled_cache, w, L_LIST)
            local_rows.append({
                "sample_id": i,
                "w_DI": w[0], "w_RI": w[1], "w_CI": w[2], "w_SI": w[3], "w_HI": w[4],
                **summary
            })
        local_df = pd.DataFrame(local_rows).dropna(subset=["mean_peak", "lead_penalty"])
        if len(local_df) == 0:
            continue

        mask = pareto_mask(local_df["mean_peak"].to_numpy(), local_df["lead_penalty"].to_numpy())
        local_pareto = local_df.loc[mask].copy()
        local_best = local_pareto.sort_values("composite_score", ascending=False).head(1)
        if len(local_best) == 0:
            continue
        row = local_best.iloc[0]
        boot_rows.append({
            "bootstrap_id": b,
            "w_DI": float(row["w_DI"]),
            "w_RI": float(row["w_RI"]),
            "w_CI": float(row["w_CI"]),
            "w_SI": float(row["w_SI"]),
            "w_HI": float(row["w_HI"]),
            "mean_peak": float(row["mean_peak"]),
            "lead_penalty": float(row["lead_penalty"]),
            "composite_score": float(row["composite_score"]),
        })

        if b % 25 == 0 or b == N_BOOTSTRAP:
            print(f"[Bootstrap] {b}/{N_BOOTSTRAP} completed")

    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(os.path.join(OUT_DIR, "bootstrap_optimal_weights.csv"), index=False, encoding="utf-8-sig")
    return boot_df


def main():
    print("BASE_DIR   :", BASE_DIR)
    print("EVENTS_CSV :", EVENTS_CSV)
    print("GW_DIR     :", GW_DIR)
    print("RAIN_DIR   :", RAIN_DIR)
    print("SMAP_DIR   :", SMAP_DIR)
    print("OUT_DIR    :", OUT_DIR)

    events_df, events_cache = build_event_cache()

    # 1) Weight search
    results_df, pareto_df, final_best_events = run_weight_search(events_cache)

    # 2) Plots
    plot_pareto_front(results_df, pareto_df, os.path.join(OUT_DIR, "pareto_front.png"))
    plot_sensitivity(results_df, os.path.join(OUT_DIR, "sensitivity_plot.png"))

    # 3) Bootstrap stability
    boot_df = run_bootstrap(events_cache, events_df)
    if len(boot_df) > 0:
        plot_bootstrap_violin(boot_df, os.path.join(OUT_DIR, "bootstrap_violin_plot.png"))
    else:
        print("[WARN] Bootstrap produced no valid rows. Check data completeness or reduce constraints.")

    # 4) Small summary text file
    final_weight = pd.read_csv(os.path.join(OUT_DIR, "final_recommended_weight.csv")).iloc[0]
    with open(os.path.join(OUT_DIR, "README_summary.txt"), "w", encoding="utf-8") as f:
        f.write("SPI weight optimization summary\n")
        f.write("================================\n")
        f.write(f"Target lead days: {TARGET_LEAD_DAYS}\n")
        f.write(f"Weight samples: {N_WEIGHT_SAMPLES}\n")
        f.write(f"Bootstrap runs: {N_BOOTSTRAP}\n\n")
        f.write("Final recommended weight (from Pareto set, highest composite score):\n")
        for k in ["w_DI", "w_RI", "w_CI", "w_SI", "w_HI", "mean_peak", "lead_penalty", "composite_score"]:
            f.write(f"- {k}: {final_weight[k]}\n")
        f.write("\nSaved outputs:\n")
        f.write("- weight_search_results.csv\n")
        f.write("- pareto_optimal_weights_top.csv\n")
        f.write("- final_recommended_weight.csv\n")
        f.write("- final_recommended_weight_bestL_by_event.csv\n")
        f.write("- pareto_front.png\n")
        f.write("- sensitivity_plot.png\n")
        f.write("- bootstrap_optimal_weights.csv\n")
        f.write("- bootstrap_violin_plot.png\n")

    print("\nDone.")
    print("Results saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
