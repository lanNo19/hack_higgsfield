"""Targeted feature builders — v2 additions.

Two functions:
  build_transaction_features_v2  — invol_churn specific signals (failure streaks,
                                   acute distress, first-failure duration)
  build_generation_features_v2   — vol_churn specific signals (silence gaps,
                                   completed-only recency, decay rates)

These return DataFrames indexed by user_id and are designed to be merged on top
of the existing feature matrix produced by churn_features.py.
The data passed in is already preprocessed by src.data.preprocess.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.churn_features import _as_utc, _safe_div
from src.utils.logger import get_logger

log = get_logger(__name__)


# ── T-v2: Invol-churn targeted transaction features ───────────────────────────

def build_transaction_features_v2(
    transactions: pd.DataFrame,
    obs_date: pd.Timestamp,
) -> pd.DataFrame:
    """New transaction features focused on the invol_churn signal.

    All inputs already preprocessed (transaction_time is UTC datetime,
    is_failed / is_successful are booleans).
    """
    log.info("Building transaction features v2 ...")
    obs_date = _as_utc(obs_date)
    t = transactions.copy()
    all_users = t["user_id"].unique()

    failed    = t[t["is_failed"]]
    successful = t[t["is_successful"]]

    # ── TV1: longest consecutive failure streak (no success in between) ────────
    # Sort global table by (user_id, transaction_time).
    # A new streak starts when: user boundary OR previous transaction was a success.
    t_sorted = t.sort_values(["user_id", "transaction_time"]).copy()
    prev_is_success = (~t_sorted["is_failed"]).shift(1).fillna(True)
    is_new_user     = t_sorted["user_id"] != t_sorted["user_id"].shift(1)
    t_sorted["_new_streak"] = (is_new_user | prev_is_success)
    t_sorted["_streak_id"]  = t_sorted["_new_streak"].cumsum()

    n_consec = (
        t_sorted[t_sorted["is_failed"]]
        .groupby(["user_id", "_streak_id"]).size()
        .groupby("user_id").max()
        .rename("n_consecutive_payment_failures")
    )

    # ── TV2: last transaction was a failure ────────────────────────────────────
    last_txn_fail = (
        t_sorted.groupby("user_id").last()["is_failed"]
        .astype(int)
        .rename("last_transaction_was_failure")
    )

    # ── TV3: user has ever had a successful payment ────────────────────────────
    has_success = (
        (successful.groupby("user_id").size() > 0)
        .astype(int)
        .rename("has_ever_successfully_paid")
    )

    # ── TV4: failed transactions in the last 7 days (acute distress) ──────────
    t_last7 = t[t["transaction_time"] >= obs_date - pd.Timedelta(days=7)]
    n_failed_7d = (
        t_last7[t_last7["is_failed"]]
        .groupby("user_id").size()
        .rename("n_failed_last_7d")
    )

    # ── TV5: days since first payment failure (how long in broken state) ───────
    first_fail_time = failed.groupby("user_id")["transaction_time"].min()
    days_first_fail = (
        (obs_date - first_fail_time).dt.days
        .rename("days_from_first_fail_to_obs")
    )

    # ── TV6: failed transactions in last 14 days (wider acute window) ─────────
    t_last14 = t[t["transaction_time"] >= obs_date - pd.Timedelta(days=14)]
    n_failed_14d = (
        t_last14[t_last14["is_failed"]]
        .groupby("user_id").size()
        .rename("n_failed_last_14d")
    )

    # ── TV7: ratio failed_last_7d / total_failed (recent concentration) ───────
    total_failed = failed.groupby("user_id").size()
    fail_concentration = _safe_div(n_failed_7d, total_failed).rename("recent_fail_concentration")

    feat = pd.concat([
        n_consec, last_txn_fail, has_success,
        n_failed_7d, n_failed_14d, days_first_fail, fail_concentration,
    ], axis=1)
    feat.index.name = "user_id"

    # Reindex — users without any transaction get NaN (models handle this)
    return feat.reindex(pd.Index(all_users, name="user_id"))


# ── G-v2: Vol-churn targeted generation features ──────────────────────────────

def build_generation_features_v2(
    gens: pd.DataFrame,
    props: pd.DataFrame,
    obs_date: pd.Timestamp,
) -> pd.DataFrame:
    """New generation features focused on the vol_churn signal.

    All inputs already preprocessed (created_at is UTC datetime,
    status / is_video / is_free_model are clean).
    """
    log.info("Building generation features v2 ...")
    obs_date   = _as_utc(obs_date)
    g          = gens.copy()
    sub_start  = props.set_index("user_id")["subscription_start_date"]
    all_users  = props["user_id"].unique()
    all_idx    = pd.Index(all_users, name="user_id")

    # ── GV1: max gap between consecutive active days in last 30 days ──────────
    # High value = user went silent for a long stretch recently (vol_churn signal).
    g_last30 = g[g["created_at"] >= obs_date - pd.Timedelta(days=30)]
    active_dates_30 = (
        g_last30.assign(
            _d=g_last30["created_at"].dt.tz_convert("UTC").dt.normalize().dt.tz_localize(None)
        )[["user_id", "_d"]]
        .drop_duplicates()
        .sort_values(["user_id", "_d"])
    )
    active_dates_30["_prev_d"] = active_dates_30.groupby("user_id")["_d"].shift(1)
    active_dates_30["_gap"]    = (active_dates_30["_d"] - active_dates_30["_prev_d"]).dt.days
    max_gap_30 = (
        active_dates_30.groupby("user_id")["_gap"].max()
        .rename("max_gap_days_last_30")
        .reindex(all_idx)
        .fillna(30)   # no activity OR only 1 active day → treat as 30-day gap
    )

    # ── GV2: calendar days with zero generations in last 30 days ─────────────
    active_day_count_30 = (
        active_dates_30.drop_duplicates(["user_id", "_d"])
        .groupby("user_id").size()
    )
    zero_gen_days_30 = (
        (30 - active_day_count_30).clip(lower=0)
        .rename("zero_gen_days_last_30")
        .reindex(all_idx)
        .fillna(30)   # no activity in last 30 days
    )

    # ── GV3: days since last COMPLETED generation (failures don't count) ──────
    completed_g = g[g["status"] == "completed"]
    last_completed = completed_g.groupby("user_id")["created_at"].max()
    days_since_completed = (
        (obs_date - last_completed).dt.days
        .rename("days_since_last_completed_gen")
        .reindex(all_idx)   # NaN for users with no completed gen
    )

    # ── GV4: completion rate in last 14 days (quality decay signal) ───────────
    g_last14 = g[g["created_at"] >= obs_date - pd.Timedelta(days=14)]
    total_last14    = g_last14.groupby("user_id").size()
    completed_last14 = g_last14[g_last14["status"] == "completed"].groupby("user_id").size()
    cr_last14 = (
        _safe_div(completed_last14, total_last14)
        .rename("completion_rate_last14")
        .reindex(all_idx)   # NaN for users inactive in last 14 days
    )

    # ── GV5: billing cycle proximity (days since last approximate renewal) ────
    # tenure_days % 30 ≈ position in current billing period.
    # Voluntary churners often cancel just before renewal; invol churners mid-cycle.
    tenure_days = (obs_date - sub_start).dt.days.clip(lower=1).reindex(all_idx)
    renewal_proximity = (tenure_days % 30).rename("plan_renewal_proximity_days")

    # ── GV6: last 7-day gens as % of personal daily average ──────────────────
    # Normalises activity drop to each user's own baseline.
    g_last7   = g[g["created_at"] >= obs_date - pd.Timedelta(days=7)]
    gens_last7 = g_last7.groupby("user_id").size()
    total_gens = g.groupby("user_id").size()
    avg_per_7d = (total_gens / (tenure_days / 7.0).replace(0, np.nan)).reindex(all_idx)
    gens_last7_pct = (
        _safe_div(gens_last7.reindex(all_idx).fillna(0), avg_per_7d + 0.1)
        .rename("gens_last7_pct_of_avg")
    )

    # ── GV7: consecutive days of inactivity immediately before obs_date ───────
    # "How many days in a row has the user been silent right now?"
    # Distinct from max_gap (which looks at the whole 30-day window).
    active_dates_all = (
        g.assign(_d=g["created_at"].dt.tz_convert("UTC").dt.normalize().dt.tz_localize(None))
        [["user_id", "_d"]]
        .drop_duplicates()
        .groupby("user_id")["_d"].max()
    )
    obs_naive = obs_date.tz_localize(None) if obs_date.tzinfo is None else obs_date.tz_convert(None)
    trailing_silence = (
        (obs_naive - active_dates_all).dt.days
        .rename("trailing_silence_days")
        .clip(lower=0)
        .reindex(all_idx)
        .fillna(tenure_days)   # never generated → silence = entire tenure
    )

    feat = pd.concat([
        max_gap_30, zero_gen_days_30, days_since_completed,
        cr_last14, renewal_proximity, gens_last7_pct, trailing_silence,
    ], axis=1)
    feat.index.name = "user_id"
    return feat
