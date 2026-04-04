"""Per-table feature builders. Each function returns a DataFrame indexed by user_id."""
import numpy as np
import pandas as pd
from scipy import stats

from src.utils.helpers import load_config, encode_top_n, normalize_country
from src.utils.logger import get_logger

log = get_logger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _as_utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_convert("UTC") if t.tzinfo is not None else t.tz_localize("UTC")


def _reindex_to_users(df: pd.DataFrame, all_users: pd.Index, fill: float = 0.0) -> pd.DataFrame:
    """Ensure every user_id in all_users appears in the result (fill missing with fill)."""
    return df.reindex(all_users, fill_value=fill)


def _safe_div(a: pd.Series, b: pd.Series, fill: float = 0.0) -> pd.Series:
    return (a / b.replace(0, np.nan)).fillna(fill)


# ── P: Properties features (P1–P7) ────────────────────────────────────────────

def build_properties_features(props: pd.DataFrame, obs_date: pd.Timestamp) -> pd.DataFrame:
    """Features from properties table. One row per user_id."""
    log.info("Building properties features ...")
    cfg = load_config()
    obs_date = _as_utc(obs_date)

    out = props.set_index("user_id")[
        ["plan_ordinal", "plan_monthly_credits", "plan_monthly_cost_usd",
         "subscription_start_date", "country_code", "subscription_plan"]
    ].copy()

    out["tenure_days"] = (obs_date - out["subscription_start_date"]).dt.days.clip(lower=1)
    out["subscription_start_month"] = out["subscription_start_date"].dt.month
    out["subscription_start_dayofweek"] = out["subscription_start_date"].dt.dayofweek  # 0=Mon

    top_countries = cfg["top_countries"]
    out["country_encoded"] = encode_top_n(out["country_code"], top_countries, fill_na="unknown")

    return out.drop(columns=["subscription_start_date", "country_code", "subscription_plan"])


# ── G: Generation features (G1–G51) ───────────────────────────────────────────

def build_generation_features(
    gens: pd.DataFrame,
    props: pd.DataFrame,
    obs_date: pd.Timestamp,
) -> pd.DataFrame:
    """Features from generations table. One row per user_id."""
    log.info("Building generation features (28M rows — may take a moment) ...")
    cfg = load_config()
    obs_date = _as_utc(obs_date)
    gen_cfg = cfg["generation"]

    all_users = props.set_index("user_id").index

    # ── Attach subscription_start_date for temporal math ──────────────────────
    sub_start = props.set_index("user_id")["subscription_start_date"]
    g = gens.join(sub_start, on="user_id")

    # ── 2a/2b: Volume and rates ───────────────────────────────────────────────
    log.info("  Volume and rates ...")
    status_counts = g.groupby(["user_id", "status"]).size().unstack(fill_value=0)
    for s in ["completed", "failed", "nsfw", "canceled", "queued", "waiting", "in_progress"]:
        if s not in status_counts.columns:
            status_counts[s] = 0

    total = g.groupby("user_id").size().rename("total_generations")
    n_completed = status_counts.get("completed", pd.Series(0, index=status_counts.index)).rename("n_completed")
    n_failed = status_counts.get("failed", pd.Series(0, index=status_counts.index)).rename("n_failed")
    n_nsfw = status_counts.get("nsfw", pd.Series(0, index=status_counts.index)).rename("n_nsfw")
    n_canceled = status_counts.get("canceled", pd.Series(0, index=status_counts.index)).rename("n_canceled")
    n_queued = (
        status_counts.get("queued", pd.Series(0, index=status_counts.index)) +
        status_counts.get("waiting", pd.Series(0, index=status_counts.index)) +
        status_counts.get("in_progress", pd.Series(0, index=status_counts.index))
    ).rename("n_queued_or_waiting")

    completion_rate = _safe_div(n_completed, total).rename("completion_rate")
    failure_rate_overall = _safe_div(n_failed, total).rename("failure_rate_overall")
    nsfw_rate = _safe_div(n_nsfw, total).rename("nsfw_rate")
    cancellation_rate = _safe_div(n_canceled, total).rename("cancellation_rate")

    # Failure rate on last 10 generations per user
    sorted_g = g.sort_values(["user_id", "created_at"])
    last_10 = sorted_g.groupby("user_id").tail(10)
    failure_rate_last_10 = (
        last_10.assign(_f=last_10["status"] == "failed")
        .groupby("user_id")["_f"].mean()
        .rename("failure_rate_last_10")
    )
    failure_rate_delta = (failure_rate_last_10 - failure_rate_overall).rename("failure_rate_delta")

    # ── 2c: Credit / monetisation ─────────────────────────────────────────────
    log.info("  Credit features ...")
    pct_free = _safe_div(g.groupby("user_id")["is_free_model"].sum(),
                         total).rename("pct_free_model_usage")
    n_credit_gens = g[~g["is_free_model"]].groupby("user_id").size().rename("n_credit_costing_gens")
    total_credits = g.groupby("user_id")["credit_cost_filled"].sum().rename("total_credits_consumed")
    avg_credit = (
        g[~g["is_free_model"]].groupby("user_id")["credit_cost"].mean().rename("avg_credit_cost_per_gen")
    )
    tenure_days = (obs_date - sub_start).dt.days.clip(lower=1)
    credit_burn_rate = _safe_div(total_credits, tenure_days.rename("tenure_days")).rename("credit_burn_rate")

    recent_win = pd.Timedelta(days=gen_cfg["recent_window_days"])
    g_last14 = g[g["created_at"] >= obs_date - recent_win]
    g_prior14 = g[(g["created_at"] >= obs_date - 2 * recent_win) & (g["created_at"] < obs_date - recent_win)]
    credits_last14 = g_last14.groupby("user_id")["credit_cost_filled"].sum()
    credits_prior14 = g_prior14.groupby("user_id")["credit_cost_filled"].sum()
    credit_burn_acc = _safe_div(credits_last14, credits_prior14 + 1).rename("credit_burn_acceleration")

    high_thresh = gen_cfg["high_credit_cost_threshold"]
    premium_gens = g[(~g["is_free_model"]) & (g["credit_cost"] > high_thresh)]
    premium_gen_ratio = _safe_div(
        premium_gens.groupby("user_id").size(),
        n_credit_gens.replace(0, np.nan)
    ).rename("premium_gen_ratio")

    # ── 2d: Content type mix ──────────────────────────────────────────────────
    log.info("  Content type features ...")
    video_count = g[g["is_video"]].groupby("user_id").size()
    image_count = g[g["is_image"]].groupby("user_id").size()
    video_gen_ratio = _safe_div(video_count, total).rename("video_generation_ratio")
    image_gen_ratio = _safe_div(image_count, total).rename("image_generation_ratio")
    n_unique_types = g.groupby("user_id")["generation_type"].nunique().rename("n_unique_generation_types")

    # Video-to-image graduation: first gen was image, later used video
    first_gen_type = sorted_g.groupby("user_id")["generation_type"].first()
    last_gen_type = sorted_g.groupby("user_id")["generation_type"].last()
    video_graduation = (
        first_gen_type.str.startswith("image_", na=False) &
        last_gen_type.str.startswith("video_", na=False)
    ).astype(int).rename("video_to_image_graduation")

    dominant_gen_type = g.groupby("user_id")["generation_type"].agg(lambda x: x.mode().iloc[0] if len(x) and len(x.mode()) else None)
    image_model_1_share = _safe_div(
        g[g["generation_type"] == "image_model_1"].groupby("user_id").size(), total
    ).rename("image_model_1_share")

    # ── 2e: Quality preferences ───────────────────────────────────────────────
    log.info("  Quality preference features ...")
    video_gens = g[g["is_video"] & g["duration"].notna()]
    avg_video_dur = video_gens.groupby("user_id")["duration"].mean().rename("avg_video_duration")
    median_video_dur = video_gens.groupby("user_id")["duration"].median().rename("median_video_duration")

    has_res = g[g["resolution"].notna()].copy()
    high_res = has_res[has_res["resolution"].isin(["4k", "2k"])]
    pct_high_res = _safe_div(
        high_res.groupby("user_id").size(),
        has_res.groupby("user_id").size()
    ).rename("pct_high_resolution")

    dominant_aspect = g.groupby("user_id")["aspect_ration"].agg(
        lambda x: x.mode().iloc[0] if x.notna().any() and len(x.mode()) else None
    ).rename("dominant_aspect_ratio")

    # ── 2f: Processing time (derived from timestamps) ─────────────────────────
    log.info("  Processing time features ...")
    completed_g = g[g["status"] == "completed"].copy()
    if "processing_time_sec" in completed_g.columns:
        # Test split has this column directly
        proc_times = completed_g["processing_time_sec"]
    else:
        proc_times = (completed_g["completed_at"] - completed_g["created_at"]).dt.total_seconds()

    completed_g = completed_g.copy()
    completed_g["proc_sec"] = proc_times.values

    avg_proc = completed_g.groupby("user_id")["proc_sec"].mean().rename("avg_processing_time_sec")
    median_proc = completed_g.groupby("user_id")["proc_sec"].median().rename("median_processing_time_sec")
    long_thresh = gen_cfg["long_wait_threshold_secs"]
    pct_long_wait = _safe_div(
        completed_g[completed_g["proc_sec"] > long_thresh].groupby("user_id").size(),
        n_completed.replace(0, np.nan)
    ).rename("pct_long_wait_gens")

    # ── 2g: Temporal engagement ───────────────────────────────────────────────
    log.info("  Temporal engagement features ...")
    last_gen = g.groupby("user_id")["created_at"].max()
    first_gen = g.groupby("user_id")["created_at"].min()

    days_since_last_gen = ((obs_date - last_gen).dt.days).rename("days_since_last_generation")
    days_to_first_gen = ((first_gen - sub_start).dt.days.clip(lower=0)).rename("days_to_first_generation")
    gen_span = ((last_gen - first_gen).dt.days.clip(lower=0)).rename("generation_span_days")

    n_active_days = g.groupby("user_id")["created_at"].apply(lambda x: x.dt.normalize().nunique()).rename("n_active_days")
    active_days_frac = _safe_div(n_active_days, tenure_days.rename("td")).rename("active_days_fraction")

    act_win = pd.Timedelta(days=gen_cfg["activation_window_days"])
    act_cutoff = (sub_start + act_win).rename("_act_cutoff")
    g_first7 = g[g["created_at"] <= g["user_id"].map(act_cutoff)]
    gens_first_7 = g_first7.groupby("user_id").size().rename("gens_first_7_days")
    gens_last_7 = g_last14.groupby("user_id").size().rename("gens_last_7_days")  # reuse last14 window halved — use 7d

    # Re-filter properly for 7d
    g_last7 = g[g["created_at"] >= obs_date - pd.Timedelta(days=7)]
    gens_last_7 = g_last7.groupby("user_id").size().rename("gens_last_7_days")

    # Engagement trajectory: gens in first half vs second half of active span
    mid_dates = (first_gen + (last_gen - first_gen) / 2).rename("mid_date")
    g_with_mid = g.join(mid_dates, on="user_id")
    gens_first_half = g_with_mid[g_with_mid["created_at"] <= g_with_mid["mid_date"]].groupby("user_id").size()
    gens_second_half = g_with_mid[g_with_mid["created_at"] > g_with_mid["mid_date"]].groupby("user_id").size()
    trajectory_ratio = _safe_div(gens_second_half, gens_first_half + 1).rename("engagement_trajectory_ratio")

    # Engagement slope via linear regression on weekly generation counts
    log.info("  Engagement slope (weekly regression) ...")
    g_slope = g[["user_id", "created_at"]].copy()
    g_slope = g_slope.join(sub_start.rename("sub_start"), on="user_id")
    g_slope["week_num"] = ((g_slope["created_at"] - g_slope["sub_start"]).dt.days // 7).clip(lower=0)
    weekly = g_slope.groupby(["user_id", "week_num"]).size().reset_index(name="wcount")

    # Vectorised OLS slope: β = (n·Σxy − Σx·Σy) / (n·Σx² − (Σx)²)
    wg = weekly.groupby("user_id")
    n    = wg["week_num"].count()
    sx   = wg["week_num"].sum()
    sy   = wg["wcount"].sum()
    sxy  = (weekly["week_num"] * weekly["wcount"]).groupby(weekly["user_id"]).sum()
    sx2  = (weekly["week_num"] ** 2).groupby(weekly["user_id"]).sum()
    denom = (n * sx2 - sx ** 2).replace(0, np.nan)
    engagement_slope = ((n * sxy - sx * sy) / denom).fillna(0.0).where(n >= 3, 0.0).rename("engagement_slope")

    # 14-day momentum: recent 14d vs prior 14d
    gens_last_14 = g_last14.groupby("user_id").size()
    gens_prior_14 = g_prior14.groupby("user_id").size()
    momentum_14 = _safe_div(gens_last_14, gens_prior_14 + 1).rename("gens_last_14_vs_prior_14")

    # ── 2h: Frequency and session patterns ───────────────────────────────────
    log.info("  Frequency features ...")
    avg_per_active = _safe_div(total, n_active_days.replace(0, np.nan)).rename("avg_gens_per_active_day")
    gen_freq_daily = _safe_div(total, tenure_days.rename("td2")).rename("generation_frequency_daily")

    daily_counts = g.groupby(["user_id", g["created_at"].dt.normalize()]).size().groupby("user_id")
    gen_freq_cv = (daily_counts.std() / daily_counts.mean().replace(0, np.nan)).rename("generation_frequency_cv")

    # Inter-generation time (completed only)
    comp_sorted = completed_g.sort_values(["user_id", "created_at"])
    comp_sorted["prev_time"] = comp_sorted.groupby("user_id")["created_at"].shift(1)
    comp_sorted["inter_hours"] = (comp_sorted["created_at"] - comp_sorted["prev_time"]).dt.total_seconds() / 3600
    inter_gen_agg = comp_sorted.groupby("user_id")["inter_hours"].agg(
        avg_inter_generation_hours="mean",
        median_inter_generation_hours="median",
    )

    # ── 2i: Time-of-day / day-of-week ────────────────────────────────────────
    log.info("  Time-of-day features ...")
    g_tod = g[["user_id", "created_at"]].copy()
    g_tod["hour"] = g_tod["created_at"].dt.hour
    g_tod["is_weekday"] = g_tod["created_at"].dt.dayofweek < 5
    bh_start = gen_cfg["business_hours_start"]
    bh_end = gen_cfg["business_hours_end"]
    g_tod["is_biz_hour"] = g_tod["is_weekday"] & g_tod["hour"].between(bh_start, bh_end)

    pct_biz = _safe_div(g_tod.groupby("user_id")["is_biz_hour"].sum(),
                        total).rename("pct_gens_business_hours")
    pct_weekday = _safe_div(g_tod.groupby("user_id")["is_weekday"].sum(),
                            total).rename("pct_gens_weekdays")
    dominant_hour = g_tod.groupby("user_id")["hour"].agg(
        lambda x: int(x.mode().iloc[0]) if len(x) and len(x.mode()) else 0
    ).rename("dominant_usage_hour")

    # ── Assemble ──────────────────────────────────────────────────────────────
    log.info("  Assembling generation feature matrix ...")
    parts = [
        total, n_completed, n_failed, n_nsfw, n_canceled, n_queued,
        completion_rate, failure_rate_overall, failure_rate_last_10,
        failure_rate_delta, nsfw_rate, cancellation_rate,
        pct_free, n_credit_gens, total_credits, avg_credit,
        credit_burn_rate, credit_burn_acc, premium_gen_ratio,
        video_gen_ratio, image_gen_ratio, n_unique_types,
        video_graduation, image_model_1_share,
        avg_video_dur, median_video_dur, pct_high_res, dominant_aspect,
        avg_proc, median_proc, pct_long_wait,
        days_since_last_gen, days_to_first_gen, gen_span,
        n_active_days, active_days_frac,
        gens_first_7, gens_last_7,
        trajectory_ratio, engagement_slope, momentum_14,
        avg_per_active, gen_freq_daily, gen_freq_cv,
        inter_gen_agg,
        pct_biz, pct_weekday, dominant_hour,
        dominant_gen_type.rename("dominant_generation_type"),
    ]

    feat = pd.concat(parts, axis=1)
    feat.index.name = "user_id"
    return _reindex_to_users(feat, all_users, fill_value=0.0)


# ── PU: Purchase features (PU1–PU18) ──────────────────────────────────────────

def build_purchase_features(purchases: pd.DataFrame, obs_date: pd.Timestamp) -> pd.DataFrame:
    """Features from purchases table. One row per user_id."""
    log.info("Building purchase features ...")
    obs_date = _as_utc(obs_date)

    pu = purchases.copy()
    pu["purchase_time"] = pd.to_datetime(pu["purchase_time"], utc=True, errors="coerce")

    grp = pu.groupby("user_id")

    n_total = grp.size().rename("n_purchases_total")
    type_counts = pu.groupby(["user_id", "purchase_type"]).size().unstack(fill_value=0)

    def _type(col: str) -> pd.Series:
        return type_counts.get(col, pd.Series(0, index=type_counts.index))

    n_sub_creates = _type("Subscription Create").rename("n_subscription_creates")
    n_sub_updates = _type("Subscription Update").rename("n_subscription_updates")
    n_credit_pkgs = _type("Credits package").rename("n_credit_package_purchases")
    n_upsells = _type("Upsell").rename("n_upsell_purchases")
    has_reactivation = (_type("Reactivation") > 0).astype(int).rename("has_reactivation_purchase")
    pct_credit_pkgs = _safe_div(n_credit_pkgs, n_total).rename("pct_credit_package_purchases")

    total_spend = grp["purchase_amount_dollars"].sum().rename("total_purchase_dollars")
    avg_spend = grp["purchase_amount_dollars"].mean().rename("avg_purchase_dollars")
    max_spend = grp["purchase_amount_dollars"].max().rename("max_purchase_dollars")
    credit_pkg_spend = (
        pu[pu["purchase_type"] == "Credits package"]
        .groupby("user_id")["purchase_amount_dollars"].sum()
        .rename("credit_package_spend_total")
    )

    first_purchase = grp["purchase_time"].min().rename("_first_purchase")
    last_purchase = grp["purchase_time"].max().rename("_last_purchase")
    days_since_last_pu = ((obs_date - last_purchase).dt.days).rename("days_since_last_purchase")

    # Avg days between purchases
    def _avg_gap(times: pd.Series) -> float:
        t = times.sort_values().dropna()
        if len(t) < 2:
            return np.nan
        return float((t.diff().dt.total_seconds() / 86400).dropna().mean())

    avg_gap = grp["purchase_time"].apply(_avg_gap).rename("avg_days_between_purchases")

    # Plan upgrade / downgrade detection via Subscription Update amount vs previous
    sub_upd = pu[pu["purchase_type"] == "Subscription Update"].sort_values(["user_id", "purchase_time"])
    sub_upd = sub_upd.copy()
    sub_upd["prev_amount"] = sub_upd.groupby("user_id")["purchase_amount_dollars"].shift(1)
    sub_upd["delta"] = sub_upd["purchase_amount_dollars"] - sub_upd["prev_amount"]
    has_upgrade = (sub_upd[sub_upd["delta"] > 0].groupby("user_id").size() > 0).astype(int).rename("has_plan_upgrade")
    has_downgrade = (sub_upd[sub_upd["delta"] < 0].groupby("user_id").size() > 0).astype(int).rename("has_plan_downgrade")
    n_plan_changes = n_sub_updates.rename("n_plan_changes")

    feat = pd.concat([
        n_total, n_sub_creates, n_sub_updates, n_credit_pkgs, n_upsells,
        has_reactivation, pct_credit_pkgs,
        total_spend, avg_spend, max_spend, credit_pkg_spend,
        days_since_last_pu, avg_gap,
        has_upgrade, has_downgrade, n_plan_changes,
        first_purchase, last_purchase,
    ], axis=1)
    feat.index.name = "user_id"
    return feat


# ── T: Transaction features (T1–T29 + T_NEW1–T_NEW4) ─────────────────────────

def build_transaction_features(
    transactions: pd.DataFrame,
    purchases: pd.DataFrame,
    obs_date: pd.Timestamp,
) -> pd.DataFrame:
    """Features from transaction_attempts table (now has user_id directly).
    One row per user_id.
    """
    log.info("Building transaction features ...")
    cfg = load_config()
    txn_cfg = cfg["transaction"]
    obs_date = _as_utc(obs_date)

    t = transactions.copy()
    t["transaction_time"] = pd.to_datetime(t["transaction_time"], utc=True, errors="coerce")
    recent_win = pd.Timedelta(days=txn_cfg["recent_window_days"])

    grp = t.groupby("user_id")

    # ── T_NEW: Features only possible now that user_id exists ─────────────────
    failed = t[t["is_failed"]]
    successful = t[t["is_successful"]]

    purchase_txn_ids = set(purchases["transaction_id"].dropna())
    failed_no_match = failed[~failed["transaction_id"].isin(purchase_txn_ids)]

    has_failed_no_success = (
        failed.groupby("user_id").size() > 0
    ) & ~(
        successful.groupby("user_id").size() > 0
    ).reindex(failed["user_id"].unique(), fill_value=False)

    t_new1 = has_failed_no_success.astype(int).rename("has_failed_but_no_successful_payment")
    t_new2 = failed_no_match.groupby("user_id").size().rename("n_failed_without_matching_purchase")
    t_new3 = (
        t.sort_values("transaction_time").groupby("user_id")
        .first()["is_failed"].astype(int).rename("first_transaction_was_failure")
    )
    t_new4 = (
        failed.groupby("user_id")["amount_in_usd"].nunique()
        .rename("n_distinct_amounts_attempted")
    )

    # ── T1–T6: Volume and failure rate ────────────────────────────────────────
    n_total_txn = grp.size().rename("n_total_transaction_attempts")
    n_successful = successful.groupby("user_id").size().rename("n_successful_transactions")
    n_failed_txn = failed.groupby("user_id").size().rename("n_failed_transactions")
    txn_fail_rate = _safe_div(n_failed_txn, n_total_txn).rename("transaction_failure_rate")

    t_recent = t[t["transaction_time"] >= obs_date - recent_win]
    n_recent = t_recent.groupby("user_id").size()
    n_recent_fail = t_recent[t_recent["is_failed"]].groupby("user_id").size()
    txn_fail_rate_recent = _safe_div(n_recent_fail, n_recent).rename("transaction_failure_rate_recent")
    fail_rate_accel = (txn_fail_rate_recent - txn_fail_rate).rename("failure_rate_acceleration")

    # ── T7–T12: Failure code profile ──────────────────────────────────────────
    n_card_declined = (
        failed[failed["failure_code"] == "card_declined"].groupby("user_id").size()
        .rename("n_card_declined")
    )
    n_cvc_fail = (
        failed[failed["failure_code"].isin(["incorrect_cvc", "invalid_cvc"])]
        .groupby("user_id").size().rename("n_cvc_failures")
    )
    n_expired = (
        failed[failed["failure_code"] == "expired_card"].groupby("user_id").size()
        .rename("n_expired_card")
    )
    n_auth_fail = (
        failed[failed["failure_code"] == "authentication_required"].groupby("user_id").size()
        .rename("n_auth_required_failures")
    )
    n_proc_err = (
        failed[failed["failure_code"] == "processing_error"].groupby("user_id").size()
        .rename("n_processing_errors")
    )
    dominant_fail_code = (
        failed.groupby("user_id")["failure_code"]
        .agg(lambda x: x.mode().iloc[0] if len(x) and len(x.mode()) else None)
        .rename("dominant_failure_code")
    )

    # T13: Payment retry (same user, same amount, failed within retry_window_hours of prev fail)
    retry_win_h = txn_cfg["retry_window_hours"]
    f_sorted = failed.sort_values(["user_id", "amount_in_usd", "transaction_time"]).copy()
    f_sorted["prev_time"] = f_sorted.groupby(["user_id", "amount_in_usd"])["transaction_time"].shift(1)
    f_sorted["hours_since_prev"] = (
        (f_sorted["transaction_time"] - f_sorted["prev_time"]).dt.total_seconds() / 3600
    )
    payment_retry_count = (
        f_sorted[f_sorted["hours_since_prev"] <= retry_win_h]
        .groupby("user_id").size().rename("payment_retry_count")
    )

    # ── T14–T19: Payment instrument risk ─────────────────────────────────────
    uses_prepaid = (t.groupby("user_id")["is_prepaid"].any()).astype(int).rename("uses_prepaid_card")
    uses_virtual = (t.groupby("user_id")["is_virtual"].any()).astype(int).rename("uses_virtual_card")
    uses_business = (t.groupby("user_id")["is_business"].any()).astype(int).rename("uses_business_card")
    uses_wallet = (
        (t["digital_wallet"] != "none") & t["digital_wallet"].notna()
    ).groupby(t["user_id"]).any().astype(int).rename("uses_digital_wallet")
    card_funding_mode = (
        grp["card_funding"].agg(lambda x: x.mode().iloc[0] if len(x) and len(x.mode()) else None)
        .rename("card_funding_type")
    )
    pct_prepaid = _safe_div(
        t[t["is_prepaid"] == True].groupby("user_id").size(), n_total_txn
    ).rename("pct_prepaid_transactions")

    # ── T20–T23: Payment friction ─────────────────────────────────────────────
    n_3d_friction = (
        t[(t["is_3d_secure"] == True) & (t["is_3d_secure_authenticated"] == False)]
        .groupby("user_id").size().rename("n_3d_secure_friction")
    )
    has_cvc_fail = (t[t["cvc_check"] == "fail"].groupby("user_id").size() > 0).astype(int).rename("has_any_cvc_failure")
    cvc_fail_rate = _safe_div(
        (t["cvc_check"] == "fail").groupby(t["user_id"]).sum(),
        n_total_txn
    ).rename("cvc_fail_rate")
    n_cvc_unavail = (
        (t["cvc_check"] == "unavailable").groupby(t["user_id"]).sum()
        .rename("n_cvc_unavailable")
    )

    # ── T24–T29: Spend and geography ─────────────────────────────────────────
    total_txn_usd = successful.groupby("user_id")["amount_in_usd"].sum().rename("total_transaction_amount_usd")
    avg_txn_usd = grp["amount_in_usd"].mean().rename("avg_transaction_amount_usd")
    hv_thresh = txn_cfg["high_value_usd"]
    n_hv_txn = (
        successful[successful["amount_in_usd"] > hv_thresh]
        .groupby("user_id").size().rename("n_high_value_transactions")
    )
    days_since_last_fail = (
        (obs_date - failed.groupby("user_id")["transaction_time"].max()).dt.days
        .rename("days_since_last_failed_transaction")
    )
    days_since_last_success = (
        (obs_date - successful.groupby("user_id")["transaction_time"].max()).dt.days
        .rename("days_since_last_successful_transaction")
    )

    feat = pd.concat([
        t_new1, t_new2, t_new3, t_new4,
        n_total_txn, n_successful, n_failed_txn,
        txn_fail_rate, txn_fail_rate_recent, fail_rate_accel,
        n_card_declined, n_cvc_fail, n_expired, n_auth_fail, n_proc_err,
        dominant_fail_code, payment_retry_count,
        uses_prepaid, uses_virtual, uses_business, uses_wallet,
        card_funding_mode, pct_prepaid,
        n_3d_friction, has_cvc_fail, cvc_fail_rate, n_cvc_unavail,
        total_txn_usd, avg_txn_usd, n_hv_txn,
        days_since_last_fail, days_since_last_success,
    ], axis=1)
    feat.index.name = "user_id"
    return feat


# ── Q: Quiz features (Q1–Q15) ─────────────────────────────────────────────────

def build_quiz_features(quizzes: pd.DataFrame) -> pd.DataFrame:
    """Features from quizzes table. One row per user_id."""
    log.info("Building quiz features ...")
    cfg = load_config()

    q = quizzes.set_index("user_id").copy()

    # Q1: Completion depth (count of non-null quiz fields)
    quiz_fields = ["source", "flow_type", "team_size", "experience", "usage_plan", "frustration", "first_feature", "role"]
    quiz_completion_depth = q[quiz_fields].notna().sum(axis=1).rename("quiz_completion_depth")

    # Q2: Acquisition channel intent tier
    _source_intent = {
        "google": 2, "chatgpt": 2, "ai-community": 2,
        "youtube": 1, "friends": 1, "twitter": 1,
        "instagram": 0, "tiktok": 0,
    }
    source_norm = q["source"].astype(str).str.lower().str.strip()
    acq_intent = source_norm.map(_source_intent).fillna(1).astype(int).rename("acquisition_channel_intent")

    # Q3: Source encoded
    top_sources = cfg["top_sources"]
    source_encoded = encode_top_n(q["source"], top_sources).rename("source_encoded")

    # Q4: Team size ordinal (already computed in preprocess, re-derive here)
    _team_map = {
        "solo": 1, "1": 1, "small": 2, "growing": 3, "2-10": 3,
        "midsize": 4, "11-50": 4, "501-2000": 5, "2001-5000": 5, "enterprise": 6,
    }
    team_size_ordinal = (
        q["team_size"].astype(str).str.lower().str.strip()
        .map(_team_map).fillna(0).astype(int).rename("team_size_ordinal")
    )

    # Q5: Experience ordinal
    _exp_map = {"beginner": 1, "intermediate": 2, "advanced": 3, "expert": 4}
    experience_ordinal = (
        q["experience"].astype(str).str.lower().str.strip()
        .map(_exp_map).fillna(0).astype(int).rename("experience_ordinal")
    )

    # Q6: Commercial usage flag
    commercial_plans = {"marketing", "filmmaking", "freelance", "education", "social"}
    usage_plan_commercial = (
        q["usage_plan"].astype(str).str.lower().str.strip().isin(commercial_plans)
        .astype(int).rename("usage_plan_commercial")
    )

    # Q7: Usage plan encoded
    usage_plan_encoded = q["usage_plan"].astype(str).str.lower().str.strip().rename("usage_plan_encoded")
    usage_plan_encoded = usage_plan_encoded.where(q["usage_plan"].notna(), other="unknown")

    # Q8–Q11: Frustration binary flags
    _frust_norm_map = {
        "high cost of top models": "high-cost",
        "inconsistent results": "inconsistent",
        "limited generations": "limited",
    }
    frust = q["frustration"].astype(str).str.lower().str.strip().replace(_frust_norm_map)
    frust = frust.where(q["frustration"].notna(), other=None)

    frustrated_cost = (frust.isin(["high-cost"])).astype(int).rename("frustrated_cost")
    frustrated_quality = (frust.isin(["inconsistent"])).astype(int).rename("frustrated_quality")
    frustrated_limited = (frust.isin(["limited"])).astype(int).rename("frustrated_limited")
    frustrated_confusing = (frust.isin(["confusing", "hard-prompt"])).astype(int).rename("frustrated_confusing")

    # Q12: First feature is video
    first_feature_video = (
        q["first_feature"].astype(str).str.lower().str.contains("video", na=False)
        .astype(int).rename("first_feature_video")
    )

    # Q13: First feature encoded
    _ff_top = [
        "Video Generations", "Commercial & Ad Videos", "Realistic AI Avatars",
        "Cinematic Visuals", "Viral Social Media Content", "Image Editing & Inpaint",
        "video-creation", "image-creation",
    ]
    first_feature_encoded = encode_top_n(q["first_feature"], _ff_top).rename("first_feature_encoded")

    # Q14: Role commitment score
    _role_score = {
        "just-for-fun": 0, "creator": 1, "designer": 1,
        "marketer": 2, "educator": 2, "prompt-engineer": 2,
        "filmmaker": 3, "brand-owner": 3,
        "founder": 4, "developer": 2,
    }
    role_commitment = (
        q["role"].astype(str).str.lower().str.strip()
        .map(_role_score).fillna(0).astype(int).rename("role_commitment_score")
    )

    # Q15: Role encoded
    role_encoded = q["role"].astype(str).str.lower().str.strip().rename("role_encoded")
    role_encoded = role_encoded.where(q["role"].notna(), other="unknown")

    feat = pd.concat([
        quiz_completion_depth, acq_intent, source_encoded,
        team_size_ordinal, experience_ordinal,
        usage_plan_commercial, usage_plan_encoded,
        frustrated_cost, frustrated_quality, frustrated_limited, frustrated_confusing,
        first_feature_video, first_feature_encoded,
        role_commitment, role_encoded,
    ], axis=1)
    feat.index.name = "user_id"
    return feat
