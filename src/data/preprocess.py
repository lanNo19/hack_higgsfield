"""Clean and normalise raw DataFrames before feature extraction."""
import pandas as pd

from src.utils.helpers import load_config
from src.utils.logger import get_logger

log = get_logger(__name__)


def _fix_year_1067(series: pd.Series) -> pd.Series:
    """Synthetic data artifact: all timestamps stored as year 1067 instead of 2023.
    Replace the year prefix before pd.to_datetime so the 956-year offset is restored.
    Dates span 1067-08 to 1067-11 → 2023-08 to 2023-11, matching obs_date 2023-11-25.
    """
    if series.dtype == object:
        return series.str.replace(r"^1067-", "2023-", regex=True)
    return series

# ── Quiz normalisation maps ────────────────────────────────────────────────────

_TEAM_SIZE_ORDINAL: dict[str, int] = {
    "solo": 1, "1": 1,
    "small": 2,
    "growing": 3, "2-10": 3,
    "midsize": 4, "11-50": 4,
    "501-2000": 5, "2001-5000": 5,
    "enterprise": 6,
}

_EXPERIENCE_ORDINAL: dict[str, int] = {
    "beginner": 1,
    "intermediate": 2,
    "advanced": 3,
    "expert": 4,
}

# Normalise messy frustration values to a canonical set
_FRUSTRATION_MAP: dict[str, str] = {
    "high cost of top models": "high-cost",
    "inconsistent results": "inconsistent",
    "limited generations": "limited",
}


def preprocess_properties(df: pd.DataFrame) -> pd.DataFrame:
    cfg = load_config()
    plan_info = cfg["plan_info"]

    out = df.copy()
    out["subscription_start_date"] = pd.to_datetime(
        _fix_year_1067(out["subscription_start_date"]), utc=True, errors="coerce"
    )
    out["plan_ordinal"] = out["subscription_plan"].map({p: v["ordinal"] for p, v in plan_info.items()})
    out["plan_monthly_credits"] = out["subscription_plan"].map({p: v["monthly_credits"] for p, v in plan_info.items()})
    out["plan_monthly_cost_usd"] = out["subscription_plan"].map({p: v["monthly_cost_usd"] for p, v in plan_info.items()})
    out["country_code"] = out["country_code"].str.upper().str.strip()
    return out


def preprocess_generations(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # credit_cost NULL = free model (not missing); set to 0 for aggregation
    out["credit_cost_filled"] = out["credit_cost"].fillna(0.0)
    out["is_free_model"] = out["credit_cost"].isna() | (out["credit_cost"] == 0)
    out["is_video"] = out["generation_type"].str.startswith("video_", na=False)
    out["is_image"] = out["generation_type"].str.startswith("image_", na=False)
    # Ensure datetimes are tz-aware UTC (fix year-1067 synthetic data offset first)
    for col in ["created_at", "completed_at", "failed_at"]:
        if col in out.columns:
            out[col] = pd.to_datetime(_fix_year_1067(out[col]), utc=True, errors="coerce")
    return out


def preprocess_purchases(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["purchase_time"] = pd.to_datetime(_fix_year_1067(out["purchase_time"]), utc=True, errors="coerce")
    return out


def preprocess_transactions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["transaction_time"] = pd.to_datetime(_fix_year_1067(out["transaction_time"]), utc=True, errors="coerce")
    out["is_failed"] = out["failure_code"].notna()
    out["is_successful"] = out["failure_code"].isna()
    # Normalise billing country to uppercase for cross-table comparison
    out["billing_address_country_norm"] = out["billing_address_country"].str.upper().str.strip()
    return out


def preprocess_quizzes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # team_size → ordinal (0 = unknown)
    out["team_size_str"] = out["team_size"].astype(str).str.lower().str.strip()
    out["team_size_ordinal"] = out["team_size_str"].map(_TEAM_SIZE_ORDINAL).fillna(0).astype(int)

    # experience → ordinal (0 = unknown)
    out["experience_ordinal"] = (
        out["experience"].astype(str).str.lower().str.strip()
        .map(_EXPERIENCE_ORDINAL).fillna(0).astype(int)
    )

    # frustration → normalised lowercase canonical value
    out["frustration_norm"] = (
        out["frustration"].astype(str).str.lower().str.strip()
        .replace(_FRUSTRATION_MAP)
    )
    out["frustration_norm"] = out["frustration_norm"].where(out["frustration"].notna(), other=None)

    # source → lowercase
    out["source_norm"] = out["source"].astype(str).str.lower().str.strip()
    out["source_norm"] = out["source_norm"].where(out["source"].notna(), other=None)

    return out


def preprocess_all(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    log.info("Preprocessing all tables ...")
    return {
        "users": tables["users"].copy(),
        "properties": preprocess_properties(tables["properties"]),
        "generations": preprocess_generations(tables["generations"]),
        "purchases": preprocess_purchases(tables["purchases"]),
        "transactions": preprocess_transactions(tables["transactions"]),
        "quizzes": preprocess_quizzes(tables["quizzes"]),
    }
