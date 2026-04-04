"""Load raw CSVs for a given split into a dict of DataFrames."""
from pathlib import Path
import pandas as pd

from src.utils.helpers import data_path
from src.utils.logger import get_logger

log = get_logger(__name__)

_DATE_COLS: dict[str, list[str]] = {
    "users": [],
    "properties": ["subscription_start_date"],
    "purchases": ["purchase_time"],
    "quizzes": [],
    "transactions": ["transaction_time"],
    "generations": ["created_at", "completed_at", "failed_at"],
}

# Suffix appended after the split prefix (e.g. "train_users" + suffix → filename)
# users is special: train_users.csv (no extra suffix), others get _<suffix>
_TABLE_SUFFIX: dict[str, str] = {
    "users": "",
    "properties": "_properties",
    "purchases": "_purchases",
    "quizzes": "_quizzes",
    "transactions": "_transaction_attempts",
    "generations": "_generations",
}


def _fix_out_of_bounds_dates(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Clamp timestamps with impossible years to NaT (logic reused from EDA notebook)."""
    for col in cols:
        if col not in df.columns:
            continue
        mask = df[col].notna()
        try:
            years = df.loc[mask, col].dt.year
            bad = years[(years < 1970) | (years > 2100)].index
            if len(bad):
                log.warning("  Fixing %d out-of-bounds dates in '%s'", len(bad), col)
                df.loc[bad, col] = pd.NaT
        except Exception:
            pass
    return df


def _load_table(base: Path, prefix: str, table: str) -> pd.DataFrame:
    fname = f"{prefix}{_TABLE_SUFFIX[table]}.csv"
    fpath = base / fname
    date_cols = _DATE_COLS[table]

    log.info("Loading %s ...", fname)
    df = pd.read_csv(fpath, parse_dates=date_cols if date_cols else False, low_memory=False)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df = _fix_out_of_bounds_dates(df, date_cols)
    log.info("  -> %s rows x %d cols", f"{len(df):,}", df.shape[1])
    return df


def load_split(split: str) -> dict[str, pd.DataFrame]:
    """Load all tables for split ('train' or 'test').

    Returns dict with keys: users, properties, purchases, quizzes, transactions, generations.
    """
    assert split in ("train", "test"), f"Unknown split: {split}"
    base = data_path(split)
    prefix = f"{split}_users"

    return {table: _load_table(base, prefix, table) for table in _TABLE_SUFFIX}
