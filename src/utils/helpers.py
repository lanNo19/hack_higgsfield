from pathlib import Path
import yaml
import pandas as pd


_ROOT = Path(__file__).resolve().parents[2]


def load_config() -> dict:
    with open(_ROOT / "config" / "config.yaml") as f:
        return yaml.safe_load(f)


def load_paths() -> dict:
    with open(_ROOT / "config" / "paths.yaml") as f:
        return yaml.safe_load(f)


def get_plan_info(cfg: dict | None = None) -> dict:
    """Return plan_name -> {ordinal, monthly_credits, monthly_cost_usd}."""
    if cfg is None:
        cfg = load_config()
    return cfg["plan_info"]


def get_plan_series(props: pd.DataFrame, field: str, cfg: dict | None = None) -> pd.Series:
    """Map subscription_plan column to a numeric plan field."""
    plan_info = get_plan_info(cfg)
    mapping = {plan: info[field] for plan, info in plan_info.items()}
    return props["subscription_plan"].map(mapping)


def normalize_country(s: pd.Series) -> pd.Series:
    """Uppercase + strip for ISO-2 comparison."""
    return s.str.upper().str.strip()


def encode_top_n(s: pd.Series, top_values: list, fill_na: str = "unknown") -> pd.Series:
    """Replace values not in top_values with 'other'; fill NaN with fill_na."""
    out = s.fillna(fill_na).str.lower().str.strip()
    top_lower = [v.lower() for v in top_values] + [fill_na]
    return out.where(out.isin(top_lower), other="other")


def root_path() -> Path:
    return _ROOT


def data_path(split: str) -> Path:
    paths = load_paths()
    return _ROOT / paths["data"]["raw"][split]


def processed_path() -> Path:
    paths = load_paths()
    return _ROOT / paths["data"]["processed"]
