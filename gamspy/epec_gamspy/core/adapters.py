from __future__ import annotations

from typing import Dict, Tuple, Any, Optional, List
import pandas as pd


def df_1d(dom_name: str, data: Dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([(k, float(v)) for k, v in data.items()], columns=[dom_name, "value"])


def df_2d(dom1: str, dom2: str, data: Dict[Tuple[str, str], float]) -> pd.DataFrame:
    return pd.DataFrame([(k1, k2, float(v)) for (k1, k2), v in data.items()], columns=[dom1, dom2, "value"])


def _level_col(df: pd.DataFrame) -> str:
    # GAMSPy typically uses "level"; older/other exports can use "value"/"l"
    for c in ["level", "value", "val", "l"]:
        if c in df.columns:
            return c
    raise KeyError(f"Cannot find a level/value column in: {list(df.columns)}")


def _domain_names_from_var(var: Any) -> Optional[List[str]]:
    # Try to read GAMSPy Variable.domain -> list of Set/Alias with .name
    try:
        dom = getattr(var, "domain", None)
        if dom is None:
            return None
        names = []
        for d in dom:
            n = getattr(d, "name", None)
            if n:
                names.append(str(n))
        return names or None
    except Exception:
        return None


def _records_df(var: Any) -> pd.DataFrame:
    if var.records is None:
        return pd.DataFrame()
    # reset_index() is safer than reset_index(drop=False) and avoids double "index" surprises
    df = var.records.reset_index()
    # Drop purely positional index columns if present
    for junk in ["index", "level_0", "Unnamed: 0"]:
        if junk in df.columns:
            df = df.drop(columns=[junk])
    return df


def var_to_dict_1d(var: Any) -> Dict[str, float]:
    df = _records_df(var)
    if df.empty:
        return {}

    lc = _level_col(df)

    dom_names = _domain_names_from_var(var)
    if dom_names:
        dom_cols = [c for c in dom_names if c in df.columns]
    else:
        # fallback heuristic
        dom_cols = [c for c in df.columns if c not in ["marginal", "lower", "upper", "scale", lc]]

    if len(dom_cols) != 1:
        raise ValueError(f"Expected 1 domain column, got {dom_cols}. Columns={list(df.columns)}")

    dcol = dom_cols[0]
    return {str(row[dcol]): float(row[lc]) for _, row in df.iterrows()}


def var_to_dict_2d(var: Any) -> Dict[Tuple[str, str], float]:
    df = _records_df(var)
    if df.empty:
        return {}

    lc = _level_col(df)

    dom_names = _domain_names_from_var(var)
    if dom_names:
        dom_cols = [c for c in dom_names if c in df.columns]
    else:
        dom_cols = [c for c in df.columns if c not in ["marginal", "lower", "upper", "scale", lc]]

    if len(dom_cols) != 2:
        raise ValueError(f"Expected 2 domain columns, got {dom_cols}. Columns={list(df.columns)}")

    d1, d2 = dom_cols
    return {(str(row[d1]), str(row[d2])): float(row[lc]) for _, row in df.iterrows()}


def scalar_level(var: Any) -> Optional[float]:
    df = _records_df(var)
    if df.empty:
        return getattr(var, "l", None)
    lc = _level_col(df)
    return float(df.iloc[0][lc])
