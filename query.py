# Python
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Dict, Optional, Tuple
import re
import pandas as pd


def _get_metric_by_date_lead(
    start_date,
    end_date,
    lead_min: float,
    lead_max: float,
    metric: str,
    stations: Iterable[str],
    *,
    data_dir: str | Path,
    path_template: str = "{data_dir}/{date}/*.{station}.csv",
    date_fmt: str = "%Y%m%d",
    quiet_missing: bool = True,
) -> pd.DataFrame:
    """
    Load per-station CSV metric files for a date interval and return a DataFrame with:
    columns = ['date', 'lead', 'station', '<metric name as passed>'].

    Parameters:
      - start_date, end_date: Any pandas-compatible date inputs (e.g., '2024-01-01', datetime, pd.Timestamp).
                              Inclusive date range at daily frequency.
      - lead_min, lead_max:   Lead-time interval (inclusive), in hours (floats accepted).
      - metric:               Metric to extract (case-insensitive, common aliases supported). Examples:
                              'bias', 'rmsd', 'rval', 'skill', 'vexp', 'peak', 'obsmax', 'plag', 'npts'
      - stations:             Iterable of station identifiers used in file names via {station}.
      - data_dir:             Base directory containing the files.
      - path_template:        A Python str.format template that must include {data_dir}, {date}, {station}.
                              Example: "{data_dir}/{date}/estofs_pac.{station}.cwl.csv"
      - date_fmt:             strftime format for {date} substitution (default "%Y%m%d").
      - quiet_missing:        If False, prints a warning for missing files; otherwise, silently skips.

    Returns:
      pd.DataFrame with columns: ['date', 'lead', 'station', '<metric name as passed>'].
      - 'date' is a pandas.Timestamp normalized to midnight (no timezone).
      - 'lead' is float.
      - 'station' is str.
      - '<metric name as passed>' is float (NaNs may appear if the value cannot be parsed).
    """
    start_ts = pd.to_datetime(start_date).normalize()
    end_ts = pd.to_datetime(end_date).normalize()
    if end_ts < start_ts:
        raise ValueError("end_date must be on or after start_date")

    date_index = pd.date_range(start_ts, end_ts, freq="D")
    stations = list(stations)

    metric_key = _normalize_key(metric)
    lead_lo, lead_hi = float(lead_min), float(lead_max)
    if lead_hi < lead_lo:
        lead_lo, lead_hi = lead_hi, lead_lo  # swap for safety

    records = []

    for d in date_index:
        date_str = d.strftime(date_fmt)
        for station in stations:
            path = _build_path(
                path_template=path_template,
                data_dir=data_dir,
                date_str=date_str,
                station=str(station),
            )

            df = _read_metrics_csv(path, quiet_missing=quiet_missing)
            if df is None:
                continue

            # Find the 'lead' and desired metric columns (robust to header variations)
            lead_col, metric_col = _resolve_columns(df, metric_key)
            if lead_col is None or metric_col is None:
                # Skip this file if columns cannot be resolved
                if not quiet_missing:
                    print(f"[warn] Skipping {path} (cannot resolve columns for metric={metric})")
                continue

            # Coerce numeric and filter lead interval
            leads = pd.to_numeric(df[lead_col], errors="coerce")
            vals = pd.to_numeric(df[metric_col], errors="coerce")

            mask = (leads >= lead_lo) & (leads <= lead_hi)
            use = mask.fillna(False)

            if not use.any():
                continue

            sub = pd.DataFrame(
                {
                    "date": d,
                    "lead": leads[use].astype(float).values,
                    "station": station,
                    metric: vals[use].astype(float).values,
                }
            )
            records.append(sub)

    if not records:
        # Return empty DataFrame with the expected schema (metric column named like the requested metric)
        return pd.DataFrame(columns=["date", "lead", "station", str(metric)])

    out = pd.concat(records, ignore_index=True)
    # Sort for convenience
    out.sort_values(["date", "station", "lead"], inplace=True, kind="stable")
    out.reset_index(drop=True, inplace=True)
    return out


# ---------------------------
# Helpers
# ---------------------------

def _extract_station_id(path: Path) -> Optional[str]:
    """
    Best-effort extraction of station ID from a metrics filename.

    Expected common patterns (examples):
      - estofs_atl.8410140.cwl.csv  -> '8410140'
      - modelX.9462620.cwl.csv      -> '9462620'

    Heuristics:
      - If the filename ends with '.cwl.csv' and has at least 4 dot-separated parts,
        return the token before 'cwl' (i.e., parts[-3]).
      - Otherwise, fall back to the penultimate token (parts[-2]) if available.
    """
    name = path.name
    parts = name.split(".")
    lower = name.lower()
    if lower.endswith(".cwl.csv") and len(parts) >= 4:
        # e.g., ["estofs_atl", "8410140", "cwl", "csv"] -> "8410140"
        return parts[-3]
    if len(parts) >= 2:
        # Fallback: take the penultimate token
        return parts[-2]
    return None


def get_available_stations(
    start_date,
    end_date,
    *,
    data_dir: str | Path,
    path_template: str = "{data_dir}/{date}/*.{station}.csv",
    date_fmt: str = "%Y%m%d",
) -> list[str]:
    """
    Return a sorted list of unique station IDs available across the given date interval.

    How it works:
      - For each date in [start_date, end_date], builds a search pattern by substituting
        {data_dir} and {date} in 'path_template' and replacing {station} with '*'.
      - Globs all matching files in that directory and extracts the station ID from each filename.

    Notes:
      - The template must include {data_dir}, {date}, and {station}.
      - Works with patterns like "{data_dir}/{date}/*.{station}.cwl.csv" (wildcards before {station} are fine).

    Parameters:
      - start_date, end_date: pandas-compatible date inputs (inclusive, daily).
      - data_dir: base directory of metrics files.
      - path_template: str.format template with {data_dir}, {date}, {station}.
      - date_fmt: strftime format for {date}.

    Returns:
      - Sorted list of unique station IDs (as strings).
    """
    start_ts = pd.to_datetime(start_date).normalize()
    end_ts = pd.to_datetime(end_date).normalize()
    if end_ts < start_ts:
        raise ValueError("end_date must be on or after start_date")

    date_index = pd.date_range(start_ts, end_ts, freq="D")
    base = str(data_dir)
    stations: set[str] = set()

    for d in date_index:
        date_str = d.strftime(date_fmt)
        # Build a path with wildcard for station
        pattern_str = path_template.format(data_dir=base, date=date_str, station="*")
        pattern_path = Path(pattern_str)

        # If the template has wildcards only in the filename part, use parent.glob(name)
        # Otherwise, use glob on the full pattern
        if any(ch in (pattern_path.name) for ch in ("*", "?", "[")):
            parent = pattern_path.parent if str(pattern_path.parent) else Path(".")
            matches = list(parent.glob(pattern_path.name))
        else:
            # No wildcard in the filename; still try to match exact path
            matches = [pattern_path] if pattern_path.exists() else []

        for p in matches:
            sid = _extract_station_id(p)
            if sid:
                stations.add(sid)

    return sorted(stations)


def summarize_metric_by_date_lead(
    start_date,
    end_date,
    lead_min: float,
    lead_max: float,
    metric: str,
    stations: Iterable[str],
    *,
    data_dir: str | Path,
    path_template: str = "{data_dir}/{date}/*.{station}.cwl.csv",
    date_fmt: str = "%Y%m%d",
    quiet_missing: bool = True,
    tol: float = 0.0,
) -> Dict[str, list[Dict[str, object]]]:
    """
    Compute min, max, and median of the metric values across the selected date/lead/station range,
    returning the matching rows' station, date, and lead. Supports ties.

    Parameters mirror get_metric_by_date_lead, plus:
      - tol: Tolerance when matching rows to the target value(s).
             For median with even N, rows at the two middle values are returned (not the average).

    Returns:
      {
        "min":    [ {metric_name, metric, station, date, lead}, ... ],
        "max":    [ {metric_name, metric, station, date, lead}, ... ],
        "median": [ {metric_name, metric, station, date, lead}, ... ],
      }
    """
    df = _get_metric_by_date_lead(
        start_date=start_date,
        end_date=end_date,
        lead_min=lead_min,
        lead_max=lead_max,
        metric=metric,
        stations=stations,
        data_dir=data_dir,
        path_template=path_template,
        date_fmt=date_fmt,
        quiet_missing=quiet_missing,
    )

    # Handle empty quickly
    if df.empty:
        return {"min": [], "max": [], "median": []}

    # Identify metric column (same as the 'metric' argument)
    if metric not in df.columns:
        raise ValueError(f"Metric column '{metric}' not present in results")

    # Coerce to numeric and drop NaNs for summary
    vals = pd.to_numeric(df[metric], errors="coerce")
    non_nan = df.loc[vals.notna()].copy()
    non_nan["_metric_num"] = vals.loc[vals.notna()].astype(float)

    if non_nan.empty:
        return {"min": [], "max": [], "median": []}

    # Compute min/max
    min_val = float(non_nan["_metric_num"].min())
    max_val = float(non_nan["_metric_num"].max())

    def match_value(target: float) -> pd.DataFrame:
        if tol == 0.0:
            return non_nan.loc[non_nan["_metric_num"] == target]
        return non_nan.loc[(non_nan["_metric_num"] - target).abs() <= tol]

    def rows_to_dicts(rows: pd.DataFrame) -> list[Dict[str, object]]:
        out: list[Dict[str, object]] = []
        for _, r in rows.iterrows():
            out.append(
                {
                    "metric_name": metric,
                    "metric": float(r["_metric_num"]),
                    "station": str(r["station"]),
                    "date": r["date"],
                    "lead": float(r["lead"]),
                }
            )
        return out

    min_rows = match_value(min_val)
    max_rows = match_value(max_val)

    # Median handling:
    # - Odd N: return rows matching the middle value
    # - Even N: return rows for the two middle values (ties supported)
    ordered = non_nan.sort_values("_metric_num", kind="stable").reset_index()
    n = len(ordered)
    if n % 2 == 1:
        mid_val = float(ordered.loc[n // 2, "_metric_num"])
        median_rows = match_value(mid_val)
    else:
        lo_val = float(ordered.loc[n // 2 - 1, "_metric_num"])
        hi_val = float(ordered.loc[n // 2, "_metric_num"])
        # Combine rows for both defining median values
        med_lo = match_value(lo_val)
        med_hi = match_value(hi_val)
        median_rows = pd.concat([med_lo, med_hi]).drop_duplicates()

    return {
        "min": rows_to_dicts(min_rows),
        "max": rows_to_dicts(max_rows),
        "median": rows_to_dicts(median_rows),
    }


# ---------------------------
# Helpers
# ---------------------------


def _build_path(*, path_template: str, data_dir: str | Path, date_str: str, station: str) -> Path:
    # Allow both str and Path for data_dir
    base = str(data_dir)
    path_str = path_template.format(data_dir=base, date=date_str, station=station)
    p = Path(path_str)

    # If the template includes glob characters, expand them and pick a deterministic match.
    # This lets you use patterns like "{data_dir}/{date}/*.{station}.cwl.csv"
    if any(ch in p.name for ch in ("*", "?", "[")):
        parent = p.parent if str(p.parent) else Path(".")
        matches = sorted(parent.glob(p.name))
        if matches:
            # Choose the first match (lexicographically). Adjust if you need a different policy.
            return matches[0]

    return p


def _read_metrics_csv(path: Path, quiet_missing: bool = True) -> Optional[pd.DataFrame]:
    if not path.exists():
        if not quiet_missing:
            print(f"[warn] Missing file: {path}")
        return None

    # Read with python engine to be tolerant of trailing commas
    try:
        df = pd.read_csv(path, engine="python")
    except Exception as exc:
        if not quiet_missing:
            print(f"[warn] Failed to read {path}: {exc}")
        return None

    # Normalize column names: strip spaces and trailing commas
    df.columns = [c.strip().rstrip(",") for c in df.columns]
    return df


_COL_ALIASES: Dict[str, Tuple[str, ...]] = {
    # normalized key -> acceptable normalized column names seen in files
    "lead": ("leadtimehrs", "leadtime", "leadhrs", "lead"),
    "obsmax": ("obsmaxmmsl", "obsmax"),
    "peak": ("peakm", "peak"),
    "plag": ("plagmin", "plag"),
    "bias": ("biasm", "bias"),
    "rmsd": ("rmsdm", "rmsd"),
    "rval": ("rval", "r"),
    "skil": ("skil", "skill"),
    "vexp": ("vexp", "vexppercent", "vexp%"),
    "npts": ("npts", "npoints", "n"),
}


def _normalize_key(name: str) -> str:
    """
    Normalize a metric or column name: lowercase, remove non-alnum.
    e.g., "LeadTime (HRS)" -> "leadtimehrs", "Bias(m)" -> "biasm"
    """
    return re.sub(r"[^0-9a-z]+", "", str(name).lower())


def _resolve_columns(df: pd.DataFrame, metric_key: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Given a DataFrame with raw headers and a desired metric key (normalized),
    return (lead_col_name, metric_col_name) if found; otherwise (None, None).
    """
    norm_to_actual: Dict[str, str] = {}
    for c in df.columns:
        norm_to_actual[_normalize_key(c)] = c

    # Resolve lead column
    lead_col = None
    for candidate in _COL_ALIASES["lead"]:
        if candidate in norm_to_actual:
            lead_col = norm_to_actual[candidate]
            break

    # Resolve metric column
    # First, convert metric_key into a canonical key found in _COL_ALIASES
    canon_metric = _canonical_metric_key(metric_key)
    if canon_metric is None:
        return (lead_col, None)

    metric_col = None
    for candidate in _COL_ALIASES[canon_metric]:
        if candidate in norm_to_actual:
            metric_col = norm_to_actual[candidate]
            break

    return (lead_col, metric_col)


def _canonical_metric_key(metric_key: str) -> Optional[str]:
    """
    Map a normalized user-provided metric key to a canonical key in _COL_ALIASES.
    Accepts common aliases (e.g., 'corr' -> 'rval', 'varianceexplained' -> 'vexp').
    """
    # Direct match
    if metric_key in _COL_ALIASES:
        return metric_key

    # Aliases
    alias_map = {
        "corr": "rval",
        "correlation": "rval",
        "r": "rval",
        "skill": "skil",
        "skillscore": "skil",
        "varianceexplained": "vexp",
        "varexplained": "vexp",
        "ve": "vexp",
        "points": "npts",
        "count": "npts",
    }
    return alias_map.get(metric_key, None)

if __name__ == "__main__":
    df = _get_metric_by_date_lead(
        start_date="2022-01-01",
        end_date="2022-01-01",
        lead_min=0,
        lead_max=150,
        metric="RMSD",
        stations=get_available_stations(start_date="2022-01-01",
                                        end_date="2022-01-01",
                                        data_dir=Path("metrics_data"),
                                        path_template="{data_dir}/{date}/*.{station}.cwl.csv"),
        data_dir=Path("metrics_data"),
        path_template="{data_dir}/{date}/*.{station}.cwl.csv",
        date_fmt="%Y%m%d",
        quiet_missing=True,
    )
    print(df.head())
    print(f"Loaded rows: {len(df)}")

    # Example 2: Summarize min, max, and median (with corresponding date/lead/station)
    summary = summarize_metric_by_date_lead(
        start_date="2022-01-01",
        end_date="2022-01-01",
        lead_min=0,
        lead_max=150,
        metric="RMSD",
        stations=get_available_stations(start_date="2022-01-01",
                                        end_date="2022-01-01",
                                        data_dir=Path("metrics_data"),
                                        path_template="{data_dir}/{date}/*.{station}.cwl.csv"),
        data_dir=Path("metrics_data"),
        path_template="{data_dir}/{date}/*.{station}.cwl.csv",
        date_fmt="%Y%m%d",
        quiet_missing=True,
        tol=0.0,  # set >0 to allow near-matches when comparing floats
    )

    print("Min rows:", summary["min"])
    print("Max rows:", summary["max"])
    print("Median rows:", summary["median"])