from __future__ import annotations

from typing import Iterable, Dict, Optional, Tuple
import re
import pandas as pd
from urllib.parse import urlparse
from pathlib import PurePosixPath

# =========================
# S3-only helpers
# =========================

def _iter_s3_paths(*, path_template: str, data_dir: str, date_str: str, station: str) -> list[str]:
    """
    Return a sorted list of s3://... object URLs matching the template.
    Requires: data_dir starts with 's3://'.
    Supports '**' recursion and other wildcards (via fsspec).
    """
    if not str(data_dir).startswith("s3://"):
        raise ValueError("data_dir must be an s3:// URL for S3-only mode")

    import fsspec  # requires s3fs installed
    fs = fsspec.filesystem("s3", anon=True)

    pattern = path_template.format(data_dir=data_dir, date=date_str, station=station)
    matches = fs.glob(pattern)
    # Ensure fully-qualified s3:// URLs
    return [m if m.startswith("s3://") else f"s3://{m}" for m in sorted(matches)]


def _read_s3_csv(s3_url: str, quiet_missing: bool = True) -> Optional[pd.DataFrame]:
    """
    Read a CSV from S3 (anonymous). Returns DataFrame or None.
    Tolerant of trailing commas and header quirks.
    """
    try:
        df = pd.read_csv(s3_url, engine="python", storage_options={"anon": True})
    except FileNotFoundError:
        if not quiet_missing:
            print(f"[warn] Missing S3 object: {s3_url}")
        return None
    except Exception as exc:
        if not quiet_missing:
            print(f"[warn] Failed to read {s3_url}: {exc}")
        return None

    df.columns = [c.strip().rstrip(",") for c in df.columns]
    return df


# =========================
# Column/metric resolution
# =========================

_COL_ALIASES: Dict[str, Tuple[str, ...]] = {
    "lead": ("leadtimehrs", "leadtime", "leadhrs", "lead"),
    "obsmax": ("obsmaxmmsl", "obsmax"),
    "peak": ("peakm", "peak"),
    "plag": ("plagmin", "plag"),
    "bias": ("biasm", "bias"),
    "rmsd": ("rmsdm", "rmsd", "rmse"),              # include RMSE variant
    "rval": ("rval", "r"),
    "skil": ("skil", "skill"),
    "vexp": ("vexp", "vexppercent", "vexp%", "varianceexplained"),
    "npts": ("npts", "npoints", "n"),
}

def _normalize_key(name: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", str(name).lower())

def _canonical_metric_key(metric_key: str) -> Optional[str]:
    if metric_key in _COL_ALIASES:
        return metric_key
    alias_map = {
        "corr": "rval", "correlation": "rval", "r": "rval",
        "skill": "skil", "skillscore": "skil",
        "varianceexplained": "vexp", "varexplained": "vexp", "ve": "vexp",
        "points": "npts", "count": "npts",
    }
    return alias_map.get(metric_key, None)

def _resolve_columns(df: pd.DataFrame, metric_key: str) -> Tuple[Optional[str], Optional[str]]:
    norm_to_actual: Dict[str, str] = {}
    for c in df.columns:
        norm_to_actual[_normalize_key(c)] = c

    # lead
    lead_col = None
    for cand in _COL_ALIASES["lead"]:
        if cand in norm_to_actual:
            lead_col = norm_to_actual[cand]
            break

    # metric
    canon_metric = _canonical_metric_key(metric_key)
    if canon_metric is None:
        return (lead_col, None)

    metric_col = None
    for cand in _COL_ALIASES[canon_metric]:
        if cand in norm_to_actual:
            metric_col = norm_to_actual[cand]
            break

    return (lead_col, metric_col)


# =========================
# Station handling
# =========================

def _extract_station_id_from_key(s3_url: str) -> Optional[str]:
    """
    Best-effort extraction of station ID from S3 object key:
    works for .../<anything>.<station>.cwl.csv (e.g., estofs_atl.8410140.cwl.csv).
    """
    key_path = PurePosixPath(urlparse(s3_url).path)  # strips bucket
    name = key_path.name
    parts = name.split(".")
    lower = name.lower()
    if lower.endswith(".cwl.csv") and len(parts) >= 4:
        return parts[-3]  # token before 'cwl'
    if len(parts) >= 2:
        return parts[-2]
    return None


def get_available_stations(
    start_date,
    end_date,
    *,
    data_dir: str,  # must be s3://...
    path_template: str = "{data_dir}/{date}/**/*.{station}.cwl.csv",
    date_fmt: str = "%Y%m%d",
) -> list[str]:
    """
    S3-only: list unique station IDs across the interval.
    """
    if not str(data_dir).startswith("s3://"):
        raise ValueError("data_dir must be s3://...")

    start_ts = pd.to_datetime(start_date).normalize()
    end_ts = pd.to_datetime(end_date).normalize()
    if end_ts < start_ts:
        raise ValueError("end_date must be on or after start_date")

    date_index = pd.date_range(start_ts, end_ts, freq="D")
    stations: set[str] = set()

    for d in date_index:
        date_str = d.strftime(date_fmt)
        matches = _iter_s3_paths(
            path_template=path_template,
            data_dir=data_dir,
            date_str=date_str,
            station="*",
        )
        for url in matches:
            sid = _extract_station_id_from_key(url)
            if sid:
                stations.add(sid)

    return sorted(stations)


# =========================
# Main data APIs (S3-only)
# =========================

def _get_metric_by_date_lead(
    start_date,
    end_date,
    lead_min: float,
    lead_max: float,
    metric: str,
    stations: Iterable[str],
    *,
    data_dir: str,  # must be s3://...
    path_template: str = "{data_dir}/{date}/**/*.{station}.cwl.csv",
    date_fmt: str = "%Y%m%d",
    quiet_missing: bool = True,
) -> pd.DataFrame:
    """
    Load per-station CSV metric files from S3 for a date interval and return:
    columns = ['date', 'lead', 'station', '<metric as passed>'].
    """
    if not str(data_dir).startswith("s3://"):
        raise ValueError("data_dir must be s3://...")

    start_ts = pd.to_datetime(start_date).normalize()
    end_ts = pd.to_datetime(end_date).normalize()
    if end_ts < start_ts:
        raise ValueError("end_date must be on or after start_date")

    date_index = pd.date_range(start_ts, end_ts, freq="D")
    stations = list(stations)

    metric_key = _normalize_key(metric)
    lead_lo, lead_hi = float(lead_min), float(lead_max)
    if lead_hi < lead_lo:
        lead_lo, lead_hi = lead_hi, lead_lo

    records = []

    for d in date_index:
        date_str = d.strftime(date_fmt)
        for station in stations:
            for s3_url in _iter_s3_paths(
                path_template=path_template,
                data_dir=data_dir,
                date_str=date_str,
                station=str(station),
            ):
                df = _read_s3_csv(s3_url, quiet_missing=quiet_missing)
                if df is None:
                    continue

                lead_col, metric_col = _resolve_columns(df, metric_key)
                if lead_col is None or metric_col is None:
                    if not quiet_missing:
                        print(f"[warn] Skipping {s3_url} (cannot resolve columns for metric={metric})")
                    continue

                leads = pd.to_numeric(df[lead_col], errors="coerce")
                vals = pd.to_numeric(df[metric_col], errors="coerce")
                use = ((leads >= lead_lo) & (leads <= lead_hi)).fillna(False)
                if not use.any():
                    continue

                sub = pd.DataFrame(
                    {
                        "date": d,  # normalized Timestamp
                        "lead": leads[use].astype(float).values,
                        "station": station,
                        metric: vals[use].astype(float).values,  # column name exactly as passed
                    }
                )
                records.append(sub)

    if not records:
        return pd.DataFrame(columns=["date", "lead", "station", str(metric)])

    out = pd.concat(records, ignore_index=True)
    out.sort_values(["date", "station", "lead"], inplace=True, kind="stable")
    out.reset_index(drop=True, inplace=True)
    return out


def summarize_metric_by_date_lead(
    start_date,
    end_date,
    lead_min: float,
    lead_max: float,
    metric: str,
    stations: Iterable[str],
    *,
    data_dir: str,  # must be s3://...
    path_template: str = "{data_dir}/{date}/**/*.{station}.cwl.csv",
    date_fmt: str = "%Y%m%d",
    quiet_missing: bool = True,
    tol: float = 0.0,
) -> Dict[str, list[Dict[str, object]]]:
    """
    Compute min, max, and median of the metric across the selected date/lead/station range on S3.
    Returns dict with lists of matching rows (ties supported).
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

    if df.empty:
        return {"min": [], "max": [], "median": []}

    if metric not in df.columns:
        raise ValueError(f"Metric column '{metric}' not present in results")

    vals = pd.to_numeric(df[metric], errors="coerce")
    non_nan = df.loc[vals.notna()].copy()
    non_nan["_metric_num"] = vals.loc[vals.notna()].astype(float)

    if non_nan.empty:
        return {"min": [], "max": [], "median": []}

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

    ordered = non_nan.sort_values("_metric_num", kind="stable").reset_index(drop=True)
    n = len(ordered)
    if n % 2 == 1:
        mid_val = float(ordered.loc[n // 2, "_metric_num"])
        median_rows = match_value(mid_val)
    else:
        lo_val = float(ordered.loc[n // 2 - 1, "_metric_num"])
        hi_val = float(ordered.loc[n // 2, "_metric_num"])
        median_rows = pd.concat([match_value(lo_val), match_value(hi_val)]).drop_duplicates()

    return {
        "min": rows_to_dicts(min_rows),
        "max": rows_to_dicts(max_rows),
        "median": rows_to_dicts(median_rows),
    }

if __name__ == "__main__":
    S3_ROOT = "s3://noaa-gestofs-pds/_post_processing/_metrics"
    TEMPLATE = "{data_dir}/{date}/**/*.{station}.cwl.csv"   # recursive

    stations = get_available_stations(
        start_date="2024-05-14",
        end_date="2024-05-14",
        data_dir=S3_ROOT,
        path_template=TEMPLATE,
        date_fmt="%Y%m%d",
    )

    print(stations)

    df = _get_metric_by_date_lead(
        start_date="2024-05-14",
        end_date="2024-05-14",
        lead_min=0,
        lead_max=150,
        metric="RMSD",
        stations=stations,
        data_dir=S3_ROOT,
        path_template=TEMPLATE,
        date_fmt="%Y%m%d",
        quiet_missing=True,
    )
    print(df.head(), len(df))

    summary = summarize_metric_by_date_lead(
        start_date="2024-05-14",
        end_date="2024-05-14",
        lead_min=0,
        lead_max=150,
        metric="RMSD",
        stations=stations,
        data_dir=S3_ROOT,
        path_template=TEMPLATE,
        date_fmt="%Y%m%d",
        quiet_missing=True,
        tol=0.0,
    )
    print(summary)
