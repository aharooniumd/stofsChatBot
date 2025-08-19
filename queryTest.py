# Python
# Save as: tests/test_query_metrics.py
import csv
from pathlib import Path
import pytest

from query import get_metric_by_date_lead


HEADER = [
    "LeadTime (HRS)",
    "obsMax (m MSL)",
    "Peak (m)",
    "PLag (min)",
    "Bias(m)",
    "RMSD (m)",
    "RVal",
    "Skil",
    "VExp (%)",
    "nPts",
]


def _write_csv(path: Path, leads: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        for lt in leads:
            # Simple, deterministic values for assertions if needed
            writer.writerow([
                lt,           # LeadTime (HRS)
                2.0,          # obsMax (m MSL)
                -0.1,         # Peak (m)
                0.0,          # PLag (min)
                0.01,         # Bias(m)
                0.10,         # RMSD (m)
                0.99,         # RVal
                0.98,         # Skil
                80.0,         # VExp (%)
                240,          # nPts
            ])


def _prepare_data(tmp_path: Path) -> Path:
    """
    Creates this layout (model prefixes vary to validate wildcard matching):
      metrics_data/
        20240101/
          estofs_atl.8410140.cwl.csv
        20240102/
          modelX.8410140.cwl.csv
    """
    base = tmp_path / "metrics_data"
    # Day 1: estofs_atl.* file
    _write_csv(
        base / "20240101" / "estofs_atl.8410140.cwl.csv",
        leads=[0.0, 6.0, 12.0],
    )
    # Day 2: different model prefix, same station
    _write_csv(
        base / "20240102" / "modelX.8410140.cwl.csv",
        leads=[0.0, 6.0, 12.0],
    )
    return base


def _get_lead_column(df):
    # Be tolerant to column naming in the implementation
    for c in df.columns:
        if str(c).lower().startswith("lead"):
            return c
    raise AssertionError("LeadTime column not found in returned DataFrame")


def test_wildcard_model_prefix_single_day(tmp_path: Path):
    data_dir = _prepare_data(tmp_path)
    df = get_metric_by_date_lead(
        start_date="2024-01-01",
        end_date="2024-01-01",
        lead_min=0,
        lead_max=6,
        metric="RMSD",
        stations=["8410140"],
        data_dir=str(data_dir),
        path_template="{data_dir}/{date}/*.{station}.cwl.csv",
        date_fmt="%Y%m%d",
        quiet_missing=False,
    )
    # Expect two rows: 0 and 6 hours for one day
    assert len(df) == 2
    lead_col = _get_lead_column(df)
    assert set(df[lead_col].astype(float).round(1).tolist()) == {0.0, 6.0}


def test_multi_day_date_interval(tmp_path: Path):
    data_dir = _prepare_data(tmp_path)
    df = get_metric_by_date_lead(
        start_date="2024-01-01",
        end_date="2024-01-02",
        lead_min=0,
        lead_max=12,
        metric="RMSD",
        stations=["8410140"],
        data_dir=str(data_dir),
        path_template="{data_dir}/{date}/*.{station}.cwl.csv",
        date_fmt="%Y%m%d",
        quiet_missing=False,
    )
    # Expect three leads per day across two days: 0, 6, 12 -> total 6 rows
    assert len(df) == 6
    lead_col = _get_lead_column(df)
    assert df[lead_col].astype(float).between(0, 12).all()


def test_leadtime_filter_boundaries(tmp_path: Path):
    data_dir = _prepare_data(tmp_path)
    df = get_metric_by_date_lead(
        start_date="2024-01-01",
        end_date="2024-01-02",
        lead_min=6,
        lead_max=6,
        metric="RMSD",
        stations=["8410140"],
        data_dir=str(data_dir),
        path_template="{data_dir}/{date}/*.{station}.cwl.csv",
        date_fmt="%Y%m%d",
        quiet_missing=False,
    )
    # Exactly one lead per day (6h), two days -> 2 rows
    assert len(df) == 2
    lead_col = _get_lead_column(df)
    assert (df[lead_col].astype(float) == 6.0).all()


@pytest.mark.parametrize("metric", ["RMSD", "Bias", "RVal", "Skil", "VExp"])
def test_each_metric_once(tmp_path: Path, metric: str):
    data_dir = _prepare_data(tmp_path)
    df = get_metric_by_date_lead(
        start_date="2024-01-01",
        end_date="2024-01-02",
        lead_min=0,
        lead_max=12,
        metric=metric,
        stations=["8410140"],
        data_dir=str(data_dir),
        path_template="{data_dir}/{date}/*.{station}.cwl.csv",
        date_fmt="%Y%m%d",
        quiet_missing=False,
    )
    # Should return rows for the two days and three leads each
    assert len(df) == 6
    # Sanity: ensure lead times are in range
    lead_col = _get_lead_column(df)
    assert df[lead_col].astype(float).between(0, 12).all()