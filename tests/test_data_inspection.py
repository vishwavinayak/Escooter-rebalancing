"""Unit tests for dataset schema, inspection, and API bounded streaming.

Uses synthetic test fixtures exclusively.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from escooter_opt.data.inspect import (
    find_dataset_path,
    inspect_dataset,
    stream_bounded_chicago_api,
)
from escooter_opt.data.schema import (
    CLEAN_COLUMN_MAP,
    DATA_DICTIONARY,
    RAW_COLUMNS,
)


@pytest.fixture
def synthetic_csv_path(tmp_path: Path) -> Path:
    """Creates a small synthetic raw CSV file for inspection tests."""
    df = pd.DataFrame(
        {
            "Trip ID": [
                "11111111-0000-0000-0000-000000000001",
                "22222222-0000-0000-0000-000000000002",
                "33333333-0000-0000-0000-000000000003",
                "{",
                '"error" : true',
                '"message" : "Internal error"',
                '"status" : 500',
                "}",
            ],
            "Start Time": [
                "08/12/2020 05:00:00 AM",
                "08/12/2020 05:00:00 AM",
                "08/12/2020 06:00:00 AM",
                None,
                None,
                None,
                None,
                None,
            ],
            "End Time": [
                "08/12/2020 05:00:00 AM",
                "08/12/2020 06:00:00 AM",
                "08/12/2020 06:00:00 AM",
                None,
                None,
                None,
                None,
                None,
            ],
            "Trip Distance": ["1,500", "2,800", "3,000", None, None, None, None, None],
            "Trip Duration": ["600", "720", "900", None, None, None, None, None],
            "Vendor": ["lime", "bird", "spin", None, None, None, None, None],
            "Start Community Area Number": [24, 24, 3, None, None, None, None, None],
            "End Community Area Number": [24, 6, 6, None, None, None, None, None],
            "Start Community Area Name": [
                "WEST TOWN",
                "WEST TOWN",
                "UPTOWN",
                None,
                None,
                None,
                None,
                None,
            ],
            "End Community Area Name": [
                "WEST TOWN",
                "LAKE VIEW",
                "LAKE VIEW",
                None,
                None,
                None,
                None,
                None,
            ],
            "Start Centroid Latitude": [
                41.90145875,
                41.90145875,
                41.96543532,
                None,
                None,
                None,
                None,
                None,
            ],
            "Start Centroid Longitude": [
                -87.67556756,
                -87.67556756,
                -87.65514456,
                None,
                None,
                None,
                None,
                None,
            ],
            "Start Centroid Location": [
                "POINT (-87.67556756 41.90145875)",
                "POINT (-87.67556756 41.90145875)",
                "POINT (-87.65514456 41.96543532)",
                None,
                None,
                None,
                None,
                None,
            ],
            "End Centroid Latitude": [
                41.90145875,
                41.9435138,
                41.9435138,
                None,
                None,
                None,
                None,
                None,
            ],
            "End Centroid Longitude": [
                -87.67556756,
                -87.65749768,
                -87.65749768,
                None,
                None,
                None,
                None,
                None,
            ],
            "End Centroid Location": [
                "POINT (-87.67556756 41.90145875)",
                "POINT (-87.65749768 41.9435138)",
                "POINT (-87.65749768 41.9435138)",
                None,
                None,
                None,
                None,
                None,
            ],
        }
    )
    p = tmp_path / "synthetic_escooter_sample.csv"
    df.to_csv(p, index=False)
    return p


def test_schema_column_definitions():
    """Ensure all 16 Chicago dataset raw columns are declared and mapped."""
    assert len(RAW_COLUMNS) == 16
    assert len(DATA_DICTIONARY) == 16
    for col in RAW_COLUMNS:
        assert col in DATA_DICTIONARY
        assert col in CLEAN_COLUMN_MAP


def test_find_dataset_path_missing(tmp_path: Path):
    """Raises FileNotFoundError with instructions when CSV is absent."""
    with pytest.raises(FileNotFoundError) as exc_info:
        find_dataset_path(workspace_root=tmp_path)
    assert "Please place the Chicago E-Scooter Trips CSV at" in str(exc_info.value)
    assert "Do not replace it with synthetic data" in str(exc_info.value)


def test_synthetic_dataset_inspection(synthetic_csv_path: Path):
    """Inspects synthetic dataset fixture without touching real 144MB file."""
    summary = inspect_dataset(csv_path=synthetic_csv_path, chunk_size=10, sample_size=2)

    assert summary.total_raw_lines == 8
    assert summary.malformed_trailer_lines == 5
    assert summary.valid_record_count == 3
    assert summary.duplicate_trip_ids == 0
    assert len(summary.columns) == 16
    assert summary.unique_start_community_areas == 2  # CA 24 and CA 3
    assert summary.centroid_lat_lon_is_1to1_with_ca is True


def test_stream_bounded_chicago_api_limit_guard():
    """Rejects request if max_records > 1000."""
    with pytest.raises(ValueError, match="at most 1,000 rows"):
        stream_bounded_chicago_api(max_records=1001)


def test_stream_bounded_chicago_api_mock_streaming():
    """Verifies that streaming bounded reader terminates after requested record count."""
    fake_header = (
        '"Trip ID","Start Time","End Time","Trip Distance","Trip Duration","Vendor"\n'
    )
    fake_rows = [
        f'"{i}","08/12/2020 05:00:00 AM","08/12/2020 05:00:00 AM","1000","500","lime"\n'
        for i in range(2000)
    ]
    all_lines = [fake_header.strip()] + [r.strip() for r in fake_rows]

    mock_resp = MagicMock()
    mock_resp.iter_lines.return_value = all_lines
    mock_resp.raise_for_status.return_value = None

    mock_sess = MagicMock()
    mock_sess.get.return_value = mock_resp

    df = stream_bounded_chicago_api(
        max_records=500,
        session=mock_sess,
    )

    assert len(df) == 500
    mock_resp.close.assert_called_once()
