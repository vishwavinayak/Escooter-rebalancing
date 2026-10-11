"""Automated tests for Policy C cleaning pipeline using synthetic fixtures.

Never depends on the complete 144MB research dataset for unit tests.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from escooter_opt.data.clean import (
    clean_dataframe,
    run_pipeline,
)
from escooter_opt.data.schema import RAW_COLUMNS


def create_base_synthetic_row(
    trip_id: str = "00000000-0000-0000-0000-000000000001",
    start_time: str = "08/12/2020 05:00:00 AM",
    end_time: str = "08/12/2020 06:00:00 AM",
    distance: str | float = "1,500",
    duration: str | float = "600",
    vendor: str = "lime",
    start_ca: float | None = 24,
    end_ca: float | None = 24,
    start_ca_name: str = "WEST TOWN",
    end_ca_name: str = "WEST TOWN",
    start_lat: float = 41.90145875,
    start_lon: float = -87.67556756,
    end_lat: float = 41.90145875,
    end_lon: float = -87.67556756,
) -> dict[str, object]:
    """Helper to generate a valid single raw CSV row dictionary."""
    return {
        "Trip ID": trip_id,
        "Start Time": start_time,
        "End Time": end_time,
        "Trip Distance": str(distance),
        "Trip Duration": str(duration),
        "Vendor": vendor,
        "Start Community Area Number": start_ca,
        "End Community Area Number": end_ca,
        "Start Community Area Name": start_ca_name,
        "End Community Area Name": end_ca_name,
        "Start Centroid Latitude": start_lat,
        "Start Centroid Longitude": start_lon,
        "Start Centroid Location": f"POINT ({start_lon} {start_lat})",
        "End Centroid Latitude": end_lat,
        "End Centroid Longitude": end_lon,
        "End Centroid Location": f"POINT ({end_lon} {end_lat})",
    }


def make_df(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Helper to convert row dictionaries to a DataFrame with full schema."""
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 1. Bounce Trip Rule Tests (duration < 60s AND distance < 100m)
# --------------------------------------------------------------------------


def test_bounce_both_conditions_met_excluded():
    """A trip with duration < 60 seconds AND distance < 100 metres is excluded."""
    # 59 seconds and 99 meters: strictly both < 60 and < 100
    row = create_base_synthetic_row(duration="59", distance="99")
    df = make_df([row])
    cleaned_df, report = clean_dataframe(df)

    assert len(cleaned_df) == 0
    assert report.sequential_policy_exclusions["step_2_bounce_trips"] == 1
    assert report.independent_policy_exclusions["bounce_trips_lt_60s_and_lt_100m"] == 1


def test_bounce_single_condition_not_excluded():
    """A trip satisfying only one bounce condition is NOT excluded by the bounce rule."""
    # Case A: duration < 60s, but distance >= 100m (e.g. 50s, 200m)
    # Speed: (200 / 50) * 2.23694 = 8.94 mph (well under 30 mph)
    row_a = create_base_synthetic_row(trip_id="A", duration="50", distance="200")

    # Case B: distance < 100m, but duration >= 60s (e.g. 120s, 50m)
    # Speed: (50 / 120) * 2.23694 = 0.93 mph
    row_b = create_base_synthetic_row(trip_id="B", duration="120", distance="50")

    df = make_df([row_a, row_b])
    cleaned_df, report = clean_dataframe(df)

    assert len(cleaned_df) == 2
    assert report.sequential_policy_exclusions["step_2_bounce_trips"] == 0
    assert report.independent_policy_exclusions["bounce_trips_lt_60s_and_lt_100m"] == 0


def test_bounce_boundary_conditions_not_excluded():
    """Trips at exactly 60 seconds or exactly 100 metres are NOT excluded by the bounce rule."""
    # Boundary 1: exactly 60s, exactly 100m
    row_exact_both = create_base_synthetic_row(
        trip_id="E1", duration="60", distance="100"
    )

    # Boundary 2: exactly 60s, 50m (< 100m)
    row_exact_time = create_base_synthetic_row(
        trip_id="E2", duration="60", distance="50"
    )

    # Boundary 3: 30s (< 60s), exactly 100m
    # Speed: (100 / 30) * 2.23694 = 7.45 mph
    row_exact_dist = create_base_synthetic_row(
        trip_id="E3", duration="30", distance="100"
    )

    df = make_df([row_exact_both, row_exact_time, row_exact_dist])
    cleaned_df, report = clean_dataframe(df)

    assert len(cleaned_df) == 3
    assert report.sequential_policy_exclusions["step_2_bounce_trips"] == 0


# --------------------------------------------------------------------------
# 2. Duration Rule Tests (duration > 4 hours = 14,400s)
# --------------------------------------------------------------------------


def test_duration_exact_4_hours_retained():
    """A trip of exactly 4 hours (14,400 seconds) is NOT excluded by the duration rule."""
    # exactly 14,400 seconds, 10,000 meters -> speed ~ 1.55 mph
    row = create_base_synthetic_row(duration="14400", distance="10000")
    df = make_df([row])
    cleaned_df, report = clean_dataframe(df)

    assert len(cleaned_df) == 1
    assert report.sequential_policy_exclusions["step_3_excessive_duration"] == 0


def test_duration_longer_than_4_hours_excluded():
    """A trip longer than 4 hours (> 14,400 seconds) is excluded."""
    # 14,401 seconds (> 4 hours)
    row = create_base_synthetic_row(duration="14401", distance="10000")
    df = make_df([row])
    cleaned_df, report = clean_dataframe(df)

    assert len(cleaned_df) == 0
    assert report.sequential_policy_exclusions["step_3_excessive_duration"] == 1
    assert report.independent_policy_exclusions["excessive_duration_gt_4h"] == 1


# --------------------------------------------------------------------------
# 3. Speed Rule Tests (speed > 30 mph)
# --------------------------------------------------------------------------


def test_speed_exact_30_mph_retained():
    """A trip at exactly 30 mph is NOT excluded by the speed rule."""
    # speed_mph = (distance_m / duration_s) * (3600 / 1609.344)
    # If duration = 3600 seconds (1 hour), distance = 30 * 1609.344 = 48,280.32 meters
    exact_dist_30mph = 30.0 * 1609.344
    row = create_base_synthetic_row(duration="3600", distance=f"{exact_dist_30mph:.2f}")
    df = make_df([row])
    cleaned_df, report = clean_dataframe(df)

    assert len(cleaned_df) == 1
    assert cleaned_df.iloc[0]["speed_mph"] == pytest.approx(30.0, rel=1e-5)
    assert report.sequential_policy_exclusions["step_4_excessive_speed"] == 0


def test_speed_above_30_mph_excluded():
    """A trip above 30 mph is excluded."""
    # If duration = 3600s, distance = 50,000m -> speed = 31.068 mph
    row = create_base_synthetic_row(duration="3600", distance="50000")
    df = make_df([row])
    cleaned_df, report = clean_dataframe(df)

    assert len(cleaned_df) == 0
    assert report.sequential_policy_exclusions["step_4_excessive_speed"] == 1
    assert report.independent_policy_exclusions["excessive_speed_gt_30mph"] == 1


# --------------------------------------------------------------------------
# 4. Community Area Handling Tests
# --------------------------------------------------------------------------


def test_missing_and_invalid_ca_handling():
    """Missing and out-of-bounds Community Area values are excluded."""
    row_valid = create_base_synthetic_row(trip_id="valid", start_ca=1, end_ca=77)
    row_missing_start = create_base_synthetic_row(
        trip_id="no_start", start_ca=None, end_ca=24
    )
    row_missing_end = create_base_synthetic_row(
        trip_id="no_end", start_ca=24, end_ca=None
    )
    row_out_of_bounds_low = create_base_synthetic_row(
        trip_id="ca_zero", start_ca=0, end_ca=24
    )
    row_out_of_bounds_high = create_base_synthetic_row(
        trip_id="ca_78", start_ca=24, end_ca=78
    )

    df = make_df(
        [
            row_valid,
            row_missing_start,
            row_missing_end,
            row_out_of_bounds_low,
            row_out_of_bounds_high,
        ]
    )
    cleaned_df, report = clean_dataframe(df)

    assert len(cleaned_df) == 1
    assert cleaned_df.iloc[0]["trip_id"] == "valid"
    assert report.sequential_policy_exclusions["step_1_missing_or_invalid_ca"] == 4


# --------------------------------------------------------------------------
# 5. Fundamental Validity Screening Tests
# --------------------------------------------------------------------------


def test_validity_policy_corrupt_records():
    """Corrupt records (unparseable timestamps, zero duration, invalid numbers) are screened."""
    row_valid = create_base_synthetic_row(trip_id="valid")
    row_bad_start_time = create_base_synthetic_row(
        trip_id="bad_time", start_time="INVALID_DATE"
    )
    row_zero_dur = create_base_synthetic_row(trip_id="zero_dur", duration="0")
    row_neg_dur = create_base_synthetic_row(trip_id="neg_dur", duration="-10")
    row_neg_dist = create_base_synthetic_row(trip_id="neg_dist", distance="-5")
    row_non_numeric_dur = create_base_synthetic_row(
        trip_id="str_dur", duration="NOT_A_NUMBER"
    )

    # Plus malformed trailer rows
    trailer_row = {col: np.nan for col in RAW_COLUMNS}
    trailer_row["Trip ID"] = '{"error": true, "status": 500}'

    df = make_df(
        [
            row_valid,
            row_bad_start_time,
            row_zero_dur,
            row_neg_dur,
            row_neg_dist,
            row_non_numeric_dur,
            trailer_row,
        ]
    )
    cleaned_df, report = clean_dataframe(df)

    assert len(cleaned_df) == 1
    assert cleaned_df.iloc[0]["trip_id"] == "valid"
    assert report.malformed_trailer_rows == 1
    assert report.validity_exclusions["unparseable_timestamps"] == 1
    assert (
        report.validity_exclusions["nonpositive_duration_le_0"] == 2
    )  # zero_dur + neg_dur
    assert report.validity_exclusions["negative_distance_lt_0"] == 1
    assert report.validity_exclusions["unparseable_metrics"] == 1
    assert report.total_validity_exclusions == 5


# --------------------------------------------------------------------------
# 6. Overlapping Exclusions and Count Reconciliation
# --------------------------------------------------------------------------


def test_overlapping_exclusions_accounting():
    """Overlapping rule triggers are tracked independently and reconciled sequentially."""
    # Trip A: triggers BOTH Missing CA AND Bounce (<60s & <100m)
    row_ca_and_bounce = create_base_synthetic_row(
        trip_id="ca_and_bounce",
        start_ca=None,
        duration="30",
        distance="50",
    )
    # Trip B: triggers BOTH Bounce AND Speed > 30 mph (e.g. 50m in 2s -> 25 m/s = 55.9 mph)
    row_bounce_and_speed = create_base_synthetic_row(
        trip_id="bounce_and_speed",
        duration="2",
        distance="50",
    )
    # Trip C: valid trip
    row_valid = create_base_synthetic_row(
        trip_id="valid", duration="600", distance="1500"
    )

    df = make_df([row_ca_and_bounce, row_bounce_and_speed, row_valid])
    cleaned_df, report = clean_dataframe(df)

    assert len(cleaned_df) == 1
    assert cleaned_df.iloc[0]["trip_id"] == "valid"

    # Independent overlaps
    assert report.pairwise_overlaps["ca_and_bounce"] == 1
    assert report.pairwise_overlaps["bounce_and_speed"] == 1

    # Sequential waterfall accounting:
    # row_ca_and_bounce is dropped at Step 1 (CA Filter)
    assert report.sequential_policy_exclusions["step_1_missing_or_invalid_ca"] == 1
    # row_bounce_and_speed is dropped at Step 2 (Bounce Filter)
    assert report.sequential_policy_exclusions["step_2_bounce_trips"] == 1
    assert report.total_policy_exclusions == 2


def test_count_reconciliation():
    """Verify exact accounting: raw_lines = trailer + validity_drops + policy_drops + retained."""
    rows = [
        create_base_synthetic_row(trip_id="valid_1"),
        create_base_synthetic_row(trip_id="valid_2"),
        create_base_synthetic_row(trip_id="missing_ca", start_ca=None),
        create_base_synthetic_row(trip_id="bounce", duration="20", distance="20"),
        create_base_synthetic_row(
            trip_id="too_long", duration="15000", distance="2000"
        ),
        create_base_synthetic_row(
            trip_id="too_fast", duration="3600", distance="60000"
        ),
        create_base_synthetic_row(trip_id="zero_dur", duration="0"),
    ]
    # Add 1 trailer line
    trailer_row = {col: np.nan for col in RAW_COLUMNS}
    trailer_row["Trip ID"] = "}"
    rows.append(trailer_row)

    df = make_df(rows)
    cleaned_df, report = clean_dataframe(df)

    assert len(cleaned_df) == 2
    # Exact reconciliation formula
    accounted_sum = (
        report.malformed_trailer_rows
        + report.total_validity_exclusions
        + report.total_policy_exclusions
        + report.final_retained_rows
    )
    assert accounted_sum == report.total_raw_rows == len(df)


# --------------------------------------------------------------------------
# 7. Parquet Round-Trip and Schema Preservation
# --------------------------------------------------------------------------


def test_parquet_export_and_read_back(tmp_path: Path):
    """Parquet output is written, read back, and retains all expected column types."""
    rows = [
        create_base_synthetic_row(trip_id="T1", distance="2,000", duration="800"),
        create_base_synthetic_row(trip_id="T2", distance="3,500", duration="1,200"),
    ]
    csv_file = tmp_path / "synthetic_raw.csv"
    make_df(rows).to_csv(csv_file, index=False)

    out_dir = tmp_path / "processed"
    _cleaned_df, report = run_pipeline(
        raw_path=csv_file,
        output_dir=out_dir,
        dry_run=False,
    )

    parquet_file = out_dir / "escooter_trips_2020_cleaned.parquet"
    report_file = out_dir / "cleaning_report.json"

    assert parquet_file.is_file()
    assert report_file.is_file()
    assert report.output_file == str(parquet_file.resolve())

    # Read back parquet
    round_trip_df = pd.read_parquet(parquet_file)
    assert len(round_trip_df) == 2
    assert "speed_mph" in round_trip_df.columns
    assert "start_time" in round_trip_df.columns
    assert pd.api.types.is_datetime64_any_dtype(round_trip_df["start_time"])
    assert round_trip_df["start_community_area_number"].dtype == "Int64"
    assert round_trip_df["trip_distance_m"].dtype == "float64"
    assert round_trip_df["trip_duration_s"].dtype == "float64"


# --------------------------------------------------------------------------
# 8. Error Handling for Missing / Empty Files
# --------------------------------------------------------------------------


def test_empty_input_raises_error():
    """Empty dataframe raises ValueError."""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="empty"):
        clean_dataframe(empty_df)


def test_missing_required_columns_raises_error():
    """Dataframe missing mandatory columns raises ValueError."""
    incomplete_df = pd.DataFrame({"Trip ID": ["1"], "Start Time": ["now"]})
    with pytest.raises(ValueError, match="missing required schema columns"):
        clean_dataframe(incomplete_df)


def test_nonexistent_file_raises_error(tmp_path: Path):
    """Non-existent file path raises FileNotFoundError."""
    missing_file = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        run_pipeline(raw_path=missing_file)


# --------------------------------------------------------------------------
# 9. Original Data Unchanged Verification
# --------------------------------------------------------------------------


def test_raw_csv_unmodified_during_pipeline(tmp_path: Path):
    """Pipeline strictly guarantees the raw input CSV is not modified."""
    rows = [create_base_synthetic_row(trip_id="T1")]
    csv_file = tmp_path / "immutable_raw.csv"
    make_df(rows).to_csv(csv_file, index=False)

    # Compute initial SHA256
    with open(csv_file, "rb") as f:
        initial_hash = hashlib.sha256(f.read()).hexdigest()

    run_pipeline(raw_path=csv_file, output_dir=tmp_path / "out", dry_run=False)

    # Compute post-run SHA256
    with open(csv_file, "rb") as f:
        post_run_hash = hashlib.sha256(f.read()).hexdigest()

    assert initial_hash == post_run_hash
