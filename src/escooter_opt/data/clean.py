"""Production cleaning pipeline and rules for Chicago E-Scooter Trips dataset."""

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from escooter_opt.data.inspect import find_dataset_path
from escooter_opt.data.schema import (
    DATETIME_FORMAT,
    END_CENTROID_LATITUDE,
    END_CENTROID_LOCATION,
    END_CENTROID_LONGITUDE,
    END_COMMUNITY_AREA_NAME,
    END_COMMUNITY_AREA_NUMBER,
    END_TIME,
    RAW_COLUMNS,
    START_CENTROID_LATITUDE,
    START_CENTROID_LOCATION,
    START_CENTROID_LONGITUDE,
    START_COMMUNITY_AREA_NAME,
    START_COMMUNITY_AREA_NUMBER,
    START_TIME,
    TRIP_DISTANCE,
    TRIP_DURATION,
    TRIP_ID,
    VALID_COMMUNITY_AREA_MAX,
    VALID_COMMUNITY_AREA_MIN,
    VENDOR,
)

DEFAULT_OUTPUT_DIR = Path("data/processed")
DEFAULT_PARQUET_FILENAME = "escooter_trips_2020_cleaned.parquet"
DEFAULT_REPORT_FILENAME = "cleaning_report.json"

# Speed conversion constant:
# 1 mile = 1609.344 meters, 1 hour = 3600 seconds
# speed_mph = (distance_m / 1609.344) / (duration_s / 3600) = (distance_m / duration_s) * (3600 / 1609.344)
MPS_TO_MPH: float = 3600.0 / 1609.344  # approx 2.2369362920544

# Expected retained count for Policy C (used solely for post-hoc verification, never forced)
POLICY_C_EXPECTED_RETAINED: int = 525_784


@dataclass
class CleaningConfig:
    """Configurable cleaning rule thresholds and parameters for Policy C.

    Policy C (Moderate Policy):
    1. Exclude records missing Community Area info (1..77) required for zone-level analysis.
    2. Exclude bounce trips ONLY when BOTH duration < 60s AND distance < 100m.
    3. Exclude trips with duration > 4 hours (14,400 seconds).
    4. Exclude trips with average speed > 30 mph.
    5. Handle corrupt records under an explicit validity screening policy.
    """

    policy_name: str = "Policy C — Moderate Policy"

    # Strict Policy C threshold parameters
    bounce_max_duration_s: float = 60.0  # strict less-than: duration < 60.0
    bounce_max_distance_m: float = 100.0  # strict less-than: distance < 100.0
    max_duration_seconds: float = (
        14400.0  # strict greater-than: duration > 14400.0 (4 hours)
    )
    max_speed_mph: float = 30.0  # strict greater-than: speed > 30.0 mph

    # Validity checks (corrupt record screening)
    drop_malformed_trailer: bool = True
    drop_unparseable_timestamps: bool = True
    drop_nonpositive_duration: bool = True
    drop_negative_distance: bool = True

    # Geographic boundary constraints
    require_valid_community_areas: bool = True
    community_area_min: int = VALID_COMMUNITY_AREA_MIN  # 1
    community_area_max: int = VALID_COMMUNITY_AREA_MAX  # 77


@dataclass
class CleaningReport:
    """Complete auditable report of cleaning execution."""

    policy_name: str
    parameters: dict[str, Any]
    input_file: str
    output_file: str | None
    execution_time_utc: str

    # Row accounting
    total_raw_rows: int
    malformed_trailer_rows: int
    syntactically_valid_rows: int

    # Validity screening exclusions (corrupt record screening)
    validity_exclusions: dict[str, int]
    total_validity_exclusions: int
    records_passing_validity: int

    # Sequential Policy C waterfall attrition
    sequential_policy_exclusions: dict[str, int]
    total_policy_exclusions: int
    final_retained_rows: int
    retention_percentage_vs_raw: float
    retention_percentage_vs_valid: float

    # Independent rule frequencies & overlaps across valid records
    independent_policy_exclusions: dict[str, int]
    pairwise_overlaps: dict[str, int]
    exclusive_policy_exclusions: dict[str, int]

    # Verification vs expected baseline
    expected_retained_rows: int
    actual_matches_expected: bool
    discrepancy_rows: int


def validate_raw_schema(df: pd.DataFrame) -> None:
    """Ensure all required columns exist in the raw dataframe."""
    if df.empty:
        raise ValueError("Input dataset is empty (0 rows).")

    missing_cols = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Input dataset is missing required schema columns: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )


def clean_dataframe(
    df: pd.DataFrame,
    config: CleaningConfig | None = None,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean the Chicago E-Scooter Trips dataset using the approved Policy C.

    Separates fundamental validity screening from domain-specific Policy C rules.
    Does not modify the input dataframe in place.
    Returns the cleaned dataframe and a complete auditable CleaningReport.
    """
    cfg = config or CleaningConfig()
    validate_raw_schema(df)

    working_df = df.copy()
    total_raw_rows = len(working_df)

    # ---------------------------------------------------------
    # STAGE 0: Strip Malformed Trailer Lines (HTTP 500 error body)
    # ---------------------------------------------------------
    trailer_mask = working_df[TRIP_ID].astype(str).str.contains(r"[\{\}\:\"]")
    malformed_trailer_count = int(trailer_mask.sum())
    working_df = working_df[~trailer_mask].copy()
    syntactically_valid_rows = len(working_df)

    # ---------------------------------------------------------
    # STAGE 1: Fundamental Validity Screening (Corrupt records)
    # ---------------------------------------------------------
    # 1. Parse timestamps
    working_df["start_dt"] = pd.to_datetime(
        working_df[START_TIME], format=DATETIME_FORMAT, errors="coerce"
    )
    working_df["end_dt"] = pd.to_datetime(
        working_df[END_TIME], format=DATETIME_FORMAT, errors="coerce"
    )
    bad_time_mask = working_df["start_dt"].isna() | working_df["end_dt"].isna()

    # 2. Parse distance & duration (strip commas)
    working_df["parsed_dist_m"] = pd.to_numeric(
        working_df[TRIP_DISTANCE].astype(str).str.replace(",", "").str.strip(),
        errors="coerce",
    )
    working_df["parsed_dur_s"] = pd.to_numeric(
        working_df[TRIP_DURATION].astype(str).str.replace(",", "").str.strip(),
        errors="coerce",
    )
    bad_metric_mask = (
        working_df["parsed_dist_m"].isna() | working_df["parsed_dur_s"].isna()
    )

    # 3. Impossible numeric values
    nonpos_dur_mask = working_df["parsed_dur_s"] <= 0
    neg_dist_mask = working_df["parsed_dist_m"] < 0

    validity_breakdown = {
        "unparseable_timestamps": int(bad_time_mask.sum()),
        "unparseable_metrics": int(bad_metric_mask.sum()),
        "nonpositive_duration_le_0": int(nonpos_dur_mask.sum()),
        "negative_distance_lt_0": int(neg_dist_mask.sum()),
    }

    # Combined validity mask
    invalid_mask = bad_time_mask | bad_metric_mask | nonpos_dur_mask | neg_dist_mask
    total_validity_exclusions = int(invalid_mask.sum())

    # Filter invalid records
    valid_records_df = working_df[~invalid_mask].copy()
    records_passing_validity = len(valid_records_df)

    # ---------------------------------------------------------
    # STAGE 2: Speed Derivation on Verified Valid Records Only
    # ---------------------------------------------------------
    # Speed is strictly derived ONLY when duration > 0 and distance >= 0
    valid_records_df["speed_mph"] = (
        valid_records_df["parsed_dist_m"] / valid_records_df["parsed_dur_s"]
    ) * MPS_TO_MPH

    # ---------------------------------------------------------
    # STAGE 3: Independent Policy C Rule Evaluation for Auditing
    # ---------------------------------------------------------
    # Rule 1: Missing or out-of-bounds Community Area
    ca_missing_mask = (
        valid_records_df[START_COMMUNITY_AREA_NUMBER].isna()
        | valid_records_df[END_COMMUNITY_AREA_NUMBER].isna()
    )
    if cfg.require_valid_community_areas:
        ca_invalid_bounds = ~(
            valid_records_df[START_COMMUNITY_AREA_NUMBER].between(
                cfg.community_area_min, cfg.community_area_max
            )
            & valid_records_df[END_COMMUNITY_AREA_NUMBER].between(
                cfg.community_area_min, cfg.community_area_max
            )
        )
        rule_ca_mask = ca_missing_mask | ca_invalid_bounds
    else:
        rule_ca_mask = ca_missing_mask

    # Rule 2: Bounce trips (strict less-than on both conditions)
    rule_bounce_mask = (
        valid_records_df["parsed_dur_s"] < cfg.bounce_max_duration_s
    ) & (valid_records_df["parsed_dist_m"] < cfg.bounce_max_distance_m)

    # Rule 3: Duration > 4 hours (strict greater-than)
    rule_duration_mask = valid_records_df["parsed_dur_s"] > cfg.max_duration_seconds

    # Rule 4: Speed > 30 mph (strict greater-than)
    rule_speed_mask = valid_records_df["speed_mph"] > cfg.max_speed_mph

    independent_exclusions = {
        "missing_or_invalid_community_area": int(rule_ca_mask.sum()),
        "bounce_trips_lt_60s_and_lt_100m": int(rule_bounce_mask.sum()),
        "excessive_duration_gt_4h": int(rule_duration_mask.sum()),
        "excessive_speed_gt_30mph": int(rule_speed_mask.sum()),
    }

    pairwise_overlaps = {
        "ca_and_bounce": int((rule_ca_mask & rule_bounce_mask).sum()),
        "ca_and_speed": int((rule_ca_mask & rule_speed_mask).sum()),
        "ca_and_duration": int((rule_ca_mask & rule_duration_mask).sum()),
        "bounce_and_speed": int((rule_bounce_mask & rule_speed_mask).sum()),
        "duration_and_speed": int((rule_duration_mask & rule_speed_mask).sum()),
    }

    exclusive_exclusions = {
        "only_missing_ca": int(
            (
                rule_ca_mask
                & ~rule_bounce_mask
                & ~rule_duration_mask
                & ~rule_speed_mask
            ).sum()
        ),
        "only_bounce": int(
            (
                rule_bounce_mask
                & ~rule_ca_mask
                & ~rule_duration_mask
                & ~rule_speed_mask
            ).sum()
        ),
        "only_excessive_duration": int(
            (
                rule_duration_mask
                & ~rule_ca_mask
                & ~rule_bounce_mask
                & ~rule_speed_mask
            ).sum()
        ),
        "only_excessive_speed": int(
            (
                rule_speed_mask
                & ~rule_ca_mask
                & ~rule_bounce_mask
                & ~rule_duration_mask
            ).sum()
        ),
    }

    # ---------------------------------------------------------
    # STAGE 4: Sequential Policy C Waterfall (Mutually Exclusive)
    # ---------------------------------------------------------
    seq_df = valid_records_df.copy()

    # Step 1: Drop Missing/Invalid CA
    drop_ca_count = int(rule_ca_mask.sum())
    seq_df = seq_df[~rule_ca_mask].copy()

    # Step 2: Drop Bounce trips
    step2_bounce_mask = (seq_df["parsed_dur_s"] < cfg.bounce_max_duration_s) & (
        seq_df["parsed_dist_m"] < cfg.bounce_max_distance_m
    )
    drop_bounce_count = int(step2_bounce_mask.sum())
    seq_df = seq_df[~step2_bounce_mask].copy()

    # Step 3: Drop Duration > 4 hours
    step3_dur_mask = seq_df["parsed_dur_s"] > cfg.max_duration_seconds
    drop_dur_count = int(step3_dur_mask.sum())
    seq_df = seq_df[~step3_dur_mask].copy()

    # Step 4: Drop Speed > 30 mph
    step4_speed_mask = seq_df["speed_mph"] > cfg.max_speed_mph
    drop_speed_count = int(step4_speed_mask.sum())
    seq_df = seq_df[~step4_speed_mask].copy()

    final_retained_rows = len(seq_df)
    total_policy_exclusions = (
        drop_ca_count + drop_bounce_count + drop_dur_count + drop_speed_count
    )

    # ---------------------------------------------------------
    # STAGE 5: Standardize Clean Schema
    # ---------------------------------------------------------
    clean_output_df = pd.DataFrame(
        {
            "trip_id": seq_df[TRIP_ID].astype(str),
            "start_time": seq_df["start_dt"],
            "end_time": seq_df["end_dt"],
            "trip_distance_m": seq_df["parsed_dist_m"].astype("float64"),
            "trip_duration_s": seq_df["parsed_dur_s"].astype("float64"),
            "speed_mph": seq_df["speed_mph"].astype("float64"),
            "vendor": seq_df[VENDOR].astype(str),
            "start_community_area_number": seq_df[START_COMMUNITY_AREA_NUMBER].astype(
                "Int64"
            ),
            "end_community_area_number": seq_df[END_COMMUNITY_AREA_NUMBER].astype(
                "Int64"
            ),
            "start_community_area_name": seq_df[START_COMMUNITY_AREA_NAME].astype(str),
            "end_community_area_name": seq_df[END_COMMUNITY_AREA_NAME].astype(str),
            "start_centroid_latitude": seq_df[START_CENTROID_LATITUDE].astype(
                "float64"
            ),
            "start_centroid_longitude": seq_df[START_CENTROID_LONGITUDE].astype(
                "float64"
            ),
            "start_centroid_location": seq_df[START_CENTROID_LOCATION].astype(str),
            "end_centroid_latitude": seq_df[END_CENTROID_LATITUDE].astype("float64"),
            "end_centroid_longitude": seq_df[END_CENTROID_LONGITUDE].astype("float64"),
            "end_centroid_location": seq_df[END_CENTROID_LOCATION].astype(str),
        }
    )

    retention_vs_raw = round((final_retained_rows / total_raw_rows) * 100, 3)
    retention_vs_valid = round(
        (final_retained_rows / syntactically_valid_rows) * 100, 3
    )

    discrepancy = final_retained_rows - POLICY_C_EXPECTED_RETAINED

    report = CleaningReport(
        policy_name=cfg.policy_name,
        parameters=asdict(cfg),
        input_file="",
        output_file=None,
        execution_time_utc=datetime.now(UTC).isoformat(),
        total_raw_rows=total_raw_rows,
        malformed_trailer_rows=malformed_trailer_count,
        syntactically_valid_rows=syntactically_valid_rows,
        validity_exclusions=validity_breakdown,
        total_validity_exclusions=total_validity_exclusions,
        records_passing_validity=records_passing_validity,
        sequential_policy_exclusions={
            "step_1_missing_or_invalid_ca": drop_ca_count,
            "step_2_bounce_trips": drop_bounce_count,
            "step_3_excessive_duration": drop_dur_count,
            "step_4_excessive_speed": drop_speed_count,
        },
        total_policy_exclusions=total_policy_exclusions,
        final_retained_rows=final_retained_rows,
        retention_percentage_vs_raw=retention_vs_raw,
        retention_percentage_vs_valid=retention_vs_valid,
        independent_policy_exclusions=independent_exclusions,
        pairwise_overlaps=pairwise_overlaps,
        exclusive_policy_exclusions=exclusive_exclusions,
        expected_retained_rows=POLICY_C_EXPECTED_RETAINED,
        actual_matches_expected=(discrepancy == 0),
        discrepancy_rows=discrepancy,
    )

    return clean_output_df, report


def verify_file_unmodified(file_path: Path, initial_hash: str) -> bool:
    """Verify that a file has not been altered during processing."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest() == initial_hash


def run_pipeline(
    raw_path: Path | None = None,
    output_dir: Path | None = None,
    config: CleaningConfig | None = None,
    dry_run: bool = False,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Execute the Policy C cleaning pipeline.

    Preserves raw data and writes cleaned data to Parquet.
    Saves a machine-readable JSON report and prints audit summary.
    """
    input_file = raw_path or find_dataset_path()
    out_dir = output_dir or DEFAULT_OUTPUT_DIR

    # Compute raw file SHA256 before reading
    hasher = hashlib.sha256()
    with open(input_file, "rb") as f:
        # Sample first/last MB for fast hash verification if large
        while chunk := f.read(2 * 1024 * 1024):
            hasher.update(chunk)
    initial_hash = hasher.hexdigest()

    raw_df = pd.read_csv(input_file, low_memory=False)
    cleaned_df, report = clean_dataframe(raw_df, config=config)
    report.input_file = str(input_file.resolve())

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        dest_parquet = out_dir / DEFAULT_PARQUET_FILENAME
        cleaned_df.to_parquet(dest_parquet, engine="pyarrow", index=False)
        report.output_file = str(dest_parquet.resolve())

        # Save machine-readable JSON cleaning report
        report_path = out_dir / DEFAULT_REPORT_FILENAME
        with open(report_path, "w", encoding="utf-8") as rf:
            json.dump(asdict(report), rf, indent=2)

    # Verify input CSV remained untouched
    if not verify_file_unmodified(input_file, initial_hash):
        raise RuntimeError(
            f"CRITICAL ERROR: Input file {input_file} was modified during pipeline execution!"
        )

    return cleaned_df, report


def main() -> None:
    """CLI entry point for Policy C cleaning pipeline."""
    parser = argparse.ArgumentParser(
        description="Execute approved Policy C cleaning pipeline for Chicago E-Scooter Trips dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to raw Chicago E-Scooter Trips CSV (defaults to finding data/ in workspace).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save cleaned parquet file and cleaning report (default: data/processed).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Audit dataset without saving output parquet or report.",
    )

    args = parser.parse_args()

    print("================================================================")
    print("Chicago E-Scooter Trips: Policy C Cleaning Pipeline Execution")
    print("================================================================")
    _cleaned_df, report = run_pipeline(
        raw_path=args.input,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )

    print(f"Policy: {report.policy_name}")
    print(f"Input File: {report.input_file}")
    print(f"Total Raw Lines: {report.total_raw_rows:,}")
    print(f"Malformed Trailer Lines: {report.malformed_trailer_rows}")
    print(f"Syntactically Valid Records: {report.syntactically_valid_rows:,}")
    print("\n--- Fundamental Validity Screening (Corrupt records) ---")
    for reason, cnt in report.validity_exclusions.items():
        print(f"  - {reason}: {cnt}")
    print(f"Total Validity Exclusions: {report.total_validity_exclusions}")
    print(f"Records Passing Validity: {report.records_passing_validity:,}")
    print("\n--- Sequential Policy C Waterfall Exclusions ---")
    for step, cnt in report.sequential_policy_exclusions.items():
        print(f"  - {step}: -{cnt:,}")
    print(f"Total Policy Exclusions: -{report.total_policy_exclusions:,}")
    print(
        f"\nFinal Retained Rows: {report.final_retained_rows:,} ({report.retention_percentage_vs_valid:.3f}%)"
    )
    print(f"Expected Retained Rows: {report.expected_retained_rows:,}")
    print(
        f"Actual Matches Expected: {report.actual_matches_expected} (Diff: {report.discrepancy_rows})"
    )

    if report.output_file:
        print(f"\nCleaned Parquet Output: {report.output_file}")
        print(f"Machine-readable Report: {args.output_dir / DEFAULT_REPORT_FILENAME}")
    print("================================================================")


if __name__ == "__main__":
    main()
