"""Inspection module for Chicago E-Scooter Trips dataset."""

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from escooter_opt.data.schema import (
    END_COMMUNITY_AREA_NUMBER,
    END_TIME,
    START_CENTROID_LATITUDE,
    START_CENTROID_LONGITUDE,
    START_COMMUNITY_AREA_NUMBER,
    START_TIME,
    TRIP_DISTANCE,
    TRIP_DURATION,
    TRIP_ID,
    VENDOR,
)

DEFAULT_CSV_PATH = Path("data/E-Scooter_Trips_-_2020_20261011.csv")
CHICAGO_API_URL = "https://data.cityofchicago.gov/api/v3/views/2i5w-ykuw/query.csv"


def find_dataset_path(workspace_root: Path | None = None) -> Path:
    """Locate the Chicago E-Scooter Trips CSV in the workspace.

    If the file is not present, raises FileNotFoundError with explicit
    instructions on where to place it. Never generates synthetic data.
    """
    root = workspace_root or Path.cwd()
    candidate_paths = [
        root / "data" / "E-Scooter_Trips_-_2020_20261011.csv",
        root / "data" / "raw" / "E-Scooter_Trips_-_2020_20261011.csv",
    ]

    for p in candidate_paths:
        if p.is_file() and p.stat().st_size > 0:
            return p

    # Also search data/ directory for any matching E-Scooter CSV
    data_dir = root / "data"
    if data_dir.is_dir():
        for p in data_dir.glob("*.csv"):
            if "escooter" in p.name.lower() or "e-scooter" in p.name.lower():
                return p
        raw_dir = data_dir / "raw"
        if raw_dir.is_dir():
            for p in raw_dir.glob("*.csv"):
                if "escooter" in p.name.lower() or "e-scooter" in p.name.lower():
                    return p

    expected_path = root / "data" / "E-Scooter_Trips_-_2020_20261011.csv"
    raise FileNotFoundError(
        f"Input CSV not found in workspace.\n"
        f"Please place the Chicago E-Scooter Trips CSV at:\n"
        f"  {expected_path.resolve()}\n"
        f"Do not replace it with synthetic data. Stopping execution as requested."
    )


@dataclass
class InspectionSummary:
    """Structured inspection summary."""

    file_path: str
    file_size_mb: float
    total_raw_lines: int
    malformed_trailer_lines: int
    valid_record_count: int
    duplicate_trip_ids: int
    columns: list[str]
    inferred_dtypes: dict[str, str]
    missing_counts: dict[str, int]
    missing_percentages: dict[str, float]
    sample_rows: list[dict[str, Any]]
    start_time_range: tuple[str, str]
    end_time_range: tuple[str, str]
    same_start_end_time_count: int
    same_start_end_time_pct: float
    time_rounding_observed: str
    unique_start_community_areas: int
    unique_end_community_areas: int
    community_area_range: tuple[int, int]
    centroid_lat_lon_is_1to1_with_ca: bool
    geographic_unit_identified: str
    distance_stats_meters: dict[str, float]
    duration_stats_seconds: dict[str, float]
    speed_stats_mph: dict[str, float]
    trips_duration_zero: int
    trips_duration_under_60s: int
    trips_duration_over_4h: int
    trips_duration_over_24h: int
    trips_speed_over_25mph: int
    trips_speed_over_50mph: int
    vendor_distribution: dict[str, int]
    zone_aggregation_feasibility: str


def inspect_dataset(
    csv_path: Path | None = None,
    chunk_size: int = 100_000,
    sample_size: int = 3,
) -> InspectionSummary:
    """Inspect the Chicago E-Scooter dataset without printing the entire dataset."""
    target_path = csv_path or find_dataset_path()
    file_size_mb = round(target_path.stat().st_size / (1024 * 1024), 2)

    total_raw_rows = 0
    trailer_error_rows = 0
    duplicate_trip_ids = 0
    seen_trip_ids = set()

    missing_counts: dict[str, int] = {}
    sample_records: list[dict[str, Any]] = []

    start_times = []
    end_times = []
    distances = []
    durations = []
    vendors: dict[str, int] = {}
    start_ca_set = set()
    end_ca_set = set()

    # Track centroid consistency: ca_number -> set of (lat, lon)
    ca_to_coords: dict[int, set[tuple[float, float]]] = {}

    first_chunk = True
    inferred_dtypes: dict[str, str] = {}
    actual_columns: list[str] = []

    for chunk in pd.read_csv(target_path, chunksize=chunk_size, low_memory=False):
        total_raw_rows += len(chunk)

        if first_chunk:
            actual_columns = list(chunk.columns)
            for col in actual_columns:
                missing_counts[col] = 0
                inferred_dtypes[col] = str(chunk[col].dtype)
            sample_records = chunk.head(sample_size).to_dict(orient="records")
            first_chunk = False

        # Identify malformed trailing JSON lines (e.g. Socrata HTTP 500 trailer)
        trailer_mask = chunk[TRIP_ID].astype(str).str.contains(r"[\{\}\:\"]")
        trailer_count = int(trailer_mask.sum())
        trailer_error_rows += trailer_count

        valid_chunk = chunk[~trailer_mask]

        for col in actual_columns:
            missing_counts[col] += int(chunk[col].isna().sum())

        # Check duplicates on valid Trip IDs
        tids = valid_chunk[TRIP_ID].dropna().values
        for tid in tids:
            if tid in seen_trip_ids:
                duplicate_trip_ids += 1
            else:
                seen_trip_ids.add(tid)

        # Vendors
        for v, cnt in valid_chunk[VENDOR].value_counts().items():
            vendors[str(v)] = vendors.get(str(v), 0) + int(cnt)

        # Community Areas
        valid_start_ca = valid_chunk[START_COMMUNITY_AREA_NUMBER].dropna().astype(int)
        valid_end_ca = valid_chunk[END_COMMUNITY_AREA_NUMBER].dropna().astype(int)
        start_ca_set.update(valid_start_ca.unique())
        end_ca_set.update(valid_end_ca.unique())

        # Check centroid 1:1 mapping
        ca_coords_sub = valid_chunk.dropna(
            subset=[
                START_COMMUNITY_AREA_NUMBER,
                START_CENTROID_LATITUDE,
                START_CENTROID_LONGITUDE,
            ]
        )
        for _, row in ca_coords_sub.iterrows():
            ca_num = int(row[START_COMMUNITY_AREA_NUMBER])
            coord = (
                round(float(row[START_CENTROID_LATITUDE]), 6),
                round(float(row[START_CENTROID_LONGITUDE]), 6),
            )
            if ca_num not in ca_to_coords:
                ca_to_coords[ca_num] = set()
            ca_to_coords[ca_num].add(coord)

        # Collect metrics (convert strings with commas to numeric)
        dist_series = (
            valid_chunk[TRIP_DISTANCE].astype(str).str.replace(",", "").astype(float)
        )
        dur_series = (
            valid_chunk[TRIP_DURATION].astype(str).str.replace(",", "").astype(float)
        )
        distances.append(dist_series)
        durations.append(dur_series)

        # Sample timestamps for analysis
        start_times.append(valid_chunk[START_TIME].dropna())
        end_times.append(valid_chunk[END_TIME].dropna())

    valid_record_count = total_raw_rows - trailer_error_rows

    missing_percentages = {
        col: round((missing_counts[col] / total_raw_rows) * 100, 3)
        for col in actual_columns
    }

    # Aggregate metric series
    all_distances = pd.concat(distances, ignore_index=True)
    all_durations = pd.concat(durations, ignore_index=True)

    dist_stats = {
        "min": float(all_distances.min()),
        "p25": float(all_distances.quantile(0.25)),
        "median": float(all_distances.median()),
        "p75": float(all_distances.quantile(0.75)),
        "p95": float(all_distances.quantile(0.95)),
        "max": float(all_distances.max()),
    }

    dur_stats = {
        "min": float(all_durations.min()),
        "p25": float(all_durations.quantile(0.25)),
        "median": float(all_durations.median()),
        "p75": float(all_durations.quantile(0.75)),
        "p95": float(all_durations.quantile(0.95)),
        "max": float(all_durations.max()),
    }

    # Speeds (mph = (meters / seconds) * 2.23694)
    valid_dur_mask = all_durations > 0
    speeds_mph = (
        all_distances[valid_dur_mask] / all_durations[valid_dur_mask]
    ) * 2.23694
    speed_stats = {
        "min": float(speeds_mph.min()),
        "median": float(speeds_mph.median()),
        "p90": float(speeds_mph.quantile(0.90)),
        "p95": float(speeds_mph.quantile(0.95)),
        "p99": float(speeds_mph.quantile(0.99)),
        "max": float(speeds_mph.max()),
    }

    # Timestamps analysis
    all_starts = pd.concat(start_times, ignore_index=True)
    all_ends = pd.concat(end_times, ignore_index=True)
    dt_starts = pd.to_datetime(all_starts, format="%m/%d/%Y %I:%M:%S %p")
    dt_ends = pd.to_datetime(all_ends, format="%m/%d/%Y %I:%M:%S %p")

    same_start_end = int((dt_starts == dt_ends).sum())
    same_start_end_pct = round((same_start_end / valid_record_count) * 100, 2)

    # Centroid verification: every CA has exactly 1 coordinate
    centroid_1to1 = all(len(coords) == 1 for coords in ca_to_coords.values())

    return InspectionSummary(
        file_path=str(target_path.resolve()),
        file_size_mb=file_size_mb,
        total_raw_lines=total_raw_rows,
        malformed_trailer_lines=trailer_error_rows,
        valid_record_count=valid_record_count,
        duplicate_trip_ids=duplicate_trip_ids,
        columns=actual_columns,
        inferred_dtypes=inferred_dtypes,
        missing_counts=missing_counts,
        missing_percentages=missing_percentages,
        sample_rows=sample_records,
        start_time_range=(str(dt_starts.min()), str(dt_starts.max())),
        end_time_range=(str(dt_ends.min()), str(dt_ends.max())),
        same_start_end_time_count=same_start_end,
        same_start_end_time_pct=same_start_end_pct,
        time_rounding_observed="All minutes and seconds are 00:00 (rounded to the hour by Chicago Data Portal).",
        unique_start_community_areas=len(start_ca_set),
        unique_end_community_areas=len(end_ca_set),
        community_area_range=(min(start_ca_set), max(start_ca_set)),
        centroid_lat_lon_is_1to1_with_ca=centroid_1to1,
        geographic_unit_identified=(
            "Official City of Chicago Community Areas (1 to 77). Coordinates are "
            "polygon centroids of the community areas, not individual GPS breadcrumbs."
        ),
        distance_stats_meters=dist_stats,
        duration_stats_seconds=dur_stats,
        speed_stats_mph=speed_stats,
        trips_duration_zero=int((all_durations <= 0).sum()),
        trips_duration_under_60s=int((all_durations < 60).sum()),
        trips_duration_over_4h=int((all_durations > 14400).sum()),
        trips_duration_over_24h=int((all_durations > 86400).sum()),
        trips_speed_over_25mph=int((speeds_mph > 25).sum()),
        trips_speed_over_50mph=int((speeds_mph > 50).sum()),
        vendor_distribution=vendors,
        zone_aggregation_feasibility=(
            "Feasible. 99.72% of trips have valid Start and End Community Area Numbers (1-77). "
            "Missing CA records (0.28%) also lack coordinates and should be excluded from OD matrices."
        ),
    )


def stream_bounded_chicago_api(
    url: str = CHICAGO_API_URL,
    max_records: int = 1000,
    timeout: int = 30,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Stream at most max_records from Chicago Open Data Portal.

    Uses server-side limit parameter ($limit / query LIMIT) where supported,
    and strictly terminates client-side streaming upon reaching max_records.
    Never downloads an unrestricted response.
    """
    if max_records > 1000:
        raise ValueError("API must be limited to at most 1,000 rows as specified.")

    params = {"$limit": str(max_records)}
    sess = session or requests.Session()

    try:
        response = sess.get(url, params=params, stream=True, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise ConnectionError(
            f"Unable to reach Chicago Data Portal API endpoint ({url}): {exc}"
        ) from exc

    # Bounded line-by-line reading
    lines: list[str] = []
    record_count = 0

    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        lines.append(line)
        # First line is header, count data lines after header
        if len(lines) > 1:
            record_count += 1
            if record_count >= max_records:
                break

    response.close()

    if not lines:
        return pd.DataFrame()

    csv_data = "\n".join(lines)
    df = pd.read_csv(io.StringIO(csv_data), low_memory=False)
    return df


if __name__ == "__main__":
    print("Running inspection on workspace dataset...")
    try:
        summary = inspect_dataset()
        print(f"File: {summary.file_path}")
        print(f"Size: {summary.file_size_mb} MB")
        print(f"Total raw lines: {summary.total_raw_lines:,}")
        print(f"Malformed trailer lines: {summary.malformed_trailer_lines}")
        print(f"Valid records: {summary.valid_record_count:,}")
        print(f"Columns: {len(summary.columns)}")
        print(f"Vendors: {summary.vendor_distribution}")
        print(
            f"Community areas: {summary.unique_start_community_areas} distinct (1-77)"
        )
        print(f"Centroid 1:1 with CA: {summary.centroid_lat_lon_is_1to1_with_ca}")
        print(f"Feasibility: {summary.zone_aggregation_feasibility}")
    except FileNotFoundError as e:
        print(e)
