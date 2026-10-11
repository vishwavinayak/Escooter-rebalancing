"""Data schemas and column definitions for Chicago E-Scooter Trips dataset."""

from dataclasses import dataclass

# Raw CSV Column Names as published by City of Chicago
TRIP_ID: str = "Trip ID"
START_TIME: str = "Start Time"
END_TIME: str = "End Time"
TRIP_DISTANCE: str = "Trip Distance"
TRIP_DURATION: str = "Trip Duration"
VENDOR: str = "Vendor"
START_COMMUNITY_AREA_NUMBER: str = "Start Community Area Number"
END_COMMUNITY_AREA_NUMBER: str = "End Community Area Number"
START_COMMUNITY_AREA_NAME: str = "Start Community Area Name"
END_COMMUNITY_AREA_NAME: str = "End Community Area Name"
START_CENTROID_LATITUDE: str = "Start Centroid Latitude"
START_CENTROID_LONGITUDE: str = "Start Centroid Longitude"
START_CENTROID_LOCATION: str = "Start Centroid Location"
END_CENTROID_LATITUDE: str = "End Centroid Latitude"
END_CENTROID_LONGITUDE: str = "End Centroid Longitude"
END_CENTROID_LOCATION: str = "End Centroid Location"

RAW_COLUMNS: list[str] = [
    TRIP_ID,
    START_TIME,
    END_TIME,
    TRIP_DISTANCE,
    TRIP_DURATION,
    VENDOR,
    START_COMMUNITY_AREA_NUMBER,
    END_COMMUNITY_AREA_NUMBER,
    START_COMMUNITY_AREA_NAME,
    END_COMMUNITY_AREA_NAME,
    START_CENTROID_LATITUDE,
    START_CENTROID_LONGITUDE,
    START_CENTROID_LOCATION,
    END_CENTROID_LATITUDE,
    END_CENTROID_LONGITUDE,
    END_CENTROID_LOCATION,
]

# Standardized Clean Column Names (snake_case)
CLEAN_COLUMN_MAP: dict[str, str] = {
    TRIP_ID: "trip_id",
    START_TIME: "start_time",
    END_TIME: "end_time",
    TRIP_DISTANCE: "trip_distance_m",
    TRIP_DURATION: "trip_duration_s",
    VENDOR: "vendor",
    START_COMMUNITY_AREA_NUMBER: "start_community_area_number",
    END_COMMUNITY_AREA_NUMBER: "end_community_area_number",
    START_COMMUNITY_AREA_NAME: "start_community_area_name",
    END_COMMUNITY_AREA_NAME: "end_community_area_name",
    START_CENTROID_LATITUDE: "start_centroid_latitude",
    START_CENTROID_LONGITUDE: "start_centroid_longitude",
    START_CENTROID_LOCATION: "start_centroid_location",
    END_CENTROID_LATITUDE: "end_centroid_latitude",
    END_CENTROID_LONGITUDE: "end_centroid_longitude",
    END_CENTROID_LOCATION: "end_centroid_location",
}

# Domain constraints for Chicago Data Portal
VALID_COMMUNITY_AREA_MIN: int = 1
VALID_COMMUNITY_AREA_MAX: int = 77
CHICAGO_LAT_MIN: float = 41.60
CHICAGO_LAT_MAX: float = 42.10
CHICAGO_LON_MIN: float = -88.05
CHICAGO_LON_MAX: float = -87.50

KNOWN_VENDORS: set[str] = {"bird", "lime", "spin"}

# Timestamp format in 2020 dataset (rounded to hour)
DATETIME_FORMAT: str = "%m/%d/%Y %I:%M:%S %p"


@dataclass(frozen=True)
class ColumnMetadata:
    """Metadata for an individual dataset column."""

    raw_name: str
    clean_name: str
    raw_dtype: str
    target_dtype: str
    unit: str | None
    description: str
    is_nullable: bool


DATA_DICTIONARY: dict[str, ColumnMetadata] = {
    TRIP_ID: ColumnMetadata(
        raw_name=TRIP_ID,
        clean_name="trip_id",
        raw_dtype="string / object",
        target_dtype="string",
        unit=None,
        description="Unique trip identifier (UUIDv4).",
        is_nullable=False,
    ),
    START_TIME: ColumnMetadata(
        raw_name=START_TIME,
        clean_name="start_time",
        raw_dtype="string / object",
        target_dtype="datetime64[ns]",
        unit=None,
        description="Trip start timestamp, truncated to the hour for privacy.",
        is_nullable=False,
    ),
    END_TIME: ColumnMetadata(
        raw_name=END_TIME,
        clean_name="end_time",
        raw_dtype="string / object",
        target_dtype="datetime64[ns]",
        unit=None,
        description="Trip end timestamp, truncated to the hour for privacy.",
        is_nullable=False,
    ),
    TRIP_DISTANCE: ColumnMetadata(
        raw_name=TRIP_DISTANCE,
        clean_name="trip_distance_m",
        raw_dtype="string / object (with commas)",
        target_dtype="float64",
        unit="meters",
        description="Distance traveled during trip in meters (string formatted with commas in raw data).",
        is_nullable=False,
    ),
    TRIP_DURATION: ColumnMetadata(
        raw_name=TRIP_DURATION,
        clean_name="trip_duration_s",
        raw_dtype="string / object (with commas)",
        target_dtype="float64",
        unit="seconds",
        description="Trip duration in seconds (string formatted with commas in raw data).",
        is_nullable=False,
    ),
    VENDOR: ColumnMetadata(
        raw_name=VENDOR,
        clean_name="vendor",
        raw_dtype="string / object",
        target_dtype="category / string",
        unit=None,
        description="Scooter operator company (bird, lime, spin).",
        is_nullable=False,
    ),
    START_COMMUNITY_AREA_NUMBER: ColumnMetadata(
        raw_name=START_COMMUNITY_AREA_NUMBER,
        clean_name="start_community_area_number",
        raw_dtype="float64 / Int64",
        target_dtype="Int64 (nullable integer)",
        unit="area number (1-77)",
        description="Official City of Chicago Community Area Number where trip began.",
        is_nullable=True,
    ),
    END_COMMUNITY_AREA_NUMBER: ColumnMetadata(
        raw_name=END_COMMUNITY_AREA_NUMBER,
        clean_name="end_community_area_number",
        raw_dtype="float64 / Int64",
        target_dtype="Int64 (nullable integer)",
        unit="area number (1-77)",
        description="Official City of Chicago Community Area Number where trip ended.",
        is_nullable=True,
    ),
    START_COMMUNITY_AREA_NAME: ColumnMetadata(
        raw_name=START_COMMUNITY_AREA_NAME,
        clean_name="start_community_area_name",
        raw_dtype="string / object",
        target_dtype="string",
        unit=None,
        description="Official City of Chicago Community Area Name where trip began (e.g. LOGAN SQUARE).",
        is_nullable=True,
    ),
    END_COMMUNITY_AREA_NAME: ColumnMetadata(
        raw_name=END_COMMUNITY_AREA_NAME,
        clean_name="end_community_area_name",
        raw_dtype="string / object",
        target_dtype="string",
        unit=None,
        description="Official City of Chicago Community Area Name where trip ended (e.g. LINCOLN PARK).",
        is_nullable=True,
    ),
    START_CENTROID_LATITUDE: ColumnMetadata(
        raw_name=START_CENTROID_LATITUDE,
        clean_name="start_centroid_latitude",
        raw_dtype="float64",
        target_dtype="float64",
        unit="degrees latitude",
        description="Geographic centroid latitude of the start community area polygon.",
        is_nullable=True,
    ),
    START_CENTROID_LONGITUDE: ColumnMetadata(
        raw_name=START_CENTROID_LONGITUDE,
        clean_name="start_centroid_longitude",
        raw_dtype="float64",
        target_dtype="float64",
        unit="degrees longitude",
        description="Geographic centroid longitude of the start community area polygon.",
        is_nullable=True,
    ),
    START_CENTROID_LOCATION: ColumnMetadata(
        raw_name=START_CENTROID_LOCATION,
        clean_name="start_centroid_location",
        raw_dtype="string / object",
        target_dtype="string",
        unit="WKT Point",
        description="Well-Known Text representation of start centroid: POINT (lon lat).",
        is_nullable=True,
    ),
    END_CENTROID_LATITUDE: ColumnMetadata(
        raw_name=END_CENTROID_LATITUDE,
        clean_name="end_centroid_latitude",
        raw_dtype="float64",
        target_dtype="float64",
        unit="degrees latitude",
        description="Geographic centroid latitude of the end community area polygon.",
        is_nullable=True,
    ),
    END_CENTROID_LONGITUDE: ColumnMetadata(
        raw_name=END_CENTROID_LONGITUDE,
        clean_name="end_centroid_longitude",
        raw_dtype="float64",
        target_dtype="float64",
        unit="degrees longitude",
        description="Geographic centroid longitude of the end community area polygon.",
        is_nullable=True,
    ),
    END_CENTROID_LOCATION: ColumnMetadata(
        raw_name=END_CENTROID_LOCATION,
        clean_name="end_centroid_location",
        raw_dtype="string / object",
        target_dtype="string",
        unit="WKT Point",
        description="Well-Known Text representation of end centroid: POINT (lon lat).",
        is_nullable=True,
    ),
}
