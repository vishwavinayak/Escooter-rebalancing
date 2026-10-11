# Chicago E-Scooter Trips Dataset - Data Dictionary

This data dictionary documents the exact columns present in the City of Chicago E-Scooter Trips dataset (`data/E-Scooter_Trips_-_2020_20261011.csv`), covering the 2020 E-Scooter Pilot Program (August 12, 2020 to November 7, 2020).

---

## Dataset Overview

- **Source File**: `data/E-Scooter_Trips_-_2020_20261011.csv`
- **Source Agency**: City of Chicago Department of Transportation (CDOT) / Chicago Data Portal
- **Raw Row Count**: 551,805 lines (551,800 valid trip rows + 5 trailing HTTP error lines)
- **Column Count**: 16 columns
- **Temporal Coverage**: 2020-08-12 05:00:00 to 2020-11-07 14:00:00

---

## Column Specifications

| # | Raw Column Name | Standardized Name | Raw Data Type | Target Clean Type | Null Count (Pct) | Units / Format | Description & Domain Rules |
|---|-----------------|-------------------|---------------|-------------------|------------------|----------------|----------------------------|
| 1 | `Trip ID` | `trip_id` | String / Object | String (UUIDv4) | 0 (0.00%)* | UUID | Unique identifier assigned to each trip. 100% unique across valid trips. |
| 2 | `Start Time` | `start_time` | String / Object | `datetime64[ns]` | 0 (0.00%)* | `MM/DD/YYYY HH:MM:SS AM/PM` | Trip start timestamp. Truncated/rounded to the top of the hour (`:00:00`) by CDOT for privacy. |
| 3 | `End Time` | `end_time` | String / Object | `datetime64[ns]` | 0 (0.00%)* | `MM/DD/YYYY HH:MM:SS AM/PM` | Trip end timestamp. Truncated/rounded to the top of the hour (`:00:00`) by CDOT for privacy. |
| 4 | `Trip Distance` | `trip_distance_m` | String (with commas) | `float64` | 0 (0.00%)* | Meters | Cumulative travel distance recorded by scooter odometer/GPS. Range: 1 to 49,997 meters. |
| 5 | `Trip Duration` | `trip_duration_s` | String (with commas) | `float64` | 0 (0.00%)* | Seconds | Exact trip duration in seconds from unlock to lock. Median: 591s (~9.8 min). |
| 6 | `Vendor` | `vendor` | String / Object | Category / String | 0 (0.00%)* | Categorical | Operating mobility company: `lime` (43.1%), `bird` (29.5%), `spin` (27.4%). |
| 7 | `Start Community Area Number` | `start_community_area_number` | Float64 / Object | `Int64` (nullable) | 845 (0.15%) | Integer: 1 to 77 | Official City of Chicago Community Area Number where trip started. |
| 8 | `End Community Area Number` | `end_community_area_number` | Float64 / Object | `Int64` (nullable) | 1,173 (0.21%) | Integer: 1 to 77 | Official City of Chicago Community Area Number where trip ended. |
| 9 | `Start Community Area Name` | `start_community_area_name` | String / Object | String | 845 (0.15%) | Uppercase String | Official Community Area Name (e.g. `LOGAN SQUARE`, `WEST TOWN`, `AUSTIN`, `LOOP`). |
| 10 | `End Community Area Name` | `end_community_area_name` | String / Object | String | 1,173 (0.21%) | Uppercase String | Official Community Area Name where trip terminated. |
| 11 | `Start Centroid Latitude` | `start_centroid_latitude` | Float64 | `float64` | 845 (0.15%) | Decimal Degrees (N) | Geographic centroid latitude of the start community area polygon. Range: 41.65789 to 42.00889. |
| 12 | `Start Centroid Longitude` | `start_centroid_longitude` | Float64 | `float64` | 845 (0.15%) | Decimal Degrees (W) | Geographic centroid longitude of the start community area polygon. Range: -87.90936 to -87.53428. |
| 13 | `Start Centroid Location` | `start_centroid_location` | String / Object | String (WKT) | 845 (0.15%) | `POINT (lon lat)` | WKT point representation of the start community area polygon centroid. |
| 14 | `End Centroid Latitude` | `end_centroid_latitude` | Float64 | `float64` | 1,173 (0.21%) | Decimal Degrees (N) | Geographic centroid latitude of the end community area polygon. Range: 41.65789 to 42.00889. |
| 15 | `End Centroid Longitude` | `end_centroid_longitude` | Float64 | `float64` | 1,173 (0.21%) | Decimal Degrees (W) | Geographic centroid longitude of the end community area polygon. Range: -87.90936 to -87.53428. |
| 16 | `End Centroid Location` | `end_centroid_location` | String / Object | String (WKT) | 1,173 (0.21%) | `POINT (lon lat)` | WKT point representation of the end community area polygon centroid. |

*\*Note: Excluding the 5 trailing HTTP error lines at the end of the file, Trip ID, Start/End Time, Trip Distance, Trip Duration, and Vendor are 100% complete (0 missing values).*

---

## Critical Geographic & Temporal Findings

### 1. Spatial Resolution: Community Area Centroids, NOT GPS Points
- The coordinate fields (`Start Centroid Latitude`, `Start Centroid Longitude`) map **1-to-1** with `Start Community Area Number`.
- There is exactly **one** unique coordinate pair for each Community Area number 1 through 77.
- **Implication**: These coordinates represent polygon centroids of Chicago's 77 designated Community Areas, NOT user device GPS breadcrumbs or dockless scooter lock locations. Dockless station IDs and Census Tracts are not present in this dataset.

### 2. Temporal Resolution: Hourly Truncation
- `Start Time` and `End Time` have **00** minutes and **00** seconds for 100% of rows.
- Consequently, in **74.73%** of trips, `Start Time == End Time` (because trips started and ended within the same calendar hour).
- However, `Trip Duration` provides exact session duration in seconds (e.g., 591 seconds), allowing precise operational analysis despite hourly timestamp binning.

### 3. Derived Columns in Cleaned Data
- `speed_mph`: Calculated as `(trip_distance_m / trip_duration_s) * 2.23694`.
  - Normal range: 4 to 15 mph (legal city limit is 15 mph).
  - Outliers exceeding 25–50 mph indicate sensor jumps, false starts, or odometer errors.
