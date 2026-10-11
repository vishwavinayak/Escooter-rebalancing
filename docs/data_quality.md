# Chicago E-Scooter Trips Dataset - Data Quality Report & Cleaning Design

**Dataset**: City of Chicago E-Scooter Trips (2020 Pilot Program)  
**File Location**: `data/E-Scooter_Trips_-_2020_20261011.csv`  
**File Size**: 143.92 MB (150,909,298 bytes)  
**Total Lines**: 551,806 (1 header + 551,805 rows)  
**Analysis Date**: October 2026 (Phase 1A Inspection)

---

## 1. Executive Summary

A comprehensive automated inspection of the Chicago E-Scooter Trips dataset was performed without loading the entire dataset uncontrollably into memory. The dataset contains **551,800 valid trip records** spanning August 12, 2020 to November 7, 2020 across three vendors (`lime`, `bird`, and `spin`). 

Key findings:
1. **Critical Parser Bug (Server Error Trailer)**: The final 5 lines of the CSV contain a truncated HTTP 500 JSON error payload emitted by the Chicago Open Data Portal during download.
2. **Geographic Ground Truth**: Location fields represent Chicago's **77 official Community Areas**. Latitude and longitude values are **polygon centroids** (1-to-1 mapping with Community Area numbers), NOT individual GPS points or dockless stations.
3. **High Spatial Feasibility**: **99.72%** of trips have complete, valid origin and destination Community Area identifiers, making Community Area zone-level aggregation highly feasible.
4. **Hourly Timestamp Truncation**: All `Start Time` and `End Time` values are rounded to the top of the hour (`:00:00`), causing 74.73% of trips to show identical start and end timestamps. `Trip Duration` in seconds provides the true trip length.
5. **Data Cleaning Policies**: We evaluated three candidate cleaning policies:
   - **Policy A (Conservative / Auditing)**: Drops only corrupt trailer lines and zero-duration records (**100.00% retained**, 551,797 rows).
   - **Policy B (Standard Optimization / Rebalancing - Recommended)**: Enforces complete OD zones, drops bounces (<60s and <100m), caps duration at 4h, and removes implausible speeds >25 mph (**85.92% retained**, 474,106 rows).
   - **Policy C (Moderate Rebalancing)**: Retains valid OD zones, drops bounces and speeds >30 mph (**95.29% retained**, 525,784 rows).

---

## 2. Real Columns & Schema Discovery

The inspection identified **16 actual columns** in the raw CSV:

```
0: 'Trip ID'
1: 'Start Time'
2: 'End Time'
3: 'Trip Distance'
4: 'Trip Duration'
5: 'Vendor'
6: 'Start Community Area Number'
7: 'End Community Area Number'
8: 'Start Community Area Name'
9: 'End Community Area Name'
10: 'Start Centroid Latitude'
11: 'Start Centroid Longitude'
12: 'Start Centroid Location'
13: 'End Centroid Latitude'
14: 'End Centroid Longitude'
15: 'End Centroid Location'
```

### Inferred vs. Stored Types:
- `Trip Distance` and `Trip Duration` are formatted as comma-separated string numerals (e.g. `"1,153"`, `"1,027"`), requiring comma stripping before numeric casting.
- `Start/End Community Area Number` contain nulls, causing standard pandas loaders to infer them as float64 rather than integer. In the cleaned schema, they are cast to nullable integer `Int64`.
- `Vendor` contains exactly three distinct strings: `lime` (237,777; 43.1%), `bird` (163,059; 29.5%), and `spin` (150,964; 27.4%).

---

## 3. Data Quality Findings

### 3.1 Malformed HTTP 500 JSON Trailer
At line 551,801 through 551,805, the raw CSV terminates with an unparsed HTTP error payload:
```json
{
  "error" : true,
  "message" : "Internal error",
  "status" : 500
}
```
This occurs when the Socrata Open Data Portal backend experiences a timeout or pipeline abort during bulk streaming. Dropping these 5 lines restores full syntactic validity.

### 3.2 Uniqueness & Duplicates
- **Duplicate Trip IDs**: Exactly **0** duplicate Trip IDs exist among the 551,800 valid trips. Each trip is a unique UUIDv4 string.
- **Exact duplicate rows**: 0.

### 3.3 Missingness Analysis (Valid 551,800 Records)
| Column Group | Missing Rows | Missing Pct | Notes |
|---|---|---|---|
| `Trip ID`, `Start Time`, `End Time`, `Trip Distance`, `Trip Duration`, `Vendor` | 0 | 0.00% | 100% complete across all valid trips |
| `Start Community Area Number`, Name, Centroid Lat/Lon/Location | 845 | 0.15% | Missing start geographic zone |
| `End Community Area Number`, Name, Centroid Lat/Lon/Location | 1,173 | 0.21% | Missing end geographic zone |
| Both Start and End Community Area Missing | 453 | 0.08% | Unzoned trips |
| Either Start or End Community Area Missing | 1,565 | 0.28% | Trips lacking full OD pair |
| Both Start and End Community Area Present | 550,235 | **99.72%** | Ready for OD matrix aggregation |

**Key Finding**: In every row where Community Area Number is missing, Centroid Latitude and Longitude are **also NaN**. No spatial coordinates exist when the community area is null.

### 3.4 Verification of Geographic Unit
- `Start Community Area Number` and `End Community Area Number` range from **1 to 77**, corresponding to Chicago's 77 official Community Areas.
- **Centroid Coordinate Invariant**: Across all 550,235 zoned records, every Community Area Number maps to **exactly 1 unique coordinate pair** (standard deviation = 0.0).
  - Example: Community Area 68 (Englewood) is always `(41.78006847, -87.64226608)`.
  - Example: Community Area 24 (West Town) is always `(41.90145875, -87.67556756)`.
- **Conclusion**: The dataset does **NOT** contain GPS breadcrumbs, GPS start/end pings, census tracts, or docking stations. All spatial coordinates are polygon centroids of the designated Community Areas.

### 3.5 Temporal Granularity & Rounding
- Range: **2020-08-12 05:00:00** to **2020-11-07 14:00:00**.
- **Minute and Second Truncation**: All timestamps end in `:00:00`.
- **Impact**: In **412,369 trips (74.73%)**, `Start Time == End Time`. This is not an error; it is privacy masking.
- `Trip Duration` (in seconds) contains the actual unrounded trip duration (median: 591s = 9.85 min) and must be used as the authoritative metric for travel time.

### 3.6 Outliers and Sensor Anomalies
- **Zero / Negative Duration**: 2 trips have duration <= 0s.
- **Short Trips / "Bounces"**: 19,952 trips (3.62%) have duration < 60s AND distance < 100m. These represent accidental unlocks, faulty scooters, or immediate cancellations.
- **Distance < 100m**: 68,167 trips (12.35%) have distance < 100m.
- **Excessive Durations**: 11 trips exceeded 4 hours (14,400s), and 1 trip exceeded 24 hours (204,182s ~ 56.7h), indicating forgotten sessions or abandoned vehicles.
- **Speed Anomalies**: City of Chicago e-scooters are hardware-governed to **15 mph**. 
  - Median implied speed: 7.14 mph.
  - Speed > 25 mph: 7,677 trips (1.39%).
  - Speed > 50 mph: 1,926 trips (0.35%), with a max of 3,584 mph (caused by odometer reading recorded over 1-2 seconds).

---

## 4. Zone-Level Aggregation Feasibility & Fallbacks

1. **Feasibility**: High. Because 99.72% (550,235) of trips possess complete Community Area start and end numbers (1–77), a $77 \times 77$ Origin-Destination (OD) transition matrix can be constructed directly.
2. **Missing Location Fallback**:
   - *For OD Rebalancing Models*: Filter out the 1,565 trips (0.28%) where either Start or End CA is missing.
   - *For Macro Demand Forecasting*: Retain all trips for citywide trip totals and vendor market share.
   - *Note on Spatial Imputation*: Because coordinates in this file are centroids populated from the Community Area, coordinates cannot be used to impute missing Community Areas within this file.

---

## 5. Approved Cleaning Implementation: Policy C (Moderate Rebalancing)

### 5.1 Sequential Waterfall Attrition & Accounting

The approved **Policy C** was executed using an auditable sequential waterfall starting from the raw CSV:

| Stage / Step | Condition / Operation | Rows Dropped | Cumulative Remaining | Retention % (vs Valid Trips) |
|---|---|---|---|---|
| **Raw Input** | Raw lines read from CSV | - | 551,805 | - |
| **Trailer Cleanup** | Drop malformed JSON/HTTP 500 error lines | -5 | 551,800 | 100.000% |
| **Step 1: CA Filter** | Drop records missing Start or End Community Area (1–77) | -1,565 | 550,235 | 99.716% |
| **Step 2: Non-positive Duration** | Drop trips with duration $\le$ 0 seconds | -2 | 550,233 | 99.716% |
| **Step 3: Bounce Trips** | Drop trips where BOTH `duration < 60s` AND `distance < 100m` | -19,907 | 530,326 | 96.108% |
| **Step 4: Excessive Duration** | Drop trips with `duration > 4 hours` (14,400s) | -11 | 530,315 | 96.106% |
| **Step 5: Excessive Speed** | Drop trips with calculated `speed > 30 mph` | -4,531 | **525,784** | **95.285%** |

- **Total valid trips excluded**: 26,016 rows (4.715%)
- **Final clean records retained**: **525,784 rows (95.285%)**
- **Discrepancy vs Expected Target (525,784)**: **0 (Exact match)**

---

### 5.2 Independent Rule Frequencies & Overlap Analysis

When each rule condition is evaluated independently across all 551,800 valid trips:

| Rule Name | Independent Exclusions | Exclusive Exclusions (Only this rule) | Overlaps with Other Rules |
|---|---|---|---|
| **Missing Community Area** | 1,565 (0.284%) | 1,510 | 43 with Bounce, 12 with Speed > 30 mph |
| **Duration $\le$ 0s** | 2 (0.00036%) | 0 | 2 with Bounce (both 0s trips had distance < 100m) |
| **Bounce (<60s & <100m)** | 19,952 (3.616%) | 19,895 | 43 with Missing CA, 12 with Speed > 30 mph, 2 with Dur $\le$ 0 |
| **Duration > 4h (14,400s)** | 11 (0.00199%) | 11 | 0 overlaps with other rules |
| **Speed > 30 mph** | 4,555 (0.825%) | 4,531 | 12 with Missing CA, 12 with Bounce |

**Key Overlap Insights**:
- **Bounces & Missing CA**: 43 bounce trips were also missing origin or destination Community Area numbers.
- **Bounces & Speed Outliers**: 12 trips triggered both the bounce filter and the >30 mph speed filter (e.g., trips where odometer registered ~50–90 meters over just 2–3 seconds, resulting in mathematically high speeds before being parked).
- **Zero Durations**: Both records with duration $\le$ 0 seconds had distance < 100m, falling into the bounce condition as well.

---

## 6. Reproducible Pipeline & Artifact Verification

1. **Raw Preservation**: The raw CSV at `data/E-Scooter_Trips_-_2020_20261011.csv` is strictly read-only and unmodified.
2. **Processed Output**: Written to `data/processed/escooter_trips_2020_cleaned.parquet` via PyArrow columnar compression (29 MB vs. 144 MB raw CSV).
3. **Data Integrity**: 0 nulls across all 17 standardized columns in the cleaned parquet file.
4. **Automated Verification**: `tests/test_data_inspection.py` verifies Policy C filtering, schema mapping, and Parquet serialization across 8/8 passing tests.
