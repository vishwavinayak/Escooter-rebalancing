# Chicago E-Scooter Trips - Phase 2 Exploratory Data Analysis & Spatial/Temporal Modeling

**Dataset**: Cleaned City of Chicago E-Scooter Trips (2020 Pilot Program)  
**Input Artifact**: `data/processed/escooter_trips_2020_cleaned.parquet` (Policy C cleaned)  
**Total Usable Trips**: 525,784  
**Date Range**: August 12, 2020 to November 7, 2020 (88 calendar days)  
**Analysis Date**: October 2026 (Phase 2 EDA)

---

## 1. Executive Summary & Dataset Scope

Phase 2 performs exploratory data analysis, spatial zone aggregation, Origin-Destination (OD) matrix construction, and temporal demand profiling using the cleaned Parquet dataset generated in Phase 1B.

| Metric | Value | Interpretation / Context |
|---|---|---|
| **Cleaned Trips Analyzed** | **525,784** | 100% of Policy C cleaned records |
| **Raw Reference Trips** | 551,800 | Syntactically valid pilot records |
| **Spatial Coverage of Cleaned Trips** | **100.0%** (525,784 / 525,784) | Every cleaned trip has valid start and end zones |
| **Spatial Coverage vs. Raw Trips** | **95.285%** (525,784 / 551,800) | Only 4.715% excluded by Policy C cleaning |
| **Usable Geographic Zones** | **77 of 77 (100%)** | All 77 Chicago Community Areas are active |
| **Temporal Coverage** | 88 days | 2020-08-12 to 2020-11-07 |
| **Mean Daily Trips** | **5,974.8** | Median: 5,717 trips/day (Min: 1,145; Max: 12,252) |
| **Intra-Zone Trips (Same CA)** | **362,556 (68.96%)** | Trips starting and ending in the same Community Area |
| **Inter-Zone Trips (Cross CA)** | **163,228 (31.04%)** | Trips traveling between different Community Areas |

---

## 2. Temporal Analysis & Demand Dynamics

### 2.1 Hourly Trip Volume (Hour of Day: 00:00 to 23:00)
All trip start timestamps in the Chicago open dataset are rounded to the top of the hour (`:00:00`). Analyzing hourly volume across the 24-hour cycle reveals a distinct evening-peak profile:

| Hour Window | Trips | Volume % | Mean Speed (mph) | Operational Classification |
|---|---|---|---|---|
| **00:00 – 04:00** | 85 | 0.02% | 5.2 | Overnight curfew / service shutdown |
| **05:00 – 07:00** | 15,494 | 2.95% | 7.6 | Early morning ramp-up |
| **08:00 – 11:00** | 71,979 | 13.69% | 7.9 | Morning commute / midday expansion |
| **12:00 – 15:00** | 151,150 | 28.75% | 7.6 | Afternoon leisure & local errand peak |
| **16:00 – 19:00** | **215,906** | **41.06%** | **7.4** | **Peak evening commute & recreation rush** |
| **20:00 – 21:00** | 70,881 | 13.48% | 7.2 | Late evening return trips |
| **22:00 – 23:00** | 289 | 0.05% | 6.8 | Overnight wind-down |

- **Peak Hour**: **18:00 (6:00 PM)** with **58,085 trips (11.05% of all trips)**, followed closely by 17:00 (57,224 trips, 10.88%).
- **Trough Hours**: 01:00–02:00 (7 to 11 trips total across 88 days).
- **Speed Consistency**: Average speed remains remarkably stable across the day (7.2 to 7.9 mph), well below the Chicago 15 mph municipal hardware governor.

### 2.2 Day of Week Dynamics
Weekly demand displays strong recreational and weekend orientation:

| Day of Week | Day Index | Trips | Volume % | Classification |
|---|---|---|---|---|
| **Monday** | 0 | 57,578 | 10.95% | Weekday trough |
| **Tuesday** | 1 | 57,751 | 10.98% | Weekday trough |
| **Wednesday** | 2 | 71,085 | 13.52% | Mid-week expansion |
| **Thursday** | 3 | 68,808 | 13.09% | Mid-week expansion |
| **Friday** | 4 | 86,827 | 16.51% | Weekend transition |
| **Saturday** | 5 | **99,971** | **19.01%** | **Peak weekend day** |
| **Sunday** | 6 | 83,764 | 15.93% | High weekend leisure |

- **Weekday vs. Weekend**:
  - Monday–Friday: **342,049 trips (65.06%)** across 5 days (mean: 68,410 trips/weekday).
  - Saturday–Sunday: **183,735 trips (34.94%)** across 2 days (mean: 91,868 trips/weekend day).
  - Average volume on a Saturday is **73.6% higher** than on a Monday.

### 2.3 Daily Volume Time Series & Seasonality
- **Pilot Seasonality**: High adoption in August and September, tapering in late October and November as temperatures dropped.
- **Single-Day Peak**: **Saturday, September 5, 2020 (Labor Day Weekend)** with **12,252 trips**.
- **Single-Day Low**: **Wednesday, October 28, 2020** with **1,145 trips** (cold rain/inclement weather).
- **Rolling Average**: The 7-day rolling average hovered between 6,500 and 8,000 trips/day through mid-September before declining to ~3,000 trips/day by November.

---

## 3. Spatial Zone Aggregation & Net Flow Imbalance

### 3.1 Mathematical Definitions
For each Community Area zone $i \in \{1, \dots, 77\}$:
1. **Departures ($D_i$)**: Number of trips whose origin is zone $i$:
   $$D_i = \sum_{j=1}^{77} T_{ij}$$
2. **Arrivals ($A_i$)**: Number of trips whose destination is zone $i$:
   $$A_i = \sum_{j=1}^{77} T_{ji}$$
3. **Turnover ($T_i^{\text{tot}}$)**: Total trip activity in zone $i$:
   $$T_i^{\text{tot}} = A_i + D_i$$
4. **Net Flow ($I_i$)**: Flow imbalance proxy:
   $$I_i = A_i - D_i$$
5. **Imbalance Ratio ($R_i$)**: Normalized flow deviation:
   $$R_i = \frac{A_i - D_i}{A_i + D_i} \in [-1.0, +1.0]$$

### 3.2 Conservation of Flow Reconciliation
Across all 77 zones, the system satisfies exact conservation of flow:
$$\sum_{i=1}^{77} D_i = 525,784, \quad \sum_{i=1}^{77} A_i = 525,784, \quad \sum_{i=1}^{77} I_i = \sum_{i=1}^{77} (A_i - D_i) = 0$$

### 3.3 Classification of Surplus and Deficit Zones
Zones are classified according to net flow accumulation:
- **Potential Surplus Zones ($I_i > 0$)**: **46 zones**. Historically accumulate more arrivals than departures.
- **Potential Deficit Zones ($I_i < 0$)**: **31 zones**. Historically lose more departures than arrivals.
- **Balanced Zones ($I_i = 0$)**: **0 zones**.

> [!IMPORTANT]
> **Operational Disclaimer on Flow Imbalances**:  
> Net flow $I_i = A_i - D_i$ is a proxy inferred strictly from historical trip origin and destination accumulation over the pilot duration. It is **not** an observation of real-time physical scooter inventory or battery state-of-charge. In actual operations, vendor rebalancing vans, battery swaps, and manual repositioning periodically reset physical fleet stocks.

#### Top 10 Potential Surplus Zones (Net Flow $I_i > 0$)
| CA # | Community Area Name | Departures ($D_i$) | Arrivals ($A_i$) | Net Flow ($I_i$) | Turnover | Imbalance Ratio ($R_i$) |
|---|---|---|---|---|---|---|
| **8** | NEAR NORTH SIDE | 23,221 | 23,784 | **+563** | 47,005 | +0.0120 |
| **22** | LOGAN SQUARE | 24,913 | 25,338 | **+425** | 50,251 | +0.0085 |
| **5** | NORTH CENTER | 7,959 | 8,369 | **+410** | 16,328 | +0.0251 |
| **2** | WEST RIDGE | 2,094 | 2,504 | **+410** | 4,598 | +0.0892 |
| **4** | LINCOLN SQUARE | 7,466 | 7,862 | **+396** | 15,328 | +0.0258 |
| **21** | AVONDALE | 5,163 | 5,496 | **+333** | 10,659 | +0.0312 |
| **17** | DUNNING | 1,789 | 2,082 | **+293** | 3,871 | +0.0757 |
| **23** | HUMBOLDT PARK | 6,845 | 7,133 | **+288** | 13,978 | +0.0206 |
| **25** | AUSTIN | 7,134 | 7,327 | **+193** | 14,461 | +0.0133 |
| **14** | ALBANY PARK | 2,189 | 2,377 | **+188** | 4,566 | +0.0412 |

#### Top 10 Potential Deficit Zones (Net Flow $I_i < 0$)
| CA # | Community Area Name | Departures ($D_i$) | Arrivals ($A_i$) | Net Flow ($I_i$) | Turnover | Imbalance Ratio ($R_i$) |
|---|---|---|---|---|---|---|
| **7** | LINCOLN PARK | 67,125 | 65,492 | **-1,633** | 132,617 | -0.0123 |
| **6** | LAKE VIEW | 80,812 | 80,167 | **-645** | 160,979 | -0.0040 |
| **41** | HYDE PARK | 15,797 | 15,164 | **-633** | 30,961 | -0.0204 |
| **3** | UPTOWN | 22,412 | 21,854 | **-558** | 44,266 | -0.0126 |
| **15** | PORTAGE PARK | 7,071 | 6,687 | **-384** | 13,758 | -0.0279 |
| **28** | NEAR WEST SIDE | 27,295 | 27,131 | **-164** | 54,426 | -0.0030 |
| **24** | WEST TOWN | 51,969 | 51,819 | **-150** | 103,788 | -0.0014 |
| **29** | NORTH LAWNDALE | 4,193 | 4,072 | **-121** | 8,265 | -0.0146 |
| **19** | BELMONT CRAGIN | 9,173 | 9,061 | **-112** | 18,234 | -0.0061 |
| **1** | ROGERS PARK | 7,417 | 7,326 | **-91** | 14,743 | -0.0062 |

#### Top 5 Busiest Activity Hubs (Turnover $A_i + D_i$)
The top 3 community areas account for **37.88%** of all trip activity in the city:
1. **Lake View (CA 6)**: 160,979 trips (30.6% of city volume)
2. **Lincoln Park (CA 7)**: 132,617 trips (25.2% of city volume)
3. **West Town (CA 24)**: 103,788 trips (19.7% of city volume)
4. **Near West Side (CA 28)**: 54,426 trips
5. **Logan Square (CA 22)**: 50,251 trips

---

## 4. Origin-Destination (OD) Matrix & Flow Patterns

- **Matrix Structure**: $77 \times 77$ cross-tabulation table saved at [`results/eda/od_matrix.csv`](../results/eda/od_matrix.csv).
- **Intra-Zone Dominance**:
  - **362,556 trips (68.96%)** start and end within the same Community Area ($i = j$).
  - For example, in Lake View (CA 6), 68,140 trips out of 80,812 departures (84.3%) remain within Lake View.
  - In Lincoln Park (CA 7), 54,411 trips out of 67,125 departures (81.1%) remain within Lincoln Park.
- **Top Inter-Zone Travel Corridors ($i \ne j$)**:
  1. Lincoln Park (CA 7) $\leftrightarrow$ Lake View (CA 6): 13,842 trips
  2. West Town (CA 24) $\leftrightarrow$ Logan Square (CA 22): 6,715 trips
  3. West Town (CA 24) $\leftrightarrow$ Near West Side (CA 28): 4,912 trips
  4. Near North Side (CA 8) $\leftrightarrow$ Lincoln Park (CA 7): 3,854 trips
  5. Uptown (CA 3) $\leftrightarrow$ Lake View (CA 6): 3,618 trips

---

## 5. Observed Centroids & Distance Approximations

### 5.1 Centroid Derivation Method
- Centroids are extracted directly from the observed coordinates (`start_centroid_latitude`, `start_centroid_longitude`).
- **Zero Fabrication**: As verified in Phase 1, each Community Area $i \in \{1, \dots, 77\}$ maps to exactly one coordinate pair in the dataset, representing the official municipal polygon centroid.
- Verified: Maximum difference between origin centroid and destination centroid for the same CA is **0.000000 degrees** across all 77 zones.
- Exported table: [`results/eda/zone_centroids.csv`](../results/eda/zone_centroids.csv).

### 5.2 Distance Metric Evaluation: Haversine vs. Manhattan Grid

For Operations Research optimization models (such as vehicle routing for rebalancing vans):

$$\text{Haversine Distance } d_{\text{hav}}(i, j) = 2 R \arcsin \sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos \phi_i \cos \phi_j \sin^2\left(\frac{\Delta \lambda}{2}\right)}$$

$$\text{Manhattan Grid Distance } d_{\text{man}}(i, j) = 111139 \cdot |\Delta \phi| + 111139 \cdot \cos\left(\frac{\phi_i + \phi_j}{2}\right) \cdot |\Delta \lambda|$$

| Distance Metric | Mathematical Property | Operational Defensibility for Chicago |
|---|---|---|
| **Straight-Line (Haversine)** | Euclidean / great-circle ($L_2$) | Unrealistic for vehicle transit; implies rebalancing vehicles drive through buildings and diagonally across city blocks. |
| **Manhattan (Orthogonal Grid)** | Taxicab / grid metric ($L_1$) | **Highly defensible.** Chicago’s street network is famously laid out on a strict rectilinear cardinal grid (North-South / East-West with 8 standard blocks per mile). |
| **Circuity-Adjusted Manhattan** | $1.05 \dots 1.15 \times d_{\text{man}}$ | Accounts for one-way streets, turns, and highway deviations during vehicle routing. |

- **Empirical Relationship**: Across all $77 \times 77$ zone pairs:
  $$\text{Mean } d_{\text{man}} = 14,821\,\text{m} \quad \text{vs.} \quad \text{Mean } d_{\text{hav}} = 11,842\,\text{m} \quad \left(\frac{d_{\text{man}}}{d_{\text{hav}}} \approx 1.25\right)$$
- **Intra-Zone Distance Nuance**: For intra-zone trips ($i = j$), inter-centroid distance is 0 meters, but observed scooter odometer trip distance averages **1,850 meters**. In Phase 3, intra-zone travel costs can be parameterized using this observed trip distance baseline.

---

## 6. Generated EDA Outputs & Visualizations

### 6.1 Summary Tables
All tables are saved in [`results/eda/`](../results/eda/):
- [`zone_flows_and_imbalance.csv`](../results/eda/zone_flows_and_imbalance.csv): 77 zones with $D_i, A_i, I_i, T_i^{\text{tot}}, R_i$, classification, and coordinates.
- [`od_matrix.csv`](../results/eda/od_matrix.csv): $77 \times 77$ Origin-Destination trip cross-tabulation.
- [`zone_centroids.csv`](../results/eda/zone_centroids.csv): Official polygon centroids for all 77 Community Areas.
- [`zone_distances_manhattan_meters.csv`](../results/eda/zone_distances_manhattan_meters.csv): Pairwise Manhattan grid distance matrix ($77 \times 77$).
- [`zone_distances_haversine_meters.csv`](../results/eda/zone_distances_haversine_meters.csv): Pairwise Haversine straight-line distance matrix ($77 \times 77$).
- [`temporal_hourly.csv`](../results/eda/temporal_hourly.csv): 24-hour trip volumes, speed, distance, and percentages.
- [`temporal_day_of_week.csv`](../results/eda/temporal_day_of_week.csv): Volume by day of week.
- [`temporal_daily.csv`](../results/eda/temporal_daily.csv): Daily time series with 7-day rolling averages.
- [`eda_summary_metrics.json`](../results/eda/eda_summary_metrics.json): Machine-readable KPI summary.

### 6.2 Visualizations
Generated in [`results/eda/figures/`](../results/eda/figures/):
1. **Hourly Volume & Speed**: `hourly_trip_volume.png` & `.html`
2. **Day of Week Profile**: `day_of_week_volume.png` & `.html`
3. **Daily Time Series**: `daily_trip_volume_timeseries.png` & `.html`
4. **Zone Net Flow Imbalances (Surplus vs. Deficit Proxies)**: `zone_net_flow_imbalances.png` & `.html`
5. **Top Activity Hubs (Departures vs. Arrivals Stacked)**: `top_activity_zones.png` & `.html`

---

## 7. Operational Limitations & Transition to Phase 3

1. **Temporal Discretization**: Start and end timestamps are binned to 1-hour windows. Phase 3 dynamic rebalancing models should formulate time periods in discrete hourly slices (e.g., $t \in \{0, \dots, 23\}$).
2. **Spatial Aggregation Resolution**: Coordinates are polygon centroids of Community Areas. Fleet rebalancing decisions in Phase 3 will operate at the macro Community Area zone level ($i \in \{1, \dots, 77\}$).
3. **Flow Imbalance as Proxy**: Imbalance $I_i = A_i - D_i$ indicates directional accumulation over time. True fleet inventory must be modeled in Phase 3 by defining initial vehicle distributions and tracking rebalancing actions.
