# Chicago E-Scooter Operations Research & Fleet Rebalancing
## Master Project Review Study Guide & Defense Sheet

> **Project Title**: Chicago E-Scooter Operations Research & Rebalancing  
> **Domain**: Operations Research, Prescriptive Analytics, Mathematical Optimization (LP & MILP), Geospatial Logistics  
> **Dataset**: City of Chicago E-Scooter Pilot Program (2020) — Chicago Department of Transportation (CDOT)  
> **Codebase**: Python 3.11+, `scipy.optimize` (HiGHS solver), `pyarrow`, `streamlit`, `plotly`, `uv`  
> **Test Suite**: 205 passing unit/integration tests (`uv run pytest`) | 100% Ruff compliant  

---

## Table of Contents
1. [30-Second Elevator Pitch & Core Motivation](#1-30-second-elevator-pitch--core-motivation)
2. [End-to-End System Architecture (The 6 Phases)](#2-end-to-end-system-architecture-the-6-phases)
3. [Data Ingestion, Schema & Data Quality Engineering (Phase 1)](#3-data-ingestion-schema--data-quality-engineering-phase-1)
4. [Exploratory Data Analysis & Spatial-Temporal Dynamics (Phase 2)](#4-exploratory-data-analysis--spatial-temporal-dynamics-phase-2)
5. [Distance Metric Science: Manhattan vs. Haversine](#5-distance-metric-science-manhattan-vs-haversine)
6. [Optimization Model 1: Transportation Problem (Fleet Redistribution LP - Phase 3)](#6-optimization-model-1-transportation-problem-fleet-redistribution-lp---phase-3)
7. [Optimization Model 2: P-Median Facility Location (Depot Siting MILP - Phase 4)](#7-optimization-model-2-p-median-facility-location-depot-siting-milp---phase-4)
8. [Scenario & Sensitivity Analysis (Phase 5)](#8-scenario--sensitivity-analysis-phase-5)
9. [Interactive Decision Support Dashboard (Streamlit - Phase 6)](#9-interactive-decision-support-dashboard-streamlit---phase-6)
10. [Key Numbers & Empirical Facts to Memorize](#10-key-numbers--empirical-facts-to-memorize)
11. [Top 15 Examiner / Viva Defense Questions & Model Answers](#11-top-15-examiner--viva-defense-questions--model-answers)
12. [CLI Commands & Reproduction Quick Reference](#12-cli-commands--reproduction-quick-reference)

---

## 1. 30-Second Elevator Pitch & Core Motivation

### The Real-World Challenge
Dockless electric scooter sharing programs suffer from severe **spatial-temporal tidal asymmetry**:
- Users unlock scooters in residential areas and ride them into commercial hubs, transit stations, and lakefront recreation areas (e.g., Lake View, Lincoln Park, West Town).
- Scooters drain from origin neighborhoods and cluster heavily at popular destinations, creating **vehicle starvation** in residential communities and **sidewalk clutter/regulatory non-compliance** in busy hubs.
- Operators must deploy rebalancing vans and service depots to maintain vehicle availability and battery health while minimizing vehicle miles traveled (VMT), fuel consumption, and operational costs.

### Our Solution
We developed a complete, reproducible **Operations Research & Decision Support Pipeline** for the City of Chicago:
1. **Data Cleaning & Auditing**: Screened 551,800 raw trips via an auditable 5-step sequential waterfall (**Policy C**), removing corrupt HTTP trailers, GPS sensor bounces, and speed anomalies to produce 525,784 clean trips.
2. **Spatial-Temporal Modeling**: Constructed origin-destination (OD) transition matrices and computed net flow imbalance proxies ($I_i = A_i - D_i$) across all 77 Chicago Community Areas.
3. **Fleet Redistribution (Transportation Problem LP)**: Formulated and solved an exact Linear Program via the **HiGHS** solver to redistribute 6,885 surplus scooters to deficit zones across 76 optimal routes in under **10 milliseconds**, achieving total dual integrality (guaranteed integer solutions).
4. **Depot Siting (P-Median Facility Location MILP)**: Formulated and solved a Mixed-Integer Linear Program with 6,006 binary variables to site $p$ maintenance/charging hubs, discovering that a 3-to-5 depot network captures over **50% of theoretical distance savings**.
5. **Interactive Dashboard**: Delivered an executive Streamlit web application providing live interactive optimization, sensitivity sweeps, and geospatial visualizations.

---

## 2. End-to-End System Architecture (The 6 Phases)

```
[Raw CSV: 551,805 lines] 
       │
       ▼ (Phase 1: Data Cleaning - Policy C Waterfall)
[Cleaned Columnar Parquet: 525,784 trips | 0 nulls | 29.1 MB]
       │
       ▼ (Phase 2: Spatial-Temporal EDA)
┌─────────────────────────────────────────────────────────────┐
│ • 77×77 OD Matrix                                           │
│ • Net Flow Proxies: I_i = A_i - D_i (46 Surplus, 31 Deficit)│
│ • 77 Observed Community Area Centroids                      │
│ • Manhattan Grid & Haversine Distance Matrices              │
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
               ▼ (Phase 3)                    ▼ (Phase 4)
┌──────────────────────────────┐┌──────────────────────────────┐
│  Transportation Problem LP   ││  P-Median Depot Siting MILP  │
│  Min transfer cost           ││  Min demand-weighted dist    │
│  Surplus S_i -> Deficit D_j  ││  Select p depot hubs         │
│  HiGHS LP (TDI Integer)      ││  HiGHS Branch-and-Cut        │
│  Result: 31,115.6 km         ││  Result: p=5 -> 3,501,234 km │
└──────────────┬───────────────┘└──────────────┬───────────────┘
               │                               │
               └───────────────┬───────────────┘
                               ▼ (Phase 5)
┌─────────────────────────────────────────────────────────────┐
│  Scenario & Sensitivity Analysis Suite                      │
│  Parametric sweeps: p ∈ {1..10}, Manhattan vs Haversine,   │
│  Demand scale s ∈ {0.25..1.5}, Elbow curve identification   │
└──────────────────────────────┬──────────────────────────────┘
                               ▼ (Phase 6)
┌─────────────────────────────────────────────────────────────┐
│  Interactive Streamlit Operations Dashboard                 │
│  Page 1: EDA & Audit  |  Page 2: Transport Redistribution   │
│  Page 3: P-Median Siting  |  Page 4: Sensitivity Sweeps     │
└─────────────────────────────────────────────────────────────┘
```

### Module & Execution Map
| Phase | Focus | Core Module | CLI Alias | Primary Artifacts |
|---|---|---|---|---|
| **1B** | Data Cleaning | `escooter_opt.data.clean` | `escooter-clean` | `data/processed/escooter_trips_2020_cleaned.parquet` |
| **2** | Spatial/Temporal EDA | `escooter_opt.analysis.eda` | `escooter-eda` | `results/eda/zone_flows_and_imbalance.csv`, `od_matrix.csv` |
| **3** | Fleet Redistribution | `escooter_opt.optimization.transport` | `escooter-transport` | `results/transport/allocations_detail.csv`, `summary.json` |
| **4** | Depot Facility Siting | `escooter_opt.optimization.pmedian` | `escooter-pmedian` | `results/pmedian/selected_depots.csv`, `zone_assignments.csv` |
| **5** | Sensitivity Analysis | `escooter_opt.scenarios.runner` | `escooter-scenarios` | `results/scenarios/scenarios_results.csv`, elbow curves |
| **6** | Web Dashboard | `escooter_opt.dashboard.app` | `escooter-dashboard` | Multi-page Streamlit application |

---

## 3. Data Ingestion, Schema & Data Quality Engineering (Phase 1)

### 3.1 Raw Dataset Context
- **Source**: City of Chicago Department of Transportation (CDOT) / Chicago Data Portal.
- **Coverage**: August 12, 2020 to November 7, 2020 (88 calendar days during the 2020 Pilot Program).
- **Size**: 143.92 MB raw CSV, 551,806 total lines (1 header + 551,805 rows).
- **Vendors**: Exactly 3 operators:
  1. `lime`: 237,777 trips (43.1%)
  2. `bird`: 163,059 trips (29.5%)
  3. `spin`: 150,964 trips (27.4%)
- **Raw Columns**: 16 columns (Trip ID, Start/End Time, Trip Distance, Trip Duration, Vendor, Community Area Numbers/Names, Centroid Lat/Lon/Location).

### 3.2 Two Crucial "Ground Truth" Discoveries (Examiners Love This!)
1. **Spatial Ground Truth: Polygon Centroids, NOT GPS Breadcrumbs**:
   - The coordinates in `Start Centroid Latitude` / `Longitude` have a strict **1-to-1 invariant mapping** with `Start Community Area Number` (standard deviation = 0.000000).
   - Each of Chicago's 77 Community Areas has exactly **one** coordinate pair representing the official municipal polygon centroid.
   - *Implication*: The public dataset contains **no** individual GPS ping breadcrumbs, street addresses, or dockless lock coordinates due to city privacy masking. Macro-logistics must operate at the Community Area level.
2. **Temporal Ground Truth: Hourly Masking**:
   - 100% of `Start Time` and `End Time` timestamps are rounded to `:00:00` (top of the hour).
   - Consequently, **74.73% of trips show `Start Time == End Time`** because trips started and ended within the same clock hour.
   - *Resolution*: `Trip Duration` (in seconds) contains the true, unrounded trip duration recorded by the vehicle hardware (median: 591 seconds $\approx$ 9.85 min) and must be used as ground truth.

### 3.3 The Parser Bug: HTTP 500 JSON Error Trailer
At lines 551,801–551,805, the raw file abruptly terminates with:
```json
{
  "error" : true,
  "message" : "Internal error",
  "status" : 500
}
```
This is a truncated JSON payload emitted by the Socrata portal backend when a server timeout aborted the HTTP stream. The pipeline screens and strips these 5 lines before parsing.

### 3.4 Approved Cleaning Waterfall: Policy C (Moderate Rebalancing)
Policy C balances strict data hygiene with maximum data retention:

| Step | Rule / Condition | Rows Excluded | Cumulative Rows | Retention % | Rationale |
|---|---|---|---|---|---|
| **0** | Raw Lines (Input) | — | 551,805 | — | Raw file read stream |
| **Trailer**| Strip HTTP 500 JSON error lines | 5 | 551,800 | 100.000% | Syntactic file validity |
| **Step 1** | Drop Missing Community Area (1–77) | 1,565 | 550,235 | 99.716% | Enforce complete OD pairs |
| **Step 2** | Drop Non-positive Duration ($\le 0$s) | 2 | 550,233 | 99.716% | Physical impossibility |
| **Step 3** | Drop Bounces (`dur < 60s` AND `dist < 100m`)| 19,907 | 530,326 | 96.108% | Accidental unlocks / faulty vehicles |
| **Step 4** | Drop Excessive Duration (`dur > 4h`) | 11 | 530,315 | 96.106% | Abandoned / forgotten sessions |
| **Step 5** | Drop Speed Outliers (`speed > 30 mph`) | 4,531 | **525,784** | **95.285%** | Sensor glitch (hardware cap is 15 mph)|

- **Final Cleaned Records**: **525,784** (95.285% retention, exactly 0 discrepancies).
- **Parquet Output**: 29.1 MB PyArrow Snappy-compressed table (80% disk savings), 17 standardized typed columns, **0 nulls**.

---

## 4. Exploratory Data Analysis & Spatial-Temporal Dynamics (Phase 2)

### 4.1 Temporal Dynamics
- **Total Duration**: 88 calendar days (2020-08-12 to 2020-11-07).
- **Daily Volume**: Mean = 5,974.8 trips/day (Median = 5,717; Min = 1,145 on rainy Oct 28; Max = 12,252 on Labor Day weekend, Sep 5).
- **Diurnal Hourly Curve**:
  - **Overnight (00:00–04:00)**: 85 trips (<0.02%) — municipal curfew / operator shutoff.
  - **Morning Commute (08:00–11:00)**: 71,979 trips (13.69%).
  - **Afternoon Leisure (12:00–15:00)**: 151,150 trips (28.75%).
  - **Evening Peak (16:00–19:00)**: **215,906 trips (41.06%)** — peak evening commute and recreational rush.
  - **Single Peak Hour**: **18:00 (6:00 PM)** with 58,085 trips (11.05%).
- **Day of Week Profile**:
  - Saturday is the absolute peak: **99,971 trips (19.01%)**.
  - Weekday mean: 68,410 trips/day vs. Weekend mean: 91,868 trips/day (**+73.6% higher on Saturday than Monday**).

### 4.2 Spatial Patterns & Intra-Zone Dominance
- **Intra-Zone Trips ($i = j$)**: **362,556 trips (68.96%)** start and end in the exact same Community Area.
- **Inter-Zone Trips ($i \ne j$)**: **163,228 trips (31.04%)** cross Community Area boundaries.
- **Top Activity Hubs (Turnover = Arrivals + Departures)**:
  - Top 3 zones account for **37.88%** of all city activity:
    1. **Lake View (CA 6)**: 160,979 trips (30.6%)
    2. **Lincoln Park (CA 7)**: 132,617 trips (25.2%)
    3. **West Town (CA 24)**: 103,788 trips (19.7%)
- **Top Inter-Zone Travel Corridors**:
  1. Lincoln Park (CA 7) $\leftrightarrow$ Lake View (CA 6): 13,842 trips
  2. West Town (CA 24) $\leftrightarrow$ Logan Square (CA 22): 6,715 trips
  3. West Town (CA 24) $\leftrightarrow$ Near West Side (CA 28): 4,912 trips

### 4.3 Flow Imbalance Proxy Mathematics
For each Community Area $i \in \{1, \dots, 77\}$:
- **Departures**: $D_i = \sum_{j=1}^{77} T_{ij}$
- **Arrivals**: $A_i = \sum_{j=1}^{77} T_{ji}$
- **Turnover**: $T_i^{\text{tot}} = A_i + D_i$
- **Net Flow**: $I_i = A_i - D_i$
- **Imbalance Ratio**: $R_i = \frac{A_i - D_i}{A_i + D_i} \in [-1.0, +1.0]$

#### Conservation of Flow
$$\sum_{i=1}^{77} D_i = 525,784, \quad \sum_{i=1}^{77} A_i = 525,784 \implies \sum_{i=1}^{77} I_i = 0$$

#### Zone Categorization
- **Potential Surplus Zones ($I_i > 0$)**: **46 zones** (Total surplus = **+6,885 units**).
  - Top Surplus: Near North Side (+563), Logan Square (+425), North Center (+410), West Ridge (+410), Lincoln Square (+396).
- **Potential Deficit Zones ($I_i < 0$)**: **31 zones** (Total deficit = **-6,885 units**).
  - Top Deficit: Lincoln Park (-1,633), Lake View (-645), Hyde Park (-633), Uptown (-558), Portage Park (-384).

> **CRITICAL OPERATIONAL DISCLAIMER**:  
> Net flow $I_i = A_i - D_i$ is an **inferred proxy** derived from trip accumulation over time. It is **not** an observation of real-time physical scooter inventory or battery state-of-charge (SoC). In commercial reality, rebalancing vans and battery swappers continuously reset physical inventories.

---

## 5. Distance Metric Science: Manhattan vs. Haversine

In urban logistics, distance approximations directly dictate routing cost and validity:

| Metric | Mathematical Formula | Property | Real-World Urban Defensibility |
|---|---|---|---|
| **Straight-Line (Haversine)** | $d_{\text{hav}} = 2 R \arcsin \sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos \phi_i \cos \phi_j \sin^2\left(\frac{\Delta \lambda}{2}\right)}$ | $L_2$ Great-circle metric | **Unrealistic** for ground vehicles; assumes vans drive through skyscrapers, rail yards, and diagonal blocks. |
| **Manhattan Grid ($L_1$)** | $d_{\text{man}} = 111,139 \cdot \|\Delta \phi\| + 111,139 \cdot \cos\left(\frac{\phi_i+\phi_j}{2}\right) \cdot \|\Delta \lambda\|$ | $L_1$ Taxicab metric | **Highly Defensible.** Chicago is famously surveyed on a strict rectilinear cardinal grid (8 standard city blocks per mile). |

### Empirical Comparison
- **Mean Pairwise Manhattan Distance**: **14,821 meters**
- **Mean Pairwise Haversine Distance**: **11,842 meters**
- **Grid Ratio**: $\frac{d_{\text{man}}}{d_{\text{hav}}} \approx 1.2519$ (+25.2% overhead for orthogonal street navigation).

---

## 6. Optimization Model 1: Transportation Problem (Fleet Redistribution LP - Phase 3)

### 6.1 Problem Statement
Given 46 surplus zones accumulating excess scooters and 31 deficit zones experiencing scooter depletion, determine the minimum-cost transfer plan to move scooters from surplus to deficit zones.

### 6.2 Mathematical Formulation
- **Sets**:
  - $S$: Set of surplus zones ($|S| = 46$, where $I_i > 0$).
  - $D$: Set of deficit zones ($|D| = 31$, where $I_j < 0$).
- **Parameters**:
  - $s_i = \max(I_i, 0)$: Available supply at zone $i \in S$.
  - $d_j = \max(-I_j, 0)$: Required demand at zone $j \in D$.
  - $c_{ij} = d_{\text{man}}(i, j)$: Distance cost in meters between zone centroids.
- **Decision Variables**:
  - $x_{ij} \ge 0$: Integer units of scooters transferred from surplus zone $i$ to deficit zone $j$.

#### Objective Function
$$\min \sum_{i \in S} \sum_{j \in D} c_{ij} \cdot x_{ij} \quad \text{[Unit: scooter-meters / transfer-cost]}$$

#### Constraints
1. **Supply Limit**: $\sum_{j \in D} x_{ij} \le s_i \quad \forall i \in S$ (Cannot ship more than available surplus)
2. **Demand Fulfillment**: $\sum_{i \in S} x_{ij} \ge d_j \quad \forall j \in D$ (Must satisfy deficit requirement)
3. **Non-negativity**: $x_{ij} \ge 0 \quad \forall i \in S, j \in D$

### 6.3 Theoretical Property: Total Dual Integrality (TDI) & Unimodularity
- **Why solve as a continuous LP?**: The constraint matrix of a bipartite transportation network is **totally unimodular** (every square submatrix has determinant $0, +1,$ or $-1$).
- **Theorem**: Because supply $s_i$ and demand $d_j$ are all integers, every basic feasible solution of the continuous LP relaxation is guaranteed to be an **integer vertex**.
- **Impact**: We solve with continuous LP algorithms (HiGHS interior-point/simplex), round trivially, and obtain an exact integer solution in **< 10 milliseconds** without branch-and-bound overhead!

### 6.4 Handling Unequal Supply & Demand
- If $\sum s_i \ne \sum d_j$, an automated **dummy slack zone** is injected with $c_{\text{dummy}} = 0$.
- In the 2020 pilot, global flow conservation holds: $\sum s_i = \sum d_j = 6,885$ units, so the problem is naturally balanced.

### 6.5 Baseline Results (Manhattan Distance)
- **Status**: Optimal (HiGHS Status 7)
- **Total Scooters Transferred**: **6,885 units**
- **Supply Utilization**: 100.0% | **Demand Satisfaction**: 100.0%
- **Total Transfer Cost / Distance**: **31,115.6 km** (31,115,644 meter-units)
- **Active Dispatch Routes**: **76 non-zero routes** (out of $46 \times 31 = 1,426$ possible links)
- **Top 3 Transfer Corridors**:
  1. Avondale $\to$ Lincoln Park: 333 units (7.32 km) $\implies$ 2,436.7 km-units
  2. West Ridge $\to$ Uptown: 304 units (7.57 km) $\implies$ 2,302.4 km-units
  3. Logan Square $\to$ Lincoln Park: 425 units (4.96 km) $\implies$ 2,109.7 km-units

---

## 7. Optimization Model 2: P-Median Facility Location (Depot Siting MILP - Phase 4)

### 7.1 Problem Statement
Where should an e-scooter operator locate $p$ central maintenance, battery-charging, and rebalancing depots among Chicago's 77 Community Areas so that the total demand-weighted transit distance from all zones to their assigned depot is minimized?

### 7.2 Mathematical Formulation
- **Sets**:
  - $I$: Set of demand zones ($|I| = 77$).
  - $J$: Set of candidate depot locations ($|J| = 77$).
- **Parameters**:
  - $w_i = T_i^{\text{tot}} = A_i + D_i$: Demand weight of zone $i$ (historical trip turnover).
  - $d_{ij}$: Distance in meters from zone $i$ to depot $j$ (Manhattan grid).
  - $p$: Exact integer number of depots to site ($1 \le p \le 77$).
- **Decision Variables**:
  - $y_j \in \{0, 1\}$: Binary variable $= 1$ if a depot is established at candidate centroid $j$; $0$ otherwise.
  - $z_{ij} \in \{0, 1\}$: Binary variable $= 1$ if demand zone $i$ is assigned to depot $j$; $0$ otherwise.

#### Objective Function
$$\min \sum_{i \in I} \sum_{j \in J} w_i \cdot d_{ij} \cdot z_{ij} \quad \text{[Demand-weighted assignment distance]}$$

#### Constraints
1. **Single Assignment**: $\sum_{j \in J} z_{ij} = 1 \quad \forall i \in I$ (Each zone assigned to exactly one depot)
2. **Linking Constraint**: $z_{ij} \le y_j \quad \forall i \in I, j \in J$ (Zone can only be assigned to an OPEN depot)
3. **Depot Budget**: $\sum_{j \in J} y_j = p$ (Exactly $p$ depots must be sited)
4. **Binary Domains**: $y_j \in \{0, 1\}, \quad z_{ij} \in \{0, 1\}$

#### Problem Dimensions
- **Binary Variables**: $|J| + |I| \cdot |J| = 77 + (77 \times 77) = \mathbf{6,006}$ binary variables.
- **Constraints**: $|I| + |I| \cdot |J| + 1 = 77 + 5,929 + 1 = \mathbf{6,007}$ linear rows.
- **Solver**: `scipy.optimize.milp` with HiGHS Branch-and-Cut (runtime: **2 to 4 seconds**).

### 7.3 Baseline Results ($p = 5$, Manhattan Distance)
- **Status**: Optimal (HiGHS Status 7)
- **Weighted Objective**: **3,501,234.3 km-weight**
- **Unweighted Total Distance**: 507.4 km
- **Mean Assignment Distance**: **6,589 meters (6.59 km)**

#### Selected Depot Locations ($p=5$)
| CA # | Community Area Name | Geographic Sector | Zones Assigned | Weighted Cost (km-wt) |
|:---:|---|---|:---:|---:|
| **6** | LAKE VIEW | North Side Corridor | 8 | 995,340 |
| **20** | HERMOSA | Northwest Corridor | 19 | 649,545 |
| **24** | WEST TOWN | Central / West Corridor | 8 | 594,461 |
| **42** | WOODLAWN | South / Southeast Sector | 25 | 796,037 |
| **63** | GAGE PARK | Southwest Sector | 17 | 465,853 |

---

## 8. Scenario & Sensitivity Analysis (Phase 5)

### 8.1 The P-Median "Elbow Curve" ($p = 1 \dots 10$)

| Depot Count ($p$) | Weighted Objective (km-wt) | Mean Assign Dist (m) | Marginal % Reduction | Cumulative % Reduction | Selected Depots |
|:---:|---:|---:|:---:|:---:|---|
| **1** | 9,034,598 | 15,599 | — | — | West Town (24) |
| **2** | 5,947,713 | 9,407 | **-34.2%** | -34.2% | Washington Park (40), Lake View (6) |
| **3** | **4,474,754** | **8,633** | **-24.8%** | **-50.5%** | Washington Park (40), Lake View (6), West Town (24) |
| **4** | 3,967,280 | 7,780 | -11.3% | -56.1% | Washington Park, Hermosa, Lake View, West Town |
| **5** | 3,501,234 | 6,589 | -11.7% | -61.2% | Woodlawn, Hermosa, Gage Park, Lake View, West Town |
| **6** | 3,039,902 | 6,534 | -13.2% | -66.4% | Woodlawn, Hermosa, Gage Park, Lake View, West Town, Lincoln Park |
| **7** | 2,696,242 | 5,507 | -11.3% | -70.2% | Woodlawn, Hermosa, Gage Park, Lake View, Near North, Austin, Roseland |
| **8** | 2,390,132 | 5,232 | -11.4% | -73.5% | (Incremental regional fine-tuning) |
| **9** | 2,112,059 | 4,891 | -11.6% | -76.6% | (Incremental regional fine-tuning) |
| **10**| 1,892,260 | 4,684 | -10.4% | -79.1% | (Incremental regional fine-tuning) |

#### Critical Insights
1. **The Economic Elbow occurs at $p = 3$**:
   - Moving from $p=1 \to p=3$ cuts travel distance by **50.5%**.
   - Beyond $p=3$, marginal gains drop sharply to ~11% per additional facility.
   - **Recommendation**: Operators on a tight capital budget should site **3 depots** (Lake View, West Town, Washington Park) to achieve tripartite North-Central-South coverage.
2. **Robustness of West Town (CA 24)**:
   - For $p=1$, West Town is selected under both Manhattan and Haversine metrics as the single optimal central hub.
3. **Equitable South Side Coverage**:
   - Expanding from $p=3 \to p=5$ splits the large southern service territory between **Woodlawn (CA 42)** in the east and **Gage Park (CA 63)** in the west, resolving transit equity barriers for South Side communities.

### 8.2 Transportation Problem Sensitivity Sweeps

| Scenario | Distance Metric | Demand Scale ($s$) | Objective (km) | Units Moved | Supply Util | Active Routes |
|---|---|:---:|---:|---:|:---:|:---:|
| `baseline_manhattan` | Manhattan | 1.00 (Full) | **31,115.6** | 6,885 | 100.0% | **76** |
| `baseline_haversine` | Haversine | 1.00 (Full) | **24,793.4** | 6,885 | 100.0% | **76** |
| `offpeak_manhattan` | Manhattan | 0.50 (50%) | **15,494.2** | 3,441 | 99.88% | **73** |
| `offpeak_haversine` | Haversine | 0.50 (50%) | **12,348.0** | 3,441 | 99.88% | **73** |

#### Crucial Topology Invariance Finding (High-Impact Viva Point!)
- The active route dispatch plan is **100% invariant** between Manhattan and Haversine distance costs!
- Both baseline models pick the **exact same 76 OD corridors** and move the **exact same 6,885 units**.
- *Takeaway*: Although Manhattan distance is ~25% longer in physical magnitude, the spatial ranking of neighbors is preserved across Chicago's grid.

---

## 9. Interactive Decision Support Dashboard (Streamlit - Phase 6)

The web dashboard is structured into four intuitive tabs/pages (`src/escooter_opt/dashboard/`):

1. **Page 1: Data Overview & Exploratory Data Analysis**:
   - Visualizes overall trip KPIs (525,784 trips, 88 days, 77 active Community Areas).
   - Shows the Policy C audit waterfall attrition chart and independent rule overlap Venn statistics.
   - Presents interactive 24-hour volume/speed diurnal profiles and weekend vs. weekday distributions.
   - Includes an in-memory CSV file uploader that applies Policy C on custom user datasets in real time.
2. **Page 2: Transportation Problem (Fleet Redistribution)**:
   - Allows users to select distance metrics (Manhattan vs. Haversine) and demand scale factor ($s \in [0.1, 2.0]$).
   - Solves the LP live via HiGHS and renders a scatter map with directional transfer vectors.
   - Displays route distance histograms, supply/demand fulfillment gauges, and exports dispatch plans to CSV.
3. **Page 3: Depot Location (P-Median Facility Siting)**:
   - Configurable depot slider ($p \in [1, 15]$) and candidate zone multi-select filter.
   - Solves the 6,006-variable MILP on the fly in ~2 seconds.
   - Plots selected depot stars, draws assignment lines to serviced Community Areas, and shows depot load distributions.
4. **Page 4: Scenario & Sensitivity Analysis**:
   - Visualizes parametric elbow curves for $p \in [1, 10]$.
   - Displays side-by-side cost and distance metric comparisons (Manhattan vs. Haversine).
   - Generates summary trade-off tables for municipal policymakers and fleet operations managers.

---

## 10. Key Numbers & Empirical Facts to Memorize

| Number / Fact | Value / Description | Why It Matters |
|---|---|---|
| **Raw Lines** | 551,806 lines | 1 header + 551,800 valid trips + 5 trailing HTTP 500 error lines |
| **Cleaned Trips** | **525,784 trips** | Policy C retention: **95.285%** of valid records |
| **Temporal Span** | 88 days (Aug 12 – Nov 7, 2020) | Covers adoption peak through late autumn cooldown |
| **Intra-Zone Trips** | **68.96% (362,556 trips)** | Vast majority of trips start and end in the same CA |
| **Top 3 Activity Zones** | **37.88% of all trips** | Lake View (161k), Lincoln Park (133k), West Town (104k) |
| **Peak Demand Hour** | **18:00 (6:00 PM)** | 58,085 trips (11.05%); 16:00–19:00 has 41.06% of volume |
| **Surplus / Deficit Split** | **46 Surplus / 31 Deficit** | Total balanced rebalancing quantity: **6,885 scooters** |
| **Largest Deficit Zone** | **Lincoln Park (CA 7)** | $-1,633$ net flow (highest scooter drain in Chicago) |
| **Largest Surplus Zone** | **Near North Side (CA 8)**| $+563$ net flow (highest scooter accumulation) |
| **Transport Baseline Obj**| **31,115.6 km** (Manhattan) | Redistributes 6,885 scooters across 76 active links |
| **P-Median $p=5$ Obj** | **3,501,234 km-wt** | Mean assignment distance = **6.59 km** |
| **Optimal Elbow** | **$p = 3$ depots** | Cuts distance by **50.5%** with only 3 facilities |
| **P-Median Binary Vars** | **6,006 variables** | $77$ selection variables ($y_j$) + $5,929$ assignment variables ($z_{ij}$) |
| **Total Unit Tests** | **205 passed** | Zero failures, 100% automated test coverage in 5.16s |

---

## 11. Top 15 Examiner / Viva Defense Questions & Model Answers

### Q1: Why didn't you use exact GPS coordinates to rebalance individual scooters?
> **Answer**: The City of Chicago Open Data Portal masks exact device coordinates for user privacy. In this dataset, coordinate fields map 1-to-1 with the municipal polygon centroids of the 77 official Community Areas (standard deviation = 0.0). No GPS breadcrumbs or dockless station coordinates exist in the public dataset. Therefore, macro-level fleet redistribution between Community Areas is the most rigorous, defensible spatial unit available.

### Q2: Why is net flow $I_i = A_i - D_i$ described as an "inferred proxy" rather than true inventory?
> **Answer**: Trip logs record where rides start and end, but they do not record vendor rebalancing van moves, battery swaps, maintenance withdrawals, or initial vehicle distributions at 05:00 AM. While $A_i - D_i$ accurately captures net cumulative flow disequilibrium caused by rider demand, real-world physical stock is periodically reset by operations vans. Labeling this as an "inferred proxy" demonstrates scientific honesty and operational awareness.

### Q3: Why is the continuous LP relaxation of the Transportation Problem guaranteed to yield integer solutions without using Branch-and-Bound?
> **Answer**: The node-arc incidence matrix of a bipartite network flow problem is **totally unimodular** (every square submatrix has determinant $0, \pm 1$). By the Hoffman-Kruskal theorem, when all supply limits ($s_i$) and demand requirements ($d_j$) are integers, the extreme points of the feasible polytope are integer vertices. The continuous LP relaxation is totally dual integral (TDI), guaranteeing that the continuous optimum is strictly integer.

### Q4: Why use Manhattan grid distance instead of Haversine or Euclidean distance?
> **Answer**: Chicago's street network is laid out on a strict Jeffersonian rectilinear grid with 8 blocks to a mile aligned to cardinal North-South and East-West axes. Straight-line Euclidean/Haversine distance assumes vehicles travel diagonally through buildings, parks, and Lake Michigan, underestimating vehicle travel distance by ~25%. Manhattan distance correctly models grid navigation:
> $$d_{\text{man}} = 111,139 \cdot |\Delta \phi| + 111,139 \cdot \cos\left(\frac{\phi_i+\phi_j}{2}\right) \cdot |\Delta \lambda|$$

### Q5: In the P-Median facility location model, why did you weight demand by turnover ($A_i + D_i$) rather than net deficit?
> **Answer**: Depots are operational hubs that service vehicles for battery recharging, preventative maintenance, brake/tire repairs, and cleaning. The volume of vehicles needing service is proportional to total trip activity (turnover = arrivals + departures), not just directional imbalance. Siting depots near high-turnover areas minimizes the distance rebalancing vans travel to collect dead or worn scooters.

### Q6: How did you choose the recommended number of depots ($p$)? Explain the elbow effect.
> **Answer**: We performed a parametric sweep for $p \in \{1 \dots 10\}$. Moving from $p=1$ to $p=2$ reduces total weighted distance by 34.2%, and $p=2$ to $p=3$ reduces it by an additional 24.8% (cumulative 50.5% reduction). Beyond $p=3$, marginal returns diminish to ~11% per depot. Thus, **$p=3$** is the cost-effective "elbow" for budget-constrained operations, while **$p=5$** is the optimal comprehensive layout ensuring equitable coverage of the Southwest (Gage Park) and Southeast (Woodlawn) corridors.

### Q7: What caused the 5 trailing lines in the raw CSV, and how did your pipeline handle them?
> **Answer**: When downloading large bulk exports from the Socrata portal, server timeouts can abort the data stream, emitting a 5-line JSON HTTP 500 error block (`{ "error": true, "message": "Internal error", "status": 500 }`). Standard CSV parsers crash on these malformed rows. Our cleaning pipeline inspects file trailers, removes non-CSV error lines, and verifies data integrity before downstream parsing.

### Q8: Why do 74.73% of trips have identical start and end timestamps?
> **Answer**: To prevent user tracking and re-identification, CDOT truncates all start and end timestamps to the top of the hour (`:00:00`). Because the median trip duration is 9.85 minutes, nearly three-quarters of trips start and end within the same clock hour (e.g., start at 14:10, end at 14:22 $\to$ both recorded as 14:00:00). However, the `Trip Duration` column records exact hardware-metered duration in seconds, which we used for all speed and duration calculations.

### Q9: How does the Transportation Problem handle unequal aggregate supply and demand?
> **Answer**: If $\sum s_i \ne \sum d_j$, our optimization module automatically inserts a dummy surplus or deficit node with transfer costs $c_{\text{dummy}} = 0$. This balances the LP without distorting routing incentives. In our baseline dataset, global flow conservation holds naturally ($\sum A_i = \sum D_i = 525,784$), resulting in exactly 6,885 surplus and deficit units.

### Q10: Why did you use exact mathematical programming (HiGHS) instead of heuristics (e.g., Genetic Algorithms or Simulated Annealing)?
> **Answer**: Heuristics do not guarantee mathematical optimality and require stochastic tuning. For linear and mixed-integer problems of this scale (77 zones, 6,006 variables), modern MIP solvers like HiGHS solve to proven global optimality in 2 to 4 seconds using presolve, simplex, and branch-and-cut. Exact methods provide an indisputable benchmark that heuristics cannot match.

### Q11: What is the computational complexity of the P-Median model, and why does HiGHS solve it so fast?
> **Answer**: Siting $p$ facilities among $n$ candidates is an NP-hard combinatorial problem ($\binom{77}{5} \approx 19.4$ million combinations). However, the mathematical formulation has 6,006 binary variables and 6,007 constraints. Modern HiGHS presolve algorithms tighten variable bounds, exploit strong LP relaxation lower bounds, and apply clique cuts, reducing branch-and-bound search time to ~2–4 seconds on commodity hardware.

### Q12: Why did you adopt Policy C over Policy B for cleaning?
> **Answer**: Policy B enforced a strict 25 mph speed cap and excluded 14.1% of trips. Policy C uses a physically defensible 30 mph speed filter (accounting for steep downhill segments and minor odometer reporting intervals) while filtering genuine bounces (<60s & <100m) and durations >4h. Policy C retains **95.285%** of trips (525,784 rows) while preserving clean, uncorrupted OD distributions.

### Q13: If nearly 69% of trips are intra-zone, what does that mean for scooter rebalancing?
> **Answer**: High intra-zone trip volume demonstrates that micro-mobility primarily serves localized, short-distance neighborhood trips (average trip distance $\approx$ 1.85 km). Consequently, rebalancing vans do not need to transport large fleets across the entire city. Over 75% of optimal transfers connect adjacent zones within 5 km, proving that localized cluster rebalancing (or swappable battery vans) is far more efficient than long-distance cross-town trucking.

### Q14: How does this optimization framework address transportation equity?
> **Answer**: In single- or dual-depot configurations ($p=1$ or $p=2$), facilities concentrate exclusively in affluent North Side and Central corridors (Lake View and West Town), forcing South Side neighborhoods to endure 15–20 km service distances. In our 5-depot configuration, dedicated hubs are established in Woodlawn (CA 42) and Gage Park (CA 63), dramatically reducing service response times and ensuring reliable scooter availability in historically underserved South and Southwest communities.

### Q15: How can this system be extended to real-time operations?
> **Answer**: The current model serves as a macro-logistics planning baseline. For real-time dispatch, it can be extended by:
> 1. Partitioning the day into rolling 2-hour or 4-hour time windows ($t \in \{1 \dots T\}$) with dynamic state-space equations.
> 2. Incorporating battery state-of-charge (SoC < 20% triggers collection demand).
> 3. Integrating with vehicle routing problem (VRP) solvers for turn-by-turn commercial van routing using real-time GPS feeds.

---

## 12. CLI Commands & Reproduction Quick Reference

All commands run through the fast Python environment manager `uv`:

```bash
# 1. Activate environment and verify dependencies
source .venv/bin/activate
uv sync

# 2. Execute Data Cleaning Pipeline (Policy C Waterfall)
uv run escooter-clean
# Input: data/E-Scooter_Trips_-_2020_20261011.csv
# Output: data/processed/escooter_trips_2020_cleaned.parquet (525,784 records)

# 3. Execute Exploratory Data Analysis & Centroid Modeling
uv run escooter-eda
# Outputs: results/eda/zone_flows_and_imbalance.csv, od_matrix.csv, distance matrices

# 4. Solve Fleet Redistribution LP (Transportation Problem)
uv run escooter-transport
# Output: results/transport/allocations_detail.csv (31,115.6 km, 76 routes)

# 5. Solve Depot Siting MILP (P-Median Facility Location)
uv run escooter-pmedian --p 5
# Output: results/pmedian/selected_depots.csv, zone_assignments.csv

# 6. Execute Scenario & Sensitivity Analysis Sweeps
uv run escooter-scenarios --distance-method both
# Output: results/scenarios/scenarios_results.csv, elbow curve plots

# 7. Launch Interactive Streamlit Operations Dashboard
uv run escooter-dashboard
# Or directly:
uv run streamlit run src/escooter_opt/dashboard/app.py

# 8. Run Automated Quality Assurance Suite (205 tests)
uv run pytest
uv run ruff check .
uv run ruff format --check .




uv run streamlit run src/escooter_opt/dashboard/app.py

```
