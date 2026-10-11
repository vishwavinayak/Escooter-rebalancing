# Chicago E-Scooter Operations Research & Rebalancing

Optimization and rebalancing model for the City of Chicago dockless e-scooter system (2020 Pilot Program dataset).

---

## Environment & Dependency Setup

This project uses [`uv`](https://docs.astral.sh/uv/) for Python execution and dependency management.

```bash
# Activate existing virtual environment or create one
source .venv/bin/activate

# Install dependencies
uv sync
```

---

## Phase 1B: Data Cleaning Pipeline (Policy C)

The pipeline cleans the raw Chicago E-Scooter Trips dataset (`data/E-Scooter_Trips_-_2020_20261011.csv`) according to the approved **Policy C (Moderate Rebalancing Policy)**:

1. **Validity Screening**: Strips malformed HTTP 500 JSON trailer lines (`{ "error": true, ... }`), unparseable timestamps, and non-positive durations ($\le 0\text{s}$).
2. **Community Area (CA) Constraint**: Excludes records missing origin or destination Community Area numbers (or outside $1 \dots 77$).
3. **Bounce Filter**: Excludes trips where **BOTH** `duration < 60s` **AND** `distance < 100m` (strict less-than).
4. **Duration Cap**: Excludes trips with `duration > 4 hours` ($14,400\text{s}$).
5. **Speed Cap**: Excludes trips with average `speed > 30 mph` (derived using physical unit conversion: $\text{m/s} \times \frac{3600}{1609.344}$).

Raw data remains untouched. Cleaned data is written to compact columnar Parquet format, accompanied by an auditable JSON report.

### Executing the Cleaning Pipeline

Run the pipeline using either the project CLI command:

```bash
uv run escooter-clean
```

Or via module invocation:

```bash
uv run python -m escooter_opt.data.clean
```

#### Optional CLI Arguments:
- `--input <PATH>`: Specify custom path to raw CSV (default: auto-detected in `data/`).
- `--output-dir <PATH>`: Specify destination directory for output (default: `data/processed`).
- `--dry-run`: Audit dataset without writing output Parquet or JSON reports to disk.

---

## Pipeline Execution Commands

```bash
# Phase 1B: Data Cleaning Pipeline
uv run escooter-clean

# Phase 2: Exploratory Data Analysis & Spatial Centroids
uv run escooter-eda

# Phase 3: Transportation Problem (Fleet Redistribution LP)
uv run escooter-transport

# Phase 4: P-Median Facility Location (Depot Siting IP)
uv run escooter-pmedian --p 5

# Phase 5: Scenario & Sensitivity Analysis Suite
uv run escooter-scenarios --distance-method both
```

---

## Project Documentation & Reports

- **Data Dictionary**: [`docs/data_dictionary.md`](docs/data_dictionary.md)
- **Data Quality Report**: [`docs/data_quality.md`](docs/data_quality.md)
- **Exploratory Data Analysis Findings**: [`docs/eda_findings.md`](docs/eda_findings.md)
- **Transportation Problem Model**: [`docs/transport_model.md`](docs/transport_model.md)
- **P-Median Facility Location Model**: [`docs/pmedian_model.md`](docs/pmedian_model.md)
- **Phase 5 Scenario & Sensitivity Analysis**: [`docs/scenario_analysis.md`](docs/scenario_analysis.md)

---

## Output Artifacts

- **Cleaned Dataset**: [`data/processed/escooter_trips_2020_cleaned.parquet`](data/processed/escooter_trips_2020_cleaned.parquet) (29.1 MB, 525,784 records)
- **EDA & Spatial Centroids**: `results/eda/` (`zone_level_summary.csv`, `distance_matrix.csv`, etc.)
- **Transportation Allocations**: `results/transport/` (`transport_allocations.csv`, `transport_summary.json`)
- **P-Median Depot Assignments**: `results/pmedian/` (`selected_depots.csv`, `zone_assignments.csv`)
- **Scenario Analysis**: `results/scenarios/` (`scenarios_results.csv`, `scenarios_results.json`, `figures/`)

---

## Quality Assurance & Automated Testing

All unit tests use lightweight synthetic fixtures and mock data:

```bash
# Run pytest test suite (166 tests)
uv run pytest

# Run ruff code linter
uv run ruff check .

# Run ruff formatter verification
uv run ruff format --check .
```
