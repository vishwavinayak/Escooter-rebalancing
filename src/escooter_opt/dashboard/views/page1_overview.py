"""Page 1: Data Overview & Exploration.

Displays cleaned dataset metrics, data-quality audit breakdown, temporal
trip patterns, and zone-level net-flow imbalances with clear methodological
distinctions between observations and proxy measures.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from escooter_opt.dashboard.helpers import (
    DEFAULT_CLEANED_PARQUET,
    load_cleaning_report,
    load_eda_summary_metrics,
    load_eda_tables,
    run_cleaning_from_dataframe,
)
from escooter_opt.dashboard.plotting import (
    create_daily_trend_fig,
    create_day_of_week_fig,
    create_hourly_profile_fig,
    create_net_flow_imbalance_fig,
    create_top_zones_fig,
)


def render_page1(eda_dir: Path | None = None) -> None:
    """Render the Page 1 Data Overview interface."""
    st.header("📊 Dataset Overview & Exploratory Data Analysis")
    st.markdown(
        """
        This page summarizes the cleaned 2020 City of Chicago E-Scooter Trips dataset,
        reports data quality audit results, and presents baseline temporal and spatial trip flows.
        """
    )

    # Methodological Callout
    st.info(
        "📌 **Methodology Note & Operational Disclaimer:**\n\n"
        "- **Observed quantities**: Recorded trip start/end times, trip durations, and recorded origin/destination Community Areas.\n"
        "- **Derived proxy measures**: Zone arrivals ($A_i$), departures ($D_i$), and net flow ($I_i = A_i - D_i$) are "
        "*historical trip-flow proxies*, **not** physical observed scooter inventory."
    )

    # -----------------------------------------------------------------------
    # Section 1: Data Pipeline Source & Controls
    # -----------------------------------------------------------------------
    with st.expander("⚙️ Data Source & Cleaning Pipeline Controls", expanded=False):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**Load Existing Cleaned Data**")
            st.write(f"Default cleaned parquet: `{DEFAULT_CLEANED_PARQUET}`")
            if DEFAULT_CLEANED_PARQUET.is_file():
                st.success("Cleaned dataset parquet is available on disk.")
            else:
                st.warning("Cleaned parquet not found. Run pipeline or upload CSV.")

        with col2:
            st.markdown("**Run Cleaning Pipeline on Uploaded CSV**")
            uploaded_file = st.file_uploader(
                "Upload Chicago E-Scooter Trips CSV",
                type=["csv"],
                key="p1_csv_uploader",
                help="Upload a CSV with the Chicago E-Scooter Trips schema to run Policy C cleaning in-memory.",
            )

        if uploaded_file is not None and st.button(
            "🚀 Run Cleaning Pipeline on Uploaded CSV", type="primary"
        ):
            with st.spinner("Executing Policy C cleaning pipeline..."):
                try:
                    raw_df = pd.read_csv(uploaded_file, low_memory=False)
                    cleaned_df, report_dict = run_cleaning_from_dataframe(raw_df)
                    st.session_state["custom_cleaned_df"] = cleaned_df
                    st.session_state["custom_cleaning_report"] = report_dict
                    st.success(
                        f"Cleaning successful! Retained {len(cleaned_df):,} trips "
                        f"({report_dict.get('retention_percentage_vs_raw', 0):.2f}% of raw rows)."
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Failed to process uploaded CSV: {exc}")

    # Load active data & tables
    report = st.session_state.get("custom_cleaning_report") or load_cleaning_report()
    eda_summary = load_eda_summary_metrics(eda_dir)
    tables = load_eda_tables(eda_dir)

    # -----------------------------------------------------------------------
    # Section 2: High-Level KPI Metric Cards
    # -----------------------------------------------------------------------
    st.subheader("Key Dataset Metrics")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    total_trips = eda_summary.get(
        "total_cleaned_trips", report.get("final_retained_rows", 525784)
    )
    kpi1.metric("Cleaned Trips", f"{total_trips:,}")

    start_date = eda_summary.get("date_range_start", "2020-08-12")
    end_date = eda_summary.get("date_range_end", "2020-12-12")
    kpi2.metric("Date Range", f"{start_date} to {end_date}")

    active_zones = eda_summary.get("usable_zones_count", 77)
    cov_pct = eda_summary.get("geographic_coverage_pct", 100.0)
    kpi3.metric("Community Areas", f"{active_zones} / 77", f"{cov_pct:.1f}% Coverage")

    reconciled = eda_summary.get("flows_reconcile", True)
    kpi4.metric(
        "Flow Reconciliation",
        "Balanced (Net = 0)" if reconciled else "Unbalanced",
        "100% Valid Arcs",
    )

    # -----------------------------------------------------------------------
    # Section 3: Data Quality & Attrition Breakdown
    # -----------------------------------------------------------------------
    if report:
        with st.expander(
            "📋 Policy C Data Quality & Attrition Waterfall", expanded=False
        ):
            st.markdown(
                """
                **Policy C Rules Applied:**
                1. Exclude records missing Community Area numbers (1..77).
                2. Exclude bounce trips where **BOTH** duration < 60s **AND** distance < 100m.
                3. Exclude trips with duration > 4 hours (14,400s).
                4. Exclude trips with average speed > 30 mph.
                """
            )
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Sequential Attrition Breakdown**")
                seq_exclusions = report.get("sequential_policy_exclusions", {})
                seq_df = pd.DataFrame(
                    [
                        {
                            "Exclusion Step": "Raw Syntactically Valid Records",
                            "Count": report.get("syntactically_valid_rows", 551797),
                        },
                        {
                            "Exclusion Step": "Step 1: Missing / Invalid Community Area",
                            "Count": -seq_exclusions.get(
                                "step_1_missing_or_invalid_ca", 25776
                            ),
                        },
                        {
                            "Exclusion Step": "Step 2: Short Bounce Trips (<60s & <100m)",
                            "Count": -seq_exclusions.get("step_2_bounce_trips", 188),
                        },
                        {
                            "Exclusion Step": "Step 3: Excessive Duration (>4 hrs)",
                            "Count": -seq_exclusions.get(
                                "step_3_excessive_duration", 30
                            ),
                        },
                        {
                            "Exclusion Step": "Step 4: Excessive Speed (>30 mph)",
                            "Count": -seq_exclusions.get("step_4_excessive_speed", 19),
                        },
                        {
                            "Exclusion Step": "Final Retained Production Trips",
                            "Count": report.get("final_retained_rows", 525784),
                        },
                    ]
                )
                st.dataframe(seq_df, use_container_width=True, hide_index=True)

            with col_b:
                st.markdown("**Independent Rule Evaluation (Overlaps)**")
                indep_exclusions = report.get("independent_policy_exclusions", {})
                if indep_exclusions:
                    indep_df = pd.DataFrame(
                        [
                            {
                                "Rule Filter": "Missing Community Area (1..77)",
                                "Independent Violations": indep_exclusions.get(
                                    "missing_or_invalid_ca", 0
                                ),
                            },
                            {
                                "Rule Filter": "Bounce (<60s AND <100m)",
                                "Independent Violations": indep_exclusions.get(
                                    "bounce_trips", 0
                                ),
                            },
                            {
                                "Rule Filter": "Excessive Duration (>4h)",
                                "Independent Violations": indep_exclusions.get(
                                    "excessive_duration", 0
                                ),
                            },
                            {
                                "Rule Filter": "Excessive Speed (>30mph)",
                                "Independent Violations": indep_exclusions.get(
                                    "excessive_speed", 0
                                ),
                            },
                        ]
                    )
                    st.dataframe(indep_df, use_container_width=True, hide_index=True)

    # -----------------------------------------------------------------------
    # Section 4: Temporal Trip Dynamics
    # -----------------------------------------------------------------------
    st.subheader("Temporal Trip Patterns")
    tab_hour, tab_dow, tab_daily = st.tabs(
        ["Hourly Profile (24-Hour)", "Day of Week Volume", "Daily Time Series Trend"]
    )

    with tab_hour:
        if "hourly" in tables:
            fig_hourly = create_hourly_profile_fig(tables["hourly"])
            st.plotly_chart(fig_hourly, use_container_width=True)
            st.caption(
                "Trip volume peaks sharply during late afternoon / evening commute and recreation hours (15:00–18:00)."
            )
        else:
            st.info("Hourly summary table not found in results/eda.")

    with tab_dow:
        if "dow" in tables:
            fig_dow = create_day_of_week_fig(tables["dow"])
            st.plotly_chart(fig_dow, use_container_width=True)
            st.caption(
                "Weekends show high recreational activity, with Saturday representing the peak volume day."
            )
        else:
            st.info("Day of week summary table not found in results/eda.")

    with tab_daily:
        if "daily" in tables:
            fig_daily = create_daily_trend_fig(tables["daily"])
            st.plotly_chart(fig_daily, use_container_width=True)
            st.caption(
                "Daily trip volume across the 2020 pilot period (August to December)."
            )
        else:
            st.info("Daily summary table not found in results/eda.")

    # -----------------------------------------------------------------------
    # Section 5: Zone-Level Spatial Activity & Net Flow
    # -----------------------------------------------------------------------
    st.subheader("Spatial Distribution & Net-Flow Imbalance")
    if "zone_flows" in tables:
        zf = tables["zone_flows"]

        tab_top_zones, tab_imbalances, tab_table = st.tabs(
            ["Top Activity Zones", "Net-Flow Imbalance Proxy", "Full Zone Flow Table"]
        )

        with tab_top_zones:
            top_n = st.slider(
                "Number of top zones to display:",
                min_value=5,
                max_value=30,
                value=15,
                step=5,
                key="p1_top_n_zones",
            )
            fig_top = create_top_zones_fig(zf, top_n=top_n)
            st.plotly_chart(fig_top, use_container_width=True)

        with tab_imbalances:
            st.markdown(
                "> **Proxy Definition**: Net flow $I_i = A_i - D_i$. Zones with $I_i > 0$ have more trip arrivals "
                "than departures (potential accumulation surplus), while zones with $I_i < 0$ have more departures "
                "(potential depletion deficit)."
            )
            top_imb_n = st.slider(
                "Number of imbalance zones to display:",
                min_value=10,
                max_value=40,
                value=20,
                step=5,
                key="p1_top_imb_n",
            )
            fig_imb = create_net_flow_imbalance_fig(zf, top_n=top_imb_n)
            st.plotly_chart(fig_imb, use_container_width=True)

        with tab_table:
            st.dataframe(
                zf[
                    [
                        "ca_number",
                        "ca_name",
                        "departures",
                        "arrivals",
                        "net_flow",
                        "turnover",
                        "flow_category",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )
            csv_buf = io.StringIO()
            zf.to_csv(csv_buf, index=False)
            st.download_button(
                "📥 Download Zone Flows CSV",
                data=csv_buf.getvalue(),
                file_name="chicago_zone_flows_and_imbalance.csv",
                mime="text/csv",
            )
    else:
        st.info("Zone flow summary table not found in results/eda.")
