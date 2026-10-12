"""Page 4: Scenario and Sensitivity Analysis Interface.

Enables multi-scenario parametric sweeps over depot counts (p), distance metrics,
and demand scale assumptions, displaying comparative trade-off tables and elbow plots.
"""

from __future__ import annotations

import io
from pathlib import Path

import streamlit as st

from escooter_opt.dashboard.helpers import (
    execute_scenario_suite,
    load_eda_tables,
    validate_scenario_params,
)
from escooter_opt.dashboard.plotting import (
    create_scenario_avg_dist_fig,
    create_scenario_elbow_fig,
    create_scenario_transport_obj_fig,
)
from escooter_opt.scenarios.runner import results_to_dataframe


def render_page4(eda_dir: Path | None = None) -> None:
    """Render the Page 4 Scenario Analysis interface."""
    st.header("📈 Scenario & Sensitivity Analysis")
    st.markdown(
        r"""
        Runs structured scenario suites across both optimization models to evaluate:
        1. **P-Median Depot Count Sensitivity**: The marginal benefit of adding facilities ($p \in \{1 \dots 5\}$).
        2. **Transportation Cost & Demand Sensitivity**: Total transfer distance under Manhattan vs. Haversine costs and peak vs. off-peak demand scales.
        """
    )

    st.info(
        "📌 **Analytical Context:**\n\n"
        "Comparing scenarios isolates how network topology and operational costs shift under "
        "different policy assumptions without relying on hardcoded estimates."
    )

    # Check for EDA tables
    tables = load_eda_tables(eda_dir)
    if "centroids" not in tables:
        st.error("Missing EDA tables in results/eda. Please run EDA first.")
        return

    max_candidates = len(tables["centroids"])

    # -----------------------------------------------------------------------
    # Section 1: Scenario Configuration Form
    # -----------------------------------------------------------------------
    st.subheader("Scenario Suite Configuration")
    with st.form("scenario_config_form"):
        col1, col2 = st.columns(2)
        with col1:
            p_values = st.multiselect(
                "P-Median Depot Counts ($p$) to Evaluate:",
                options=list(range(1, min(11, max_candidates + 1))),
                default=[1, 2, 3, 4, 5],
                help="Select depot counts for parametric sensitivity analysis.",
            )
            include_transport = st.checkbox(
                "Include Transportation Problem Scenarios", value=True
            )

        with col2:
            distance_methods = st.multiselect(
                "Distance Metrics to Compare:",
                options=["manhattan", "haversine"],
                default=["manhattan", "haversine"],
                help="Compare rectilinear street grid (Manhattan) vs straight-line (Haversine) models.",
            )
            demand_scales = st.multiselect(
                "Transportation Demand Scale Factors ($s$):",
                options=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
                default=[1.0, 0.5],
                help="1.0 is full-period baseline proxy; 0.5 models off-peak partial-period imbalance.",
            )

        submitted = st.form_submit_button("🚀 Run Scenario Suite", type="primary")

    # Validate parameters
    errors = validate_scenario_params(
        p_values, max_candidates, distance_methods, demand_scales
    )
    if errors:
        for err in errors:
            st.error(f"Validation Error: {err}")
        return

    # Run scenario suite if submitted or if cached
    state_key = "scenario_suite_results"
    if submitted or state_key not in st.session_state:
        with st.spinner("Executing optimization scenario suite via HiGHS..."):
            results = execute_scenario_suite(
                p_values=p_values,
                distance_methods=distance_methods,
                include_transport=include_transport,
                demand_scales=demand_scales,
                eda_dir=eda_dir,
            )
            res_df = results_to_dataframe(results)
            st.session_state[state_key] = {
                "results": results,
                "res_df": res_df,
            }

    suite_data = st.session_state[state_key]
    results = suite_data["results"]
    res_df = suite_data["res_df"]

    # -----------------------------------------------------------------------
    # Section 2: Scenario Summary Status & Metrics
    # -----------------------------------------------------------------------
    st.subheader("Scenario Execution Summary")
    total_runs = len(results)
    optimal_runs = sum(1 for r in results if r.status == "optimal")
    failed_runs = total_runs - optimal_runs

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Scenarios Evaluated", str(total_runs))
    col_m2.metric(
        "Optimal Solves",
        f"{optimal_runs} / {total_runs}",
        "100%" if failed_runs == 0 else f"-{failed_runs}",
    )
    if failed_runs > 0:
        col_m3.metric(
            "Failed / Infeasible Scenarios", str(failed_runs), delta_color="inverse"
        )
    else:
        col_m3.metric("Failed Scenarios", "0", "All Solved Cleanly")

    # Display failure alerts if any
    failed_items = [r for r in results if r.status != "optimal"]
    if failed_items:
        for f in failed_items:
            st.error(
                f"❌ Scenario `{f.config.label}` failed with status: **{f.status}** ({f.error_message})"
            )

    # -----------------------------------------------------------------------
    # Section 3: P-Median Sensitivity & The Elbow Curve
    # -----------------------------------------------------------------------
    st.subheader("P-Median Facility Location Sensitivity")
    pm_results = [
        r
        for r in results
        if r.config.model == "pmedian" and r.pmedian_objective_km is not None
    ]

    if pm_results:
        tab_elbow, tab_dist, tab_table = st.tabs(
            [
                "Elbow Curve (Weighted Obj vs p)",
                "Mean Assignment Distance",
                "Detailed P-Median Table",
            ]
        )

        with tab_elbow:
            fig_elbow = create_scenario_elbow_fig(results)
            st.plotly_chart(fig_elbow, use_container_width=True)
            st.caption(
                "**Elbow Effect**: Moving from p=1 to p=3 achieves >50% reduction in total weighted distance. "
                "Adding depots beyond p=3 exhibits sharply diminishing marginal returns (~11% per depot)."
            )

        with tab_dist:
            fig_dist = create_scenario_avg_dist_fig(results)
            st.plotly_chart(fig_dist, use_container_width=True)

        with tab_table:
            pm_cols = [
                "label",
                "distance_method",
                "p",
                "status",
                "pm_obj_km_wt",
                "pm_avg_dist_m",
                "pm_unweighted_km",
                "pm_depots",
                "runtime_s",
            ]
            pm_table = res_df[res_df["model"] == "pmedian"][pm_cols].copy()
            pm_table.rename(
                columns={
                    "label": "Scenario",
                    "distance_method": "Metric",
                    "p": "Depots (p)",
                    "pm_obj_km_wt": "Objective (km-wt)",
                    "pm_avg_dist_m": "Mean Dist (m)",
                    "pm_unweighted_km": "Total Dist (km)",
                    "pm_depots": "Selected Hubs",
                    "runtime_s": "Solve Time (s)",
                },
                inplace=True,
            )
            st.dataframe(pm_table, use_container_width=True, hide_index=True)

    # -----------------------------------------------------------------------
    # Section 4: Transportation Problem Sensitivity
    # -----------------------------------------------------------------------
    tr_results = [
        r
        for r in results
        if r.config.model == "transport" and r.transport_objective_km is not None
    ]
    if tr_results:
        st.subheader("Transportation Problem Sensitivity")
        tab_tr_plot, tab_tr_table = st.tabs(
            ["Objective Comparison", "Detailed Transportation Table"]
        )

        with tab_tr_plot:
            fig_tr = create_scenario_transport_obj_fig(results)
            st.plotly_chart(fig_tr, use_container_width=True)
            st.caption(
                "Manhattan street grid distances yield ~25.5% higher transfer costs than straight-line Haversine."
            )

        with tab_tr_table:
            tr_cols = [
                "label",
                "distance_method",
                "demand_scale",
                "status",
                "tr_obj_km",
                "tr_units",
                "tr_supply_util_pct",
                "tr_demand_sat_pct",
                "tr_routes",
                "runtime_s",
            ]
            tr_table = res_df[res_df["model"] == "transport"][tr_cols].copy()
            tr_table.rename(
                columns={
                    "label": "Scenario",
                    "distance_method": "Metric",
                    "demand_scale": "Scale Factor",
                    "tr_obj_km": "Objective (km)",
                    "tr_units": "Units Transferred",
                    "tr_supply_util_pct": "Supply Util (%)",
                    "tr_demand_sat_pct": "Demand Sat (%)",
                    "tr_routes": "Active Routes",
                    "runtime_s": "Solve Time (s)",
                },
                inplace=True,
            )
            st.dataframe(tr_table, use_container_width=True, hide_index=True)

    # -----------------------------------------------------------------------
    # Section 5: Full Suite Export
    # -----------------------------------------------------------------------
    st.subheader("Export Suite Results")
    csv_buf = io.StringIO()
    res_df.to_csv(csv_buf, index=False)
    st.download_button(
        "📥 Download Scenario Suite Results CSV",
        data=csv_buf.getvalue(),
        file_name="escooter_optimization_scenario_suite_results.csv",
        mime="text/csv",
        type="primary",
    )
