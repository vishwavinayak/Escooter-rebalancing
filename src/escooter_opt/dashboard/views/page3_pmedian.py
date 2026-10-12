"""Page 3: Depot Location (P-Median Integer Programming).

Configures, solves, and visualizes the P-Median Facility Location Integer
Program for optimal e-scooter depot siting across Chicago Community Areas
using HiGHS MILP via scipy.optimize.milp.
"""

from __future__ import annotations

import io
from pathlib import Path

import streamlit as st

from escooter_opt.dashboard.helpers import (
    execute_pmedian_solver,
    load_eda_tables,
    validate_pmedian_params,
)
from escooter_opt.dashboard.plotting import (
    create_pmedian_dist_hist_fig,
    create_pmedian_load_fig,
    create_pmedian_map_fig,
)


def render_page3(eda_dir: Path | None = None) -> None:
    """Render the Page 3 P-Median Depot Location interface."""
    st.header("🏢 Depot Facility Location (P-Median IP)")
    st.markdown(
        r"""
        Formulates depot siting as an Integer Program to select exactly $p$ depot locations
        from candidate Community Areas to **minimize total demand-weighted zone-to-depot distance**:
        $$\min \sum_{i \in I} \sum_{j \in J} w_i \cdot d_{ij} \cdot z_{ij}$$
        where demand weight $w_i = \text{arrivals}_i + \text{departures}_i$ (total activity turnover).
        """
    )

    st.warning(
        "⚠️ **Facility Location Disclaimer:**\n\n"
        "Selected depot locations are **modelled optimal centroid candidates** derived from historical "
        "trip-flow density. They are **NOT** approved, acquired, or physically available municipal depot sites. "
        "Coordinates represent geographic Community Area centroids, not surveyed building parcels."
    )

    # Load required EDA tables
    tables = load_eda_tables(eda_dir)
    if not all(k in tables for k in ("zone_flows", "centroids")):
        st.error(
            "Missing required tables (`zone_flows`, `centroids`) in results/eda. Please run EDA first."
        )
        return

    zone_flows = tables["zone_flows"]
    centroids = tables["centroids"]
    n_total_candidates = len(centroids)

    # -----------------------------------------------------------------------
    # Section 1: Configuration Form
    # -----------------------------------------------------------------------
    st.subheader("Model Configuration")
    with st.form("pmedian_config_form"):
        col1, col2 = st.columns(2)
        with col1:
            p = st.slider(
                "Number of Depots to Site ($p$):",
                min_value=1,
                max_value=min(15, n_total_candidates),
                value=5,
                step=1,
                help="The exact number of candidate depot locations to open.",
            )
        with col2:
            distance_method = st.selectbox(
                "Distance Metric ($d_{ij}$):",
                options=["manhattan", "haversine"],
                index=0,
                help="Manhattan reflects rectilinear Chicago street grid distances. Haversine is great-circle distance.",
            )

        with st.expander("Advanced Candidate Restrictions (Optional)", expanded=False):
            st.caption(
                "Restrict candidate sites to a specific subset of Community Areas. By default, all 77 are eligible."
            )
            ca_options = dict(zip(centroids["ca_number"], centroids["ca_name"]))
            selected_candidates = st.multiselect(
                "Eligible Candidate Depots:",
                options=list(ca_options.keys()),
                format_func=lambda x: f"CA {x}: {ca_options[x]}",
                default=None,
                help="Leave empty to consider all 77 Community Areas as candidate depot sites.",
            )

        submitted = st.form_submit_button("⚡ Solve P-Median Model", type="primary")

    # Validate parameters
    cand_cas = selected_candidates if selected_candidates else None
    max_cand = len(selected_candidates) if selected_candidates else n_total_candidates
    errors = validate_pmedian_params(p, max_cand, distance_method, cand_cas)
    if errors:
        for err in errors:
            st.error(f"Validation Error: {err}")
        return

    # Select distance matrix
    dist_matrix_key = f"dist_{distance_method}"
    if dist_matrix_key not in tables:
        st.error(f"Distance matrix `{dist_matrix_key}` not found in results/eda.")
        return
    dist_matrix = tables[dist_matrix_key]

    # Run solver if submitted or if cached in session
    state_key = "pmedian_solution"
    if submitted or state_key not in st.session_state:
        with st.spinner(
            "Solving P-Median Integer Program via HiGHS branch-and-bound..."
        ):
            result, depots_df, assign_df, inputs = execute_pmedian_solver(
                zone_flows=zone_flows,
                centroids=centroids,
                dist_matrix=dist_matrix,
                p=p,
                distance_method=distance_method,
                candidate_cas=cand_cas,
            )
            st.session_state[state_key] = {
                "result": result,
                "depots_df": depots_df,
                "assign_df": assign_df,
                "inputs": inputs,
                "p": p,
                "distance_method": distance_method,
            }

    data = st.session_state[state_key]
    result = data["result"]
    depots_df = data["depots_df"]
    assign_df = data["assign_df"]

    # -----------------------------------------------------------------------
    # Section 2: Solution KPI Metrics
    # -----------------------------------------------------------------------
    st.subheader("Optimization Solution Summary")
    if result.status == "optimal":
        st.success(
            f"Solver Status: **{result.status.upper()}** — Optimal {data['p']}-Depot Configuration Found"
        )
    else:
        st.error(f"Solver Status: **{result.status.upper()}**")
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Weighted Objective", f"{result.objective_value_km:,.1f} km-wt", "Minimised"
    )
    m2.metric(
        "Mean Assignment Distance",
        f"{result.average_assignment_distance_meters:,.0f} m",
        f"{result.average_assignment_distance_meters / 1000.0:.2f} km",
    )
    m3.metric(
        "Selected Depots",
        f"{len(result.selected_depots)} / {p}",
        "Exact Count Satisfied",
    )
    m4.metric(
        "Demand Zones Assigned",
        f"{len(result.assignments)} Zones",
        "100% Demand Covered",
    )

    # -----------------------------------------------------------------------
    # Section 3: Visualizations
    # -----------------------------------------------------------------------
    st.subheader("Spatial Assignment & Depot Network Maps")
    tab_map, tab_load, tab_hist = st.tabs(
        [
            "Geographic Assignment Map",
            "Depot Load Distribution",
            "Zone Distance Histogram",
        ]
    )

    with tab_map:
        fig_map = create_pmedian_map_fig(assign_df, depots_df, centroids)
        st.plotly_chart(fig_map, use_container_width=True)
        st.caption(
            "Circles represent Chicago Community Area centroids (sized by trip turnover, colored by assigned depot). Stars mark selected depot hubs."
        )

    with tab_load:
        fig_load = create_pmedian_load_fig(depots_df)
        st.plotly_chart(fig_load, use_container_width=True)

    with tab_hist:
        fig_hist = create_pmedian_dist_hist_fig(assign_df)
        st.plotly_chart(fig_hist, use_container_width=True)

    # -----------------------------------------------------------------------
    # Section 4: Result Tables & CSV Exports
    # -----------------------------------------------------------------------
    st.subheader("Selected Depots & Assignments")
    col_d, col_a = st.columns([1, 1])

    with col_d:
        st.markdown("**Selected Depot Locations**")
        st.dataframe(depots_df, use_container_width=True, hide_index=True)
        depots_csv = io.StringIO()
        depots_df.to_csv(depots_csv, index=False)
        st.download_button(
            "📥 Download Selected Depots CSV",
            data=depots_csv.getvalue(),
            file_name=f"escooter_pmedian_depots_p{data['p']}_{data['distance_method']}.csv",
            mime="text/csv",
        )

    with col_a:
        st.markdown("**Zone-to-Depot Assignments Preview**")
        st.dataframe(assign_df.head(15), use_container_width=True, hide_index=True)
        assign_csv = io.StringIO()
        assign_df.to_csv(assign_csv, index=False)
        st.download_button(
            "📥 Download All Zone Assignments CSV",
            data=assign_csv.getvalue(),
            file_name=f"escooter_pmedian_assignments_p{data['p']}_{data['distance_method']}.csv",
            mime="text/csv",
            type="primary",
        )
