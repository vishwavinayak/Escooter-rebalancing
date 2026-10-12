"""Page 2: Transportation Problem Interface.

Configures, executes, and visualizes the classical Transportation Problem
Linear Program for estimated e-scooter fleet redistribution across Chicago
Community Areas using HiGHS via scipy.optimize.linprog.
"""

from __future__ import annotations

import io
from pathlib import Path

import streamlit as st

from escooter_opt.dashboard.helpers import (
    execute_transport_solver,
    load_eda_tables,
    validate_transport_params,
)
from escooter_opt.dashboard.plotting import (
    create_transport_allocation_fig,
    create_transport_dist_hist_fig,
)


def render_page2(eda_dir: Path | None = None) -> None:
    """Render the Page 2 Transportation Problem interface."""
    st.header("🚚 Transportation Problem (Fleet Redistribution LP)")
    st.markdown(
        r"""
        Solves the balanced/unbalanced Transportation Problem to determine the cost-minimizing
        relocation of e-scooters from **potential surplus zones** ($s_i = \max(I_i, 0)$) to
        **potential deficit zones** ($d_j = \max(-I_j, 0)$).
        """
    )

    st.info(
        "📌 **Operational Research Disclaimer:**\n\n"
        "Rebalancing quantities are **estimated proxies** derived from historical trip-flow imbalances. "
        "They do **not** represent measured physical on-street scooter inventory or live battery levels. "
        "Costs are based on centroid-to-centroid distance approximations."
    )

    # Load required EDA tables
    tables = load_eda_tables(eda_dir)
    if "zone_flows" not in tables:
        st.error(
            "Missing `zone_flows_and_imbalance.csv` in results/eda. Please run EDA first."
        )
        return

    zone_flows = tables["zone_flows"]

    # -----------------------------------------------------------------------
    # Section 1: Configuration Form
    # -----------------------------------------------------------------------
    st.subheader("Model Configuration")
    with st.form("transport_config_form"):
        col1, col2 = st.columns(2)
        with col1:
            distance_method = st.selectbox(
                "Distance Metric (Cost per unit $c_{ij}$):",
                options=["manhattan", "haversine"],
                index=0,
                help="Manhattan reflects rectilinear street grid distances. Haversine is great-circle straight-line distance.",
            )
        with col2:
            demand_scale = st.slider(
                "Demand / Imbalance Scale Factor ($s$):",
                min_value=0.1,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="1.0 is full historical proxy imbalance. Values < 1.0 model partial-day or off-peak redistribution.",
            )

        submitted = st.form_submit_button(
            "⚡ Solve Transportation Problem", type="primary"
        )

    # Validate parameters
    errors = validate_transport_params(distance_method, demand_scale)
    if errors:
        for err in errors:
            st.error(f"Validation Error: {err}")
        return

    # Select the distance matrix
    dist_matrix_key = f"dist_{distance_method}"
    if dist_matrix_key not in tables:
        st.error(f"Distance matrix `{dist_matrix_key}` not found in results/eda.")
        return
    dist_matrix = tables[dist_matrix_key]

    # Run solver if submitted or if cached in session
    state_key = "transport_solution"
    if submitted or state_key not in st.session_state:
        with st.spinner(
            "Solving Transportation Problem via HiGHS simplex/interior-point..."
        ):
            result, alloc_df, inputs = execute_transport_solver(
                zone_flows=zone_flows,
                dist_matrix=dist_matrix,
                distance_method=distance_method,
                demand_scale=demand_scale,
            )
            st.session_state[state_key] = {
                "result": result,
                "alloc_df": alloc_df,
                "inputs": inputs,
                "distance_method": distance_method,
                "demand_scale": demand_scale,
            }

    data = st.session_state[state_key]
    result = data["result"]
    alloc_df = data["alloc_df"]
    inputs = data["inputs"]

    # -----------------------------------------------------------------------
    # Section 2: Solution Summary Cards
    # -----------------------------------------------------------------------
    st.subheader("Optimization Solution Summary")
    if result.status == "optimal":
        st.success(
            f"Solver Status: **{result.status.upper()}** — Optimal Redistribution Schedule Identified"
        )
    else:
        st.error(
            f"Solver Status: **{result.status.upper()}** — {result.solver_message}"
        )
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Total Transfer Cost",
        f"{result.objective_value_km:,.1f} km",
        f"{result.objective_value_meters:,.0f} m",
    )
    m2.metric(
        "Units Transferred",
        f"{result.total_units_transferred:,}",
        f"{len(result.allocations)} Active Routes",
    )
    m3.metric(
        "Supply Utilisation",
        f"{result.supply_utilisation_pct:.1f}%",
        f"{result.total_supply:,} Supply Units",
    )
    m4.metric(
        "Demand Satisfaction",
        f"{result.demand_satisfaction_pct:.1f}%",
        f"{result.total_demand:,} Demand Units",
    )

    # Second row of metrics: Slacks and network scale
    s1, s2, s3, s4 = st.columns(4)
    unused_supply = max(0, inputs.total_supply - result.total_units_transferred)
    unmet_demand = max(0, inputs.total_demand - result.total_units_transferred)
    s1.metric("Unused Supply", f"{unused_supply:,} units")
    s2.metric("Unmet Demand", f"{unmet_demand:,} units")
    s3.metric("Surplus Zones ($|S|$)", f"{len(inputs.surplus_zones)}")
    s4.metric("Deficit Zones ($|D|$)", f"{len(inputs.deficit_zones)}")

    # -----------------------------------------------------------------------
    # Section 3: Visualizations
    # -----------------------------------------------------------------------
    st.subheader("Redistribution Allocations & Diagnostics")
    tab_routes, tab_dist, tab_sources = st.tabs(
        [
            "Top Transfer Routes",
            "Route Distance Distribution",
            "Source & Sink Summaries",
        ]
    )

    with tab_routes:
        top_k = st.slider(
            "Routes to display:",
            min_value=5,
            max_value=30,
            value=15,
            step=5,
            key="p2_top_k",
        )
        fig_alloc = create_transport_allocation_fig(alloc_df, top_n=top_k)
        st.plotly_chart(fig_alloc, use_container_width=True)

    with tab_dist:
        fig_dist = create_transport_dist_hist_fig(alloc_df)
        st.plotly_chart(fig_dist, use_container_width=True)
        st.caption(
            "Most recommended transfers connect adjacent or near-neighbor Community Areas (< 5 km)."
        )

    with tab_sources:
        col_s, col_d = st.columns(2)
        with col_s:
            st.markdown("**Top Surplus Origins (Sources)**")
            if not alloc_df.empty:
                s_sum = (
                    alloc_df.groupby(["Surplus CA", "Surplus Name"])[
                        "Units Transferred"
                    ]
                    .sum()
                    .reset_index()
                )
                s_sum = s_sum.sort_values(by="Units Transferred", ascending=False).head(
                    10
                )
                st.dataframe(s_sum, use_container_width=True, hide_index=True)
        with col_d:
            st.markdown("**Top Deficit Destinations (Sinks)**")
            if not alloc_df.empty:
                d_sum = (
                    alloc_df.groupby(["Deficit CA", "Deficit Name"])[
                        "Units Transferred"
                    ]
                    .sum()
                    .reset_index()
                )
                d_sum = d_sum.sort_values(by="Units Transferred", ascending=False).head(
                    10
                )
                st.dataframe(d_sum, use_container_width=True, hide_index=True)

    # -----------------------------------------------------------------------
    # Section 4: Allocations Table & CSV Export
    # -----------------------------------------------------------------------
    st.subheader("Detailed Transfer Plan Table")
    st.dataframe(alloc_df, use_container_width=True, hide_index=True)

    csv_buf = io.StringIO()
    alloc_df.to_csv(csv_buf, index=False)
    st.download_button(
        "📥 Download Transfer Plan CSV",
        data=csv_buf.getvalue(),
        file_name=f"escooter_transport_allocations_{data['distance_method']}_scale{data['demand_scale']}.csv",
        mime="text/csv",
        type="primary",
    )
