"""Interactive Plotly chart generators for the Streamlit dashboard.

Phase 6 – Streamlit Dashboard Visualizations
=============================================
Provides clean, performant, and interactive Plotly figures for:
- Page 1: Temporal trip distributions and spatial flow imbalances.
- Page 2: Transportation problem allocations, supply-demand balances.
- Page 3: P-Median geographic depot assignments and load distributions.
- Page 4: Scenario elbow curves, distance sensitivity, and cost trade-offs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from escooter_opt.scenarios.runner import ScenarioResult

# Standardized academic / executive palette
PRIMARY_BLUE = "#1f77b4"
ACCENT_ORANGE = "#ff7f0e"
SURPLUS_GREEN = "#2ca02c"
DEFICIT_RED = "#d62728"
NEUTRAL_GRAY = "#7f7f7f"


# ---------------------------------------------------------------------------
# Page 1: Data Overview Plots
# ---------------------------------------------------------------------------


def create_hourly_profile_fig(hourly_df: pd.DataFrame) -> go.Figure:
    """Hourly volume bar chart with dual-axis mean speed line."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=hourly_df["hour"],
            y=hourly_df["trip_count"],
            name="Trip Volume",
            marker_color=PRIMARY_BLUE,
            opacity=0.85,
            hovertemplate="Hour %{x}:00<br>Trips: %{y:,.0f}<extra></extra>",
        )
    )
    if "mean_speed_mph" in hourly_df.columns:
        fig.add_trace(
            go.Scatter(
                x=hourly_df["hour"],
                y=hourly_df["mean_speed_mph"],
                name="Mean Speed (mph)",
                yaxis="y2",
                mode="lines+markers",
                line={"color": ACCENT_ORANGE, "width": 2.5},
                marker={"size": 6},
                hovertemplate="Hour %{x}:00<br>Speed: %{y:.1f} mph<extra></extra>",
            )
        )

    fig.update_layout(
        title="24-Hour Trip Volume & Speed Profile",
        xaxis={"title": "Hour of Day (0–23)", "tickmode": "linear", "dtick": 2},
        yaxis={"title": "Total Trips"},
        yaxis2={
            "title": "Mean Speed (mph)",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
        hovermode="x unified",
    )
    return fig


def create_day_of_week_fig(dow_df: pd.DataFrame) -> go.Figure:
    """Day of week trip volume bar chart."""
    df = dow_df.copy()
    if "day_name" in df.columns:
        x_col = "day_name"
    else:
        x_col = "day_of_week"

    colors = [
        ACCENT_ORANGE if day in ("Saturday", "Sunday", 5, 6) else PRIMARY_BLUE
        for day in df[x_col]
    ]
    fig = go.Figure(
        go.Bar(
            x=df[x_col],
            y=df["trip_count"],
            marker_color=colors,
            hovertemplate="%{x}<br>Trips: %{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Trip Volume by Day of Week (Orange = Weekend)",
        xaxis_title="Day of Week",
        yaxis_title="Total Trips",
        margin={"l": 40, "r": 40, "t": 50, "b": 40},
    )
    return fig


def create_daily_trend_fig(daily_df: pd.DataFrame) -> go.Figure:
    """Daily trip volume time series."""
    fig = go.Figure(
        go.Scatter(
            x=daily_df["date"],
            y=daily_df["trip_count"],
            mode="lines",
            line={"color": PRIMARY_BLUE, "width": 2},
            hovertemplate="%{x}<br>Trips: %{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Daily Trip Volume Trend (2020 Pilot)",
        xaxis_title="Date",
        yaxis_title="Trips per Day",
        margin={"l": 40, "r": 40, "t": 50, "b": 40},
    )
    return fig


def create_top_zones_fig(zone_flows_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Departures vs Arrivals for top high-activity zones."""
    df = zone_flows_df.sort_values(by="turnover", ascending=False).head(top_n).copy()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=df["ca_name"],
            x=df["departures"],
            name="Departures",
            orientation="h",
            marker_color=PRIMARY_BLUE,
            hovertemplate="%{y}<br>Departures: %{x:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            y=df["ca_name"],
            x=df["arrivals"],
            name="Arrivals",
            orientation="h",
            marker_color=ACCENT_ORANGE,
            hovertemplate="%{y}<br>Arrivals: %{x:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Top {top_n} Community Areas by Trip Turnover (Activity)",
        barmode="group",
        yaxis={"autorange": "reversed"},
        xaxis_title="Trip Count",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        margin={"l": 120, "r": 40, "t": 60, "b": 40},
    )
    return fig


def create_net_flow_imbalance_fig(
    zone_flows_df: pd.DataFrame, top_n: int = 20
) -> go.Figure:
    """Net flow imbalance (Surplus vs Deficit) proxy."""
    # Top surplus and top deficit
    surplus = zone_flows_df.sort_values(by="net_flow", ascending=False).head(top_n // 2)
    deficit = zone_flows_df.sort_values(by="net_flow", ascending=True).head(top_n // 2)
    combo = (
        pd.concat([surplus, deficit])
        .drop_duplicates(subset=["ca_number"])
        .sort_values(by="net_flow")
    )

    colors = [SURPLUS_GREEN if nf > 0 else DEFICIT_RED for nf in combo["net_flow"]]
    fig = go.Figure(
        go.Bar(
            y=combo["ca_name"],
            x=combo["net_flow"],
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}<br>Net Flow: %{x:+,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Zone Net-Flow Imbalance (Green = Surplus, Red = Deficit Proxy)",
        xaxis_title="Net Flow (Arrivals − Departures)",
        yaxis_title="Community Area",
        margin={"l": 140, "r": 40, "t": 50, "b": 40},
    )
    return fig


# ---------------------------------------------------------------------------
# Page 2: Transportation Problem Plots
# ---------------------------------------------------------------------------


def create_transport_allocation_fig(
    alloc_df: pd.DataFrame, top_n: int = 15
) -> go.Figure:
    """Top transfer arcs by units transferred."""
    if alloc_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Transfer Allocations")
        return fig

    df = (
        alloc_df.sort_values(by="Units Transferred", ascending=False).head(top_n).copy()
    )
    route_labels = [
        f"{s} → {d}" for s, d in zip(df["Surplus Name"], df["Deficit Name"])
    ]

    fig = go.Figure(
        go.Bar(
            y=route_labels,
            x=df["Units Transferred"],
            orientation="h",
            marker_color=PRIMARY_BLUE,
            hovertemplate="%{y}<br>Units: %{x:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Top {len(df)} Recommended Transfer Routes by Unit Volume",
        xaxis_title="Estimated Units Transferred",
        yaxis={"autorange": "reversed"},
        margin={"l": 200, "r": 40, "t": 50, "b": 40},
    )
    return fig


def create_transport_dist_hist_fig(alloc_df: pd.DataFrame) -> go.Figure:
    """Histogram of transfer route distances."""
    if alloc_df.empty:
        return go.Figure()

    fig = px.histogram(
        alloc_df,
        x="Distance (km)",
        nbins=20,
        title="Distribution of Recommended Transfer Route Distances",
        color_discrete_sequence=[ACCENT_ORANGE],
    )
    fig.update_layout(
        xaxis_title="Transfer Route Distance (km)",
        yaxis_title="Number of Active Routes",
        margin={"l": 40, "r": 40, "t": 50, "b": 40},
    )
    return fig


# ---------------------------------------------------------------------------
# Page 3: P-Median Facility Location Plots
# ---------------------------------------------------------------------------


def create_pmedian_map_fig(
    assign_df: pd.DataFrame,
    depots_df: pd.DataFrame,
    centroids_df: pd.DataFrame,
) -> go.Figure:
    """Interactive geographic scatter of Chicago Community Areas & selected depots."""
    fig = go.Figure()

    # Join centroids with assignments if needed
    merged = assign_df.merge(
        centroids_df[["ca_number", "latitude", "longitude"]],
        left_on="Zone CA",
        right_on="ca_number",
        how="left",
    )

    # 1. Plot demand zones as circles sized by demand weight and colored by assigned depot
    fig.add_trace(
        go.Scatter(
            x=merged["longitude"],
            y=merged["latitude"],
            mode="markers",
            marker={
                "size": np.clip(np.sqrt(merged["Demand Weight"]) * 0.08, 6, 25),
                "color": merged["Assigned Depot CA"],
                "colorscale": "Turbo",
                "opacity": 0.7,
                "line": {"color": "white", "width": 1},
            },
            text=merged["Zone Name"]
            + "<br>Assigned to: "
            + merged["Assigned Depot Name"]
            + "<br>Distance: "
            + merged["Distance (km)"].astype(str)
            + " km",
            name="Demand Zones",
            hovertemplate="%{text}<extra></extra>",
        )
    )

    # 2. Plot selected depots with distinct star markers
    fig.add_trace(
        go.Scatter(
            x=depots_df["Longitude"],
            y=depots_df["Latitude"],
            mode="markers+text",
            marker={
                "symbol": "star",
                "size": 18,
                "color": "black",
                "line": {"color": "gold", "width": 2},
            },
            text=depots_df["Depot Name"],
            textposition="top center",
            name="Selected Depots",
            hovertemplate="DEPOT: %{text}<br>Zones Served: %{customdata[0]}<br>Total Weight: %{customdata[1]:,.0f}<extra></extra>",
            customdata=depots_df[
                ["Assigned Demand Zones", "Assigned Total Weight"]
            ].values,
        )
    )

    fig.update_layout(
        title="Chicago Community Areas & Selected P-Median Depots (Centroid Proxies)",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        xaxis={"scaleanchor": "y", "scaleratio": 1.3},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
    )
    return fig


def create_pmedian_load_fig(depots_df: pd.DataFrame) -> go.Figure:
    """Depot load comparison (demand weight and zones served)."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=depots_df["Depot Name"],
            y=depots_df["Assigned Total Weight"],
            name="Assigned Demand Weight",
            marker_color=PRIMARY_BLUE,
            hovertemplate="%{x}<br>Weight: %{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Total Demand Activity Assigned per Depot",
        xaxis_title="Selected Depot",
        yaxis_title="Total Assigned Turnover Weight",
        margin={"l": 40, "r": 40, "t": 50, "b": 40},
    )
    return fig


def create_pmedian_dist_hist_fig(assign_df: pd.DataFrame) -> go.Figure:
    """Histogram of zone-to-depot assignment distances."""
    fig = px.histogram(
        assign_df,
        x="Distance (km)",
        nbins=15,
        title="Distribution of Zone-to-Depot Assignment Distances",
        color_discrete_sequence=[SURPLUS_GREEN],
    )
    fig.update_layout(
        xaxis_title="Assignment Distance (km)",
        yaxis_title="Number of Zones",
        margin={"l": 40, "r": 40, "t": 50, "b": 40},
    )
    return fig


# ---------------------------------------------------------------------------
# Page 4: Scenario Analysis Plots
# ---------------------------------------------------------------------------


def create_scenario_elbow_fig(results: list[ScenarioResult]) -> go.Figure:
    """P-Median Elbow Curve: Weighted objective vs p by distance metric."""
    pm_res = [
        r
        for r in results
        if r.config.model == "pmedian" and r.pmedian_objective_km is not None
    ]
    if not pm_res:
        return go.Figure()

    fig = go.Figure()
    dm_groups: dict[str, list[ScenarioResult]] = {}
    for r in pm_res:
        dm_groups.setdefault(r.config.distance_method, []).append(r)

    palette = {"manhattan": PRIMARY_BLUE, "haversine": ACCENT_ORANGE}
    for dm, grp in dm_groups.items():
        sorted_grp = sorted(grp, key=lambda x: x.config.p or 0)
        p_vals = [r.config.p for r in sorted_grp]
        objs = [r.pmedian_objective_km for r in sorted_grp]
        fig.add_trace(
            go.Scatter(
                x=p_vals,
                y=objs,
                mode="lines+markers",
                name=f"{dm.capitalize()} Distance",
                line={"color": palette.get(dm, NEUTRAL_GRAY), "width": 2.5},
                marker={"size": 8},
                hovertemplate="p=%{x}<br>Obj: %{y:,.1f} km-wt<extra></extra>",
            )
        )

    fig.update_layout(
        title="P-Median Facility Location: Objective Elbow Curve",
        xaxis_title="Number of Depots (p)",
        yaxis_title="Weighted Objective (km-weight)",
        xaxis={"tickmode": "linear", "dtick": 1},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        margin={"l": 50, "r": 40, "t": 60, "b": 40},
    )
    return fig


def create_scenario_avg_dist_fig(results: list[ScenarioResult]) -> go.Figure:
    """Mean zone-to-depot assignment distance vs p."""
    pm_res = [
        r
        for r in results
        if r.config.model == "pmedian" and r.pmedian_avg_dist_m is not None
    ]
    if not pm_res:
        return go.Figure()

    dm_groups: dict[str, list[ScenarioResult]] = {}
    for r in pm_res:
        dm_groups.setdefault(r.config.distance_method, []).append(r)

    fig = go.Figure()
    palette = {"manhattan": PRIMARY_BLUE, "haversine": ACCENT_ORANGE}
    for dm, grp in dm_groups.items():
        sorted_grp = sorted(grp, key=lambda x: x.config.p or 0)
        p_vals = [f"p={r.config.p}" for r in sorted_grp]
        dist_km = [round(r.pmedian_avg_dist_m / 1000.0, 2) for r in sorted_grp]
        fig.add_trace(
            go.Bar(
                x=p_vals,
                y=dist_km,
                name=f"{dm.capitalize()}",
                marker_color=palette.get(dm, NEUTRAL_GRAY),
                hovertemplate="%{x}<br>Mean Dist: %{y:.2f} km<extra></extra>",
            )
        )

    fig.update_layout(
        title="Mean Zone-to-Depot Assignment Distance vs Depot Count",
        xaxis_title="Depot Count Configuration",
        yaxis_title="Mean Assignment Distance (km)",
        barmode="group",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        margin={"l": 50, "r": 40, "t": 60, "b": 40},
    )
    return fig


def create_scenario_transport_obj_fig(results: list[ScenarioResult]) -> go.Figure:
    """Transportation Problem objective across evaluated scenarios."""
    tr_res = [
        r
        for r in results
        if r.config.model == "transport" and r.transport_objective_km is not None
    ]
    if not tr_res:
        return go.Figure()

    labels = [r.config.label for r in tr_res]
    objs = [r.transport_objective_km for r in tr_res]
    colors = [
        PRIMARY_BLUE if "baseline" in r.config.label else DEFICIT_RED for r in tr_res
    ]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=objs,
            marker_color=colors,
            hovertemplate="%{x}<br>Cost: %{y:,.1f} km<extra></extra>",
        )
    )
    fig.update_layout(
        title="Transportation Problem: Objective Value Comparison (Total Transfer Cost)",
        xaxis_title="Scenario",
        yaxis_title="Transfer Cost (km)",
        margin={"l": 50, "r": 40, "t": 50, "b": 40},
    )
    return fig
