"""Visualizations for exploratory data analysis (Matplotlib static + Plotly interactive)."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_hourly_volume(
    hourly_df: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Generate static PNG and interactive HTML for 24-hour trip volume."""
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "hourly_trip_volume.png"
    html_path = output_dir / "hourly_trip_volume.html"

    # Static Matplotlib plot
    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=150)
    ax1.bar(
        hourly_df["hour"],
        hourly_df["trip_count"],
        color="#2b5c8f",
        alpha=0.85,
        edgecolor="#1b3957",
        width=0.8,
    )
    ax1.set_xlabel("Hour of Day (24-Hour)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Total Trip Volume", fontsize=11, fontweight="bold", color="#2b5c8f")
    ax1.set_xticks(range(24))
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Secondary axis: average speed
    ax2 = ax1.twinx()
    ax2.plot(
        hourly_df["hour"],
        hourly_df["mean_speed_mph"],
        color="#e6550d",
        marker="o",
        linewidth=2,
        label="Mean Speed (mph)",
    )
    ax2.set_ylabel("Mean Speed (mph)", fontsize=11, fontweight="bold", color="#e6550d")
    ax2.set_ylim(0, max(hourly_df["mean_speed_mph"].max() * 1.3, 10))

    plt.title(
        "Chicago E-Scooter Trips: Hourly Volume & Speed Profile (2020 Pilot)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    # Interactive Plotly chart
    pfig = go.Figure()
    pfig.add_trace(
        go.Bar(
            x=hourly_df["hour"],
            y=hourly_df["trip_count"],
            name="Trip Volume",
            marker_color="#2b5c8f",
            hovertemplate="Hour: %{x}:00<br>Trips: %{y:,}<extra></extra>",
        )
    )
    pfig.add_trace(
        go.Scatter(
            x=hourly_df["hour"],
            y=hourly_df["mean_speed_mph"],
            name="Mean Speed (mph)",
            yaxis="y2",
            line={"color": "#e6550d", "width": 3},
            hovertemplate="Hour: %{x}:00<br>Speed: %{y:.2f} mph<extra></extra>",
        )
    )
    pfig.update_layout(
        title="Chicago E-Scooter Trips: Hourly Volume & Speed Profile",
        xaxis={"title": "Hour of Day", "tickmode": "linear", "tick0": 0, "dtick": 1},
        yaxis={"title": "Trip Volume"},
        yaxis2={"title": "Mean Speed (mph)", "overlaying": "y", "side": "right"},
        template="plotly_white",
        legend={"x": 0.02, "y": 0.98},
    )
    pfig.write_html(str(html_path))

    return png_path, html_path


def plot_day_of_week_volume(
    dow_df: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Generate static PNG and interactive HTML for trip volume by day of week."""
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "day_of_week_volume.png"
    html_path = output_dir / "day_of_week_volume.html"

    # Static Matplotlib plot
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    colors = [
        "#4393c3" if d in ["Saturday", "Sunday"] else "#2166ac"
        for d in dow_df["day_of_week"]
    ]
    ax.bar(
        dow_df["day_of_week"],
        dow_df["trip_count"],
        color=colors,
        edgecolor="#053061",
        width=0.65,
    )
    ax.set_ylabel("Total Trip Volume", fontsize=11, fontweight="bold")
    ax.set_title(
        "Trip Volume by Day of Week (Weekend Highlighted)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for i, v in enumerate(dow_df["trip_count"]):
        pct = dow_df.loc[i, "volume_pct"]
        ax.text(i, v + 1200, f"{v:,}\n({pct:.1f}%)", ha="center", fontsize=9)

    ax.set_ylim(0, dow_df["trip_count"].max() * 1.15)
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    # Interactive Plotly chart
    pfig = px.bar(
        dow_df,
        x="day_of_week",
        y="trip_count",
        color="day_of_week",
        color_discrete_map={"Saturday": "#4393c3", "Sunday": "#4393c3"},
        title="Trip Volume by Day of Week",
        labels={"day_of_week": "Day of Week", "trip_count": "Trip Volume"},
        template="plotly_white",
    )
    pfig.update_layout(showlegend=False)
    pfig.write_html(str(html_path))

    return png_path, html_path


def plot_daily_time_series(
    daily_df: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Generate static PNG and interactive HTML for daily time series with 7-day rolling mean."""
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "daily_trip_volume_timeseries.png"
    html_path = output_dir / "daily_trip_volume_timeseries.html"

    # Static Matplotlib plot
    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    ax.plot(
        daily_df["date"],
        daily_df["trip_count"],
        color="#92c5de",
        alpha=0.6,
        linewidth=1.5,
        label="Daily Trips",
    )
    ax.plot(
        daily_df["date"],
        daily_df["rolling_7d_mean"],
        color="#0571b0",
        linewidth=2.5,
        label="7-Day Rolling Mean",
    )
    ax.set_ylabel("Trips per Day", fontsize=11, fontweight="bold")
    ax.set_title(
        "Daily E-Scooter Trips: 2020 Pilot Program Duration",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True)
    fig.autofmt_xdate()
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    # Interactive Plotly chart
    pfig = go.Figure()
    pfig.add_trace(
        go.Scatter(
            x=daily_df["date"],
            y=daily_df["trip_count"],
            name="Daily Trips",
            line={"color": "#92c5de", "width": 1.5},
            opacity=0.6,
        )
    )
    pfig.add_trace(
        go.Scatter(
            x=daily_df["date"],
            y=daily_df["rolling_7d_mean"],
            name="7-Day Rolling Mean",
            line={"color": "#0571b0", "width": 3},
        )
    )
    pfig.update_layout(
        title="Daily Trip Volume: Chicago 2020 E-Scooter Pilot",
        xaxis={"title": "Date"},
        yaxis={"title": "Daily Trips"},
        template="plotly_white",
    )
    pfig.write_html(str(html_path))

    return png_path, html_path


def plot_zone_imbalances(
    zone_flows_df: pd.DataFrame,
    output_dir: Path,
    top_n: int = 15,
) -> tuple[Path, Path]:
    """Generate diverging horizontal bar chart of top surplus vs deficit zones."""
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "zone_net_flow_imbalances.png"
    html_path = output_dir / "zone_net_flow_imbalances.html"

    # Select top surplus (most positive net flow) and top deficit (most negative net flow)
    top_surplus = zone_flows_df.sort_values("net_flow", ascending=False).head(top_n)
    top_deficit = (
        zone_flows_df.sort_values("net_flow", ascending=True)
        .head(top_n)
        .sort_values("net_flow", ascending=False)
    )
    combined = (
        pd.concat([top_surplus, top_deficit])
        .drop_duplicates(subset=["ca_number"])
        .sort_values("net_flow")
    )

    combined["label"] = combined["ca_number"].astype(str) + " - " + combined["ca_name"]

    # Static Matplotlib plot
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    colors = ["#238b45" if val > 0 else "#cb181d" for val in combined["net_flow"]]
    bars = ax.barh(
        combined["label"],
        combined["net_flow"],
        color=colors,
        alpha=0.85,
        edgecolor="#252525",
    )
    ax.axvline(0, color="black", linewidth=1.2, linestyle="-")
    ax.set_xlabel(
        "Net Historical Flow I_i = A_i - D_i (Trips)", fontsize=11, fontweight="bold"
    )
    ax.set_title(
        f"Zone Imbalance Proxies: Top {top_n} Potential Surplus vs Deficit Zones\n(Note: Inferred from historical trip flows, not observed inventory)",
        fontsize=11,
        fontweight="bold",
        pad=14,
    )
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    for bar in bars:
        width = bar.get_width()
        ha = "left" if width > 0 else "right"
        offset = 15 if width > 0 else -15
        ax.text(
            width + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{int(width):+d}",
            va="center",
            ha=ha,
            fontsize=8,
        )

    ax.set_xlim(combined["net_flow"].min() * 1.15, combined["net_flow"].max() * 1.25)
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    # Interactive Plotly chart
    pfig = px.bar(
        combined,
        x="net_flow",
        y="label",
        color="flow_category",
        color_discrete_map={
            "potential_surplus": "#238b45",
            "potential_deficit": "#cb181d",
        },
        orientation="h",
        title="Zone Imbalance Proxies: Net Historical Trip Flow (A_i - D_i)",
        labels={
            "net_flow": "Net Flow (Arrivals - Departures)",
            "label": "Community Area",
        },
        template="plotly_white",
    )
    pfig.update_layout(yaxis={"categoryorder": "total ascending"})
    pfig.write_html(str(html_path))

    return png_path, html_path


def plot_top_activity_zones(
    zone_flows_df: pd.DataFrame,
    output_dir: Path,
    top_n: int = 15,
) -> tuple[Path, Path]:
    """Generate stacked bar chart of top active zones by total turnover (A_i + D_i)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "top_activity_zones.png"
    html_path = output_dir / "top_activity_zones.html"

    top_active = (
        zone_flows_df.sort_values("turnover", ascending=False)
        .head(top_n)
        .sort_values("turnover", ascending=True)
    )
    labels = top_active["ca_number"].astype(str) + " - " + top_active["ca_name"]

    # Static Matplotlib plot
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    ax.barh(
        labels,
        top_active["departures"],
        color="#2171b5",
        label="Departures (D_i)",
        alpha=0.85,
    )
    ax.barh(
        labels,
        top_active["arrivals"],
        left=top_active["departures"],
        color="#6baed6",
        label="Arrivals (A_i)",
        alpha=0.85,
    )
    ax.set_xlabel(
        "Total Trip Activity (Turnover A_i + D_i)", fontsize=11, fontweight="bold"
    )
    ax.set_title(
        f"Top {top_n} Busiest Community Areas: Departures vs Arrivals",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.legend(loc="lower right", frameon=True)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    # Interactive Plotly chart
    pfig = go.Figure()
    pfig.add_trace(
        go.Bar(
            y=labels,
            x=top_active["departures"],
            name="Departures (D_i)",
            orientation="h",
            marker_color="#2171b5",
        )
    )
    pfig.add_trace(
        go.Bar(
            y=labels,
            x=top_active["arrivals"],
            name="Arrivals (A_i)",
            orientation="h",
            marker_color="#6baed6",
        )
    )
    pfig.update_layout(
        barmode="stack",
        title=f"Top {top_n} Busiest Community Areas (Turnover A_i + D_i)",
        xaxis={"title": "Trip Volume"},
        template="plotly_white",
    )
    pfig.write_html(str(html_path))

    return png_path, html_path
