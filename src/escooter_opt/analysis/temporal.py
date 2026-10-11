"""Temporal trip pattern analysis: hour of day, day of week, and date trends."""

from typing import Any

import pandas as pd

DAY_OF_WEEK_ORDER: list[str] = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def calculate_hourly_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze trip volume and characteristics across 24 hours of the day (0-23)."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "hour",
                "trip_count",
                "volume_pct",
                "mean_distance_m",
                "mean_duration_s",
                "mean_speed_mph",
            ]
        )

    working = df.copy()
    working["hour"] = working["start_time"].dt.hour

    grouped = (
        working.groupby("hour")
        .agg(
            trip_count=("trip_id", "count"),
            mean_distance_m=("trip_distance_m", "mean"),
            mean_duration_s=("trip_duration_s", "mean"),
            mean_speed_mph=("speed_mph", "mean"),
        )
        .reset_index()
    )

    # Ensure all 24 hours (0..23) are represented
    full_hours = pd.DataFrame({"hour": range(24)})
    result = full_hours.merge(grouped, on="hour", how="left").fillna(
        {
            "trip_count": 0,
            "mean_distance_m": 0.0,
            "mean_duration_s": 0.0,
            "mean_speed_mph": 0.0,
        }
    )

    total_trips = len(df)
    result["trip_count"] = result["trip_count"].astype(int)
    result["volume_pct"] = (result["trip_count"] / total_trips) * 100.0

    return result.sort_values("hour").reset_index(drop=True)


def calculate_day_of_week_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze trip volume across days of the week (Monday-Sunday)."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "day_of_week",
                "day_index",
                "trip_count",
                "volume_pct",
                "mean_distance_m",
                "mean_duration_s",
            ]
        )

    working = df.copy()
    working["day_of_week"] = working["start_time"].dt.day_name()
    working["day_index"] = working["start_time"].dt.dayofweek

    grouped = (
        working.groupby(["day_of_week", "day_index"])
        .agg(
            trip_count=("trip_id", "count"),
            mean_distance_m=("trip_distance_m", "mean"),
            mean_duration_s=("trip_duration_s", "mean"),
            mean_speed_mph=("speed_mph", "mean"),
        )
        .reset_index()
    )

    total_trips = len(df)
    grouped["volume_pct"] = (grouped["trip_count"] / total_trips) * 100.0

    # Sort canonically Monday (0) to Sunday (6)
    grouped["day_order"] = grouped["day_of_week"].map(
        lambda d: DAY_OF_WEEK_ORDER.index(d)
    )
    result = (
        grouped.sort_values("day_order")
        .drop(columns=["day_order"])
        .reset_index(drop=True)
    )
    return result


def calculate_daily_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze daily trip volume over time, including rolling averages."""
    if df.empty:
        return pd.DataFrame(
            columns=["date", "day_of_week", "trip_count", "rolling_7d_mean"]
        )

    working = df.copy()
    working["date"] = working["start_time"].dt.date

    daily = (
        working.groupby("date")
        .agg(
            trip_count=("trip_id", "count"),
            mean_distance_m=("trip_distance_m", "mean"),
            mean_duration_s=("trip_duration_s", "mean"),
        )
        .reset_index()
    )

    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["day_of_week"] = daily["date"].dt.day_name()
    daily["rolling_7d_mean"] = (
        daily["trip_count"].rolling(window=7, min_periods=1).mean().round(1)
    )

    return daily


def summarize_temporal_patterns(df: pd.DataFrame) -> dict[str, Any]:
    """Compute high-level summary metrics of temporal patterns."""
    if df.empty:
        return {}

    hourly = calculate_hourly_volume(df)
    dow = calculate_day_of_week_volume(df)
    daily = calculate_daily_volume(df)

    start_date = df["start_time"].min()
    end_date = df["start_time"].max()
    total_days = int((end_date.date() - start_date.date()).days) + 1

    peak_hour_row = hourly.loc[hourly["trip_count"].idxmax()]
    trough_hour_row = hourly.loc[hourly["trip_count"].idxmin()]

    weekend_mask = df["start_time"].dt.dayofweek.isin([5, 6])  # Sat, Sun
    weekend_trips = int(weekend_mask.sum())
    weekday_trips = len(df) - weekend_trips

    peak_date_row = daily.loc[daily["trip_count"].idxmax()]
    lowest_date_row = daily.loc[daily["trip_count"].idxmin()]

    return {
        "start_date": str(start_date.date()),
        "end_date": str(end_date.date()),
        "total_calendar_days": total_days,
        "mean_daily_trips": round(len(df) / total_days, 1),
        "peak_hour": int(peak_hour_row["hour"]),
        "peak_hour_volume": int(peak_hour_row["trip_count"]),
        "peak_hour_pct": round(float(peak_hour_row["volume_pct"]), 2),
        "trough_hour": int(trough_hour_row["hour"]),
        "trough_hour_volume": int(trough_hour_row["trip_count"]),
        "weekday_trips": weekday_trips,
        "weekday_pct": round((weekday_trips / len(df)) * 100.0, 2),
        "weekend_trips": weekend_trips,
        "weekend_pct": round((weekend_trips / len(df)) * 100.0, 2),
        "peak_day_of_week": str(dow.loc[dow["trip_count"].idxmax()]["day_of_week"]),
        "peak_calendar_date": str(peak_date_row["date"].date()),
        "peak_calendar_date_trips": int(peak_date_row["trip_count"]),
        "lowest_calendar_date": str(lowest_date_row["date"].date()),
        "lowest_calendar_date_trips": int(lowest_date_row["trip_count"]),
    }
