"""Spatial analysis, zone aggregation, flow imbalance, and distance calculations."""

import math
from typing import Any

import numpy as np
import pandas as pd

EARTH_RADIUS_METERS: float = 6_371_008.8  # WGS-84 mean earth radius
METERS_PER_DEGREE_LAT: float = 111_139.0


def derive_zone_centroids(df: pd.DataFrame) -> pd.DataFrame:
    """Derive observed geographic centroids for each Community Area zone.

    Extracts ground-truth polygon centroids provided in the dataset.
    Never invents coordinates.
    """
    if df.empty:
        return pd.DataFrame(
            columns=["ca_number", "ca_name", "latitude", "longitude", "location_wkt"]
        )

    # Aggregate observed coordinates by zone
    start_obs = (
        df.groupby(["start_community_area_number", "start_community_area_name"])[
            [
                "start_centroid_latitude",
                "start_centroid_longitude",
                "start_centroid_location",
            ]
        ]
        .first()
        .reset_index()
    )

    start_obs.columns = [
        "ca_number",
        "ca_name",
        "latitude",
        "longitude",
        "location_wkt",
    ]
    start_obs["ca_number"] = start_obs["ca_number"].astype(int)
    start_obs = start_obs.sort_values("ca_number").reset_index(drop=True)
    return start_obs


def calculate_zone_flows(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate departures (D_i), arrivals (A_i), and net flow (I_i = A_i - D_i) per zone.

    Classifies zones:
      - Potential surplus zone (I_i > 0): More trips arrive than depart historically.
      - Potential deficit zone (I_i < 0): More trips depart than arrive historically.
      - Balanced zone (I_i == 0): Arrivals equal departures.

    NOTE: Net-flow surplus/deficit classifications are proxies inferred from
    historical trip origin-destination flow accumulation, NOT observed vehicle inventory.
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "ca_number",
                "ca_name",
                "departures",
                "arrivals",
                "net_flow",
                "turnover",
                "imbalance_ratio",
                "flow_category",
                "latitude",
                "longitude",
            ]
        )

    # Departures D_i (from start CA)
    dep_counts = (
        df.groupby(["start_community_area_number", "start_community_area_name"])
        .size()
        .rename("departures")
    )
    dep_counts.index.names = ["ca_number", "ca_name"]

    # Arrivals A_i (to end CA)
    arr_counts = (
        df.groupby(["end_community_area_number", "end_community_area_name"])
        .size()
        .rename("arrivals")
    )
    arr_counts.index.names = ["ca_number", "ca_name"]

    # Outer join to capture any zone with either departures or arrivals
    flows = (
        pd.DataFrame({"departures": dep_counts})
        .join(arr_counts, how="outer")
        .fillna(0)
        .astype(int)
    )

    flows["net_flow"] = flows["arrivals"] - flows["departures"]
    flows["turnover"] = flows["arrivals"] + flows["departures"]

    # Imbalance ratio: net_flow / turnover, bounded in [-1.0, 1.0]
    flows["imbalance_ratio"] = np.where(
        flows["turnover"] > 0,
        flows["net_flow"] / flows["turnover"],
        0.0,
    )

    # Classification
    conditions = [
        flows["net_flow"] > 0,
        flows["net_flow"] < 0,
        flows["net_flow"] == 0,
    ]
    choices = ["potential_surplus", "potential_deficit", "balanced"]
    flows["flow_category"] = np.select(conditions, choices, default="balanced")

    flows = flows.reset_index()
    flows["ca_number"] = flows["ca_number"].astype(int)

    # Attach observed centroid coordinates
    centroids = derive_zone_centroids(df)
    flows = flows.merge(
        centroids[["ca_number", "latitude", "longitude"]],
        on="ca_number",
        how="left",
    )

    flows = flows.sort_values("ca_number").reset_index(drop=True)
    return flows


def calculate_od_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate complete Origin-Destination trip flow matrix between zones."""
    if df.empty:
        return pd.DataFrame()

    od = pd.crosstab(
        df["start_community_area_number"].astype(int),
        df["end_community_area_number"].astype(int),
        rownames=["origin_ca"],
        colnames=["dest_ca"],
    )
    return od


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle straight-line distance in meters between two lat/lon pairs."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_METERS * c


def manhattan_grid_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Calculate Manhattan (L1 grid) distance in meters.

    Chicago follows a regular orthogonal grid aligned North-South and East-West.
    Manhattan distance provides a realistic street-network lower bound for road transit.
    """
    mean_lat = math.radians((lat1 + lat2) / 2.0)
    meters_per_degree_lon = METERS_PER_DEGREE_LAT * math.cos(mean_lat)

    dy = abs(lat2 - lat1) * METERS_PER_DEGREE_LAT
    dx = abs(lon2 - lon1) * meters_per_degree_lon
    return dy + dx


def compute_zone_distance_matrix(
    centroids_df: pd.DataFrame,
    method: str = "manhattan",
) -> pd.DataFrame:
    """Compute pairwise distance matrix in meters between all zone centroids.

    Methods:
      - 'haversine': Great-circle straight-line distance.
      - 'manhattan': L1 grid distance matching Chicago's orthogonal street network.
    """
    if centroids_df.empty:
        return pd.DataFrame()

    ca_list = centroids_df["ca_number"].tolist()
    coords = {
        row["ca_number"]: (row["latitude"], row["longitude"])
        for _, row in centroids_df.iterrows()
    }

    n = len(ca_list)
    dist_mat = np.zeros((n, n), dtype=float)

    for i, ca_i in enumerate(ca_list):
        lat_i, lon_i = coords[ca_i]
        for j, ca_j in enumerate(ca_list):
            if i == j:
                dist_mat[i, j] = 0.0
            else:
                lat_j, lon_j = coords[ca_j]
                if method.lower() == "haversine":
                    dist_mat[i, j] = haversine_distance(lat_i, lon_i, lat_j, lon_j)
                else:
                    dist_mat[i, j] = manhattan_grid_distance(lat_i, lon_i, lat_j, lon_j)

    return pd.DataFrame(dist_mat, index=ca_list, columns=ca_list)


def summarize_spatial_coverage(
    cleaned_df: pd.DataFrame,
    raw_valid_trip_count: int = 551_800,
) -> dict[str, Any]:
    """Compute geographic coverage and usability statistics."""
    total_cleaned = len(cleaned_df)
    unique_start_zones = int(cleaned_df["start_community_area_number"].nunique())
    unique_end_zones = int(cleaned_df["end_community_area_number"].nunique())

    all_zones = set(cleaned_df["start_community_area_number"]).union(
        set(cleaned_df["end_community_area_number"])
    )
    total_usable_zones = len(all_zones)

    intra_zone_count = int(
        (
            cleaned_df["start_community_area_number"]
            == cleaned_df["end_community_area_number"]
        ).sum()
    )
    inter_zone_count = total_cleaned - intra_zone_count

    return {
        "total_cleaned_trips": total_cleaned,
        "raw_valid_trips_reference": raw_valid_trip_count,
        "usable_spatial_records": total_cleaned,
        "spatial_coverage_cleaned_pct": 100.0,
        "spatial_coverage_vs_raw_valid_pct": round(
            (total_cleaned / raw_valid_trip_count) * 100, 3
        ),
        "total_chicago_zones": 77,
        "usable_geographic_zones": total_usable_zones,
        "unique_origin_zones": unique_start_zones,
        "unique_destination_zones": unique_end_zones,
        "intra_zone_trips": intra_zone_count,
        "intra_zone_pct": round((intra_zone_count / total_cleaned) * 100, 2),
        "inter_zone_trips": inter_zone_count,
        "inter_zone_pct": round((inter_zone_count / total_cleaned) * 100, 2),
    }
