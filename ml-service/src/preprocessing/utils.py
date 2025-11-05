"""
Utility functions for geospatial calculations
"""

import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great circle distance between two points using Haversine formula

    Args:
        lat1: Latitude of point 1 (degrees)
        lon1: Longitude of point 1 (degrees)
        lat2: Latitude of point 2 (degrees)
        lon2: Longitude of point 2 (degrees)

    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth radius in km

    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    distance = R * c

    return distance


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate bearing from point 1 to point 2

    Args:
        lat1: Latitude of point 1 (degrees)
        lon1: Longitude of point 1 (degrees)
        lat2: Latitude of point 2 (degrees)
        lon2: Longitude of point 2 (degrees)

    Returns:
        Bearing in degrees (0-360)
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1

    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)

    bearing = np.arctan2(x, y)
    bearing = np.degrees(bearing)
    bearing = (bearing + 360) % 360

    return bearing


def calculate_route_waypoints(start_lat: float, start_lon: float,
                              end_lat: float, end_lon: float,
                              num_points: int = 10) -> List[Dict]:
    """
    Calculate waypoints along great circle route

    Args:
        start_lat: Starting latitude
        start_lon: Starting longitude
        end_lat: Ending latitude
        end_lon: Ending longitude
        num_points: Number of waypoints to generate

    Returns:
        List of waypoint dicts with 'lat' and 'lon' keys
    """
    waypoints = []

    for i in range(num_points):
        fraction = i / (num_points - 1) if num_points > 1 else 0

        # Simple linear interpolation (for more accuracy, use great circle interpolation)
        lat = start_lat + (end_lat - start_lat) * fraction
        lon = start_lon + (end_lon - start_lon) * fraction

        waypoints.append({'lat': lat, 'lon': lon})

    return waypoints


def find_surrounding_points(data: List[Dict], target_time: datetime) -> Tuple[Dict, Dict]:
    """
    Find the two data points surrounding a target timestamp

    Args:
        data: List of dicts with 'timestamp' key
        target_time: Target timestamp

    Returns:
        Tuple of (before_point, after_point)
    """
    before = None
    after = None

    for point in data:
        point_time = point['timestamp']

        if isinstance(point_time, str):
            point_time = datetime.fromisoformat(point_time.replace('Z', '+00:00'))

        if point_time <= target_time:
            before = point
        elif point_time > target_time and after is None:
            after = point
            break

    return before, after


def normalize_angle(angle: float) -> float:
    """
    Normalize angle to 0-360 range

    Args:
        angle: Angle in degrees

    Returns:
        Normalized angle (0-360)
    """
    return angle % 360


def knots_to_kmh(knots: float) -> float:
    """Convert knots to km/h"""
    return knots * 1.852


def kmh_to_knots(kmh: float) -> float:
    """Convert km/h to knots"""
    return kmh / 1.852
