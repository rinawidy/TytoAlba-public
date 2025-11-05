

"""
Data preprocessing pipeline for vessel arrival prediction
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler

from .utils import haversine_distance, calculate_bearing, calculate_route_waypoints, find_surrounding_points


class VoyageDataPreprocessor:
    """
    Handles all data preprocessing for vessel arrival prediction

    Main responsibilities:
    - Fetch AIS and weather data
    - Clean and validate data
    - Engineer features
    - Create sequences for LSTM
    - Normalize data
    """

    def __init__(self, sequence_length: int = 48):
        """
        Initialize preprocessor

        Args:
            sequence_length: Number of timesteps in sequence (default: 48 = 24 hours at 30min intervals)
        """
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()


    def prepare_inference_data(self, vessel_mmsi: str,
                              destination_lat: float,
                              destination_lon: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Main preprocessing pipeline for inference

        Args:
            vessel_mmsi: Vessel identifier (MMSI number)
            destination_lat: Target latitude
            destination_lon: Target longitude

        Returns:
            Tuple of (sequence_data, static_features)
            - sequence_data: [48, 8] Combined trajectory + weather
            - static_features: [10] Vessel and route metadata
        """
        # Fetch data
        ais_data = self.fetch_ais_history(vessel_mmsi, hours=24)

        if not ais_data or len(ais_data) == 0:
            raise ValueError(f"No AIS data found for vessel {vessel_mmsi}")

        current_pos = ais_data[-1]
        weather_data = self.fetch_weather_along_route(
            start_lat=current_pos['latitude'],
            start_lon=current_pos['longitude'],
            end_lat=destination_lat,
            end_lon=destination_lon
        )

        # Clean data
        ais_data = self.remove_outliers(ais_data)
        ais_data = self.interpolate_missing(ais_data)

        # Extract features
        features = self.extract_features(ais_data, weather_data, destination_lat, destination_lon)

        # Create sequences
        sequence_data = self.create_sequence(ais_data, weather_data)

        # Normalize
        sequence_data = self.normalize_sequence(sequence_data)

        # Static features
        static_features = self.create_static_features(features)

        return sequence_data, static_features


    def fetch_ais_history(self, mmsi: str, hours: int = 24) -> List[Dict]:
        """
        Fetch AIS positions from database or API

        Args:
            mmsi: Vessel MMSI identifier
            hours: Number of hours of history to fetch

        Returns:
            List of position dicts
        """
        # TODO: Replace with actual database query or API call
        # This is a placeholder that should be replaced with real data source

        print(f"[INFO] Fetching AIS data for MMSI {mmsi} (last {hours} hours)")

        # Placeholder: Return mock data structure
        # In production, this would query a database like:
        #
        # query = """
        #     SELECT timestamp, latitude, longitude,
        #            speed_over_ground as speed,
        #            course_over_ground as course
        #     FROM ais_positions
        #     WHERE mmsi = %s
        #     AND timestamp >= NOW() - INTERVAL %s HOUR
        #     ORDER BY timestamp ASC
        # """
        # result = database.execute(query, [mmsi, hours])

        raise NotImplementedError(
            "AIS data fetching not implemented. "
            "Please implement database connection or API integration in fetch_ais_history()"
        )


    def fetch_weather_along_route(self, start_lat: float, start_lon: float,
                                  end_lat: float, end_lon: float) -> List[Dict]:
        """
        Fetch weather forecast along planned route

        Args:
            start_lat: Starting latitude
            start_lon: Starting longitude
            end_lat: Ending latitude
            end_lon: Ending longitude

        Returns:
            List of weather data dicts
        """
        # TODO: Replace with actual weather API call
        # This is a placeholder for weather data integration

        print(f"[INFO] Fetching weather data from ({start_lat}, {start_lon}) to ({end_lat}, {end_lon})")

        # Calculate waypoints
        waypoints = calculate_route_waypoints(start_lat, start_lon, end_lat, end_lon, num_points=10)

        # Placeholder: In production, call weather API for each waypoint
        # Example using OpenWeatherMap or marine weather service:
        #
        # weather_forecasts = []
        # for waypoint in waypoints:
        #     forecast = weather_api.get_marine_forecast(
        #         lat=waypoint['lat'],
        #         lon=waypoint['lon'],
        #         hours=48
        #     )
        #     weather_forecasts.append(forecast)

        raise NotImplementedError(
            "Weather data fetching not implemented. "
            "Please implement weather API integration in fetch_weather_along_route()"
        )


    def remove_outliers(self, ais_data: List[Dict]) -> List[Dict]:
        """
        Remove impossible or erroneous values

        Args:
            ais_data: List of AIS position dicts

        Returns:
            Cleaned data
        """
        cleaned = []

        for i, point in enumerate(ais_data):
            if not self._is_valid_speed(point):
                continue

            if i > 0 and not self._is_valid_transition(ais_data[i-1], point):
                continue

            cleaned.append(point)

        return cleaned

    def _is_valid_speed(self, point: Dict) -> bool:
        """
        Check if speed is within valid range for cargo ships

        Args:
            point: AIS data point

        Returns:
            True if speed is valid (0-30 knots)
        """
        speed = point['speed']
        if speed < 0 or speed > 30:
            print(f"[WARNING] Removing outlier: invalid speed {speed}")
            return False
        return True

    def _is_valid_transition(self, prev: Dict, current: Dict) -> bool:
        """
        Check if position jump between consecutive points is plausible

        Args:
            prev: Previous AIS data point
            current: Current AIS data point

        Returns:
            True if transition is valid
        """
        distance = haversine_distance(
            prev['latitude'], prev['longitude'],
            current['latitude'], current['longitude']
        )

        time_diff_hours = self._calculate_time_diff(prev, current)

        if time_diff_hours <= 0:
            return True

        implied_speed_kmh = distance / time_diff_hours
        if implied_speed_kmh > 50:
            print(f"[WARNING] Removing outlier: impossible position jump ({implied_speed_kmh:.1f} km/h)")
            return False

        return True

    def _calculate_time_diff(self, prev: Dict, current: Dict) -> float:
        """
        Calculate time difference between two points in hours

        Args:
            prev: Previous AIS data point
            current: Current AIS data point

        Returns:
            Time difference in hours
        """
        prev_time = self._normalize_timestamp(prev['timestamp'])
        current_time = self._normalize_timestamp(current['timestamp'])

        return (current_time - prev_time).total_seconds() / 3600

    def _normalize_timestamp(self, timestamp) -> datetime:
        """
        Convert timestamp to datetime object

        Args:
            timestamp: String or datetime timestamp

        Returns:
            datetime object
        """
        if isinstance(timestamp, str):
            return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return timestamp


    def interpolate_missing(self, ais_data: List[Dict]) -> List[Dict]:
        """
        Fill gaps in trajectory using linear interpolation

        Args:
            ais_data: List of AIS position dicts

        Returns:
            Interpolated data with regular 30-minute intervals
        """
        if len(ais_data) < 2:
            return ais_data

        # Convert timestamps
        for point in ais_data:
            if isinstance(point['timestamp'], str):
                point['timestamp'] = datetime.fromisoformat(point['timestamp'].replace('Z', '+00:00'))

        # Create regular 30-minute intervals
        start_time = ais_data[0]['timestamp']
        end_time = ais_data[-1]['timestamp']

        regular_intervals = []
        current = start_time

        while current <= end_time:
            regular_intervals.append(current)
            current += timedelta(minutes=30)

        # Interpolate for each regular interval
        interpolated = []

        for target_time in regular_intervals:
            # Find closest points before and after
            before, after = find_surrounding_points(ais_data, target_time)

            if before and after:
                # Linear interpolation
                time_before = before['timestamp']
                time_after = after['timestamp']

                total_duration = (time_after - time_before).total_seconds()
                elapsed = (target_time - time_before).total_seconds()

                ratio = elapsed / total_duration if total_duration > 0 else 0

                interpolated_point = {
                    'timestamp': target_time,
                    'latitude': before['latitude'] + ratio * (after['latitude'] - before['latitude']),
                    'longitude': before['longitude'] + ratio * (after['longitude'] - before['longitude']),
                    'speed': before['speed'] + ratio * (after['speed'] - before['speed']),
                    'course': before['course'] + ratio * (after['course'] - before['course'])
                }
                interpolated.append(interpolated_point)
            elif before:
                # Use the before point
                interpolated.append({**before, 'timestamp': target_time})

        return interpolated


    def extract_features(self, ais_data: List[Dict], weather_data: List[Dict],
                        dest_lat: float, dest_lon: float) -> Dict:
        """
        Calculate derived features from raw data

        Args:
            ais_data: List of AIS positions
            weather_data: List of weather observations
            dest_lat: Destination latitude
            dest_lon: Destination longitude

        Returns:
            Dict of extracted features
        """
        features = {}

        # Current position
        current = ais_data[-1]

        # Distance to destination
        features['distance_remaining'] = haversine_distance(
            current['latitude'], current['longitude'],
            dest_lat, dest_lon
        )

        # Speed statistics
        speeds = [p['speed'] for p in ais_data]
        features['avg_speed'] = np.mean(speeds)
        features['std_speed'] = np.std(speeds)
        features['max_speed'] = np.max(speeds)
        features['min_speed'] = np.min(speeds)

        # Speed changes (acceleration/deceleration patterns)
        speed_changes = [speeds[i] - speeds[i-1] for i in range(1, len(speeds))]
        features['avg_acceleration'] = np.mean(speed_changes) if speed_changes else 0
        features['speed_volatility'] = np.std(speed_changes) if speed_changes else 0

        # Course stability
        courses = [p['course'] for p in ais_data]
        course_changes = [abs(courses[i] - courses[i-1]) for i in range(1, len(courses))]
        features['course_stability'] = 1 / (np.mean(course_changes) + 1) if course_changes else 1

        # Weather features (if available)
        if weather_data:
            wind_speeds = [w.get('wind_speed', 0) for w in weather_data]
            wave_heights = [w.get('wave_height', 0) for w in weather_data]

            features['avg_wind_speed'] = np.mean(wind_speeds)
            features['max_wave_height'] = np.max(wave_heights)

            # Weather severity score (0-1)
            features['weather_severity'] = min(1.0, (
                (features['avg_wind_speed'] / 20) * 0.6 +
                (features['max_wave_height'] / 5) * 0.4
            ))
        else:
            features['avg_wind_speed'] = 0
            features['max_wave_height'] = 0
            features['weather_severity'] = 0

        # Temporal features
        current_time = current['timestamp'] if isinstance(current['timestamp'], datetime) else datetime.now()
        features['hour_of_day'] = current_time.hour
        features['day_of_week'] = current_time.weekday()
        features['month'] = current_time.month

        # Bearing to destination
        features['bearing_to_dest'] = calculate_bearing(
            current['latitude'], current['longitude'],
            dest_lat, dest_lon
        )

        # Course alignment (how well aligned is vessel with destination?)
        course_diff = abs(current['course'] - features['bearing_to_dest'])
        features['course_alignment'] = 1 - min(course_diff, 360 - course_diff) / 180

        return features


    def create_sequence(self, ais_data: List[Dict], weather_data: List[Dict]) -> np.ndarray:
        """
        Combine AIS and weather into sequence matrix

        Args:
            ais_data: List of AIS positions
            weather_data: List of weather observations

        Returns:
            Numpy array of shape [sequence_length, 8]
            Features: [lat, lon, speed, course, wind, wave, current, temp]
        """
        # Take last N points
        ais_seq = ais_data[-self.sequence_length:]

        # Pad if not enough data
        if len(ais_seq) < self.sequence_length:
            padding_needed = self.sequence_length - len(ais_seq)
            ais_seq = [ais_seq[0]] * padding_needed + ais_seq

        # Align weather data (or use defaults if not available)
        if weather_data and len(weather_data) > 0:
            weather_seq = weather_data[-self.sequence_length:]
            if len(weather_seq) < self.sequence_length:
                padding_needed = self.sequence_length - len(weather_seq)
                weather_seq = [weather_seq[0]] * padding_needed + weather_seq
        else:
            # Default weather values
            default_weather = {
                'wind_speed': 0,
                'wave_height': 0,
                'current_speed': 0,
                'temperature': 20
            }
            weather_seq = [default_weather] * self.sequence_length

        # Build sequence matrix
        sequence = []

        for ais, weather in zip(ais_seq, weather_seq):
            row = [
                ais['latitude'],
                ais['longitude'],
                ais['speed'],
                ais['course'],
                weather.get('wind_speed', 0),
                weather.get('wave_height', 0),
                weather.get('current_speed', 0),
                weather.get('temperature', 20)
            ]
            sequence.append(row)

        return np.array(sequence)


    def normalize_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        Normalize sequence to standard scale

        Args:
            sequence: Input sequence [timesteps, features]

        Returns:
            Normalized sequence
        """
        # Reshape for scaler
        original_shape = sequence.shape
        flattened = sequence.reshape(-1, original_shape[-1])

        # Normalize
        normalized = self.scaler.fit_transform(flattened)

        # Reshape back
        return normalized.reshape(original_shape)


    def create_static_features(self, features: Dict) -> np.ndarray:
        """
        Create static feature vector from extracted features

        Args:
            features: Dict of extracted features

        Returns:
            Numpy array of shape [10]
        """
        static = [
            features['distance_remaining'] / 10000,  # Normalize by max expected distance
            features['avg_speed'] / 25,              # Normalize by typical max speed
            features['weather_severity'],            # Already 0-1
            features['course_alignment'],            # Already 0-1
            np.sin(2 * np.pi * features['hour_of_day'] / 24),      # Cyclical time encoding
            np.cos(2 * np.pi * features['hour_of_day'] / 24),
            np.sin(2 * np.pi * features['day_of_week'] / 7),
            np.cos(2 * np.pi * features['day_of_week'] / 7),
            np.sin(2 * np.pi * features['month'] / 12),
            np.cos(2 * np.pi * features['month'] / 12)
        ]

        return np.array(static, dtype=np.float32)
