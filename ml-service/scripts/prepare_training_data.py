"""
Prepare training data from historical voyage records
Converts raw voyage data into ML-ready features and labels
"""
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

def load_historical_data(file_path):
    """Load historical voyage data"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data['data'])

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points (Haversine formula) in nautical miles"""
    from math import radians, cos, sin, asin, sqrt

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    nm = 3440.065 * c  # Nautical miles
    return nm

def prepare_eta_data(df):
    """Prepare data for ETA prediction model"""
    print("\n📊 Preparing ETA prediction data...")

    # Group by ship (MMSI) to create voyage sequences
    eta_features = []
    eta_labels = []

    for mmsi, group in df.groupby('mmsi'):
        group = group.sort_values('timestamp').reset_index(drop=True)

        if len(group) < 2:  # Need at least 2 points
            continue

        # Get destination (last point in the voyage)
        destination_lat = group.iloc[-1]['latitude']
        destination_lon = group.iloc[-1]['longitude']

        # For each point in the voyage, calculate ETA to destination
        for i in range(len(group) - 1):  # Exclude last point (already at destination)
            current = group.iloc[i]

            # Calculate distance to destination
            distance_to_dest = calculate_distance(
                current['latitude'], current['longitude'],
                destination_lat, destination_lon
            )

            # Calculate ETA using speed
            # ETA (hours) = Distance (nm) / Speed (knots)
            speed = current['speed_knots']
            if speed > 0.5:  # Only calculate if ship is moving
                eta_hours = distance_to_dest / speed
            else:
                # If ship is stationary, use average speed of 10 knots for estimation
                eta_hours = distance_to_dest / 10.0

            # Features: lat, lon, speed, course, distance_remaining
            features = [
                current['latitude'],
                current['longitude'],
                speed,
                current['course'],
                distance_to_dest
            ]

            # Label: calculated ETA to destination (hours)
            label = eta_hours

            eta_features.append(features)
            eta_labels.append(label)

    eta_df = pd.DataFrame(eta_features, columns=[
        'latitude', 'longitude', 'speed_knots', 'course', 'distance_nm'
    ])
    eta_df['eta_hours'] = eta_labels

    print(f"  ✓ Created {len(eta_df)} training samples for ETA prediction")
    print(f"  📈 ETA range: {eta_df['eta_hours'].min():.2f} - {eta_df['eta_hours'].max():.2f} hours")
    print(f"  📊 Average ETA: {eta_df['eta_hours'].mean():.2f} hours (std: {eta_df['eta_hours'].std():.2f})")
    return eta_df

def prepare_fuel_data(df):
    """Prepare data for fuel consumption prediction"""
    print("\n⛽ Preparing fuel consumption data...")

    fuel_features = []
    fuel_labels = []

    for mmsi, group in df.groupby('mmsi'):
        group = group.sort_values('timestamp').reset_index(drop=True)

        if len(group) < 2:
            continue

        for i in range(len(group) - 1):
            current = group.iloc[i]
            next_point = group.iloc[i + 1]

            # Calculate distance traveled
            distance = calculate_distance(
                current['latitude'], current['longitude'],
                next_point['latitude'], next_point['longitude']
            )

            # Calculate time elapsed
            current_time = pd.to_datetime(current['timestamp'])
            next_time = pd.to_datetime(next_point['timestamp'])
            time_hours = (next_time - current_time).total_seconds() / 3600

            # Features: speed, distance, course changes
            avg_speed = (current['speed_knots'] + next_point['speed_knots']) / 2
            course_change = abs(next_point['course'] - current['course'])
            if course_change > 180:
                course_change = 360 - course_change

            features = [
                avg_speed,
                distance,
                time_hours,
                course_change
            ]

            # Label: estimated fuel consumption (liters)
            # Rough estimate: ~200 liters/hour at 12 knots for bulk carrier
            # Scale by speed squared (fuel consumption increases exponentially)
            fuel_consumption = time_hours * 200 * (avg_speed / 12.0) ** 2

            fuel_features.append(features)
            fuel_labels.append(fuel_consumption)

    fuel_df = pd.DataFrame(fuel_features, columns=[
        'avg_speed', 'distance_nm', 'time_hours', 'course_change'
    ])
    fuel_df['fuel_liters'] = fuel_labels

    print(f"  ✓ Created {len(fuel_df)} training samples for fuel prediction")
    return fuel_df

def prepare_anomaly_data(df):
    """Prepare data for anomaly detection"""
    print("\n🔍 Preparing anomaly detection data...")

    anomaly_features = []

    for mmsi, group in df.groupby('mmsi'):
        group = group.sort_values('timestamp').reset_index(drop=True)

        for i in range(len(group)):
            current = group.iloc[i]

            # Features: speed, course, position
            features = [
                current['latitude'],
                current['longitude'],
                current['speed_knots'],
                current['course']
            ]

            anomaly_features.append(features)

    anomaly_df = pd.DataFrame(anomaly_features, columns=[
        'latitude', 'longitude', 'speed_knots', 'course'
    ])

    # For anomaly detection, we'll use all data as "normal"
    # In production, you'd label actual anomalies
    anomaly_df['is_anomaly'] = 0  # 0 = normal, 1 = anomaly

    print(f"  ✓ Created {len(anomaly_df)} samples for anomaly detection")
    return anomaly_df

def prepare_route_data(df):
    """Prepare data for route optimization"""
    print("\n🗺️  Preparing route optimization data...")

    route_features = []
    route_labels = []

    for mmsi, group in df.groupby('mmsi'):
        group = group.sort_values('timestamp').reset_index(drop=True)

        if len(group) < 3:
            continue

        # For each voyage, create features for route waypoints
        for i in range(len(group) - 2):
            p1 = group.iloc[i]
            p2 = group.iloc[i + 1]
            p3 = group.iloc[i + 2]

            # Features: current position, destination, current heading
            features = [
                p1['latitude'],
                p1['longitude'],
                p3['latitude'],  # destination
                p3['longitude'],
                p1['course']
            ]

            # Label: next waypoint (optimal route point)
            label = [p2['latitude'], p2['longitude']]

            route_features.append(features)
            route_labels.append(label)

    route_df = pd.DataFrame(route_features, columns=[
        'start_lat', 'start_lon', 'dest_lat', 'dest_lon', 'heading'
    ])
    route_df['next_lat'] = [l[0] for l in route_labels]
    route_df['next_lon'] = [l[1] for l in route_labels]

    print(f"  ✓ Created {len(route_df)} training samples for route optimization")
    return route_df

def main():
    """Main function to prepare all training data"""
    print("="*60)
    print("🔧 PREPARING TRAINING DATA")
    print("="*60)

    # Load historical data
    data_path = Path(__file__).parent.parent.parent / 'backend' / 'data' / 'historical_voyages_15min.json'
    print(f"\n📂 Loading data from: {data_path}")
    df = load_historical_data(data_path)
    print(f"  ✓ Loaded {len(df)} historical records for {df['mmsi'].nunique()} ships")

    # Prepare data for each model
    eta_data = prepare_eta_data(df)
    fuel_data = prepare_fuel_data(df)
    anomaly_data = prepare_anomaly_data(df)
    route_data = prepare_route_data(df)

    # Save prepared data
    output_dir = Path(__file__).parent.parent / 'data' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)

    eta_data.to_csv(output_dir / 'eta_training_data.csv', index=False)
    fuel_data.to_csv(output_dir / 'fuel_training_data.csv', index=False)
    anomaly_data.to_csv(output_dir / 'anomaly_training_data.csv', index=False)
    route_data.to_csv(output_dir / 'route_training_data.csv', index=False)

    print("\n" + "="*60)
    print("✅ TRAINING DATA PREPARED")
    print("="*60)
    print(f"📁 Output directory: {output_dir}")
    print(f"  • eta_training_data.csv ({len(eta_data)} samples)")
    print(f"  • fuel_training_data.csv ({len(fuel_data)} samples)")
    print(f"  • anomaly_training_data.csv ({len(anomaly_data)} samples)")
    print(f"  • route_training_data.csv ({len(route_data)} samples)")
    print("\n🎯 Next step: Run training scripts for each model")

if __name__ == '__main__':
    main()
