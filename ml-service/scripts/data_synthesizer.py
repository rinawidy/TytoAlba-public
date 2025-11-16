"""
Data Synthesizer for Ship Positions
Fills gaps in historical data with interpolated/extrapolated positions
"""
import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict
import math

HISTORICAL_DATA_FILE = Path(__file__).parent.parent.parent / 'backend' / 'data' / 'historical_voyages_15min.json'


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in kilometers"""
    R = 6371  # Earth radius in km

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return R * c


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate bearing between two points in degrees"""
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lon = math.radians(lon2 - lon1)

    x = math.sin(delta_lon) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon)

    bearing = math.atan2(x, y)
    bearing_deg = math.degrees(bearing)

    return (bearing_deg + 360) % 360


def interpolate_position(record1: Dict, record2: Dict, target_time: datetime) -> Dict:
    """
    Linearly interpolate position between two records

    Args:
        record1: Earlier record
        record2: Later record
        target_time: Time to interpolate for

    Returns:
        Interpolated record
    """
    time1 = datetime.strptime(record1['timestamp'], "%Y-%m-%d %H:%M:%S")
    time2 = datetime.strptime(record2['timestamp'], "%Y-%m-%d %H:%M:%S")

    # Calculate interpolation ratio
    total_duration = (time2 - time1).total_seconds()
    elapsed = (target_time - time1).total_seconds()

    if total_duration == 0:
        ratio = 0
    else:
        ratio = elapsed / total_duration

    # Interpolate position
    lat = record1['latitude'] + ratio * (record2['latitude'] - record1['latitude'])
    lon = record1['longitude'] + ratio * (record2['longitude'] - record1['longitude'])

    # Interpolate speed
    speed = record1['speed_knots'] + ratio * (record2['speed_knots'] - record1['speed_knots'])

    # Calculate course based on movement direction
    course = calculate_bearing(record1['latitude'], record1['longitude'], lat, lon)

    # Create synthetic record
    synthetic_record = {
        "mmsi": record1.get('mmsi', ''),
        "vessel_name": record1.get('vessel_name', ''),
        "latitude": round(lat, 6),
        "longitude": round(lon, 6),
        "speed_knots": round(speed, 2),
        "course": int(course),
        "timestamp": target_time.strftime("%Y-%m-%d %H:%M:%S"),
        "last_port": record1.get('last_port', ''),
        "destination": record1.get('destination', ''),
        "eta": record1.get('eta', ''),
        "synthetic": True  # Mark as synthesized data
    }

    return synthetic_record


def find_gaps_for_vessel(records: List[Dict], mmsi: str, interval_minutes: int = 15) -> List[tuple]:
    """
    Find time gaps in vessel data

    Args:
        records: All historical records
        mmsi: Vessel MMSI to check
        interval_minutes: Expected interval between records

    Returns:
        List of (start_record, end_record, missing_timestamps) tuples
    """
    # Filter records for this vessel (only real data, not already synthetic)
    vessel_records = [r for r in records if r['mmsi'] == mmsi and not r.get('synthetic', False)]

    if len(vessel_records) < 2:
        return []

    # Sort by timestamp
    vessel_records.sort(key=lambda x: datetime.strptime(x['timestamp'], "%Y-%m-%d %H:%M:%S"))

    gaps = []
    interval_delta = timedelta(minutes=interval_minutes)

    for i in range(len(vessel_records) - 1):
        time1 = datetime.strptime(vessel_records[i]['timestamp'], "%Y-%m-%d %H:%M:%S")
        time2 = datetime.strptime(vessel_records[i+1]['timestamp'], "%Y-%m-%d %H:%M:%S")

        time_diff = time2 - time1

        # Check if gap is larger than expected interval
        if time_diff > interval_delta * 1.5:  # Allow 50% tolerance
            # Calculate missing timestamps
            missing_times = []
            current_time = time1 + interval_delta

            while current_time < time2:
                missing_times.append(current_time)
                current_time += interval_delta

            if missing_times:
                gaps.append((vessel_records[i], vessel_records[i+1], missing_times))

    return gaps


def synthesize_missing_data():
    """
    Main function to synthesize missing data for all vessels
    """
    print("=" * 60)
    print(f"🔬 DATA SYNTHESIS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Load historical data
    if not HISTORICAL_DATA_FILE.exists():
        print("❌ Historical data file not found")
        return

    with open(HISTORICAL_DATA_FILE, 'r') as f:
        historical = json.load(f)

    print(f"\n📊 Current dataset:")
    print(f"   • Total records: {len(historical['data'])}")
    print(f"   • Total vessels: {historical['metadata']['total_ships']}")

    # Get unique vessels
    vessels = list(set(r['mmsi'] for r in historical['data']))
    print(f"\n🔍 Checking for gaps in {len(vessels)} vessels...")

    synthesized_records = []

    for mmsi in vessels:
        gaps = find_gaps_for_vessel(historical['data'], mmsi, interval_minutes=15)

        if gaps:
            vessel_name = next((r.get('vessel_name', 'Unknown') for r in historical['data'] if r['mmsi'] == mmsi), 'Unknown')
            print(f"\n⚠️  {vessel_name} (MMSI: {mmsi}): Found {len(gaps)} gap(s)")

            for gap_idx, (record1, record2, missing_times) in enumerate(gaps):
                time1 = record1['timestamp']
                time2 = record2['timestamp']
                print(f"   Gap {gap_idx + 1}: {time1} → {time2} ({len(missing_times)} missing intervals)")

                # Interpolate missing positions
                for missing_time in missing_times:
                    synthetic_record = interpolate_position(record1, record2, missing_time)
                    synthesized_records.append(synthetic_record)

    if synthesized_records:
        # Add synthesized records to dataset
        old_count = len(historical['data'])
        historical['data'].extend(synthesized_records)

        # Re-sort by timestamp
        historical['data'].sort(key=lambda x: datetime.strptime(x['timestamp'], "%Y-%m-%d %H:%M:%S"))

        # Update metadata
        new_count = len(historical['data'])
        historical['metadata']['total_records'] = new_count
        historical['metadata']['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Add synthesis metadata
        synthetic_count = len([r for r in historical['data'] if r.get('synthetic', False)])
        historical['metadata']['synthetic_records'] = synthetic_count
        historical['metadata']['real_records'] = new_count - synthetic_count

        # Save updated data
        with open(HISTORICAL_DATA_FILE, 'w') as f:
            json.dump(historical, f, indent=2)

        print(f"\n✅ Synthesis complete:")
        print(f"   • Synthesized: {len(synthesized_records)} new records")
        print(f"   • Total records: {old_count} → {new_count}")
        print(f"   • Real data: {new_count - synthetic_count} ({(new_count - synthetic_count) / new_count * 100:.1f}%)")
        print(f"   • Synthetic: {synthetic_count} ({synthetic_count / new_count * 100:.1f}%)")
    else:
        print("\n✅ No gaps found. Data is complete!")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    synthesize_missing_data()
