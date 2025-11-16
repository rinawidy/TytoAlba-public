"""
Fetch PLN Ship Tracking API data and append to historical dataset
Runs every 15 minutes via cron job
"""
import requests
import json
from datetime import datetime
from pathlib import Path

# Configuration
PLN_API_URL = "https://shiptracking.plnbag.co.id/api/vessel-position"
PLN_API_KEY = "9a710fe5-a3ef-4fbd-9540-6f1af31573df"
HISTORICAL_DATA_FILE = Path(__file__).parent.parent.parent / 'backend' / 'data' / 'historical_voyages_15min.json'
BACKUP_DIR = Path(__file__).parent.parent / 'data' / 'snapshots'

def fetch_pln_api():
    """Fetch current vessel positions from PLN API"""
    headers = {
        'X-API-Key': PLN_API_KEY,
        'Content-Type': 'application/json'
    }

    try:
        response = requests.get(PLN_API_URL, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        if 'data' in data and isinstance(data['data'], list):
            return data['data']
        else:
            print(f"⚠️  Unexpected API response format: {data}")
            return []

    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching PLN API: {e}")
        return []

def load_historical_data():
    """Load existing historical voyage data"""
    if not HISTORICAL_DATA_FILE.exists():
        print(f"⚠️  Historical data file not found: {HISTORICAL_DATA_FILE}")
        return None

    with open(HISTORICAL_DATA_FILE, 'r') as f:
        return json.load(f)

def save_snapshot(vessels):
    """Save snapshot of current data"""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_file = BACKUP_DIR / f"pln_snapshot_{timestamp}.json"

    snapshot_data = {
        "timestamp": datetime.now().isoformat(),
        "total_vessels": len(vessels),
        "data": vessels
    }

    with open(snapshot_file, 'w') as f:
        json.dump(snapshot_data, f, indent=2)

    print(f"📸 Snapshot saved: {snapshot_file}")
    return snapshot_file

def convert_to_historical_format(vessels):
    """Convert PLN API format to historical voyage format"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    records = []

    for vessel in vessels:
        # Skip vessels without position data
        lat = vessel.get('lat')
        lon = vessel.get('lon')

        if not lat or not lon:
            continue

        # Convert string coordinates to float
        try:
            latitude = float(lat)
            longitude = float(lon)
        except (ValueError, TypeError):
            continue

        # Create historical record
        record = {
            "mmsi": vessel.get('mmsi', ''),
            "vessel_name": vessel.get('shipname', ''),
            "latitude": latitude,
            "longitude": longitude,
            "speed_knots": float(vessel.get('speed', 0)),
            "course": int(vessel.get('course', 0)),
            "timestamp": current_time,
            "last_port": vessel.get('last_port', ''),
            "destination": vessel.get('destination', ''),
            "eta": vessel.get('eta', '')
        }

        records.append(record)

    return records

def append_to_historical_data(new_records):
    """Append new records to historical dataset"""
    # Load existing data
    historical = load_historical_data()
    if not historical:
        print("❌ Cannot load historical data. Creating new file...")
        historical = {
            "metadata": {
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "interval_minutes": 15,
                "total_records": 0,
                "total_ships": 0,
                "time_span_start": "",
                "time_span_end": ""
            },
            "data": []
        }

    # Get current counts
    old_count = len(historical['data'])
    old_ships = len(set(r['mmsi'] for r in historical['data']))

    # Append new records
    historical['data'].extend(new_records)

    # Update metadata
    new_count = len(historical['data'])
    new_ships = len(set(r['mmsi'] for r in historical['data']))

    # Update time span
    if historical['data']:
        timestamps = [r['timestamp'] for r in historical['data']]
        historical['metadata']['time_span_start'] = min(timestamps)
        historical['metadata']['time_span_end'] = max(timestamps)

    historical['metadata']['total_records'] = new_count
    historical['metadata']['total_ships'] = new_ships
    historical['metadata']['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save updated data
    with open(HISTORICAL_DATA_FILE, 'w') as f:
        json.dump(historical, f, indent=2)

    print(f"✅ Updated historical data:")
    print(f"   • Added: {len(new_records)} new records")
    print(f"   • Total records: {old_count} → {new_count}")
    print(f"   • Total ships: {old_ships} → {new_ships}")
    print(f"   • Time span: {historical['metadata']['time_span_start']} to {historical['metadata']['time_span_end']}")

def main():
    """Main function to fetch and append data"""
    print("=" * 60)
    print(f"🔄 PLN DATA FETCH - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Fetch current data from PLN API
    print("\n📡 Fetching data from PLN API...")
    vessels = fetch_pln_api()

    if not vessels:
        print("❌ No vessel data received. Exiting.")
        return

    print(f"✅ Received {len(vessels)} vessels")

    # Save snapshot
    save_snapshot(vessels)

    # Convert to historical format
    print("\n🔄 Converting to historical format...")
    records = convert_to_historical_format(vessels)
    print(f"✅ Converted {len(records)} records")

    # Append to historical data
    print("\n💾 Appending to historical dataset...")
    append_to_historical_data(records)

    print("\n" + "=" * 60)
    print("✅ DATA FETCH COMPLETE")
    print("=" * 60)

if __name__ == '__main__':
    main()
