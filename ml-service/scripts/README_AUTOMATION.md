# Automated Data Collection & Synthesis

## Overview

This system automatically:
1. **Fetches** real ship positions from PLN API every 15 minutes
2. **Synthesizes** missing data points to fill gaps (interpolation)
3. **Runs** only during working hours (10:30 - 16:00 WIB)

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  CRON (Every 15 minutes)                                │
│  */15 * * * * cron_wrapper.sh                           │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  cron_wrapper.sh     │ ← Time window check (10:30-16:00)
        └──────────┬───────────┘
                   │
                   ├─► Step 1: fetch_pln_data.py
                   │   • Fetch from PLN API
                   │   • Convert format
                   │   • Append to historical_voyages_15min.json
                   │   • Save snapshot
                   │
                   └─► Step 2: data_synthesizer.py
                       • Find gaps in data
                       • Interpolate missing positions
                       • Mark synthetic records
                       • Update historical file
```

## Files

| File | Purpose |
|------|---------|
| `cron_wrapper.sh` | Main cron wrapper (time window check) |
| `fetch_pln_data.py` | Fetch real data from PLN API |
| `data_synthesizer.py` | Fill gaps with interpolated data |
| `automated_data_collection.py` | Combined pipeline (optional standalone) |

## Setup

### Already Configured!

Your cron job is already set up:
```bash
*/15 * * * * /mnt/c/Users/angga.suryabrata/VisCode/TytoAlba/ml-service/scripts/cron_wrapper.sh >> /tmp/tytoalba/logs/pln-fetch.log 2>&1
```

### Verify Cron

```bash
# Check cron is running
crontab -l

# Check logs
tail -f /tmp/tytoalba/logs/pln-fetch.log
```

## Manual Testing

### Test Full Pipeline
```bash
cd /mnt/c/Users/angga.suryabrata/VisCode/TytoAlba/ml-service/scripts
./cron_wrapper.sh
```

### Test Individual Scripts

**Fetch only:**
```bash
cd /mnt/c/Users/angga.suryabrata/VisCode/TytoAlba/ml-service
source venv/bin/activate
python scripts/fetch_pln_data.py
```

**Synthesis only:**
```bash
cd /mnt/c/Users/angga.suryabrata/VisCode/TytoAlba/ml-service
source venv/bin/activate
python scripts/data_synthesizer.py
```

## Data Format

### Historical Data Structure

```json
{
  "metadata": {
    "generated_at": "2025-11-14 10:30:00",
    "interval_minutes": 15,
    "total_records": 1234,
    "total_ships": 15,
    "real_records": 1100,
    "synthetic_records": 134,
    "time_span_start": "2025-11-14 10:30:00",
    "time_span_end": "2025-11-14 15:45:00"
  },
  "data": [
    {
      "mmsi": "525012815",
      "vessel_name": "TB SAHABAT 3",
      "latitude": -6.123456,
      "longitude": 106.789012,
      "speed_knots": 8.5,
      "course": 145,
      "timestamp": "2025-11-14 10:30:00",
      "last_port": "Tanjung Priok",
      "destination": "Labuan Bajo",
      "eta": "2025-11-15 08:00:00"
      // No "synthetic" field = REAL data
    },
    {
      "mmsi": "525012815",
      "vessel_name": "TB SAHABAT 3",
      "latitude": -6.234567,
      "longitude": 106.890123,
      "speed_knots": 8.3,
      "course": 147,
      "timestamp": "2025-11-14 10:45:00",
      "last_port": "Tanjung Priok",
      "destination": "Labuan Bajo",
      "eta": "2025-11-15 08:00:00",
      "synthetic": true  // ← SYNTHESIZED (interpolated)
    }
  ]
}
```

## How Synthesis Works

### Interpolation Algorithm

For a missing timestamp between two real records:

```python
# Example: Missing 10:45 between 10:30 and 11:00
record_1030 = {"lat": -6.0, "lon": 106.0, "time": "10:30"}
record_1100 = {"lat": -6.5, "lon": 106.5, "time": "11:00"}

# Calculate ratio
elapsed = (10:45 - 10:30) = 15 minutes
total = (11:00 - 10:30) = 30 minutes
ratio = 15 / 30 = 0.5

# Interpolate position
lat_1045 = -6.0 + 0.5 * (-6.5 - (-6.0)) = -6.25
lon_1045 = 106.0 + 0.5 * (106.5 - 106.0) = 106.25

# Interpolate speed
speed_1045 = speed_1030 + 0.5 * (speed_1100 - speed_1030)

# Calculate course (bearing between points)
course_1045 = calculate_bearing(lat_1030, lon_1030, lat_1045, lon_1045)
```

### Gap Detection

- **Expected interval:** 15 minutes
- **Gap threshold:** 22.5 minutes (15 × 1.5 tolerance)
- **Action:** If gap > 22.5 minutes, synthesize missing points

## Monitoring

### Check if Pipeline is Running

```bash
# View recent logs
tail -n 50 /tmp/tytoalba/logs/pln-fetch.log

# Monitor in real-time
tail -f /tmp/tytoalba/logs/pln-fetch.log

# Check data file
ls -lh /mnt/c/Users/angga.suryabrata/VisCode/TytoAlba/backend/data/historical_voyages_15min.json
```

### Expected Log Output

```
[2025-11-14 10:30:01] ✅ Within time window. Running data collection pipeline...
[2025-11-14 10:30:01] 📡 Fetching PLN API data...
=============================================================
🔄 PLN DATA FETCH - 2025-11-14 10:30:01
=============================================================

📡 Fetching data from PLN API...
✅ Received 15 vessels
📸 Snapshot saved: data/snapshots/pln_snapshot_20251114_103001.json

🔄 Converting to historical format...
✅ Converted 15 records

💾 Appending to historical dataset...
✅ Updated historical data:
   • Added: 15 new records
   • Total records: 420 → 435
   • Total ships: 15 → 15

[2025-11-14 10:30:02] ✅ Data fetch successful
[2025-11-14 10:30:02] 🔬 Synthesizing missing data...
=============================================================
🔬 DATA SYNTHESIS - 2025-11-14 10:30:02
=============================================================

📊 Current dataset:
   • Total records: 435
   • Total vessels: 15

🔍 Checking for gaps in 15 vessels...

⚠️  TB SAHABAT 3 (MMSI: 525012815): Found 1 gap(s)
   Gap 1: 2025-11-14 09:30:00 → 2025-11-14 10:00:00 (1 missing intervals)

✅ Synthesis complete:
   • Synthesized: 1 new records
   • Total records: 435 → 436
   • Real data: 435 (99.8%)
   • Synthetic: 1 (0.2%)

[2025-11-14 10:30:03] ✅ Data synthesis complete
[2025-11-14 10:30:03] ✅ Pipeline complete
```

## Data Quality Metrics

Track these in metadata:
- **Real records:** Data from PLN API
- **Synthetic records:** Interpolated data
- **Synthetic ratio:** Should be < 20% ideally

## Troubleshooting

### Cron Not Running

```bash
# Check cron service
sudo service cron status

# Restart cron
sudo service cron restart

# Check cron logs
grep CRON /var/log/syslog
```

### API Fetch Failing

```bash
# Test API directly
curl -H "X-API-Key: 9a710fe5-a3ef-4fbd-9540-6f1af31573df" \
  https://shiptracking.plnbag.co.id/api/vessel-position
```

### Synthesis Errors

```bash
# Check historical data file exists
ls -la /mnt/c/Users/angga.suryabrata/VisCode/TytoAlba/backend/data/historical_voyages_15min.json

# Validate JSON
python -m json.tool backend/data/historical_voyages_15min.json > /dev/null
```

## Backup Strategy

- **Snapshots:** Every fetch creates snapshot in `ml-service/data/snapshots/`
- **Retention:** Keep snapshots for debugging
- **Restore:** Can rebuild from snapshots if main file corrupted

## Next Steps

1. ✅ Cron running every 15 minutes
2. ✅ Data synthesis filling gaps
3. 📊 Monitor data quality (synthetic ratio)
4. 🤖 Use complete dataset for ML training
5. 📈 Visualize trails on frontend

---

**Last Updated:** 2025-11-14
**Author:** Angga Suryabrata
