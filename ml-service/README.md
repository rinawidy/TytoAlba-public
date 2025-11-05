# TytoAlba ML Service v2.0

**LSTM-based Deep Learning Service** for vessel arrival time prediction using AIS data and weather forecasts.

## Features

- **Deep Learning Architecture**: CNN + Attention + Bidirectional LSTM
- **Multi-Modal Data**: Combines AIS trajectory and weather data
- **Confidence Scoring**: Monte Carlo Dropout for uncertainty estimation
- **RESTful API**: FastAPI-based endpoints for easy integration
- **Production Ready**: Inference-only service with pre-trained models

## Architecture

The model uses a hybrid deep learning architecture inspired by VATP (Vessel Arrival Time Prediction) research:

```
AIS Data + Weather → Preprocessing → CNN → Attention → LSTM → Prediction
```

See `ARCHITECTURE_DESIGN.md` for detailed architecture documentation.

## Project Structure

```
ml-service/
├── src/
│   ├── preprocessing/
│   │   ├── data_pipeline.py         # Main preprocessing pipeline
│   │   ├── utils.py                 # Geospatial utilities
│   │   └── __init__.py
│   │
│   ├── models/
│   │   ├── lstm_arrival_predictor.py # LSTM model architecture
│   │   └── __init__.py
│   │
│   └── api_inference.py             # FastAPI inference service
│
├── models/
│   ├── vessel_arrival_lstm.h5       # Pre-trained model (after training)
│   └── .gitkeep
│
├── config/
│   ├── model_config.yaml            # Model hyperparameters
│   └── api_config.yaml              # API configuration
│
├── data/
│   └── .gitkeep                     # Training data goes here
│
├── train_lstm_model.py              # Training script (run separately)
├── run_inference_server.py          # Start inference server
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment variables template
├── .gitignore
├── ARCHITECTURE_DESIGN.md           # Detailed architecture docs
└── README.md
```

## Installation

### 1. Create Virtual Environment

```bash
cd ml-service
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: TensorFlow installation may require specific versions for your system. See [TensorFlow installation guide](https://www.tensorflow.org/install) if you encounter issues.

### 3. Setup Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and configure:
- `MODEL_PATH`: Path to your trained model (default: `models/vessel_arrival_lstm.h5`)
- `API_HOST` and `API_PORT`: Server configuration
- Database credentials (when AIS data source is ready)
- Weather API key (when weather service is integrated)

## Quick Start

### Option A: Using Pre-trained Model (Inference Only)

If you have a pre-trained model file:

```bash
# 1. Place your model in models/ directory
mv your_model.h5 models/vessel_arrival_lstm.h5

# 2. Start the inference server
python run_inference_server.py
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### Option B: Train Model First (Full Pipeline)

If you need to train a model from historical data:

```bash
# 1. Prepare your training data
# Place CSV file in data/ directory
# See "Training Data Format" section below

# 2. Train the model
python train_lstm_model.py --data data/historical_voyages.csv --epochs 100

# Or use synthetic data for testing:
python train_lstm_model.py --synthetic --n_samples 1000 --epochs 50

# 3. Start the inference server
python run_inference_server.py
```

## API Endpoints

### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_info": {
    "model_name": "VesselArrivalLSTM",
    "total_parameters": 145729
  }
}
```

### Vessel Arrival Prediction
```http
POST /predict/arrival
```

**Request Body:**
```json
{
  "vessel_mmsi": "563012345",
  "destination_lat": 1.2644,
  "destination_lon": 103.8229
}
```

**Response:**
```json
{
  "vessel_mmsi": "563012345",
  "current_position": {
    "latitude": 1.1234,
    "longitude": 103.5678,
    "timestamp": "2024-10-21T10:00:00Z"
  },
  "destination": {
    "latitude": 1.2644,
    "longitude": 103.8229
  },
  "estimated_arrival_time": "2024-10-21T18:30:00Z",
  "travel_time_minutes": 487.5,
  "confidence": 0.87,
  "distance_km": 125.4,
  "avg_speed_knots": 12.3,
  "weather_impact": "moderate"
}
```

### Model Information
```http
GET /model/info
```

**Response:**
```json
{
  "model_name": "VesselArrivalLSTM",
  "architecture": "CNN + Attention + Bidirectional LSTM",
  "total_parameters": 145729,
  "sequence_length": 48,
  "features": 8
}
```

## Data Requirements

### Input Features (Inference)

The model requires two types of data:

**1. Sequence Data (48 timesteps × 8 features):**
- Latitude position
- Longitude position
- Speed over ground (knots)
- Course over ground (degrees)
- Wind speed (m/s)
- Wave height (meters)
- Sea current speed (m/s)
- Water temperature (°C)

**2. Static Features (10 features):**
- Distance remaining to destination
- Average vessel speed
- Weather severity score
- Course alignment with destination
- Temporal features (hour, day, month - cyclically encoded)

### Training Data Format

For model training, prepare a CSV with these columns:

```csv
vessel_mmsi,voyage_id,ais_data,weather_data,destination_lat,destination_lon,actual_arrival_time
563012345,V001,"[{...}]","[{...}]",1.2644,103.8229,487.5
```

Where:
- `ais_data`: JSON array of position reports
- `weather_data`: JSON array of weather observations
- `actual_arrival_time`: Ground truth in minutes

See `train_lstm_model.py` for detailed data preparation examples.

## Model Performance

Expected performance metrics:

- **MAE** (Mean Absolute Error): ~30 minutes
- **MAPE** (Mean Absolute Percentage Error): ~5-8%
- **Confidence Score**: 0.75-0.95 for most predictions

Performance depends heavily on training data quality and quantity.

## Integration with Frontend

### JavaScript/TypeScript Example

```javascript
// Predict vessel arrival time
async function predictArrival(vesselMMSI, destLat, destLon) {
  const response = await fetch('http://localhost:8000/predict/arrival', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      vessel_mmsi: vesselMMSI,
      destination_lat: destLat,
      destination_lon: destLon
    })
  });

  const data = await response.json();

  console.log(`ETA: ${data.estimated_arrival_time}`);
  console.log(`Travel time: ${data.travel_time_minutes} minutes`);
  console.log(`Confidence: ${(data.confidence * 100).toFixed(1)}%`);

  return data;
}

// Usage
predictArrival('563012345', 1.2644, 103.8229);
```

### Python Client Example

```python
import requests

url = "http://localhost:8000/predict/arrival"
payload = {
    "vessel_mmsi": "563012345",
    "destination_lat": 1.2644,
    "destination_lon": 103.8229
}

response = requests.post(url, json=payload)
data = response.json()

print(f"ETA: {data['estimated_arrival_time']}")
print(f"Confidence: {data['confidence']:.2%}")
```

## Data Source Integration

### Implementing AIS Data Fetching

Edit `src/preprocessing/data_pipeline.py` and implement the `fetch_ais_history()` method:

```python
def fetch_ais_history(self, mmsi: str, hours: int = 24):
    # Example with PostgreSQL
    import psycopg2
    conn = psycopg2.connect(...)
    cursor = conn.cursor()

    query = """
        SELECT timestamp, latitude, longitude, speed, course
        FROM ais_positions
        WHERE mmsi = %s AND timestamp >= NOW() - INTERVAL '%s hours'
        ORDER BY timestamp ASC
    """

    cursor.execute(query, [mmsi, hours])
    return cursor.fetchall()
```

### Implementing Weather API

Edit `src/preprocessing/data_pipeline.py` and implement the `fetch_weather_along_route()` method:

```python
def fetch_weather_along_route(self, start_lat, start_lon, end_lat, end_lon):
    # Example with OpenWeatherMap
    import requests

    api_key = os.getenv('WEATHER_API_KEY')
    waypoints = calculate_route_waypoints(start_lat, start_lon, end_lat, end_lon)

    weather_data = []
    for waypoint in waypoints:
        url = f"https://api.openweathermap.org/data/2.5/forecast"
        params = {
            'lat': waypoint['lat'],
            'lon': waypoint['lon'],
            'appid': api_key
        }
        response = requests.get(url, params=params)
        weather_data.append(response.json())

    return weather_data
```

## Production Deployment

### Using Docker

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "run_inference_server.py"]
```

Build and run:
```bash
docker build -t tytoalba-ml .
docker run -p 8000:8000 -v $(pwd)/models:/app/models tytoalba-ml
```

### Using Gunicorn

```bash
pip install gunicorn

gunicorn -w 4 -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  src.api_inference:app
```

### Deployment Checklist

- [ ] Configure AIS database connection
- [ ] Set up weather API integration
- [ ] Train model on production data
- [ ] Update CORS settings for your domain
- [ ] Set up logging and monitoring
- [ ] Configure environment variables
- [ ] Add authentication if needed
- [ ] Set up SSL/TLS
- [ ] Configure firewall rules

## Troubleshooting

### Model not found error
```
[WARNING] Model file not found at models/vessel_arrival_lstm.h5
```
**Solution**: Train the model first using `python train_lstm_model.py` or place a pre-trained model in the `models/` directory.

### TensorFlow installation issues
```
ERROR: Could not find a version that satisfies the requirement tensorflow==2.15.0
```
**Solution**: Check your Python version (requires 3.9-3.11). For Apple Silicon Macs, use `tensorflow-macos`.

### Data source not implemented
```
NotImplementedError: AIS data fetching not implemented
```
**Solution**: Implement the `fetch_ais_history()` and `fetch_weather_along_route()` methods in `src/preprocessing/data_pipeline.py`. See "Data Source Integration" section.

### Import errors
```
ModuleNotFoundError: No module named 'tensorflow'
```
**Solution**:
- Activate virtual environment: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
- Install dependencies: `pip install -r requirements.txt`

### Port already in use
```
ERROR: [Errno 48] Address already in use
```
**Solution**:
- Change `API_PORT` in `.env` file
- Or find and kill the process: `lsof -ti:8000 | xargs kill -9` (Unix) or `netstat -ano | findstr :8000` (Windows)

### CUDA/GPU issues (optional)
If you want to use GPU acceleration:
- Install CUDA-enabled TensorFlow: `pip install tensorflow[and-cuda]`
- Verify GPU: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

## Next Steps

1. **Integrate Data Sources**: Implement AIS and weather data fetching
2. **Collect Training Data**: Gather historical voyage data
3. **Train Model**: Run `train_lstm_model.py` with your data
4. **Deploy Service**: Start the inference server
5. **Connect Frontend**: Integrate with TytoAlba web application

## Architecture Documentation

For detailed information about the model architecture, see:
- `ARCHITECTURE_DESIGN.md` - Complete architecture documentation with pseudocode
- `config/model_config.yaml` - Model hyperparameters
- `src/models/lstm_arrival_predictor.py` - Model implementation

## Contributing

When contributing to this project:
1. Follow PEP 8 style guidelines
2. Add docstrings to all functions
3. Update tests when adding features
4. Document any new configuration options

## License

MIT License - see LICENSE file for details

## Contact

For questions or support:
- GitHub Issues: [TytoAlba Issues](https://github.com/your-org/tytoalba)
- Documentation: See `ARCHITECTURE_DESIGN.md`
- Model Architecture: Inspired by VATP research paper

---

**Built with**: TensorFlow, FastAPI, Python 3.10+
