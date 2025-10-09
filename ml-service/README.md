# TytoAlba ML Service

Machine Learning service for **fuel consumption prediction** and **arrival time prediction** using Random Forest models.

## Features

- **Fuel Prediction**: Predict fuel consumption based on trip parameters
- **Arrival Time Prediction**: Predict travel time and estimated arrival
- **RESTful API**: FastAPI-based endpoints for easy integration
- **Random Forest Models**: Simple, effective, and easy to train

## Project Structure

```
ml-service/
├── src/
│   ├── models/
│   │   ├── fuel_predictor.py       # Fuel prediction model
│   │   └── arrival_predictor.py    # Arrival time prediction model
│   ├── api.py                       # FastAPI application
│   └── schemas.py                   # Pydantic schemas
├── models/                          # Trained model files (.pkl)
├── train_fuel_model.py             # Training script for fuel model
├── train_arrival_model.py          # Training script for arrival model
├── run_server.py                   # Start API server
├── test_api.py                     # Test API endpoints
├── requirements.txt                # Python dependencies
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

### 3. Setup Environment Variables

```bash
cp .env.example .env
```

Edit `.env` if needed (default settings should work).

## Quick Start

### Step 1: Train Models

Train the models with sample data:

```bash
# Train fuel prediction model
python train_fuel_model.py

# Train arrival time prediction model
python train_arrival_model.py
```

This will create `models/fuel_prediction_model.pkl` and `models/arrival_prediction_model.pkl`.

### Step 2: Start API Server

```bash
python run_server.py
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

### Step 3: Test API

In a new terminal (with the server running):

```bash
python test_api.py
```

## API Endpoints

### Health Check
```
GET /health
```

### Fuel Prediction
```
POST /predict/fuel
```

**Request Body:**
```json
{
  "distance": 150.5,
  "vehicle_weight": 8000,
  "avg_speed": 65,
  "vehicle_type": 1,
  "terrain_type": 1
}
```

**Response:**
```json
{
  "predicted_fuel_liters": 23.45,
  "features_used": { ... }
}
```

### Arrival Time Prediction
```
POST /predict/arrival
```

**Request Body:**
```json
{
  "distance": 120.0,
  "departure_time": "2024-10-09T14:30:00",
  "route_id": 5,
  "avg_traffic_level": 1,
  "historical_avg_time": 95.5
}
```

**Response:**
```json
{
  "travel_time_minutes": 98.25,
  "estimated_arrival_time": "2024-10-09T16:08:15",
  "features_used": { ... }
}
```

### Feature Importance
```
GET /models/fuel/importance
GET /models/arrival/importance
```

## Feature Descriptions

### Fuel Prediction Features

| Feature | Type | Description | Example |
|---------|------|-------------|---------|
| `distance` | float | Distance in kilometers | 150.5 |
| `vehicle_weight` | float | Vehicle weight in kg | 8000 |
| `avg_speed` | float | Average speed in km/h | 65 |
| `vehicle_type` | int | 0=small, 1=medium, 2=large | 1 |
| `terrain_type` | int | 0=flat, 1=hilly, 2=mountain | 1 |

### Arrival Prediction Features

| Feature | Type | Description | Example |
|---------|------|-------------|---------|
| `distance` | float | Distance in kilometers | 120.0 |
| `departure_time` | string (ISO) | Departure datetime | "2024-10-09T14:30:00" |
| `departure_hour` | int (optional) | Hour 0-23 | 14 |
| `day_of_week` | int (optional) | 0=Mon, 6=Sun | 2 |
| `route_id` | int | Route identifier | 5 |
| `avg_traffic_level` | int | 0=low, 1=medium, 2=high | 1 |
| `historical_avg_time` | float | Historical avg in minutes | 95.5 |

## Training with Your Own Data

### Fuel Model

Create a CSV file with columns: `distance`, `vehicle_weight`, `avg_speed`, `vehicle_type`, `terrain_type`, `fuel_consumption`

```python
# Edit train_fuel_model.py
train_model(data_source='file', data_path='data/your_fuel_data.csv')
```

### Arrival Model

Create a CSV file with columns: `distance`, `departure_hour`, `day_of_week`, `route_id`, `avg_traffic_level`, `historical_avg_time`, `arrival_time`

```python
# Edit train_arrival_model.py
train_model(data_source='file', data_path='data/your_arrival_data.csv')
```

## Model Performance

After training, you'll see metrics like:

- **MAE** (Mean Absolute Error): Average prediction error
- **RMSE** (Root Mean Squared Error): Penalizes large errors
- **R²** (R-squared): How well the model explains variance (closer to 1 is better)

## Integration with Frontend

### Example: Fetch from JavaScript

```javascript
// Fuel prediction
const response = await fetch('http://localhost:8000/predict/fuel', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    distance: 150.5,
    vehicle_weight: 8000,
    avg_speed: 65,
    vehicle_type: 1,
    terrain_type: 1
  })
});
const data = await response.json();
console.log(`Predicted fuel: ${data.predicted_fuel_liters} liters`);
```

## Development

### Running in Development Mode

The server runs with auto-reload enabled by default:

```bash
python run_server.py
```

### Running Tests

```bash
# Make sure server is running first
python test_api.py
```

## Production Deployment

For production, consider:

1. **Disable auto-reload** in `run_server.py`
2. **Use a production WSGI server** (e.g., Gunicorn)
3. **Update CORS settings** in `src/api.py`
4. **Use environment variables** for sensitive data
5. **Add authentication/authorization**

Example with Gunicorn:
```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.api:app
```

## Upgrading Models

If you want to use more advanced models later:

1. Uncomment `xgboost` or `lightgbm` in `requirements.txt`
2. Modify the predictor classes to use those models
3. Retrain with your data

## Troubleshooting

### Model not found error
- Make sure you've run the training scripts first
- Check that `.pkl` files exist in `models/` directory

### Import errors
- Activate virtual environment
- Install all requirements: `pip install -r requirements.txt`

### Port already in use
- Change `API_PORT` in `.env` file
- Or kill the process using port 8000

## License

MIT

## Support

For issues or questions, please contact the TytoAlba development team.
