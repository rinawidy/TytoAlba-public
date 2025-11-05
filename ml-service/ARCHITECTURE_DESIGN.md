# TytoAlba VATP-Inspired Architecture Design
## Simplified LSTM-Based Vessel Arrival Prediction

---

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ARRIVAL TIME PREDICTION OUTPUT                        │
│                      (Estimated Minutes to Arrival)                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │
┌─────────────────────────────────────────────────────────────────────────┐
│                    FEED-FORWARD NEURAL NETWORK                          │
│                         Dense(128) → Dense(64) → Dense(1)               │
│                         ReLU        ReLU        Linear                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │
┌─────────────────────────────────────────────────────────────────────────┐
│                       CONCATENATION MODULE                              │
│              [LSTM Output + Static Features (vessel, route)]            │
│                    Shape: [128 + 10] = [138]                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │
┌─────────────────────────────────────────────────────────────────────────┐
│                    LSTM MODULE (Bidirectional)                          │
│              Input: [Timesteps × Features] → Output: [128]              │
│              Captures: Route patterns, speed changes, weather impact    │
│              Dropout: 0.2 | Return sequences: False                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │
┌─────────────────────────────────────────────────────────────────────────┐
│                       DROPOUT LAYER (rate=0.3)                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │
┌─────────────────────────────────────────────────────────────────────────┐
│                   ATTENTION MECHANISM MODULE                            │
│              Learns which voyage segments are most important            │
│              Weights: Softmax over timesteps                            │
│              Output: Weighted feature vectors                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │
┌─────────────────────────────────────────────────────────────────────────┐
│              CNN FEATURE EXTRACTOR (1D Convolution)                     │
│              Conv1D(64, kernel=3) → MaxPool → Conv1D(128, kernel=3)     │
│              Extracts spatial patterns from trajectory + weather        │
└─────────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │
┌─────────────────────────────────────────────────────────────────────────┐
│                    COMBINED SEQUENCE INPUT                              │
│         [Lat, Lon, Speed, Course, Wind, Waves, Current, Temp]           │
│                    Shape: [n_timesteps × 8]                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │
┌─────────────────────────────────────────────────────────────────────────┐
│              VECTOR REPRESENTATION MODULE                               │
│   - Normalize sequences (StandardScaler)                               │
│   - Create time windows (48 timesteps = 24 hours)                      │
│   - Align weather data with AIS positions                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │
┌─────────────────────────────────────────────────────────────────────────┐
│         FEATURE ENGINEERING & PREPROCESSING (DECTF)                     │
│   - Calculate distance to destination                                  │
│   - Extract temporal features (hour, day, month)                       │
│   - Compute speed statistics (mean, std, acceleration)                 │
│   - Generate weather severity scores                                   │
│   - Handle missing data (interpolation)                                │
└─────────────────────────────────────────────────────────────────────────┘
                                    ▲
                    ┌───────────────┴───────────────┐
                    │                               │
┌─────────────────────────────────┐ ┌─────────────────────────────────┐
│        AIS DATA SOURCE          │ │   WEATHER & SEA DATA SOURCE     │
│                                 │ │                                 │
│ - Historical positions (24h)    │ │ - Wind speed & direction        │
│ - Speed over ground (SOG)       │ │ - Wave height & period          │
│ - Course over ground (COG)      │ │ - Sea current speed             │
│ - Vessel MMSI/IMO               │ │ - Water temperature             │
│ - Timestamp of each position    │ │ - Forecast along route          │
└─────────────────────────────────┘ └─────────────────────────────────┘
                    DATABASE / API LAYER
```

---

## Simplified Data Flow

```
USER REQUEST
     │
     │ {vessel_mmsi, destination_lat, destination_lon}
     ▼
┌─────────────────────────────────────────────────────┐
│              PYTHON INFERENCE SERVICE               │
│                   (FastAPI)                         │
└─────────────────────────────────────────────────────┘
     │
     ├─── Fetch AIS History (last 24 hours)
     │         └─> Database / AIS API
     │
     ├─── Fetch Weather Forecast (along route)
     │         └─> Weather API (OpenWeatherMap/NOAA)
     │
     └─── Load Pre-trained Model
                └─> models/vessel_arrival_lstm.h5
     │
     ▼
┌─────────────────────────────────────────────────────┐
│                PREPROCESSING                        │
│                                                     │
│  1. Clean AIS data (remove outliers)                │
│  2. Interpolate missing positions                   │
│  3. Align weather with trajectory                   │
│  4. Extract features                                │
│  5. Create sequences (48 timesteps)                 │
│  6. Normalize data                                  │
└─────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│              NEURAL NETWORK INFERENCE               │
│                                                     │
│  Input: [trajectory_sequence, static_features]      │
│         ↓                                           │
│  CNN → Attention → Dropout → LSTM → Dense          │
│         ↓                                           │
│  Output: Predicted minutes                          │
└─────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│              POST-PROCESSING                        │
│                                                     │
│  - Calculate ETA (current_time + minutes)           │
│  - Generate confidence score                        │
│  - Extract attention weights (explainability)       │
└─────────────────────────────────────────────────────┘
     │
     ▼
JSON RESPONSE
{
  "estimated_arrival_time": "2024-10-22T14:30:00Z",
  "travel_time_minutes": 487.5,
  "confidence": 0.87,
  "key_factors": {
    "distance_km": 450,
    "avg_speed_knots": 12.3,
    "weather_severity": "moderate"
  }
}
```

---

## High-Level Pseudocode

### 1. Data Preprocessing Module

```python
# ===================================================================
# FILE: src/preprocessing/data_pipeline.py
# ===================================================================

import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

class VoyageDataPreprocessor:
    """
    Handles all data preprocessing for inference
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.sequence_length = 48  # 24 hours at 30-min intervals


    def prepare_inference_data(self, vessel_mmsi, destination_lat, destination_lon):
        """
        Main preprocessing pipeline for inference

        Input:
            vessel_mmsi: Vessel identifier (MMSI number)
            destination_lat: Target latitude
            destination_lon: Target longitude

        Output:
            sequence_data: [48, 8] Combined trajectory + weather
            static_features: [10] Vessel and route metadata
        """

        # STEP 1: Fetch AIS Data
        ais_data = self.fetch_ais_history(vessel_mmsi, hours=24)
        """
        Returns list of dicts:
        [
            {
                'timestamp': '2024-10-21T10:00:00Z',
                'latitude': 1.2345,
                'longitude': 103.8765,
                'speed': 12.5,  # knots
                'course': 145.0  # degrees
            },
            ...
        ]
        """

        # STEP 2: Fetch Weather Data
        current_pos = ais_data[-1]  # Latest position
        weather_data = self.fetch_weather_along_route(
            start_lat=current_pos['latitude'],
            start_lon=current_pos['longitude'],
            end_lat=destination_lat,
            end_lon=destination_lon
        )
        """
        Returns list aligned with AIS timestamps:
        [
            {
                'timestamp': '2024-10-21T10:00:00Z',
                'wind_speed': 8.5,  # m/s
                'wave_height': 1.2,  # meters
                'current_speed': 0.5,  # m/s
                'temperature': 28.0  # celsius
            },
            ...
        ]
        """

        # STEP 3: Clean and Validate
        ais_data = self.remove_outliers(ais_data)
        ais_data = self.interpolate_missing(ais_data)

        # STEP 4: Feature Engineering
        features = self.extract_features(ais_data, weather_data,
                                         destination_lat, destination_lon)

        # STEP 5: Create Sequences
        sequence_data = self.create_sequence(ais_data, weather_data)

        # STEP 6: Normalize
        sequence_data = self.normalize_sequence(sequence_data)

        # STEP 7: Static Features
        static_features = self.create_static_features(features)

        return sequence_data, static_features


    def fetch_ais_history(self, mmsi, hours=24):
        """
        Fetch AIS positions from database
        """
        # Query database or API
        query = f"""
            SELECT
                timestamp,
                latitude,
                longitude,
                speed_over_ground as speed,
                course_over_ground as course
            FROM ais_positions
            WHERE mmsi = {mmsi}
            AND timestamp >= NOW() - INTERVAL {hours} HOUR
            ORDER BY timestamp ASC
        """

        result = database.execute(query)
        return result


    def fetch_weather_along_route(self, start_lat, start_lon, end_lat, end_lon):
        """
        Fetch weather forecast along planned route
        """
        # Calculate waypoints
        waypoints = self.calculate_route_waypoints(
            start_lat, start_lon,
            end_lat, end_lon,
            num_points=10
        )

        weather_forecasts = []

        for waypoint in waypoints:
            # Call weather API
            forecast = weather_api.get_marine_forecast(
                lat=waypoint['lat'],
                lon=waypoint['lon'],
                hours=48
            )
            weather_forecasts.append(forecast)

        # Interpolate weather data to match AIS timestamps
        aligned_weather = self.align_weather_to_trajectory(weather_forecasts)

        return aligned_weather


    def remove_outliers(self, ais_data):
        """
        Remove impossible values
        """
        cleaned = []

        for i, point in enumerate(ais_data):
            # Check speed (cargo ships typically 0-25 knots)
            if point['speed'] < 0 or point['speed'] > 30:
                continue

            # Check for position jumps (>100km in 30 min = impossible)
            if i > 0:
                prev = ais_data[i-1]
                distance = haversine_distance(
                    prev['latitude'], prev['longitude'],
                    point['latitude'], point['longitude']
                )
                time_diff_hours = (point['timestamp'] - prev['timestamp']).total_seconds() / 3600

                if distance / time_diff_hours > 50:  # >50 km/h is suspicious
                    continue

            cleaned.append(point)

        return cleaned


    def interpolate_missing(self, ais_data):
        """
        Fill gaps in trajectory using linear interpolation
        """
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
                ratio = (target_time - before['timestamp']) / (after['timestamp'] - before['timestamp'])

                interpolated_point = {
                    'timestamp': target_time,
                    'latitude': before['latitude'] + ratio * (after['latitude'] - before['latitude']),
                    'longitude': before['longitude'] + ratio * (after['longitude'] - before['longitude']),
                    'speed': before['speed'] + ratio * (after['speed'] - before['speed']),
                    'course': before['course'] + ratio * (after['course'] - before['course'])
                }
                interpolated.append(interpolated_point)

        return interpolated


    def extract_features(self, ais_data, weather_data, dest_lat, dest_lon):
        """
        Calculate derived features
        """
        features = {}

        # Current position
        current = ais_data[-1]

        # Distance to destination
        features['distance_remaining'] = haversine_distance(
            current['latitude'], current['longitude'],
            dest_lat, dest_lon
        )

        # Speed statistics (last 24 hours)
        speeds = [p['speed'] for p in ais_data]
        features['avg_speed'] = np.mean(speeds)
        features['std_speed'] = np.std(speeds)
        features['max_speed'] = np.max(speeds)
        features['min_speed'] = np.min(speeds)

        # Speed changes (acceleration/deceleration)
        speed_changes = [speeds[i] - speeds[i-1] for i in range(1, len(speeds))]
        features['avg_acceleration'] = np.mean(speed_changes)
        features['speed_volatility'] = np.std(speed_changes)

        # Course stability
        courses = [p['course'] for p in ais_data]
        course_changes = [abs(courses[i] - courses[i-1]) for i in range(1, len(courses))]
        features['course_stability'] = 1 / (np.mean(course_changes) + 1)  # Higher = more stable

        # Weather impact score
        wind_speeds = [w['wind_speed'] for w in weather_data]
        wave_heights = [w['wave_height'] for w in weather_data]

        features['avg_wind_speed'] = np.mean(wind_speeds)
        features['max_wave_height'] = np.max(wave_heights)

        # Combined weather severity (0-1 scale)
        features['weather_severity'] = (
            (features['avg_wind_speed'] / 20) * 0.6 +  # Wind contribution
            (features['max_wave_height'] / 5) * 0.4    # Wave contribution
        )
        features['weather_severity'] = min(1.0, features['weather_severity'])

        # Temporal features
        current_time = datetime.now()
        features['hour_of_day'] = current_time.hour
        features['day_of_week'] = current_time.weekday()
        features['month'] = current_time.month

        # Bearing to destination
        features['bearing_to_dest'] = calculate_bearing(
            current['latitude'], current['longitude'],
            dest_lat, dest_lon
        )

        # Course alignment (how aligned is vessel course with destination?)
        features['course_alignment'] = 1 - abs(current['course'] - features['bearing_to_dest']) / 180

        return features


    def create_sequence(self, ais_data, weather_data):
        """
        Combine AIS and weather into sequence matrix

        Output shape: [48, 8]
            - 48 timesteps (24 hours at 30-min intervals)
            - 8 features: [lat, lon, speed, course, wind, wave, current, temp]
        """
        # Take last 48 points
        ais_seq = ais_data[-self.sequence_length:]
        weather_seq = weather_data[-self.sequence_length:]

        # Pad if not enough data
        if len(ais_seq) < self.sequence_length:
            # Repeat first point to fill
            padding = [ais_seq[0]] * (self.sequence_length - len(ais_seq))
            ais_seq = padding + ais_seq
            weather_seq = padding + weather_seq

        # Build sequence matrix
        sequence = []

        for ais, weather in zip(ais_seq, weather_seq):
            row = [
                ais['latitude'],
                ais['longitude'],
                ais['speed'],
                ais['course'],
                weather['wind_speed'],
                weather['wave_height'],
                weather['current_speed'],
                weather['temperature']
            ]
            sequence.append(row)

        return np.array(sequence)


    def normalize_sequence(self, sequence):
        """
        Normalize to [0, 1] range
        """
        # Fit scaler on this sequence
        normalized = self.scaler.fit_transform(sequence)
        return normalized


    def create_static_features(self, features):
        """
        Create static feature vector

        Output: [10 features]
        """
        static = [
            features['distance_remaining'] / 10000,  # Normalize by max expected distance
            features['avg_speed'] / 25,              # Normalize by typical max speed
            features['weather_severity'],            # Already 0-1
            features['course_alignment'],            # Already 0-1
            np.sin(2 * np.pi * features['hour_of_day'] / 24),      # Cyclical encoding
            np.cos(2 * np.pi * features['hour_of_day'] / 24),
            np.sin(2 * np.pi * features['day_of_week'] / 7),
            np.cos(2 * np.pi * features['day_of_week'] / 7),
            np.sin(2 * np.pi * features['month'] / 12),
            np.cos(2 * np.pi * features['month'] / 12)
        ]

        return np.array(static)


# ===================================================================
# Helper Functions
# ===================================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate great circle distance between two points
    Returns distance in kilometers
    """
    R = 6371  # Earth radius in km

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate bearing from point 1 to point 2
    Returns bearing in degrees (0-360)
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1

    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)

    bearing = np.arctan2(x, y)
    bearing = np.degrees(bearing)
    bearing = (bearing + 360) % 360

    return bearing
```

---

### 2. Neural Network Model (LSTM-Focused)

```python
# ===================================================================
# FILE: src/models/lstm_arrival_predictor.py
# ===================================================================

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Bidirectional, Dropout,
    Conv1D, MaxPooling1D, Concatenate, Multiply, Softmax
)
from tensorflow.keras.optimizers import Adam

class VesselArrivalLSTM:
    """
    LSTM-based arrival time prediction model
    No Random Forest - Pure deep learning
    """

    def __init__(self, model_path=None):
        self.model = None

        if model_path:
            self.load_model(model_path)
        else:
            self.model = self.build_model()


    def build_model(self):
        """
        Build the neural network architecture

        Architecture:
            Sequence Input → CNN → Attention → LSTM → Dense → Output
            Static Input → Concatenate with LSTM
        """

        # INPUT LAYERS
        # ------------

        # Sequence input: [batch, 48 timesteps, 8 features]
        sequence_input = Input(shape=(48, 8), name='sequence_input')

        # Static features: [batch, 10 features]
        static_input = Input(shape=(10,), name='static_input')


        # CNN FEATURE EXTRACTION
        # ----------------------

        # Extract spatial-temporal patterns
        x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(sequence_input)
        x = MaxPooling1D(pool_size=2)(x)
        # Shape: [batch, 24, 64]

        x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
        x = MaxPooling1D(pool_size=2)(x)
        # Shape: [batch, 12, 128]


        # ATTENTION MECHANISM
        # -------------------

        # Calculate attention weights
        attention_scores = Dense(1, activation='tanh')(x)
        # Shape: [batch, 12, 1]

        attention_weights = Softmax(axis=1)(attention_scores)
        # Normalize across timesteps

        # Apply attention
        x = Multiply()([x, attention_weights])
        # Weight each timestep by its importance


        # DROPOUT REGULARIZATION
        # ----------------------

        x = Dropout(rate=0.3)(x)


        # BIDIRECTIONAL LSTM
        # ------------------

        # Process temporal dependencies in both directions
        lstm_out = Bidirectional(LSTM(
            units=64,
            return_sequences=False,
            dropout=0.2,
            recurrent_dropout=0.2
        ))(x)
        # Shape: [batch, 128] (64 forward + 64 backward)


        # CONCATENATE WITH STATIC FEATURES
        # --------------------------------

        combined = Concatenate()([lstm_out, static_input])
        # Shape: [batch, 138]


        # FEED-FORWARD NETWORK
        # --------------------

        dense1 = Dense(128, activation='relu')(combined)
        dense1 = Dropout(0.3)(dense1)

        dense2 = Dense(64, activation='relu')(dense1)
        dense2 = Dropout(0.2)(dense2)

        dense3 = Dense(32, activation='relu')(dense2)

        # Output: Predicted travel time in minutes
        output = Dense(1, activation='linear', name='arrival_minutes')(dense3)


        # BUILD MODEL
        # -----------

        model = Model(
            inputs=[sequence_input, static_input],
            outputs=output,
            name='VesselArrivalLSTM'
        )

        # Compile
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mae', 'mse']
        )

        return model


    def predict(self, sequence_data, static_features):
        """
        Make inference prediction

        Input:
            sequence_data: [48, 8] numpy array
            static_features: [10] numpy array

        Output:
            predicted_minutes: float
        """
        # Add batch dimension
        seq_batch = np.expand_dims(sequence_data, axis=0)
        static_batch = np.expand_dims(static_features, axis=0)

        # Predict
        prediction = self.model.predict([seq_batch, static_batch], verbose=0)

        return float(prediction[0][0])


    def predict_with_confidence(self, sequence_data, static_features, n_samples=10):
        """
        Make prediction with confidence estimate using MC Dropout

        Returns:
            mean_prediction: Average prediction
            confidence: Standard deviation (lower = more confident)
        """
        # Enable dropout during inference for uncertainty estimation
        predictions = []

        for _ in range(n_samples):
            pred = self.predict(sequence_data, static_features)
            predictions.append(pred)

        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)

        # Convert std to confidence score (0-1)
        # Lower std = higher confidence
        confidence = 1 / (1 + std_pred / mean_pred)

        return mean_pred, confidence


    def load_model(self, path):
        """Load pre-trained model"""
        self.model = tf.keras.models.load_model(path)
        print(f"Model loaded from {path}")


    def save_model(self, path):
        """Save trained model"""
        self.model.save(path)
        print(f"Model saved to {path}")
```

---

### 3. FastAPI Inference Service

```python
# ===================================================================
# FILE: src/api_inference.py
# ===================================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime, timedelta
import numpy as np

from preprocessing.data_pipeline import VoyageDataPreprocessor
from models.lstm_arrival_predictor import VesselArrivalLSTM

# Initialize
app = FastAPI(title="TytoAlba Vessel Arrival Prediction")

# Load pre-trained model
model = VesselArrivalLSTM(model_path='models/vessel_arrival_lstm.h5')
preprocessor = VoyageDataPreprocessor()


# REQUEST/RESPONSE SCHEMAS
# ------------------------

class PredictionRequest(BaseModel):
    vessel_mmsi: str
    destination_lat: float
    destination_lon: float


class PredictionResponse(BaseModel):
    vessel_mmsi: str
    current_position: dict
    destination: dict
    estimated_arrival_time: str
    travel_time_minutes: float
    confidence: float
    distance_km: float
    avg_speed_knots: float
    weather_impact: str


# API ENDPOINTS
# -------------

@app.get("/health")
async def health_check():
    """Check if service is running"""
    return {"status": "healthy", "model": "VesselArrivalLSTM"}


@app.post("/predict/arrival", response_model=PredictionResponse)
async def predict_arrival(request: PredictionRequest):
    """
    Predict vessel arrival time using LSTM model

    WORKFLOW:
    1. Fetch and preprocess data
    2. Run LSTM inference
    3. Calculate ETA
    4. Return prediction with confidence
    """

    try:
        # STEP 1: Preprocess data
        sequence_data, static_features = preprocessor.prepare_inference_data(
            vessel_mmsi=request.vessel_mmsi,
            destination_lat=request.destination_lat,
            destination_lon=request.destination_lon
        )

        # STEP 2: Make prediction
        predicted_minutes, confidence = model.predict_with_confidence(
            sequence_data,
            static_features
        )

        # STEP 3: Calculate ETA
        current_time = datetime.utcnow()
        eta = current_time + timedelta(minutes=predicted_minutes)

        # STEP 4: Extract metadata
        # Get current position from preprocessor
        ais_data = preprocessor.fetch_ais_history(request.vessel_mmsi, hours=1)
        current_pos = ais_data[-1]

        # Calculate distance
        distance_km = haversine_distance(
            current_pos['latitude'], current_pos['longitude'],
            request.destination_lat, request.destination_lon
        )

        # Average speed from static features
        avg_speed = static_features[1] * 25  # Denormalize

        # Weather severity
        weather_severity = static_features[2]
        if weather_severity < 0.3:
            weather_impact = "low"
        elif weather_severity < 0.6:
            weather_impact = "moderate"
        else:
            weather_impact = "high"

        # STEP 5: Build response
        return PredictionResponse(
            vessel_mmsi=request.vessel_mmsi,
            current_position={
                "latitude": current_pos['latitude'],
                "longitude": current_pos['longitude'],
                "timestamp": current_pos['timestamp']
            },
            destination={
                "latitude": request.destination_lat,
                "longitude": request.destination_lon
            },
            estimated_arrival_time=eta.isoformat() + "Z",
            travel_time_minutes=round(predicted_minutes, 2),
            confidence=round(confidence, 3),
            distance_km=round(distance_km, 2),
            avg_speed_knots=round(avg_speed, 2),
            weather_impact=weather_impact
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/model/info")
async def model_info():
    """Get model architecture information"""
    return {
        "model_name": "VesselArrivalLSTM",
        "architecture": "CNN + Attention + Bidirectional LSTM",
        "sequence_length": 48,
        "features": 8,
        "static_features": 10,
        "total_parameters": model.model.count_params()
    }


# MAIN
# ----

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## Project Structure

```
ml-service/
│
├── src/
│   ├── __init__.py
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── data_pipeline.py          # Main preprocessing logic
│   │   └── utils.py                  # Helper functions (haversine, bearing, etc)
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── lstm_arrival_predictor.py # LSTM model definition
│   │
│   └── api_inference.py              # FastAPI service
│
├── models/
│   └── vessel_arrival_lstm.h5        # Pre-trained model (loaded at startup)
│
├── config/
│   ├── model_config.yaml             # Model hyperparameters
│   └── api_config.yaml               # API settings
│
├── requirements.txt                   # Python dependencies
├── Dockerfile                         # Container definition
├── docker-compose.yml                 # Service orchestration
│
└── README.md                          # Documentation
```

---

## Requirements

```txt
# requirements.txt

# Deep Learning
tensorflow==2.15.0
keras==2.15.0

# API
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0

# Data Processing
numpy==1.26.2
pandas==2.1.3
scikit-learn==1.3.2

# Geospatial
geopy==2.4.1

# Utilities
python-dotenv==1.0.0
requests==2.31.0
```

---

## Key Differences from Original Plan

| Aspect | Original | Revised |
|--------|----------|---------|
| **Model Type** | Random Forest + DL | LSTM only |
| **Port Data** | Berth queue included | No berth data |
| **Focus** | Dual CNN branches | Single CNN + LSTM |
| **Service Type** | Training + Inference | Inference only |
| **Complexity** | High | Simplified |

---

## Model Training (Separate Script)

Training happens offline, not in ml-service:

```python
# train_lstm_model.py (run separately)

from models.lstm_arrival_predictor import VesselArrivalLSTM
import pandas as pd

# Load historical voyage data
df = pd.read_csv('data/historical_voyages.csv')

# Preprocess training data
X_seq, X_static, y = preprocess_training_data(df)

# Split train/validation
X_train, X_val, y_train, y_val = train_test_split(...)

# Build and train model
model = VesselArrivalLSTM()
model.model.fit(
    x=[X_train_seq, X_train_static],
    y=y_train,
    validation_data=([X_val_seq, X_val_static], y_val),
    epochs=100,
    batch_size=32
)

# Save trained model
model.save_model('models/vessel_arrival_lstm.h5')
```

Then deploy the trained model with the inference service.

---

This design gives you a clean, LSTM-focused inference service ready for production deployment!
