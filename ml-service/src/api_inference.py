"""
FastAPI Inference Service for Vessel Arrival Prediction
LSTM-based deep learning model
"""

import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from preprocessing.data_pipeline import VoyageDataPreprocessor
from models.lstm_arrival_predictor import VesselArrivalLSTM
from preprocessing.utils import haversine_distance

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="TytoAlba Vessel Arrival Prediction",
    description="LSTM-based deep learning service for ship arrival time prediction",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv('CORS_ORIGINS', '*').split(','),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model and preprocessor
MODEL_PATH = os.getenv('MODEL_PATH', 'models/vessel_arrival_lstm.h5')
model = None
preprocessor = VoyageDataPreprocessor()


# ============================================================================
# REQUEST/RESPONSE SCHEMAS
# ============================================================================

class PredictionRequest(BaseModel):
    """Request schema for arrival prediction"""
    vessel_mmsi: str = Field(..., description="Vessel MMSI identifier")
    destination_lat: float = Field(..., description="Destination latitude", ge=-90, le=90)
    destination_lon: float = Field(..., description="Destination longitude", ge=-180, le=180)

    class Config:
        json_schema_extra = {
            "example": {
                "vessel_mmsi": "563012345",
                "destination_lat": 1.2644,
                "destination_lon": 103.8229
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for arrival prediction"""
    vessel_mmsi: str
    current_position: dict
    destination: dict
    estimated_arrival_time: str
    travel_time_minutes: float
    confidence: float = Field(..., description="Confidence score (0-1)")
    distance_km: float
    avg_speed_knots: float
    weather_impact: str = Field(..., description="Weather severity: low, moderate, high")

    class Config:
        json_schema_extra = {
            "example": {
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
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_info: Optional[dict] = None


class ModelInfoResponse(BaseModel):
    """Model information response"""
    model_name: str
    architecture: str
    total_parameters: int
    sequence_length: int
    features: int


# ============================================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def load_model():
    """Load pre-trained LSTM model on startup"""
    global model

    print(f"[INFO] Loading model from {MODEL_PATH}")

    if os.path.exists(MODEL_PATH):
        try:
            model = VesselArrivalLSTM(model_path=MODEL_PATH)
            print(f"[SUCCESS] Model loaded successfully")
            print(f"[INFO] Model parameters: {model.get_model_info()['total_parameters']:,}")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            print(f"[WARNING] Service will start but predictions will fail")
    else:
        print(f"[WARNING] Model file not found at {MODEL_PATH}")
        print(f"[WARNING] Please train model and place it at {MODEL_PATH}")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    print("[INFO] Shutting down service")


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with service information"""
    return {
        "service": "TytoAlba Vessel Arrival Prediction",
        "version": "2.0.0",
        "model": "LSTM-based Deep Learning",
        "endpoints": {
            "health": "/health",
            "predict": "/predict/arrival",
            "model_info": "/model/info"
        },
        "documentation": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Check service health and model status
    """
    model_loaded = model is not None and model.model is not None

    response = HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded
    )

    if model_loaded:
        response.model_info = model.get_model_info()

    return response


@app.post("/predict/arrival", response_model=PredictionResponse, tags=["Prediction"])
async def predict_arrival(request: PredictionRequest):
    """
    Predict vessel arrival time using LSTM model

    This endpoint:
    1. Fetches historical AIS data for the vessel
    2. Fetches weather forecast along the route
    3. Preprocesses and engineers features
    4. Runs LSTM inference
    5. Returns ETA with confidence score

    **Note**: Requires AIS and weather data sources to be configured
    """
    # Check if model is loaded
    if model is None or model.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs and ensure model file exists."
        )

    try:
        # STEP 1: Preprocess data
        print(f"[INFO] Processing prediction request for MMSI: {request.vessel_mmsi}")

        sequence_data, static_features = preprocessor.prepare_inference_data(
            vessel_mmsi=request.vessel_mmsi,
            destination_lat=request.destination_lat,
            destination_lon=request.destination_lon
        )

        # STEP 2: Make prediction with confidence
        predicted_minutes, confidence = model.predict_with_confidence(
            sequence_data,
            static_features,
            n_samples=10
        )

        # STEP 3: Calculate ETA
        current_time = datetime.utcnow()
        eta = current_time + timedelta(minutes=predicted_minutes)

        # STEP 4: Extract metadata for response
        # Get current position (would come from AIS data in production)
        # For now, using placeholder
        # TODO: Replace with actual AIS data fetch
        current_pos = {
            "latitude": 0.0,
            "longitude": 0.0,
            "timestamp": current_time.isoformat() + "Z"
        }

        # Calculate distance (using static features)
        distance_km = static_features[0] * 10000  # Denormalize

        # Average speed (using static features)
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
        response = PredictionResponse(
            vessel_mmsi=request.vessel_mmsi,
            current_position=current_pos,
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

        print(f"[SUCCESS] Prediction completed: ETA = {eta.isoformat()}, Confidence = {confidence:.3f}")

        return response

    except NotImplementedError as e:
        raise HTTPException(
            status_code=501,
            detail=f"Data source not configured: {str(e)}"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input data: {str(e)}"
        )
    except Exception as e:
        print(f"[ERROR] Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """
    Get information about the loaded LSTM model
    """
    if model is None or model.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    info = model.get_model_info()

    return ModelInfoResponse(
        model_name=info['model_name'],
        architecture="CNN + Attention + Bidirectional LSTM",
        total_parameters=info['total_parameters'],
        sequence_length=info['sequence_length'],
        features=info['sequence_features']
    )


@app.get("/model/summary", tags=["Model"])
async def get_model_summary():
    """
    Get detailed model architecture summary
    """
    if model is None or model.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    # Capture model summary as string
    import io
    import sys

    buffer = io.StringIO()
    sys.stdout = buffer
    model.summary()
    sys.stdout = sys.__stdout__
    summary = buffer.getvalue()

    return {
        "summary": summary,
        "info": model.get_model_info()
    }


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', 8000))

    print(f"[INFO] Starting TytoAlba Vessel Arrival Prediction Service")
    print(f"[INFO] Server: http://{host}:{port}")
    print(f"[INFO] Docs: http://{host}:{port}/docs")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )
