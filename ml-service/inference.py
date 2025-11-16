"""
TytoAlba ML Service - Inference Server
FastAPI-based REST API for vessel arrival prediction

Supports:
- Single vessel prediction
- Batch prediction (up to 30 bulk carrier vessels in parallel)
- GPU/CPU auto-detection
"""

import os
import sys
import asyncio
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.models.pytorch_arrival_predictor import VesselArrivalPredictor
from src.preprocessing.data_pipeline import VoyageDataPreprocessor
from src.preprocessing.utils import haversine_distance


# ============================================================================
# GPU/CPU Configuration
# ============================================================================

def configure_device():
    """Auto-detect and configure GPU/CPU for PyTorch"""
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_count = torch.cuda.device_count()
        print(f"✓ GPU detected: {gpu_count} GPU(s) available")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        return device
    else:
        print("ℹ Using CPU for inference")
        return 'cpu'


# Configure device at startup
DEVICE = configure_device()


# ============================================================================
# Request/Response Schemas
# ============================================================================

class VesselPredictionRequest(BaseModel):
    """Single vessel prediction request"""
    vessel_mmsi: str = Field(..., description="Vessel MMSI identifier")
    ship_type: str = Field(default="bulk_carrier", description="Ship type (bulk_carrier or pusher)")
    destination_lat: float = Field(..., description="Destination latitude", ge=-90, le=90)
    destination_lon: float = Field(..., description="Destination longitude", ge=-180, le=180)


class BatchPredictionRequest(BaseModel):
    """Batch prediction request (up to 30 vessels)"""
    vessels: List[VesselPredictionRequest] = Field(
        ...,
        description="List of vessels to predict",
        max_items=30
    )


class VesselPredictionResponse(BaseModel):
    """Single vessel prediction response"""
    vessel_mmsi: str
    ship_type: str
    current_position: dict
    destination: dict
    estimated_arrival_time: str
    travel_time_minutes: float
    confidence: float
    distance_km: float
    avg_speed_knots: float
    weather_impact: str
    predicted_at: str


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    total_vessels: int
    successful_predictions: int
    failed_predictions: int
    predictions: List[VesselPredictionResponse]
    errors: List[dict]
    processing_time_seconds: float


class ModelInfo(BaseModel):
    """Model information"""
    model_name: str
    architecture: str
    sequence_length: int
    features: int
    static_features: int
    total_parameters: int
    device: str
    supported_ship_types: List[str]
    max_batch_size: int


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="TytoAlba Vessel Arrival Prediction API",
    description="LSTM-based arrival time prediction for bulk carrier vessels",
    version="2.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Global Model and Preprocessor
# ============================================================================

MODEL = None
PREPROCESSOR = None


def load_model():
    """Load pre-trained PyTorch LSTM model"""
    global MODEL, PREPROCESSOR

    model_path = os.getenv('MODEL_PATH', 'models/vessel_arrival_lstm.pth')

    # Initialize model (will work even without trained weights for testing)
    try:
        MODEL = VesselArrivalPredictor(model_path=model_path if os.path.exists(model_path) else None, device=DEVICE)
        PREPROCESSOR = VoyageDataPreprocessor()

        total_params = sum(p.numel() for p in MODEL.model.parameters())

        if os.path.exists(model_path):
            print(f"✓ Model loaded successfully from {model_path}")
        else:
            print(f"⚠ WARNING: Model file not found at {model_path}")
            print("  Using untrained model architecture. Train with: python train.py --synthetic --epochs 50")

        print(f"  Device: {DEVICE}")
        print(f"  Parameters: {total_params:,}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        MODEL = None
        PREPROCESSOR = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    print("=" * 70)
    print("  TytoAlba ML Inference Server Starting...")
    print("=" * 70)
    load_model()


# ============================================================================
# Helper Functions
# ============================================================================

def validate_ship_type(ship_type: str) -> bool:
    """
    Validate ship type - only bulk carriers are supported for predictions

    Args:
        ship_type: Ship type string

    Returns:
        True if bulk_carrier, False otherwise
    """
    return ship_type.lower() == 'bulk_carrier'


async def predict_single_vessel(request: VesselPredictionRequest) -> VesselPredictionResponse:
    """
    Make prediction for a single vessel

    Args:
        request: VesselPredictionRequest

    Returns:
        VesselPredictionResponse

    Raises:
        HTTPException: If prediction fails
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")

    # Validate ship type
    if not validate_ship_type(request.ship_type):
        raise HTTPException(
            status_code=400,
            detail=f"Ship type '{request.ship_type}' not supported. Only 'bulk_carrier' is supported for predictions."
        )

    try:
        # Prepare data
        sequence_data, static_features = PREPROCESSOR.prepare_inference_data(
            vessel_mmsi=request.vessel_mmsi,
            destination_lat=request.destination_lat,
            destination_lon=request.destination_lon
        )

        # Make prediction with confidence
        predicted_minutes, confidence = MODEL.predict_with_confidence(
            sequence_data,
            static_features,
            n_samples=10
        )

        # Calculate ETA
        current_time = datetime.utcnow()
        eta = current_time + timedelta(minutes=predicted_minutes)

        # Get current position
        ais_data = PREPROCESSOR.fetch_ais_history(request.vessel_mmsi, hours=1)
        current_pos = ais_data[-1]

        # Calculate distance
        distance_km = haversine_distance(
            current_pos['latitude'], current_pos['longitude'],
            request.destination_lat, request.destination_lon
        )

        # Average speed from static features (denormalized)
        avg_speed = static_features[1] * 25

        # Weather severity
        weather_severity = static_features[2]
        if weather_severity < 0.3:
            weather_impact = "low"
        elif weather_severity < 0.6:
            weather_impact = "moderate"
        else:
            weather_impact = "high"

        # Build response
        return VesselPredictionResponse(
            vessel_mmsi=request.vessel_mmsi,
            ship_type=request.ship_type,
            current_position={
                "latitude": float(current_pos['latitude']),
                "longitude": float(current_pos['longitude']),
                "timestamp": current_pos['timestamp']
            },
            destination={
                "latitude": request.destination_lat,
                "longitude": request.destination_lon
            },
            estimated_arrival_time=eta.isoformat() + "Z",
            travel_time_minutes=round(float(predicted_minutes), 2),
            confidence=round(float(confidence), 3),
            distance_km=round(float(distance_km), 2),
            avg_speed_knots=round(float(avg_speed), 2),
            weather_impact=weather_impact,
            predicted_at=current_time.isoformat() + "Z"
        )

    except NotImplementedError as e:
        raise HTTPException(
            status_code=501,
            detail=f"Data source not implemented: {str(e)}. Please implement AIS/weather data fetching."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_loaded = MODEL is not None

    if model_loaded:
        total_params = sum(p.numel() for p in MODEL.model.parameters())
        return {
            "status": "healthy",
            "model_loaded": True,
            "device": DEVICE,
            "model_info": {
                "model_name": "VesselArrivalLSTM",
                "framework": "PyTorch",
                "total_parameters": total_params,
                "supported_ship_types": ["bulk_carrier"]
            }
        }
    else:
        return {
            "status": "degraded",
            "model_loaded": False,
            "message": "Model not loaded. Train the model first.",
            "device": DEVICE
        }


@app.post("/predict/arrival", response_model=VesselPredictionResponse)
async def predict_arrival(request: VesselPredictionRequest):
    """
    Predict arrival time for a single bulk carrier vessel

    Only bulk carrier ships are supported for predictions.
    Pusher ships are not supported.
    """
    return await predict_single_vessel(request)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict arrival times for multiple vessels in parallel (max 30)

    Only bulk carrier ships will be processed.
    Other ship types will be skipped with errors reported.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(request.vessels) > 30:
        raise HTTPException(status_code=400, detail="Maximum 30 vessels per batch")

    start_time = datetime.utcnow()

    # Filter bulk carriers only
    bulk_carriers = [v for v in request.vessels if validate_ship_type(v.ship_type)]
    non_bulk_carriers = [v for v in request.vessels if not validate_ship_type(v.ship_type)]

    predictions = []
    errors = []

    # Process bulk carriers in parallel
    tasks = [predict_single_vessel(vessel) for vessel in bulk_carriers]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for vessel, result in zip(bulk_carriers, results):
        if isinstance(result, Exception):
            errors.append({
                "vessel_mmsi": vessel.vessel_mmsi,
                "error": str(result)
            })
        else:
            predictions.append(result)

    # Add errors for non-bulk carriers
    for vessel in non_bulk_carriers:
        errors.append({
            "vessel_mmsi": vessel.vessel_mmsi,
            "ship_type": vessel.ship_type,
            "error": f"Ship type '{vessel.ship_type}' not supported. Only bulk_carrier vessels are supported."
        })

    end_time = datetime.utcnow()
    processing_time = (end_time - start_time).total_seconds()

    return BatchPredictionResponse(
        total_vessels=len(request.vessels),
        successful_predictions=len(predictions),
        failed_predictions=len(errors),
        predictions=predictions,
        errors=errors,
        processing_time_seconds=round(processing_time, 3)
    )


@app.get("/model/info", response_model=ModelInfo)
async def model_info():
    """Get model architecture and configuration information"""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    total_params = sum(p.numel() for p in MODEL.model.parameters())

    return ModelInfo(
        model_name="VesselArrivalLSTM",
        architecture="CNN + Attention + Bidirectional LSTM (PyTorch)",
        sequence_length=48,
        features=8,
        static_features=10,
        total_parameters=total_params,
        device=DEVICE,
        supported_ship_types=["bulk_carrier"],
        max_batch_size=30
    )


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "TytoAlba Vessel Arrival Prediction",
        "version": "2.0.0",
        "framework": "PyTorch",
        "model": "LSTM-based arrival time prediction",
        "supported_ship_types": ["bulk_carrier"],
        "device": DEVICE,
        "endpoints": {
            "health": "/health",
            "predict_single": "/predict/arrival",
            "predict_batch": "/predict/batch",
            "model_info": "/model/info",
            "docs": "/docs"
        }
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', '8000'))

    print(f"\n🚀 Starting inference server on {host}:{port}")
    print(f"   Device: {DEVICE}")
    print(f"   API Docs: http://{host}:{port}/docs")

    uvicorn.run(app, host=host, port=port)
