"""
FastAPI Application for ML Service
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from src.models.fuel_predictor import FuelConsumptionPredictor
from src.models.pytorch_arrival_predictor import VesselArrivalPredictor
from src.schemas import (
    FuelPredictionRequest,
    FuelPredictionResponse,
    ArrivalPredictionRequest,
    ArrivalPredictionResponse,
    HealthResponse
)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="TytoAlba ML Service",
    description="Machine Learning service for fuel and arrival time predictions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictors
fuel_predictor = FuelConsumptionPredictor()
arrival_predictor = VesselArrivalPredictor()

# Model paths
FUEL_MODEL_PATH = os.getenv('FUEL_MODEL_PATH', 'models/fuel_model.pth')
ARRIVAL_MODEL_PATH = os.getenv('ARRIVAL_MODEL_PATH', 'models/eta_model.pth')


@app.on_event("startup")
async def load_models():
    """Load pre-trained models on startup"""
    global fuel_predictor, arrival_predictor

    # Try to load fuel prediction model
    if os.path.exists(FUEL_MODEL_PATH):
        try:
            fuel_predictor.load_model(FUEL_MODEL_PATH)
            print(f"✓ Fuel prediction model loaded from {FUEL_MODEL_PATH}")
        except Exception as e:
            print(f"✗ Failed to load fuel model: {e}")
    else:
        print(f"⚠ Fuel model not found at {FUEL_MODEL_PATH}. Train the model first.")

    # Try to load arrival prediction model
    if os.path.exists(ARRIVAL_MODEL_PATH):
        try:
            arrival_predictor.load_model(ARRIVAL_MODEL_PATH)
            print(f"✓ Arrival prediction model loaded from {ARRIVAL_MODEL_PATH}")
        except Exception as e:
            print(f"✗ Failed to load arrival model: {e}")
    else:
        print(f"⚠ Arrival model not found at {ARRIVAL_MODEL_PATH}. Train the model first.")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "service": "TytoAlba ML Service",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "fuel_prediction": "/predict/fuel",
            "arrival_prediction": "/predict/arrival"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check service and model health"""
    return HealthResponse(
        status="healthy",
        fuel_model_loaded=fuel_predictor.model is not None,
        arrival_model_loaded=arrival_predictor.model is not None
    )


@app.post("/predict/fuel", response_model=FuelPredictionResponse, tags=["Predictions"])
async def predict_fuel(request: FuelPredictionRequest):
    """
    Predict fuel consumption based on trip parameters

    - **distance**: Distance in kilometers
    - **vehicle_weight**: Vehicle weight in kilograms
    - **avg_speed**: Average speed in km/h
    - **vehicle_type**: 0=small, 1=medium, 2=large
    - **terrain_type**: 0=flat, 1=hilly, 2=mountain
    """
    if fuel_predictor.model is None:
        raise HTTPException(
            status_code=503,
            detail="Fuel prediction model not loaded. Please train the model first."
        )

    try:
        # Convert request to dict
        features = request.model_dump()

        # Make prediction
        predicted_fuel = fuel_predictor.predict(features)

        return FuelPredictionResponse(
            predicted_fuel_liters=round(predicted_fuel, 2),
            features_used=features
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/arrival", response_model=ArrivalPredictionResponse, tags=["Predictions"])
async def predict_arrival(request: ArrivalPredictionRequest):
    """
    Predict arrival time based on trip parameters

    - **distance**: Distance in kilometers
    - **departure_time**: ISO format datetime (optional, will extract hour/day)
    - **departure_hour**: Hour of departure 0-23 (optional if departure_time provided)
    - **day_of_week**: Day of week 0-6 (optional if departure_time provided)
    - **route_id**: Encoded route identifier
    - **avg_traffic_level**: 0=low, 1=medium, 2=high
    - **historical_avg_time**: Historical average time in minutes
    """
    if arrival_predictor.model is None:
        raise HTTPException(
            status_code=503,
            detail="Arrival prediction model not loaded. Please train the model first."
        )

    try:
        # Convert request to dict
        features = request.model_dump()

        # Make prediction with datetime handling
        if request.departure_time:
            result = arrival_predictor.predict_with_datetime(features)
        else:
            travel_time = arrival_predictor.predict(features)
            result = {'travel_time_minutes': round(travel_time, 2)}

        return ArrivalPredictionResponse(
            travel_time_minutes=result['travel_time_minutes'],
            estimated_arrival_time=result.get('estimated_arrival_time'),
            features_used=features
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/models/fuel/importance", tags=["Models"])
async def get_fuel_feature_importance():
    """Get feature importance for fuel prediction model"""
    if fuel_predictor.model is None:
        raise HTTPException(
            status_code=503,
            detail="Fuel prediction model not loaded."
        )

    try:
        importance = fuel_predictor.get_feature_importance()
        return {"feature_importance": importance}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/arrival/importance", tags=["Models"])
async def get_arrival_feature_importance():
    """Get feature importance for arrival prediction model"""
    if arrival_predictor.model is None:
        raise HTTPException(
            status_code=503,
            detail="Arrival prediction model not loaded."
        )

    try:
        importance = arrival_predictor.get_feature_importance()
        return {"feature_importance": importance}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', 8000))

    uvicorn.run(app, host=host, port=port)
