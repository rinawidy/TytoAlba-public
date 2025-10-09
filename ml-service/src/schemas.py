"""
Pydantic schemas for API request/response validation
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class FuelPredictionRequest(BaseModel):
    """Request schema for fuel prediction"""
    distance: float = Field(..., description="Distance in kilometers", gt=0)
    vehicle_weight: float = Field(..., description="Vehicle weight in kilograms", gt=0)
    avg_speed: float = Field(..., description="Average speed in km/h", gt=0, le=200)
    vehicle_type: int = Field(..., description="Vehicle type: 0=small, 1=medium, 2=large", ge=0, le=2)
    terrain_type: int = Field(..., description="Terrain type: 0=flat, 1=hilly, 2=mountain", ge=0, le=2)

    class Config:
        json_schema_extra = {
            "example": {
                "distance": 150.5,
                "vehicle_weight": 8000,
                "avg_speed": 65,
                "vehicle_type": 1,
                "terrain_type": 1
            }
        }


class FuelPredictionResponse(BaseModel):
    """Response schema for fuel prediction"""
    predicted_fuel_liters: float = Field(..., description="Predicted fuel consumption in liters")
    features_used: dict = Field(..., description="Features used for prediction")

    class Config:
        json_schema_extra = {
            "example": {
                "predicted_fuel_liters": 23.45,
                "features_used": {
                    "distance": 150.5,
                    "vehicle_weight": 8000,
                    "avg_speed": 65,
                    "vehicle_type": 1,
                    "terrain_type": 1
                }
            }
        }


class ArrivalPredictionRequest(BaseModel):
    """Request schema for arrival time prediction"""
    distance: float = Field(..., description="Distance in kilometers", gt=0)
    departure_time: Optional[str] = Field(None, description="Departure time in ISO format")
    departure_hour: Optional[int] = Field(None, description="Hour of departure (0-23)", ge=0, le=23)
    day_of_week: Optional[int] = Field(None, description="Day of week (0=Monday, 6=Sunday)", ge=0, le=6)
    route_id: int = Field(..., description="Encoded route identifier", ge=0)
    avg_traffic_level: int = Field(..., description="Traffic level: 0=low, 1=medium, 2=high", ge=0, le=2)
    historical_avg_time: Optional[float] = Field(0, description="Historical average time in minutes")

    class Config:
        json_schema_extra = {
            "example": {
                "distance": 120.0,
                "departure_time": "2024-10-09T14:30:00",
                "route_id": 5,
                "avg_traffic_level": 1,
                "historical_avg_time": 95.5
            }
        }


class ArrivalPredictionResponse(BaseModel):
    """Response schema for arrival time prediction"""
    travel_time_minutes: float = Field(..., description="Predicted travel time in minutes")
    estimated_arrival_time: Optional[str] = Field(None, description="Estimated arrival time in ISO format")
    features_used: dict = Field(..., description="Features used for prediction")

    class Config:
        json_schema_extra = {
            "example": {
                "travel_time_minutes": 98.25,
                "estimated_arrival_time": "2024-10-09T16:08:15",
                "features_used": {
                    "distance": 120.0,
                    "departure_hour": 14,
                    "day_of_week": 2,
                    "route_id": 5,
                    "avg_traffic_level": 1,
                    "historical_avg_time": 95.5
                }
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    fuel_model_loaded: bool
    arrival_model_loaded: bool
