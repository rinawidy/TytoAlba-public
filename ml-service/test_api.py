"""
Simple test script to verify API endpoints
Run this after starting the server with run_server.py
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint"""
    print("\n" + "=" * 50)
    print("Testing Health Endpoint")
    print("=" * 50)

    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_fuel_prediction():
    """Test fuel prediction endpoint"""
    print("\n" + "=" * 50)
    print("Testing Fuel Prediction")
    print("=" * 50)

    payload = {
        "distance": 150.5,
        "vehicle_weight": 8000,
        "avg_speed": 65,
        "vehicle_type": 1,
        "terrain_type": 1
    }

    print(f"Request: {json.dumps(payload, indent=2)}")

    response = requests.post(f"{BASE_URL}/predict/fuel", json=payload)
    print(f"\nStatus Code: {response.status_code}")

    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"Error: {response.text}")


def test_arrival_prediction():
    """Test arrival prediction endpoint"""
    print("\n" + "=" * 50)
    print("Testing Arrival Prediction")
    print("=" * 50)

    payload = {
        "distance": 120.0,
        "departure_time": "2024-10-09T14:30:00",
        "route_id": 5,
        "avg_traffic_level": 1,
        "historical_avg_time": 95.5
    }

    print(f"Request: {json.dumps(payload, indent=2)}")

    response = requests.post(f"{BASE_URL}/predict/arrival", json=payload)
    print(f"\nStatus Code: {response.status_code}")

    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"Error: {response.text}")


def test_feature_importance():
    """Test feature importance endpoints"""
    print("\n" + "=" * 50)
    print("Testing Feature Importance")
    print("=" * 50)

    # Fuel model importance
    print("\nFuel Model Feature Importance:")
    response = requests.get(f"{BASE_URL}/models/fuel/importance")
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error: {response.text}")

    # Arrival model importance
    print("\nArrival Model Feature Importance:")
    response = requests.get(f"{BASE_URL}/models/arrival/importance")
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error: {response.text}")


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("ML SERVICE API TESTS")
    print("=" * 50)
    print("Make sure the server is running on http://localhost:8000")
    print("Start server with: python run_server.py")
    print("=" * 50)

    try:
        # Run tests
        test_health()
        test_fuel_prediction()
        test_arrival_prediction()
        test_feature_importance()

        print("\n" + "=" * 50)
        print("ALL TESTS COMPLETED")
        print("=" * 50)

    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to server.")
        print("Make sure the server is running: python run_server.py")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
