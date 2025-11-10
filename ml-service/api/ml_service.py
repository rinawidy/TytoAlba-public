"""
TytoAlba ML Service API
Unified REST API for all 4 LSTM prediction models

Endpoints:
- POST /api/predict/eta - ETA/Arrival prediction
- POST /api/predict/fuel - Fuel consumption prediction
- POST /api/detect/anomaly - Anomaly detection
- POST /api/optimize/route - Route optimization

Run: python ml_service.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import all models
try:
    from models.pytorch_arrival_predictor import VesselArrivalPredictor
    from models.fuel_predictor import FuelConsumptionPredictor
    from models.anomaly_detector import VesselAnomalyDetector
    from models.route_optimizer import RouteOptimizer
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Models not available: {e}")
    MODELS_AVAILABLE = False

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Global model instances
eta_model = None
fuel_model = None
anomaly_model = None
route_model = None

def initialize_models():
    """Load all LSTM models"""
    global eta_model, fuel_model, anomaly_model, route_model

    print("[INFO] Initializing ML models...")

    try:
        eta_model = VesselArrivalPredictor(device='cpu')
        print("✓ ETA model loaded")
    except Exception as e:
        print(f"✗ ETA model failed: {e}")

    try:
        fuel_model = FuelConsumptionPredictor(device='cpu')
        print("✓ Fuel model loaded")
    except Exception as e:
        print(f"✗ Fuel model failed: {e}")

    try:
        anomaly_model = VesselAnomalyDetector(device='cpu')
        anomaly_model.threshold = 0.5  # Default threshold
        print("✓ Anomaly model loaded")
    except Exception as e:
        print(f"✗ Anomaly model failed: {e}")

    try:
        route_model = RouteOptimizer(device='cpu')
        print("✓ Route model loaded")
    except Exception as e:
        print(f"✗ Route model failed: {e}")

    print("[INFO] Model initialization complete!")


# =======================
# HEALTH CHECK ENDPOINTS
# =======================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models': {
            'eta': eta_model is not None,
            'fuel': fuel_model is not None,
            'anomaly': anomaly_model is not None,
            'route': route_model is not None
        }
    })

@app.route('/api/models/info', methods=['GET'])
def models_info():
    """Get information about all models"""
    info = {}

    if eta_model:
        info['eta'] = eta_model.get_model_info()

    if fuel_model:
        total_params = sum(p.numel() for p in fuel_model.model.parameters())
        info['fuel'] = {
            'model_name': 'FuelConsumptionLSTM',
            'framework': 'PyTorch',
            'total_parameters': total_params
        }

    if anomaly_model:
        total_params = sum(p.numel() for p in anomaly_model.model.parameters())
        info['anomaly'] = {
            'model_name': 'LSTMAutoencoder',
            'framework': 'PyTorch',
            'total_parameters': total_params,
            'threshold': anomaly_model.threshold
        }

    if route_model:
        total_params = sum(p.numel() for p in route_model.model.parameters())
        info['route'] = {
            'model_name': 'RouteOptimizationLSTM',
            'framework': 'PyTorch',
            'total_parameters': total_params
        }

    return jsonify(info)


# =======================
# ETA PREDICTION
# =======================

@app.route('/api/predict/eta', methods=['POST'])
def predict_eta():
    """
    Predict vessel arrival time

    Request body:
    {
        "mmsi": "525001001",
        "sequence_data": [[lat, lon, speed, ...], ...],  // 48 timesteps, 8 features
        "static_features": [dwt, loa, beam, ...]  // 10 features
    }

    Response:
    {
        "mmsi": "525001001",
        "eta_minutes": 1234.5,
        "eta_hours": 20.6,
        "confidence": 0.92
    }
    """
    if not eta_model:
        return jsonify({'error': 'ETA model not initialized'}), 500

    try:
        data = request.get_json()

        sequence_data = np.array(data['sequence_data'], dtype=np.float32)
        static_features = np.array(data['static_features'], dtype=np.float32)

        # Validate shapes
        if sequence_data.shape != (48, 8):
            return jsonify({'error': f'Invalid sequence shape: {sequence_data.shape}, expected (48, 8)'}), 400
        if static_features.shape != (10,):
            return jsonify({'error': f'Invalid static shape: {static_features.shape}, expected (10,)'}), 400

        # Predict
        eta_minutes = eta_model.predict(sequence_data, static_features)
        mean_eta, confidence = eta_model.predict_with_confidence(sequence_data, static_features)

        return jsonify({
            'mmsi': data.get('mmsi', 'unknown'),
            'eta_minutes': float(eta_minutes),
            'eta_hours': float(eta_minutes / 60),
            'mean_eta_minutes': float(mean_eta),
            'confidence': float(confidence)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =======================
# FUEL CONSUMPTION
# =======================

@app.route('/api/predict/fuel', methods=['POST'])
def predict_fuel():
    """
    Predict fuel consumption

    Request body:
    {
        "mmsi": "525001001",
        "sequence_data": [[speed, rpm, load, ...], ...],  // 48 timesteps, 10 features
        "static_features": [dwt, engine_power, ...]  // 8 features
    }

    Response:
    {
        "mmsi": "525001001",
        "fuel_consumption_lph": 1250.5,
        "confidence": 0.88
    }
    """
    if not fuel_model:
        return jsonify({'error': 'Fuel model not initialized'}), 500

    try:
        data = request.get_json()

        sequence_data = np.array(data['sequence_data'], dtype=np.float32)
        static_features = np.array(data['static_features'], dtype=np.float32)

        # Validate shapes
        if sequence_data.shape != (48, 10):
            return jsonify({'error': f'Invalid sequence shape: {sequence_data.shape}, expected (48, 10)'}), 400
        if static_features.shape != (8,):
            return jsonify({'error': f'Invalid static shape: {static_features.shape}, expected (8,)'}), 400

        # Predict
        fuel_consumption = fuel_model.predict(sequence_data, static_features)
        mean_fuel, confidence = fuel_model.predict_with_confidence(sequence_data, static_features)

        return jsonify({
            'mmsi': data.get('mmsi', 'unknown'),
            'fuel_consumption_lph': float(fuel_consumption),
            'mean_fuel_consumption_lph': float(mean_fuel),
            'confidence': float(confidence)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =======================
# ANOMALY DETECTION
# =======================

@app.route('/api/detect/anomaly', methods=['POST'])
def detect_anomaly():
    """
    Detect anomalous vessel behavior

    Request body:
    {
        "mmsi": "525001001",
        "sequence_data": [[lat, lon, speed, ...], ...]  // 48 timesteps, 12 features
    }

    Response:
    {
        "mmsi": "525001001",
        "is_anomaly": true,
        "anomaly_score": 0.75,
        "severity": "moderate",
        "threshold": 0.50,
        "confidence": 0.95
    }
    """
    if not anomaly_model:
        return jsonify({'error': 'Anomaly model not initialized'}), 500

    try:
        data = request.get_json()

        sequence_data = np.array(data['sequence_data'], dtype=np.float32)

        # Validate shape
        if sequence_data.shape != (48, 12):
            return jsonify({'error': f'Invalid sequence shape: {sequence_data.shape}, expected (48, 12)'}), 400

        # Detect
        result = anomaly_model.detect_anomaly(sequence_data)

        return jsonify({
            'mmsi': data.get('mmsi', 'unknown'),
            'is_anomaly': result['is_anomaly'],
            'anomaly_score': result['anomaly_score'],
            'severity': result['severity'],
            'threshold': result['threshold'],
            'confidence': result['confidence']
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =======================
# ROUTE OPTIMIZATION
# =======================

@app.route('/api/optimize/route', methods=['POST'])
def optimize_route():
    """
    Optimize vessel route

    Request body:
    {
        "mmsi": "525001001",
        "trajectory_history": [[lat, lon, speed, heading], ...],  // 24 timesteps, 4 features
        "environment_history": [[wave, wind_speed, ...], ...],  // 24 timesteps, 6 features
        "vessel_specs": [loa, beam, draft, max_speed, fuel_capacity],  // 5 features
        "destination": [dest_lat, dest_lon]  // 2 features
    }

    Response:
    {
        "mmsi": "525001001",
        "waypoints": [[lat1, lon1], [lat2, lon2], ...],  // 12 waypoints
        "fuel_consumption": 15000.0,
        "eta_hours": 8.5
    }
    """
    if not route_model:
        return jsonify({'error': 'Route model not initialized'}), 500

    try:
        data = request.get_json()

        trajectory_history = np.array(data['trajectory_history'], dtype=np.float32)
        environment_history = np.array(data['environment_history'], dtype=np.float32)
        vessel_specs = np.array(data['vessel_specs'], dtype=np.float32)
        destination = np.array(data['destination'], dtype=np.float32)

        # Validate shapes
        if trajectory_history.shape != (24, 4):
            return jsonify({'error': f'Invalid trajectory shape: {trajectory_history.shape}, expected (24, 4)'}), 400
        if environment_history.shape != (24, 6):
            return jsonify({'error': f'Invalid environment shape: {environment_history.shape}, expected (24, 6)'}), 400
        if vessel_specs.shape != (5,):
            return jsonify({'error': f'Invalid vessel specs shape: {vessel_specs.shape}, expected (5,)'}), 400
        if destination.shape != (2,):
            return jsonify({'error': f'Invalid destination shape: {destination.shape}, expected (2,)'}), 400

        # Optimize
        result = route_model.predict_route(
            trajectory_history, environment_history, vessel_specs, destination
        )

        return jsonify({
            'mmsi': data.get('mmsi', 'unknown'),
            'waypoints': result['waypoints'].tolist(),
            'fuel_consumption': float(result['fuel_consumption']),
            'eta_hours': float(result['eta_hours'])
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =======================
# MAIN
# =======================

if __name__ == '__main__':
    if not MODELS_AVAILABLE:
        print("[ERROR] Models not available. Install PyTorch: pip install torch numpy")
        print("[WARNING] Starting server anyway (will return errors)")
    else:
        initialize_models()

    print("\n" + "="*70)
    print("TytoAlba ML Service API")
    print("="*70)
    print("\nEndpoints:")
    print("  GET  /health                - Health check")
    print("  GET  /api/models/info       - Model information")
    print("  POST /api/predict/eta       - ETA prediction")
    print("  POST /api/predict/fuel      - Fuel consumption")
    print("  POST /api/detect/anomaly    - Anomaly detection")
    print("  POST /api/optimize/route    - Route optimization")
    print("\nStarting server on http://0.0.0.0:5000")
    print("="*70 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=True)
