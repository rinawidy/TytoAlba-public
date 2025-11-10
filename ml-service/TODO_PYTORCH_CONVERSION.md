# TytoAlba ML-Service PyTorch Conversion TODO

**Deadline:** November 9, 2025 (Live Demo)
**Unit Tests Deadline:** November 17, 2025

---

## Phase 1: Setup & Environment ✅

- [x] Install PyTorch CPU version
- [x] Update requirements.txt to use PyTorch instead of TensorFlow
- [x] Create Python virtual environment
- [x] Verify PyTorch installation
- [x] Test basic PyTorch LSTM

---

## Phase 2: Convert ETA Prediction Model (TensorFlow → PyTorch) ✅

### 2.1 Model Architecture Conversion ✅

- [x] **Implement AttentionLayer class** (`src/models/pytorch_arrival_predictor.py`)
  - [x] Create learnable parameters (W and b)
  - [x] Implement forward() method
  - [x] Test attention mechanism independently

- [x] **Implement VesselArrivalLSTM class**
  - [x] Define CNN layers (Conv1d, MaxPool1d)
  - [x] Define Attention layer
  - [x] Define Dropout layers
  - [x] Define Bidirectional LSTM
  - [x] Define Dense (Fully Connected) layers
  - [x] Implement forward() method
  - [x] Handle tensor shape transformations (permute for Conv1d)

### 2.2 Wrapper Class Implementation ✅

- [x] **Implement VesselArrivalPredictor class**
  - [x] Implement predict() method
  - [x] Implement predict_with_confidence() (Monte Carlo Dropout)
  - [x] Implement train() method (full training loop)
  - [x] Implement evaluate() method
  - [x] Test save_model() and load_model()

### 2.3 Testing & Validation

- [ ] Test model with dummy data (pending package installation)
- [ ] Verify output shapes match TensorFlow version
- [ ] Compare parameter count with TensorFlow model
- [ ] Test forward pass end-to-end

---

## Phase 3: Add New Models (Your Homework Requirements)

### 3.1 Fuel Consumption Prediction

- [ ] Create `src/models/fuel_predictor.py`
- [ ] Define LSTM architecture for fuel prediction
  - [ ] Input: speed, distance, engine_rpm, cargo_load (sequence data)
  - [ ] Output: predicted fuel consumption (liters)
- [ ] Implement training script
- [ ] Generate synthetic fuel consumption data
- [ ] Train model (5-10 min on CPU)
- [ ] Test predictions
- [ ] Save trained model

### 3.2 Anomaly Detection

- [ ] Create `src/models/anomaly_detector.py`
- [ ] Implement LSTM Autoencoder architecture
  - [ ] Encoder: LSTM → compressed representation
  - [ ] Decoder: LSTM → reconstruct input
  - [ ] Calculate reconstruction error for anomaly score
- [ ] Generate synthetic normal/anomalous ship behavior data
- [ ] Train autoencoder
- [ ] Define anomaly threshold
- [ ] Test anomaly detection
- [ ] Save trained model

### 3.3 Route Optimization

- [ ] Create `src/models/route_optimizer.py`
- [ ] Define LSTM architecture for route prediction
  - [ ] Input: historical routes (positions, fuel, time)
  - [ ] Output: optimal next waypoint or route score
- [ ] Generate synthetic route data with efficiency scores
- [ ] Train model
- [ ] Test route recommendations
- [ ] Save trained model

---

## Phase 4: Data Generation

- [ ] Create `data/generate_synthetic_data.py`
- [ ] Generate 30 ships × 30 days historical data
  - [ ] Ship positions (lat/lon trajectories)
  - [ ] Speed variations (realistic patterns)
  - [ ] Fuel consumption (based on speed/distance)
  - [ ] Engine RPM data
  - [ ] Weather conditions (wind, waves)
  - [ ] Timestamps
- [ ] Save data as CSV or pickle files
- [ ] Create train/validation/test splits

---

## Phase 5: API Integration

### 5.1 Update FastAPI Endpoints

- [ ] Update `src/api_inference.py` to use PyTorch models
- [ ] Add endpoint: `POST /predict/fuel`
- [ ] Add endpoint: `POST /predict/anomaly`
- [ ] Add endpoint: `POST /predict/route`
- [ ] Update existing `POST /predict/arrival` for PyTorch
- [ ] Update `GET /model/info` to show PyTorch details

### 5.2 Testing API

- [ ] Test health check endpoint
- [ ] Test all prediction endpoints with sample data
- [ ] Verify response formats match documentation
- [ ] Test error handling

---

## Phase 6: Training Scripts

- [ ] Update `train_lstm_model.py` for PyTorch
- [ ] Create `training/train_fuel.py`
- [ ] Create `training/train_anomaly.py`
- [ ] Create `training/train_route.py`
- [ ] Add training progress logging
- [ ] Add model checkpointing
- [ ] Add early stopping

---

## Phase 7: Integration with TytoAlba Backend

- [ ] Update Go backend to call ml-service API
- [ ] Test backend → ml-service communication
- [ ] Handle API errors gracefully
- [ ] Add timeout handling

---

## Phase 8: Testing & Documentation

- [ ] End-to-end testing (frontend → backend → ml-service)
- [ ] Update README.md with PyTorch instructions
- [ ] Document model architectures
- [ ] Create usage examples
- [ ] Test with 29 ships data from `frontend/src/data/ships.json`

---

## Phase 9: Unit Tests (Due Nov 17)

- [ ] Write unit tests for AttentionLayer
- [ ] Write unit tests for VesselArrivalLSTM
- [ ] Write unit tests for fuel predictor
- [ ] Write unit tests for anomaly detector
- [ ] Write unit tests for route optimizer
- [ ] Write unit tests for data preprocessing
- [ ] Write unit tests for API endpoints
- [ ] Write integration tests
- [ ] Achieve >80% code coverage

---

## Phase 10: Test Cases Documentation (Due Nov 24)

- [ ] Document 50+ test cases
- [ ] Include test data samples
- [ ] Document expected outputs
- [ ] Document edge cases
- [ ] Create test execution report

---

## Current Blockers

- [ ] None currently

---

## Notes & Decisions

- **Framework:** PyTorch (CPU only, no GPU)
- **Python Version:** 3.13
- **Device:** CPU (no external GPU access)
- **Approach:** Full conversion from TensorFlow to PyTorch
- **Data:** Synthetic data generation for training

---

## Quick Commands Reference

### Activate Environment
```bash
cd /mnt/c/Users/angga.suryabrata/VisCode/TytoAlba/ml-service
source venv/bin/activate
```

### Test Model
```bash
python src/models/pytorch_arrival_predictor.py
```

### Start API Server
```bash
python run_inference_server.py
```

### Run Training
```bash
python train.py --model fuel --epochs 50
```

### Run Tests
```bash
pytest tests/ -v
```

---

**Last Updated:** November 5, 2025
