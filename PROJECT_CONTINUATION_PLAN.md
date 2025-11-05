# TytoAlba Project Continuation Plan
## Predictive Monitoring System for Coal Delivery Vessel

**Document Owner:** Kelompok 2 (Angga, Rina, Putri)
**Current Sprint:** 8-9 (December 2025) - Development Sprint 2
**Target Completion:** February 2026 (UAS)
**Last Updated:** October 31, 2025

---

## Executive Summary

This document provides a comprehensive roadmap to complete the TytoAlba Maritime Vessel Tracking & Prediction System from its current state (UTS - Mid-term) to final delivery (UAS - Final exam). The project is currently in Sprint 8-9 with core backend completed and frontend/ML services in progress.

**Current Status:**
- ✅ Backend API (Go) - **COMPLETED**
- ✅ MQTT Broker Setup - **COMPLETED**
- 🔄 Frontend (Vue.js) - **IN PROGRESS**
- 🔄 ML Service (Python/TensorFlow) - **IN PROGRESS**
- 🔄 LSTM Model Training - **92% accuracy (target: 95%)**
- ⏳ Testing & QA - **PENDING**
- ⏳ Production Deployment - **PENDING**

---

## Phase 1: Code Quality Improvements (Week 1-2)
### Priority: HIGH | Timeline: Nov 1-14, 2025

Based on cyclomatic complexity analysis, we need to refactor high-complexity functions to improve maintainability and testability.

### 1.1 Refactor High-Complexity Functions

#### Task 1.1.1: Refactor `remove_outliers` (CC=10 → CC=3)
**Owner:** Angga
**File:** `ml-service/src/preprocessing/data_pipeline.py:164-209`
**Current CC:** 10 (highest in project)
**Target CC:** 3

**Action Items:**
1. Extract `_is_valid_speed(point)` helper function (CC=1)
2. Extract `_is_valid_transition(prev, current)` helper function (CC=2)
3. Extract `_calculate_time_diff(prev, current)` helper function (CC=1)
4. Extract `_normalize_timestamp(timestamp)` helper function (CC=2)
5. Rewrite main function to use helper functions

**Before:**
```python
def remove_outliers(self, ais_data):
    # Nested if statements, CC=10
    cleaned = []
    for i, point in enumerate(ais_data):
        if point['speed'] < 0 or point['speed'] > 30:
            continue
        if i > 0:
            # More nested conditions...
```

**After:**
```python
def remove_outliers(self, ais_data):
    # Simplified, CC=3
    cleaned = []
    for i, point in enumerate(ais_data):
        if not self._is_valid_speed(point):
            continue
        if i > 0 and not self._is_valid_transition(ais_data[i-1], point):
            continue
        cleaned.append(point)
    return cleaned
```

**Acceptance Criteria:**
- [ ] CC reduced to ≤ 3
- [ ] All existing functionality preserved
- [ ] Unit tests pass
- [ ] Code review approved

---

#### Task 1.1.2: Extract Parameter Parsing in `GetShipHistory` (CC=7 → CC=5)
**Owner:** Angga
**File:** `backend/internal/handlers/mqtt_ships.go:100-137`
**Current CC:** 7
**Target CC:** 5

**Action Items:**
1. Create `parseHoursParameter(hoursStr string, defaultValue int) int` function
2. Move parameter parsing logic out of handler
3. Update GetShipHistory to use helper function

**Acceptance Criteria:**
- [ ] CC reduced to ≤ 5
- [ ] Helper function is reusable
- [ ] Unit tests for parameter parsing
- [ ] Handler tests pass

---

#### Task 1.1.3: Create CORS and Error Handling Middleware
**Owner:** Angga
**File:** `backend/internal/middleware/` (new)

**Action Items:**
1. Create `withCORS(next http.HandlerFunc) http.HandlerFunc` middleware
2. Create `withShipStore(next http.HandlerFunc) http.HandlerFunc` middleware
3. Update all handlers to use middleware
4. Remove duplicated CORS code from handlers

**Benefits:**
- Reduces CC in each handler by ~2 points
- DRY principle (Don't Repeat Yourself)
- Centralized error handling
- Easier to modify CORS policy globally

**Acceptance Criteria:**
- [ ] Middleware functions created
- [ ] All 5 handlers updated to use middleware
- [ ] Integration tests pass
- [ ] No CORS-related code duplication

---

### 1.2 Add Comprehensive Unit Tests

#### Task 1.2.1: Unit Tests for `remove_outliers`
**Owner:** Putri
**File:** `ml-service/tests/test_data_pipeline.py`
**Test Count:** 10+

**Test Cases:**
1. Test valid speed range (0-30 knots) - PASS
2. Test speed too low (< 0) - REMOVE
3. Test speed too high (> 30) - REMOVE
4. Test first point (no previous point) - KEEP
5. Test valid position change - KEEP
6. Test implausible position jump - REMOVE
7. Test string timestamp conversion - KEEP
8. Test datetime timestamp handling - KEEP
9. Test zero time difference - KEEP
10. Test edge case: exactly 30 knots - KEEP

**Acceptance Criteria:**
- [ ] All 10 test cases written
- [ ] Code coverage for remove_outliers ≥ 95%
- [ ] Tests run in CI/CD pipeline
- [ ] Documentation updated

---

#### Task 1.2.2: Unit Tests for `GetShipHistory`
**Owner:** Angga
**File:** `backend/internal/handlers/mqtt_ships_test.go`
**Test Count:** 7+

**Test Cases:**
1. Test OPTIONS request (CORS preflight) - HTTP 200
2. Test nil ShipStore - HTTP 500
3. Test missing MMSI parameter - HTTP 400
4. Test valid request with default hours (24) - HTTP 200
5. Test valid request with custom hours - HTTP 200
6. Test invalid hours parameter (non-numeric) - Default to 24
7. Test JSON encoding error simulation - HTTP 500

**Acceptance Criteria:**
- [ ] All 7 test cases written
- [ ] Mock ShipStore for testing
- [ ] Code coverage ≥ 90%
- [ ] Tests integrated in CI/CD

---

### 1.3 CI/CD Integration for Complexity Monitoring

#### Task 1.3.1: Add gocyclo to CI Pipeline
**Owner:** Angga
**File:** `.github/workflows/code-quality.yml`

**Action Items:**
1. Install gocyclo in GitHub Actions
2. Run `gocyclo -over 10 backend/` in CI
3. Fail build if any function has CC > 15
4. Generate complexity report as artifact

**Configuration:**
```yaml
name: Go Code Quality

on: [push, pull_request]

jobs:
  complexity:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-go@v2
      - name: Install gocyclo
        run: go install github.com/fzipp/gocyclo/cmd/gocyclo@latest
      - name: Check complexity
        run: gocyclo -over 15 backend/ || exit 1
      - name: Generate report
        run: gocyclo backend/ > complexity-report.txt
      - name: Upload report
        uses: actions/upload-artifact@v2
        with:
          name: complexity-report
          path: complexity-report.txt
```

**Acceptance Criteria:**
- [ ] CI workflow created
- [ ] Complexity check runs on every push
- [ ] Build fails if CC > 15
- [ ] Reports accessible in GitHub Actions

---

#### Task 1.3.2: Add radon to CI Pipeline
**Owner:** Putri
**File:** `.github/workflows/code-quality.yml`

**Action Items:**
1. Install radon in GitHub Actions
2. Run `radon cc -nc ml-service/` in CI (no grade C or worse)
3. Fail build if any function has CC > 10
4. Generate complexity and maintainability reports

**Configuration:**
```yaml
name: Python Code Quality

on: [push, pull_request]

jobs:
  complexity:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - name: Install radon
        run: pip install radon
      - name: Check complexity
        run: radon cc -nc ml-service/ || exit 1
      - name: Maintainability index
        run: radon mi ml-service/ -s
      - name: Generate reports
        run: |
          radon cc ml-service/ -a > cc-report.txt
          radon mi ml-service/ > mi-report.txt
      - name: Upload reports
        uses: actions/upload-artifact@v2
        with:
          name: python-quality-reports
          path: "*-report.txt"
```

**Acceptance Criteria:**
- [ ] CI workflow created
- [ ] Complexity check runs on every push
- [ ] Build fails if any function has grade C or worse
- [ ] Maintainability index tracked

---

## Phase 2: Complete Core Features (Week 3-6)
### Priority: CRITICAL | Timeline: Nov 15 - Dec 12, 2025

### 2.1 Frontend Development (Vue.js 3)

#### Task 2.1.1: Complete Dashboard View
**Owner:** Putri
**File:** `frontend/src/views/Dashboard.vue`

**Features:**
1. Real-time fleet map showing all vessels
2. Color-coded ship status indicators:
   - 🟢 Green: On schedule
   - 🟡 Yellow: Delayed (< 1 hour)
   - 🔴 Red: Significantly delayed (> 1 hour)
3. Arrival prediction panel with confidence scores
4. Fleet statistics summary

**Technology Stack:**
- Vue 3 Composition API
- TypeScript for type safety
- Tailwind CSS for styling
- Leaflet.js for mapping

**Acceptance Criteria:**
- [ ] Dashboard displays all active vessels
- [ ] Real-time updates via WebSocket/polling
- [ ] Color-coded status working correctly
- [ ] Responsive design (mobile-friendly)
- [ ] Loading states and error handling

---

#### Task 2.1.2: Implement Route Visualization
**Owner:** Putri
**File:** `frontend/src/components/RouteMap.vue`

**Features:**
1. Interactive map with zoom/pan
2. Dual-line rendering:
   - Solid green line: Path already traversed
   - Dotted gray line: Remaining route to destination
3. Ship icon showing current position and heading
4. Waypoint markers
5. Destination port information popup

**Technical Requirements:**
- Use Leaflet.js or Mapbox GL
- Custom line rendering with SVG
- Real-time position updates
- Smooth animations for ship movement

**Acceptance Criteria:**
- [ ] Map loads correctly with ship data
- [ ] Dual-line rendering works as specified
- [ ] Ship icon rotates based on heading
- [ ] Waypoints are clickable
- [ ] Performance optimized (smooth at 60fps)

---

#### Task 2.1.3: Integrate Weather Overlay
**Owner:** Putri
**File:** `frontend/src/components/WeatherOverlay.vue`

**Features:**
1. Weather layer toggle on/off
2. Weather icons along route (cloud, rain, wind)
3. Weather forecast timeline
4. Wind direction arrows
5. Wave height indicators

**Data Source:**
- OpenWeatherMap API (or alternative)
- Weather data fetched from backend API
- Cache weather data (5-minute refresh)

**Acceptance Criteria:**
- [ ] Weather layer toggleable
- [ ] Weather icons display correctly
- [ ] Forecast data accurate
- [ ] Layer doesn't obscure ship routes
- [ ] API rate limiting handled

---

### 2.2 ML Service Integration

#### Task 2.2.1: Complete ML Service API
**Owner:** Angga + Putri
**File:** `ml-service/src/api_inference.py`

**Endpoints to Implement:**
1. `POST /predict/arrival` - Single vessel ETA prediction
2. `POST /predict/batch` - Batch prediction (up to 30 vessels)
3. `GET /model/info` - Model metadata and version
4. `GET /health` - Service health check
5. `POST /predict/uncertainty` - Confidence interval estimation

**Request/Response Schema:**
```json
POST /predict/arrival
Request:
{
  "vessel_mmsi": "563012345",
  "ship_type": "bulk_carrier",
  "destination_lat": 1.2644,
  "destination_lon": 103.8229,
  "current_weather": { ... }
}

Response:
{
  "predicted_eta": "2025-11-05T14:30:00Z",
  "confidence_score": 0.92,
  "confidence_interval": {
    "lower": "2025-11-05T14:00:00Z",
    "upper": "2025-11-05T15:00:00Z"
  },
  "mae_minutes": 28,
  "prediction_timestamp": "2025-11-04T10:00:00Z"
}
```

**Acceptance Criteria:**
- [ ] All 5 endpoints implemented
- [ ] OpenAPI/Swagger documentation
- [ ] Input validation with Pydantic schemas
- [ ] Error handling (400, 404, 500)
- [ ] Response time < 500ms (p95)

---

#### Task 2.2.2: Improve LSTM Model Accuracy (92% → 95%)
**Owner:** Putri
**File:** `ml-service/src/models/lstm_arrival_predictor.py`

**Current Performance:**
- Accuracy: 92%
- MAE: ~35 minutes
- MAPE: ~7%

**Target Performance:**
- Accuracy: ≥ 95%
- MAE: < 30 minutes
- MAPE: < 5%

**Improvement Strategies:**
1. **Hyperparameter Tuning:**
   - Grid search for LSTM units (64, 128, 256)
   - Dropout rates (0.1, 0.2, 0.3, 0.4)
   - Learning rates (0.0001, 0.001, 0.01)
   - Batch sizes (16, 32, 64)

2. **Feature Engineering:**
   - Add distance-to-destination feature
   - Cyclical time encoding (hour, day, month)
   - Weather severity composite score
   - Vessel-specific features (tonnage, draft)

3. **Model Architecture:**
   - Add attention mechanism (Bahdanau attention)
   - Try bidirectional LSTM
   - Add residual connections
   - Experiment with CNN + LSTM hybrid

4. **Data Augmentation:**
   - Gather more training data (5000+ voyages)
   - Synthetic data generation for edge cases
   - Data balancing (oversampling rare scenarios)

5. **Ensemble Methods:**
   - Train 3 models with different architectures
   - Weighted averaging of predictions
   - Variance-based confidence scoring

**Acceptance Criteria:**
- [ ] MAE < 30 minutes on validation set
- [ ] MAPE < 5%
- [ ] Confidence intervals cover 95% of actual arrivals
- [ ] Model generalizes to unseen routes
- [ ] Hyperparameter tuning results documented

---

#### Task 2.2.3: Implement Uncertainty Quantification
**Owner:** Putri
**File:** `ml-service/src/models/uncertainty.py`

**Methods:**
1. **Monte Carlo Dropout:**
   - Run inference 100 times with dropout enabled
   - Calculate mean and std of predictions
   - Convert to 95% confidence interval

2. **Bayesian LSTM:**
   - Use TensorFlow Probability
   - Variational inference for weight uncertainty
   - Predictive distribution instead of point estimate

**Implementation:**
```python
def predict_with_uncertainty(self, X, n_samples=100):
    predictions = []
    for _ in range(n_samples):
        pred = self.model(X, training=True)  # Keep dropout active
        predictions.append(pred)

    predictions = np.array(predictions)
    mean = np.mean(predictions, axis=0)
    std = np.std(predictions, axis=0)

    # 95% confidence interval
    lower = mean - 1.96 * std
    upper = mean + 1.96 * std

    return {
        'predicted_eta': mean,
        'confidence_interval': {'lower': lower, 'upper': upper},
        'confidence_score': 1 / (1 + std)  # Higher std = lower confidence
    }
```

**Acceptance Criteria:**
- [ ] Uncertainty estimation implemented
- [ ] Confidence intervals calibrated (95% coverage)
- [ ] Confidence scores correlate with actual error
- [ ] Inference time < 2 seconds for batch

---

### 2.3 Database Implementation

#### Task 2.3.1: Set Up PostgreSQL with TimescaleDB
**Owner:** Rina
**Files:** `database/schema.sql`, `database/migrations/`

**Schema Design:**

**Table: ships**
```sql
CREATE TABLE ships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    mmsi VARCHAR(20) UNIQUE NOT NULL,
    name VARCHAR(255),
    ship_type VARCHAR(50),
    tonnage INTEGER,
    max_speed DECIMAL(5,2),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

**Table: ais_positions (Hypertable)**
```sql
CREATE TABLE ais_positions (
    time TIMESTAMPTZ NOT NULL,
    ship_id UUID REFERENCES ships(id),
    latitude DECIMAL(10,7),
    longitude DECIMAL(10,7),
    speed DECIMAL(5,2),
    course DECIMAL(5,2),
    heading DECIMAL(5,2),
    status VARCHAR(50)
);

SELECT create_hypertable('ais_positions', 'time');
```

**Table: predictions**
```sql
CREATE TABLE predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ship_id UUID REFERENCES ships(id),
    created_at TIMESTAMPTZ NOT NULL,
    predicted_eta TIMESTAMPTZ NOT NULL,
    confidence_score DECIMAL(3,2),
    actual_eta TIMESTAMPTZ,
    mae_minutes INTEGER,
    model_version VARCHAR(50)
);
```

**Table: voyages**
```sql
CREATE TABLE voyages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ship_id UUID REFERENCES ships(id),
    departure_port VARCHAR(255),
    destination_port VARCHAR(255),
    departure_time TIMESTAMPTZ,
    estimated_arrival TIMESTAMPTZ,
    actual_arrival TIMESTAMPTZ,
    status VARCHAR(50)
);
```

**Indexes:**
```sql
CREATE INDEX idx_ais_positions_ship_time ON ais_positions (ship_id, time DESC);
CREATE INDEX idx_predictions_ship_created ON predictions (ship_id, created_at DESC);
CREATE INDEX idx_voyages_ship_status ON voyages (ship_id, status);
```

**Acceptance Criteria:**
- [ ] All tables created with proper relationships
- [ ] TimescaleDB hypertable for time-series data
- [ ] Indexes optimized for common queries
- [ ] Migration scripts work forward and backward
- [ ] Data retention policy configured (1 year)

---

#### Task 2.3.2: Implement Data Access Layer (Go)
**Owner:** Angga + Rina
**File:** `backend/internal/database/`

**Modules:**
1. `database/connection.go` - Connection pooling
2. `database/ships.go` - Ship CRUD operations
3. `database/ais.go` - AIS position queries
4. `database/predictions.go` - Prediction logging
5. `database/voyages.go` - Voyage management

**Key Functions:**
```go
// Ship operations
func GetShipByMMSI(mmsi string) (*Ship, error)
func GetAllActiveShips() ([]Ship, error)
func UpdateShipPosition(mmsi string, position AISPosition) error

// AIS position queries
func GetShipHistory(shipID uuid.UUID, hours int) ([]AISPosition, error)
func GetLatestPosition(shipID uuid.UUID) (*AISPosition, error)

// Prediction logging
func LogPrediction(prediction *Prediction) error
func GetPredictionAccuracy(shipID uuid.UUID, days int) (float64, error)

// Voyage management
func CreateVoyage(voyage *Voyage) error
func UpdateVoyageETA(voyageID uuid.UUID, eta time.Time) error
func CompleteVoyage(voyageID uuid.UUID, actualArrival time.Time) error
```

**Acceptance Criteria:**
- [ ] All CRUD operations implemented
- [ ] Connection pooling configured
- [ ] Prepared statements for security
- [ ] Error handling and logging
- [ ] Integration tests with test database

---

#### Task 2.3.3: Migrate In-Memory Store to PostgreSQL
**Owner:** Rina
**Files:** `backend/internal/storage/ship_store.go` (refactor)

**Migration Plan:**
1. Keep in-memory cache for performance (Redis-like)
2. Persist all data to PostgreSQL
3. Cache expiration: 5 minutes
4. Write-through cache strategy
5. Fallback to DB if cache miss

**Implementation:**
```go
type ShipStore struct {
    db    *sql.DB
    cache map[string]*Ship
    mutex sync.RWMutex
}

func (s *ShipStore) GetShip(mmsi string) (*Ship, error) {
    // Try cache first
    s.mutex.RLock()
    ship, exists := s.cache[mmsi]
    s.mutex.RUnlock()

    if exists && !ship.IsExpired() {
        return ship, nil
    }

    // Cache miss or expired, fetch from DB
    ship, err := s.db.GetShipByMMSI(mmsi)
    if err != nil {
        return nil, err
    }

    // Update cache
    s.mutex.Lock()
    s.cache[mmsi] = ship
    s.mutex.Unlock()

    return ship, nil
}
```

**Acceptance Criteria:**
- [ ] In-memory cache for hot data
- [ ] All writes persisted to PostgreSQL
- [ ] Cache invalidation strategy implemented
- [ ] Performance maintained (< 100ms API latency)
- [ ] Data consistency guaranteed

---

## Phase 3: Testing & Quality Assurance (Week 7-8)
### Priority: CRITICAL | Timeline: Dec 13-26, 2025

### 3.1 Unit Testing

#### Task 3.1.1: Backend Unit Tests (Go)
**Owner:** Angga
**Target Coverage:** 80%+

**Test Files:**
- `backend/internal/handlers/mqtt_ships_test.go`
- `backend/internal/mqtt/broker_test.go`
- `backend/internal/storage/ship_store_test.go`
- `backend/internal/database/*_test.go`

**Tools:**
- Go testing framework (`testing` package)
- Testify for assertions
- Mock database with `sqlmock`

**Acceptance Criteria:**
- [ ] Code coverage ≥ 80%
- [ ] All critical paths tested
- [ ] Mocks for external dependencies
- [ ] Tests run in < 5 seconds

---

#### Task 3.1.2: Frontend Unit Tests (Vue.js)
**Owner:** Putri
**Target Coverage:** 70%+

**Test Files:**
- `frontend/src/views/__tests__/Dashboard.spec.ts`
- `frontend/src/components/__tests__/RouteMap.spec.ts`
- `frontend/src/components/__tests__/WeatherOverlay.spec.ts`

**Tools:**
- Vitest (Vite-native test runner)
- Vue Test Utils
- Happy-DOM or jsdom

**Acceptance Criteria:**
- [ ] Code coverage ≥ 70%
- [ ] Component rendering tests
- [ ] User interaction tests
- [ ] Snapshot tests for UI

---

#### Task 3.1.3: ML Service Unit Tests (Python)
**Owner:** Putri
**Target Coverage:** 85%+

**Test Files:**
- `ml-service/tests/test_data_pipeline.py`
- `ml-service/tests/test_lstm_model.py`
- `ml-service/tests/test_api_inference.py`

**Tools:**
- pytest
- unittest.mock
- TensorFlow test utilities

**Acceptance Criteria:**
- [ ] Code coverage ≥ 85%
- [ ] Model inference tests
- [ ] Data preprocessing tests
- [ ] API endpoint tests

---

### 3.2 Integration Testing

#### Task 3.2.1: API Integration Tests
**Owner:** Angga
**Tool:** Postman/Newman + Go integration tests

**Test Scenarios:**
1. Complete prediction flow:
   - Frontend → Backend API → ML Service → Database
2. MQTT message flow:
   - Ship publishes → MQTT Broker → Backend → Database
3. Real-time updates:
   - Database change → Backend notification → Frontend WebSocket

**Test Collection:**
```
Collection: TytoAlba API Integration Tests
├── Authentication
│   ├── Login with valid credentials
│   └── Login with invalid credentials
├── Ship Management
│   ├── Get all ships
│   ├── Get ship by MMSI
│   ├── Get ship history
│   └── Update ship position
├── Predictions
│   ├── Single vessel prediction
│   ├── Batch prediction (5 vessels)
│   ├── Batch prediction (30 vessels - max)
│   └── Prediction with invalid data
└── Health Checks
    ├── Backend health
    └── ML service health
```

**Acceptance Criteria:**
- [ ] All integration tests pass
- [ ] End-to-end data flow validated
- [ ] Error handling tested
- [ ] Tests automated in CI/CD

---

#### Task 3.2.2: MQTT Integration Tests
**Owner:** Angga
**Tool:** Mosquitto test clients

**Test Scenarios:**
1. Publish AIS position data → Backend receives → Database updated
2. Publish sensor data (fuel, RPM) → Backend receives → Stored
3. Connection loss → Reconnect → Resume subscription
4. Message QoS levels (0, 1, 2) validation
5. Wildcard topic subscriptions (+, #)

**Test Script:**
```bash
# Test 1: Publish ship position
mosquitto_pub -h localhost -t "tytoalba/ships/563012345/ais" -m '{
  "vessel_mmsi": "563012345",
  "ship_type": "bulk_carrier",
  "timestamp": "2025-11-04T10:00:00Z",
  "latitude": -5.5,
  "longitude": 112.5,
  "speed": 12.5,
  "course": 145.0
}'

# Verify in database
psql -U tytoalba -c "SELECT * FROM ais_positions WHERE ship_id = (SELECT id FROM ships WHERE mmsi = '563012345') ORDER BY time DESC LIMIT 1;"
```

**Acceptance Criteria:**
- [ ] Message publishing works reliably
- [ ] Backend receives all messages
- [ ] Data persisted correctly
- [ ] Reconnection logic works
- [ ] QoS levels honored

---

### 3.3 Performance Testing

#### Task 3.3.1: Load Testing
**Owner:** Angga
**Tool:** k6 (Grafana k6)

**Test Scenarios:**

**Scenario 1: Normal Load**
- 50 concurrent users
- Duration: 5 minutes
- Target: API response time p95 < 500ms

**Scenario 2: Peak Load**
- 100 concurrent users
- Duration: 10 minutes
- Target: API response time p95 < 1000ms

**Scenario 3: Stress Test**
- Ramp up from 0 to 500 users over 10 minutes
- Find breaking point
- Target: Graceful degradation, no crashes

**k6 Script:**
```javascript
import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
  stages: [
    { duration: '2m', target: 50 },  // Ramp-up
    { duration: '5m', target: 50 },  // Sustain
    { duration: '2m', target: 0 },   // Ramp-down
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'], // 95% of requests < 500ms
    http_req_failed: ['rate<0.01'],   // Error rate < 1%
  },
};

export default function () {
  // Test GET all ships
  let res = http.get('http://localhost:8080/api/mqtt/ships');
  check(res, {
    'status is 200': (r) => r.status === 200,
    'response time < 500ms': (r) => r.timings.duration < 500,
  });

  sleep(1);
}
```

**Acceptance Criteria:**
- [ ] p95 response time < 500ms (normal load)
- [ ] System handles 100 concurrent users
- [ ] No crashes under stress
- [ ] Database connection pooling adequate

---

#### Task 3.3.2: ML Inference Latency Testing
**Owner:** Putri
**Tool:** Python pytest + time profiling

**Test Scenarios:**
1. Single prediction latency
   - Target: < 500ms
2. Batch prediction (10 vessels)
   - Target: < 2 seconds
3. Batch prediction (30 vessels - max)
   - Target: < 5 seconds
4. Cold start latency (model loading)
   - Target: < 10 seconds

**Test Code:**
```python
import pytest
import time
from src.api_inference import predict_arrival

def test_single_prediction_latency():
    vessel_data = {...}  # Test data

    start = time.time()
    result = predict_arrival(vessel_data)
    end = time.time()

    latency_ms = (end - start) * 1000

    assert latency_ms < 500, f"Latency {latency_ms}ms exceeds 500ms threshold"
    assert result['predicted_eta'] is not None
```

**Profiling:**
- Use cProfile to identify bottlenecks
- Optimize data preprocessing (vectorization)
- Optimize model inference (batch processing)
- Consider GPU acceleration if available

**Acceptance Criteria:**
- [ ] Single prediction < 500ms
- [ ] Batch (10) < 2s
- [ ] Batch (30) < 5s
- [ ] Bottlenecks identified and optimized

---

### 3.4 User Acceptance Testing (UAT)

#### Task 3.4.1: Pilot with 2 Vessels
**Owner:** All team members
**Duration:** 2 weeks (Dec 16-26, 2025)
**Participants:** Ship captains, fleet managers

**UAT Plan:**

**Week 1: Training & Setup**
- Day 1-2: User training sessions (captains & managers)
- Day 3-4: System deployment to pilot vessels
- Day 5-7: Hands-on usage with support

**Week 2: Feedback & Iteration**
- Day 8-11: Daily usage with issue tracking
- Day 12-13: Feedback sessions
- Day 14: Final sign-off

**UAT Test Cases (50+ total):**

**Category 1: Ship Captain Use Cases (20 tests)**
1. View current ship position on map
2. View route to destination (solid/dotted lines)
3. View predicted ETA with confidence score
4. View weather forecast along route
5. Compare predicted vs. actual ETA from previous trip
6. Update destination port
7. View fuel consumption prediction
8. Access system from ship's bridge computer
9. Access system from mobile device
10. Receive ETA alert notifications
... (continue to 20)

**Category 2: Fleet Manager Use Cases (15 tests)**
1. View all ships on fleet dashboard
2. Filter ships by status (on-time, delayed)
3. View ETA predictions for all active voyages
4. Compare vessel performance
5. Export daily trip summary report (PDF)
6. Export weekly analytics (Excel)
7. View prediction accuracy metrics
8. Identify vessels with low confidence predictions
9. Receive alerts for significant delays
10. Access historical voyage data
... (continue to 15)

**Category 3: System Admin Use Cases (10 tests)**
1. Monitor system health dashboard
2. View MQTT broker status
3. View database connection status
4. View ML model performance metrics
5. Restart services via admin panel
6. View system logs
7. Manage user accounts
8. Configure system settings
9. Run manual model retraining
10. Export system metrics

**Category 4: Edge Cases & Error Handling (5 tests)**
1. System behavior during internet connectivity loss
2. Handling of invalid sensor data
3. Prediction when weather data unavailable
4. Handling of extreme weather conditions
5. System recovery after crash

**Acceptance Criteria:**
- [ ] All 50+ test cases executed
- [ ] User satisfaction score ≥ 4.0/5.0
- [ ] Critical bugs fixed before UAT completion
- [ ] Sign-off from PT Bahtera stakeholders
- [ ] Feedback incorporated into final release

---

## Phase 4: Documentation & UAS Preparation (Week 9-10)
### Priority: HIGH | Timeline: Dec 27, 2025 - Jan 9, 2026

### 4.1 UAS Report Enhancements

Based on the UTS document Section 6, we need to create comprehensive additions for the final UAS report.

#### Task 4.1.1: Risk Management Section
**Owner:** Angga
**Deliverable:** `docs/UAS_Section_RiskManagement.md`

**Content:**
1. **Risk Identification Matrix**
   - Technical risks (8 identified)
   - Operational risks (4 identified)
   - Project risks (3 identified)
   - External risks (2 identified)

2. **Risk Assessment**
   - Probability: Low, Medium, High
   - Impact: Low, Medium, High
   - Risk Score = Probability × Impact

3. **Mitigation Strategies**
   - For each high/medium risk
   - Preventive actions
   - Contingency plans
   - Risk owners

4. **Risk Monitoring**
   - Weekly risk review meetings
   - Risk register updates
   - Escalation procedures

**Acceptance Criteria:**
- [ ] All 17+ risks documented
- [ ] Mitigation strategies defined
- [ ] Risk register created in spreadsheet
- [ ] Incorporated into UAS report

---

#### Task 4.1.2: Enhanced Stakeholder Analysis
**Owner:** Rina
**Deliverable:** `docs/UAS_Section_Stakeholders.md`

**Content:**
1. **Stakeholder Matrix**
   - 7 stakeholder groups identified
   - Interest level (Low, Medium, High)
   - Influence level (Low, Medium, High)
   - Engagement strategy

2. **Power-Interest Grid**
   - Quadrant 1: High Power, High Interest (Manage Closely)
   - Quadrant 2: High Power, Low Interest (Keep Satisfied)
   - Quadrant 3: Low Power, High Interest (Keep Informed)
   - Quadrant 4: Low Power, Low Interest (Monitor)

3. **Communication Plan**
   - Stakeholder-specific communication frequency
   - Communication channels (email, meetings, demos)
   - Meeting templates and agendas

4. **Stakeholder Engagement Log**
   - Meeting minutes from Sept 2025 - Jan 2026
   - Decisions made
   - Action items tracking

**Acceptance Criteria:**
- [ ] Stakeholder matrix completed
- [ ] Power-interest grid visualized
- [ ] Communication plan documented
- [ ] Engagement log comprehensive

---

#### Task 4.1.3: Project Governance Structure
**Owner:** Angga
**Deliverable:** `docs/UAS_Section_Governance.md`

**Content:**
1. **Governance Hierarchy Diagram**
   - Steering Committee (PT Bahtera + PLN)
   - Project Sponsor (PT Bahtera CTO)
   - Project Manager (Angga)
   - Development Team (Angga, Rina, Putri)

2. **RACI Matrix**
   - All major activities mapped
   - R = Responsible, A = Accountable, C = Consulted, I = Informed
   - Activities: Requirements, Design, Development, Testing, Deployment

3. **Decision-Making Framework**
   - Decision authority levels
   - Escalation paths
   - Decision log (all major decisions since Sept 2025)

4. **Meeting Cadence**
   - Daily standups (team)
   - Weekly sprint reviews (team)
   - Bi-weekly stakeholder demos
   - Monthly steering committee

**Acceptance Criteria:**
- [ ] Governance diagram created
- [ ] RACI matrix completed (15+ activities)
- [ ] Decision log comprehensive
- [ ] Meeting schedules documented

---

#### Task 4.1.4: Enhanced Testing Strategy
**Owner:** Putri
**Deliverable:** `docs/UAS_Section_Testing.md`

**Content:**
1. **Testing Pyramid**
   - Unit tests: 60% (target 80% coverage)
   - Integration tests: 30%
   - E2E tests: 10% (5 critical journeys)

2. **Test Coverage Report**
   - Backend: X% coverage
   - Frontend: Y% coverage
   - ML Service: Z% coverage
   - Overall: Target 80%+

3. **Test Cases Documentation**
   - 50+ detailed test cases
   - Expected results
   - Actual results
   - Pass/Fail status

4. **Bug Tracking & Resolution**
   - Bug severity levels (Critical, High, Medium, Low)
   - Bug lifecycle (New → Assigned → Fixed → Verified → Closed)
   - Resolution time SLAs

5. **UAT Results**
   - Pilot feedback summary
   - User satisfaction scores
   - Issues raised and resolutions

**Acceptance Criteria:**
- [ ] Testing pyramid documented
- [ ] Coverage reports generated
- [ ] All 50+ test cases documented
- [ ] UAT results comprehensive

---

#### Task 4.1.5: LSTM Architecture Deep Dive
**Owner:** Putri
**Deliverable:** `docs/UAS_Section_MLArchitecture.md`

**Content:**
1. **Model Architecture Specification**
   - Layer-by-layer description
   - Parameter counts
   - Activation functions
   - Regularization techniques

2. **Training Configuration**
   - Optimizer (Adam with hyperparameters)
   - Loss function (Gaussian NLL)
   - Batch size, epochs, learning rate
   - Early stopping, learning rate scheduling

3. **Hyperparameter Tuning Results**
   - Grid search results
   - Hyperparameter importance
   - Final configuration justification

4. **Model Performance Analysis**
   - Training/validation loss curves
   - MAE, RMSE, MAPE, R² scores
   - Prediction error distribution
   - Residual plots

5. **Feature Importance**
   - SHAP values for feature explanation
   - Top 10 most important features
   - Feature correlation analysis

6. **Model Versioning Strategy**
   - MLflow integration
   - Model registry
   - A/B testing framework

**Acceptance Criteria:**
- [ ] Architecture fully documented
- [ ] Training results comprehensive
- [ ] Performance visualizations included
- [ ] Feature importance analyzed

---

#### Task 4.1.6: Data Pipeline Documentation
**Owner:** Rina
**Deliverable:** `docs/UAS_Section_DataPipeline.md`

**Content:**
1. **End-to-End Data Flow Diagram**
   - IoT Sensors → MQTT → Raw Storage → Validation → Cleaned Data → Feature Store → ML Model

2. **Data Quality Rules**
   - Consistency rules (15+ rules)
   - Timeliness rules
   - Accuracy rules
   - Completeness rules

3. **Data Quality Dashboard**
   - Metrics: Data freshness, completeness, accuracy
   - Alerts for quality violations
   - Historical trends

4. **Anomaly Detection**
   - Statistical methods (Z-score, IQR)
   - ML-based anomaly detection
   - Alert mechanisms

5. **Data Retention Policy**
   - Hot storage: Last 7 days (high performance)
   - Warm storage: 8-90 days (standard)
   - Cold storage: 91-365 days (archival)
   - Purge after 1 year (GDPR compliance)

**Acceptance Criteria:**
- [ ] Data flow diagram comprehensive
- [ ] Quality rules documented
- [ ] Retention policy defined
- [ ] Anomaly detection explained

---

#### Task 4.1.7: Change Management & User Adoption
**Owner:** Angga
**Deliverable:** `docs/UAS_Section_ChangeManagement.md`

**Content:**
1. **Adoption Strategy (4 Phases)**
   - Phase 1: Awareness (Sprint 8-9)
   - Phase 2: Training (Sprint 10)
   - Phase 3: Pilot (2 weeks)
   - Phase 4: Rollout (Phased)

2. **Training Materials**
   - User manual for ship captains (20+ pages)
   - User manual for fleet managers (15+ pages)
   - Video tutorials (5+ videos)
   - Quick reference guides

3. **Support Plan**
   - Help desk setup
   - Support tiers (L1, L2, L3)
   - Support hours (24/7 for critical issues)
   - Ticket SLAs

4. **Success Metrics**
   - User adoption rate > 90% within 3 months
   - Support tickets < 5/week after month 2
   - User satisfaction > 4.0/5.0

**Acceptance Criteria:**
- [ ] Training materials created
- [ ] Support plan documented
- [ ] Adoption metrics defined
- [ ] Rollout plan detailed

---

#### Task 4.1.8: Post-Implementation Review Framework
**Owner:** All
**Deliverable:** `docs/UAS_Section_PostImplementation.md`

**Content:**
1. **Success Metrics (3 Months Post-Launch)**
   - Accuracy: ETA MAE < 30 min ✅
   - Performance: Uptime > 99.5% ✅
   - Adoption: Active users > 90% ✅
   - Business: Planning accuracy +20% ✅

2. **Review Cadence**
   - Week 1-4: Daily reviews
   - Month 2-3: Weekly reviews
   - Month 4+: Monthly reviews
   - Quarter 1: Formal retrospective

3. **Lessons Learned**
   - What went well
   - What didn't go well
   - What to improve next time

4. **Phase 2 Roadmap**
   - Route optimization feature
   - Multi-objective prediction (ETA + fuel)
   - Mobile app for captains
   - Advanced analytics dashboard

**Acceptance Criteria:**
- [ ] Success metrics tracked
- [ ] Review cadence defined
- [ ] Lessons learned documented
- [ ] Phase 2 roadmap created

---

#### Task 4.1.9: Technical Debt & Maintenance Plan
**Owner:** Angga
**Deliverable:** `docs/UAS_Section_Maintenance.md`

**Content:**
1. **Known Technical Debt**
   - 4 items identified
   - Prioritization (High, Medium, Low)
   - Remediation timeline

2. **Maintenance Schedule**
   - Daily: Database backups verification
   - Weekly: Model performance review
   - Monthly: Security patches, sensor calibration
   - Quarterly: Full system audit

3. **Disaster Recovery Plan**
   - Backup strategy (daily, weekly, monthly)
   - RTO (Recovery Time Objective): 4 hours
   - RPO (Recovery Point Objective): 1 hour
   - Failover procedures

4. **Monitoring & Alerting**
   - System health dashboard
   - Alert rules (CPU, memory, disk, latency)
   - On-call rotation

**Acceptance Criteria:**
- [ ] Technical debt backlog created
- [ ] Maintenance schedule documented
- [ ] DR plan comprehensive
- [ ] Monitoring configured

---

#### Task 4.1.10: Academic Contribution
**Owner:** Putri
**Deliverable:** `docs/UAS_Section_AcademicContribution.md`

**Content:**
1. **Research Questions Addressed**
   - RQ1: LSTM vs traditional models (comparison study)
   - RQ2: Weather feature impact (ablation study)
   - RQ3: Uncertainty quantification utility (survey)

2. **Literature Review**
   - 10+ relevant papers
   - Comparison table (this work vs prior art)
   - Contributions and gaps filled

3. **Experimental Results**
   - Comparative analysis results
   - Statistical significance testing
   - Performance benchmarks

4. **Limitations & Future Work**
   - Current limitations (data, scope, generalization)
   - Future research directions
   - Potential extensions

5. **Ethical Considerations**
   - Data privacy (GDPR compliance)
   - Algorithmic bias (fairness)
   - Environmental impact (carbon footprint)

**Acceptance Criteria:**
- [ ] Research questions answered
- [ ] Literature review comprehensive
- [ ] Experimental results rigorous
- [ ] Ethics addressed

---

### 4.2 Technical Documentation

#### Task 4.2.1: API Documentation (OpenAPI/Swagger)
**Owner:** Angga
**Deliverable:** `docs/api/openapi.yaml`

**Content:**
- All backend endpoints documented
- Request/response schemas
- Authentication requirements
- Error codes
- Example requests/responses
- Interactive Swagger UI

**Acceptance Criteria:**
- [ ] OpenAPI 3.0 specification complete
- [ ] Swagger UI accessible at /docs
- [ ] Examples for all endpoints
- [ ] Authentication documented

---

#### Task 4.2.2: Database Schema Documentation
**Owner:** Rina
**Deliverable:** `docs/database/ERD.png`, `docs/database/schema.md`

**Content:**
- Entity-Relationship Diagram (ERD)
- Table descriptions
- Column definitions
- Relationships and constraints
- Indexes and optimization notes
- Sample queries

**Acceptance Criteria:**
- [ ] ERD diagram clear and professional
- [ ] All tables and relationships documented
- [ ] Sample queries provided
- [ ] Migration guide included

---

#### Task 4.2.3: Deployment Guide
**Owner:** Angga
**Deliverable:** `docs/deployment/DEPLOYMENT.md`

**Content:**
1. **Prerequisites**
   - Hardware requirements
   - Software requirements
   - Network requirements

2. **Installation Steps**
   - Step-by-step instructions
   - Docker Compose setup
   - Kubernetes deployment (optional)
   - Environment configuration

3. **Configuration**
   - Environment variables
   - Secrets management
   - SSL/TLS setup
   - Monitoring setup

4. **Verification**
   - Health checks
   - Smoke tests
   - Troubleshooting guide

**Acceptance Criteria:**
- [ ] Deployment reproducible by following guide
- [ ] All configurations documented
- [ ] Troubleshooting section comprehensive
- [ ] Tested on clean environment

---

#### Task 4.2.4: User Manual
**Owner:** Putri
**Deliverable:** `docs/user_manual/ShipCaptain.pdf`, `docs/user_manual/FleetManager.pdf`

**Content:**

**Ship Captain Manual:**
1. Getting Started (login, dashboard overview)
2. Viewing Your Ship's Route
3. Understanding ETA Predictions
4. Viewing Weather Forecasts
5. Accessing Historical Data
6. Troubleshooting Common Issues
7. FAQ

**Fleet Manager Manual:**
1. Getting Started
2. Fleet Dashboard Overview
3. Monitoring Multiple Vessels
4. Prediction Accuracy Metrics
5. Generating Reports
6. Configuring Alerts
7. User Management
8. FAQ

**Acceptance Criteria:**
- [ ] Both manuals 15-20 pages
- [ ] Screenshots for all features
- [ ] Step-by-step instructions
- [ ] PDF format, professional design

---

### 4.3 Appendices for UAS Report

#### Task 4.3.1: Create All 10 Appendices
**Owner:** All team members
**Deliverables:** `docs/appendices/Appendix_A.md` through `Appendix_J.md`

**Appendix A: Sprint Retrospectives (Owner: Angga)**
- Sprint 1-2: Initiation
- Sprint 3-5: Analysis & Design
- Sprint 6-7: Development Sprint 1
- Sprint 8-9: Development Sprint 2
- Sprint 10: Testing & QA
- Sprint 11: Deployment

**Appendix B: Complete API Documentation (Owner: Angga)**
- OpenAPI spec embedded
- All endpoint examples
- Postman collection export

**Appendix C: Database Schema (Owner: Rina)**
- Full ERD diagram
- Table creation scripts
- Sample data scripts

**Appendix D: ML Model Training Logs (Owner: Putri)**
- Training history (epochs, loss, metrics)
- Hyperparameter tuning logs
- Model evaluation results
- TensorBoard screenshots

**Appendix E: User Acceptance Test Cases (Owner: All)**
- All 50+ test cases with results
- Test execution log
- Bug reports and resolutions

**Appendix F: Training Materials (Owner: Putri)**
- User manual excerpts
- Training slides
- Video tutorial transcripts

**Appendix G: Deployment Checklist (Owner: Angga)**
- Pre-deployment checklist
- Deployment steps verification
- Post-deployment verification

**Appendix H: Cost-Benefit Analysis (Owner: Rina)**
- Development costs (labor, infrastructure)
- Operational costs (maintenance, support)
- Benefits (time savings, accuracy improvement)
- ROI calculation

**Appendix I: Lessons Learned Log (Owner: All)**
- Technical lessons
- Project management lessons
- Stakeholder management lessons
- Recommendations for future projects

**Appendix J: Future Roadmap (Owner: All)**
- Phase 2 features (6-month plan)
- Phase 3 features (1-year plan)
- Research opportunities

**Acceptance Criteria:**
- [ ] All 10 appendices completed
- [ ] Professional formatting
- [ ] Cross-referenced in main report
- [ ] Submitted with UAS report

---

## Phase 5: Deployment & Go-Live (Week 11)
### Priority: CRITICAL | Timeline: Jan 10-16, 2026

### 5.1 Production Deployment

#### Task 5.1.1: Prepare Production Environment
**Owner:** Angga
**Platform:** DigitalOcean/AWS/On-premise PT Bahtera servers

**Infrastructure:**
1. **Application Servers**
   - Backend API: 2 instances (load balanced)
   - Frontend: 1 instance (Nginx)
   - ML Service: 1 instance (GPU optional)

2. **Data Stores**
   - PostgreSQL: Primary + replica for HA
   - Redis: Cache layer
   - MQTT Broker: Mosquitto cluster

3. **Monitoring**
   - Prometheus + Grafana for metrics
   - ELK stack for logging
   - Uptime monitoring (UptimeRobot)

4. **Security**
   - SSL/TLS certificates (Let's Encrypt)
   - Firewall rules (UFW/security groups)
   - SSH key-based authentication
   - Secrets management (Vault/environment variables)

**Acceptance Criteria:**
- [ ] All servers provisioned
- [ ] Networking configured
- [ ] Monitoring operational
- [ ] Security hardened

---

#### Task 5.1.2: Deploy Application
**Owner:** Angga
**Method:** Docker Compose or Kubernetes

**Docker Compose Configuration:**
```yaml
version: '3.8'
services:
  postgres:
    image: timescale/timescaledb:latest-pg15
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    ports:
      - "5432:5432"

  mosquitto:
    image: eclipse-mosquitto:2.0
    volumes:
      - ./mosquitto/config:/mosquitto/config
      - ./mosquitto/data:/mosquitto/data
    ports:
      - "1883:1883"
      - "9001:9001"

  backend:
    image: tytoalba/backend:latest
    depends_on:
      - postgres
      - mosquitto
    environment:
      DATABASE_URL: ${DATABASE_URL}
      MQTT_BROKER_URL: mqtt://mosquitto:1883
    ports:
      - "8080:8080"

  ml-service:
    image: tytoalba/ml-service:latest
    depends_on:
      - postgres
    environment:
      MODEL_PATH: /models/vessel_arrival_lstm.h5
    ports:
      - "8000:8000"
    volumes:
      - ./models:/models

  frontend:
    image: tytoalba/frontend:latest
    depends_on:
      - backend
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./ssl:/etc/nginx/ssl

volumes:
  postgres_data:
```

**Deployment Steps:**
1. Build Docker images (CI/CD)
2. Push images to registry
3. SSH to production server
4. Pull latest images
5. Run database migrations
6. Start services with `docker-compose up -d`
7. Verify health checks
8. Configure Nginx reverse proxy
9. Set up SSL certificates

**Acceptance Criteria:**
- [ ] All services running
- [ ] Health checks passing
- [ ] SSL configured
- [ ] Zero downtime deployment

---

#### Task 5.1.3: Production Smoke Tests
**Owner:** All
**Duration:** 2 hours post-deployment

**Test Checklist:**
- [ ] Frontend accessible via HTTPS
- [ ] Backend API health check returns 200
- [ ] ML service health check returns 200
- [ ] Database connections successful
- [ ] MQTT broker accepting connections
- [ ] Ship data flowing end-to-end
- [ ] Predictions generating correctly
- [ ] Monitoring dashboards populating
- [ ] Alerts firing correctly
- [ ] Logs centralized and searchable

**Rollback Plan:**
If smoke tests fail:
1. Identify failing component
2. Check logs for errors
3. Attempt quick fix if obvious
4. If not resolvable in 30 min, rollback to previous version
5. Investigate in staging environment

---

### 5.2 Go-Live Activities

#### Task 5.2.1: Go-Live Communication
**Owner:** Angga
**Audience:** PT Bahtera stakeholders, ship captains, fleet managers

**Communication Plan:**
1. **T-7 days:** Email announcement of go-live date
2. **T-3 days:** Reminder email with training resources
3. **T-1 day:** Final check-in, support contact info
4. **Go-Live Day:** Launch announcement, support on standby
5. **T+1 day:** Follow-up email, feedback request

**Support Readiness:**
- On-call support 24/7 for first week
- Daily check-in calls with key users
- Issue tracking system (Jira/GitHub Issues)
- Escalation path defined

---

#### Task 5.2.2: Hypercare Period (Week 1)
**Owner:** All team members
**Duration:** Jan 10-16, 2026

**Activities:**
- Daily morning sync meetings (15 min)
- Monitor system metrics continuously
- Respond to user issues within 2 hours
- Collect feedback daily
- Daily standup with PT Bahtera stakeholders
- Document all issues and resolutions

**Success Metrics for Week 1:**
- System uptime ≥ 99% ✅
- Response time < 500ms p95 ✅
- No critical bugs ✅
- User satisfaction ≥ 3.5/5.0 ✅

---

#### Task 5.2.3: Handover to Operations Team
**Owner:** Angga
**Recipient:** PT Bahtera IT team

**Handover Deliverables:**
1. **Runbooks**
   - System architecture overview
   - Common issues and resolutions
   - Restart procedures
   - Backup/restore procedures
   - Monitoring dashboard guide

2. **Access & Credentials**
   - Server SSH access
   - Database credentials
   - Monitoring dashboards
   - Admin panel access
   - Git repository access

3. **Knowledge Transfer Sessions**
   - Session 1: System architecture (2 hours)
   - Session 2: Deployment & operations (2 hours)
   - Session 3: Troubleshooting (2 hours)
   - Session 4: Monitoring & alerting (1 hour)

4. **Support SLA Agreement**
   - Response times for different severity levels
   - On-call rotation schedule
   - Escalation procedures
   - Maintenance windows

**Acceptance Criteria:**
- [ ] All runbooks delivered
- [ ] Knowledge transfer sessions completed
- [ ] Operations team can deploy independently
- [ ] Operations team can troubleshoot common issues
- [ ] Sign-off from PT Bahtera IT lead

---

## Phase 6: UAS Report Finalization (Week 12)
### Priority: CRITICAL | Timeline: Jan 17-23, 2026

### 6.1 Consolidate All Sections

#### Task 6.1.1: Compile Final UAS Report
**Owner:** All team members
**Deliverable:** `TytoAlba_UAS_Report_Final.pdf`

**Report Structure:**
```
TytoAlba UAS Report - Final Submission

Cover Page
- Project title
- Team members
- Submission date
- University logo

Table of Contents

Executive Summary (2 pages)

1. Introduction (5 pages)
   1.1 Background and Context
   1.2 Problem Statement
   1.3 Objectives
   1.4 Success Criteria

2. Literature Review (8 pages)
   2.1 Maritime Monitoring Systems
   2.2 Machine Learning for ETA Prediction
   2.3 LSTM Neural Networks
   2.4 IoT in Maritime Industry

3. System Architecture (12 pages)
   3.1 UML Diagrams (Use Case, Sequence, Class)
   3.2 Technology Stack
   3.3 Data Flow Architecture
   3.4 ML Pipeline Design

4. Implementation (20 pages)
   4.1 Backend Development (Go)
   4.2 Frontend Development (Vue.js)
   4.3 ML Service (Python/TensorFlow)
   4.4 MQTT Integration
   4.5 Database Design

5. LSTM Model Deep Dive (10 pages)
   5.1 Model Architecture
   5.2 Training Methodology
   5.3 Hyperparameter Tuning
   5.4 Performance Evaluation
   5.5 Feature Importance Analysis

6. Testing & Quality Assurance (10 pages)
   6.1 Testing Strategy
   6.2 Unit Testing Results
   6.3 Integration Testing Results
   6.4 Performance Testing Results
   6.5 User Acceptance Testing

7. Project Management (15 pages)
   7.1 Risk Management
   7.2 Stakeholder Analysis
   7.3 Governance Structure
   7.4 Timeline & Milestones
   7.5 Sprint Retrospectives

8. Deployment & Operations (8 pages)
   8.1 Production Deployment
   8.2 Go-Live Activities
   8.3 Maintenance Plan
   8.4 Technical Debt Management

9. Results & Evaluation (8 pages)
   9.1 Success Metrics Achievement
   9.2 User Feedback
   9.3 Performance Benchmarks
   9.4 Comparative Analysis (LSTM vs Alternatives)

10. Academic Contribution (6 pages)
    10.1 Research Questions Addressed
    10.2 Novel Contributions
    10.3 Limitations
    10.4 Future Work

11. Change Management (5 pages)
    11.1 User Adoption Strategy
    11.2 Training Program
    11.3 Support Plan

12. Lessons Learned (4 pages)
    12.1 Technical Lessons
    12.2 Project Management Lessons
    12.3 Recommendations for Future Projects

13. Conclusion (3 pages)

References (5 pages)

Appendices (50+ pages)
- Appendix A: Sprint Retrospectives
- Appendix B: API Documentation
- Appendix C: Database Schema
- Appendix D: ML Training Logs
- Appendix E: UAT Test Cases
- Appendix F: Training Materials
- Appendix G: Deployment Checklist
- Appendix H: Cost-Benefit Analysis
- Appendix I: Lessons Learned Log
- Appendix J: Future Roadmap

Total: ~170 pages (excluding appendices)
```

**Acceptance Criteria:**
- [ ] All sections written
- [ ] Professional formatting
- [ ] Consistent citations (APA/IEEE)
- [ ] Figures/tables numbered
- [ ] Proofread and edited

---

#### Task 6.1.2: Create Presentation Slides
**Owner:** All team members
**Deliverable:** `TytoAlba_UAS_Presentation.pptx`

**Presentation Structure (30 minutes):**

Slide 1-2: Title & Team Introduction (2 min)

Slide 3-5: Problem & Motivation (3 min)
- PT Bahtera Adhiguna challenge
- Impact of ETA uncertainty
- Business value proposition

Slide 6-8: Solution Overview (3 min)
- System architecture diagram
- Technology stack
- Key features

Slide 9-12: LSTM Model (5 min)
- Model architecture
- Training approach
- Performance results (92% → 95%)
- Comparison with baselines

Slide 13-15: Implementation Highlights (4 min)
- Backend (Go) + Frontend (Vue.js)
- MQTT real-time data ingestion
- Database design (TimescaleDB)

Slide 16-18: Testing & QA (3 min)
- Testing pyramid
- Coverage results
- UAT outcomes

Slide 19-21: Project Management (3 min)
- Agile/Scrum methodology
- Risk management
- Timeline adherence

Slide 22-24: Results & Impact (4 min)
- Success metrics achieved
- User feedback
- Business impact quantification

Slide 25-27: Lessons Learned (2 min)
- Key takeaways
- Challenges overcome
- Recommendations

Slide 28-30: Future Work & Conclusion (2 min)
- Phase 2 roadmap
- Research opportunities
- Closing remarks

**Acceptance Criteria:**
- [ ] 30 slides max
- [ ] Professional design
- [ ] Data visualizations clear
- [ ] Demo video embedded (optional)
- [ ] Rehearsed for 30-minute delivery

---

### 6.2 Final Review & Submission

#### Task 6.2.1: Internal Review
**Owner:** All team members
**Duration:** 2 days

**Review Checklist:**
- [ ] All sections complete
- [ ] No grammatical errors
- [ ] Consistent formatting
- [ ] All figures/tables referenced
- [ ] All citations correct
- [ ] Appendices attached
- [ ] Page numbers correct
- [ ] Table of contents accurate
- [ ] PDF generated correctly
- [ ] File size reasonable (< 50MB)

---

#### Task 6.2.2: Supervisor Review
**Owner:** Angga (coordinator)
**Duration:** 3 days

**Activities:**
1. Submit draft to Dosen Pembimbing
2. Receive feedback
3. Incorporate changes
4. Re-submit for final approval
5. Obtain sign-off

---

#### Task 6.2.3: Final Submission
**Owner:** Angga
**Deadline:** January 23, 2026

**Submission Package:**
1. **UAS Report PDF** (TytoAlba_UAS_Report_Final.pdf)
2. **Presentation Slides** (TytoAlba_UAS_Presentation.pptx)
3. **Source Code** (GitHub repository link)
4. **Demo Video** (5-minute overview, uploaded to YouTube/Drive)
5. **User Manuals** (Ship Captain + Fleet Manager PDFs)
6. **Deployment Guide** (DEPLOYMENT.md)

**Acceptance Criteria:**
- [ ] All deliverables submitted
- [ ] Submission deadline met
- [ ] Confirmation receipt received
- [ ] Presentation scheduled

---

## Summary: Project Roadmap at a Glance

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TYTOALBA PROJECT TIMELINE                        │
└─────────────────────────────────────────────────────────────────────┘

PHASE 1: Code Quality Improvements (Nov 1-14, 2025)
├── Refactor high-complexity functions
├── Add comprehensive unit tests
└── Integrate CI/CD complexity monitoring

PHASE 2: Complete Core Features (Nov 15 - Dec 12, 2025)
├── Frontend: Dashboard, route visualization, weather overlay
├── ML Service: API completion, model accuracy improvement
└── Database: PostgreSQL + TimescaleDB implementation

PHASE 3: Testing & QA (Dec 13-26, 2025)
├── Unit testing (80%+ coverage)
├── Integration testing
├── Performance testing (load, stress, ML latency)
└── User Acceptance Testing (2-week pilot)

PHASE 4: Documentation & UAS Prep (Dec 27, 2025 - Jan 9, 2026)
├── UAS report enhancements (10 new sections)
├── Technical documentation (API, DB, deployment)
└── Appendices creation (A-J)

PHASE 5: Deployment & Go-Live (Jan 10-16, 2026)
├── Production deployment
├── Go-live activities
├── Hypercare period (Week 1)
└── Handover to operations

PHASE 6: UAS Report Finalization (Jan 17-23, 2026)
├── Compile final report (~170 pages)
├── Create presentation (30 slides)
├── Internal & supervisor review
└── Final submission (Jan 23, 2026)
```

---

## Critical Success Factors

1. **Team Collaboration**
   - Daily standups (even if brief)
   - Clear task ownership
   - Open communication on blockers

2. **Stakeholder Engagement**
   - Weekly demos to PT Bahtera
   - Proactive risk communication
   - Manage expectations on timeline

3. **Quality Focus**
   - No shortcuts on testing
   - Code reviews for all PRs
   - Continuous integration enforced

4. **Documentation Discipline**
   - Document as you go (not at the end)
   - ADRs for major decisions
   - Keep todo list updated

5. **Realistic Scoping**
   - MVP first, enhancements later
   - Cut features if timeline slips
   - Buffer time for unknowns

---

## Risk Mitigation

**Top 5 Risks & Mitigations:**

1. **Risk:** LSTM model doesn't reach 95% accuracy
   - **Mitigation:** Ensemble methods, more data, accept 92% if business approves

2. **Risk:** Timeline delays due to academic commitments
   - **Mitigation:** Buffer week in schedule, prioritize ruthlessly

3. **Risk:** UAT reveals critical usability issues
   - **Mitigation:** Early prototypes, frequent user feedback

4. **Risk:** Production deployment issues
   - **Mitigation:** Staging environment, comprehensive smoke tests, rollback plan

5. **Risk:** Insufficient data for training
   - **Mitigation:** Synthetic data generation, data augmentation

---

## Success Metrics Tracking

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| LSTM Model Accuracy | ≥ 95% | 92% | 🟡 In Progress |
| Backend Code Coverage | ≥ 80% | TBD | ⏳ Pending |
| Frontend Code Coverage | ≥ 70% | TBD | ⏳ Pending |
| ML Service Coverage | ≥ 85% | TBD | ⏳ Pending |
| API Response Time (p95) | < 500ms | TBD | ⏳ Pending |
| System Uptime | ≥ 99.5% | TBD | ⏳ Pending |
| User Adoption | > 90% | TBD | ⏳ Pending |
| User Satisfaction | ≥ 4.0/5.0 | TBD | ⏳ Pending |
| UAT Test Cases Passed | 100% | TBD | ⏳ Pending |
| UAS Report Completion | 100% | 60% | 🟡 In Progress |

---

## Next Steps (Immediate)

**This Week (Nov 1-7, 2025):**
1. ✅ Refactor `remove_outliers` function (Angga/Putri)
2. ✅ Extract parameter parsing in `GetShipHistory` (Angga)
3. ✅ Add unit tests for both functions (Angga/Putri)
4. ✅ Set up CI/CD for complexity monitoring (Angga)
5. 🔄 Continue LSTM model accuracy improvement (Putri)
6. 🔄 Complete frontend dashboard (Putri)

**Next Week (Nov 8-14, 2025):**
1. Complete route visualization with Leaflet.js
2. Integrate weather API
3. Finalize ML Service API endpoints
4. Begin PostgreSQL migration
5. Start writing UAT test cases

---

## Contact & Escalation

**Project Team:**
- Angga Pratama Suryabrata (Project Manager / Backend) - angga@example.com
- Rina Widyasti Habibah (Database / Data Pipeline) - rina@example.com
- Putri Nur Meilisa (Frontend / ML Service) - putri@example.com

**Escalation Path:**
1. Team internal resolution (< 24 hours)
2. Escalate to Dosen Pembimbing (< 48 hours)
3. Escalate to PT Bahtera CTO (project sponsor)

**Meeting Schedule:**
- Daily Standups: 9:00 AM (15 min)
- Weekly Sprint Review: Friday 2:00 PM (1 hour)
- Bi-weekly Client Demo: Every other Tuesday 3:00 PM (1 hour)

---

**Document Version:** 1.0
**Last Updated:** October 31, 2025
**Next Review:** November 7, 2025
**Status:** ACTIVE

---

**Approval Signatures:**
- [ ] Angga Pratama Suryabrata (Project Manager)
- [ ] Rina Widyasti Habibah (Team Member)
- [ ] Putri Nur Meilisa (Team Member)
- [ ] Dosen Pembimbing (Academic Supervisor)
- [ ] PT Bahtera CTO (Project Sponsor)

