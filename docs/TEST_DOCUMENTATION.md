# TytoAlba Test Documentation

**Created:** November 2, 2025
**Due Date:** November 24, 2025 (University Assignment)
**Owner:** Angga Suryabrata & Team
**Related Tasks:** TODO #15, #16, #17

---

## 📋 Document Overview

This comprehensive test documentation covers all testing activities for the TytoAlba Maritime Vessel Tracking & Prediction System, including test cases, test results, and quality assurance procedures.

---

## 📊 Testing Strategy

### Testing Pyramid

```
          /\
         /E2E\        10% - End-to-End Tests
        /------\
       /  INT   \     30% - Integration Tests
      /----------\
     /   UNIT     \   60% - Unit Tests
    /--------------\
```

**Total Target: 50+ Test Cases**

- **Unit Tests:** 30 test cases (60%)
- **Integration Tests:** 15 test cases (30%)
- **End-to-End Tests:** 5 test cases (10%)

---

## 🧪 Test Case Catalog

### CATEGORY 1: Backend Unit Tests (10 cases)

#### TC-BE-001: GetShipHistory - Valid MMSI
**Priority:** HIGH
**Type:** Unit Test
**Component:** `backend/internal/handlers/mqtt_ships.go`

**Preconditions:**
- Backend server running
- Ship with MMSI "563012345" exists in store
- Ship has 10+ history records

**Test Steps:**
1. Send GET request to `/api/mqtt/history?mmsi=563012345`
2. Verify response status code is 200
3. Verify response contains array of history records
4. Verify all records have required fields (timestamp, lat, lon, speed, course)

**Expected Result:**
```json
{
  "mmsi": "563012345",
  "history": [
    {
      "timestamp": "2025-11-02T10:00:00Z",
      "latitude": -5.5,
      "longitude": 112.5,
      "speed": 12.5,
      "course": 145.0
    }
  ]
}
```

**Pass Criteria:** Status 200, valid JSON array returned

---

#### TC-BE-002: GetShipHistory - Invalid MMSI Format
**Priority:** MEDIUM
**Type:** Unit Test
**Component:** `backend/internal/handlers/mqtt_ships.go`

**Test Steps:**
1. Send GET request to `/api/mqtt/history?mmsi=INVALID`
2. Verify response status code is 400
3. Verify error message contains "invalid MMSI format"

**Expected Result:**
```json
{
  "error": "invalid MMSI format"
}
```

**Pass Criteria:** Status 400, error message present

---

#### TC-BE-003: GetShipHistory - Non-Existent Ship
**Priority:** MEDIUM
**Type:** Unit Test

**Test Steps:**
1. Send GET request to `/api/mqtt/history?mmsi=999999999`
2. Verify response status code is 404
3. Verify error message contains "ship not found"

**Expected Result:** 404 Not Found

**Pass Criteria:** Correct error handling

---

#### TC-BE-004: GetBulkCarriers - Filter Pusher Ships
**Priority:** HIGH
**Type:** Unit Test
**Component:** `backend/internal/handlers/mqtt_ships.go`

**Preconditions:**
- Store contains 3 bulk carriers
- Store contains 2 pusher ships

**Test Steps:**
1. Send GET request to `/api/mqtt/bulk-carriers`
2. Verify response contains only 3 ships
3. Verify all ships have `ship_type: "bulk_carrier"`
4. Verify no pusher ships in response

**Pass Criteria:** Only bulk carriers returned

---

#### TC-BE-005: Ship Store - Concurrent Write Safety
**Priority:** HIGH
**Type:** Unit Test
**Component:** `backend/internal/storage/ship_store.go`

**Test Steps:**
1. Spawn 10 goroutines
2. Each goroutine writes ship data to same MMSI
3. Wait for all goroutines to complete
4. Verify no data corruption
5. Verify final state is consistent

**Pass Criteria:** Thread-safe, no race conditions

---

#### TC-BE-006: Ship Store - 24-Hour Data Retention
**Priority:** MEDIUM
**Type:** Unit Test

**Test Steps:**
1. Add ship data with timestamp 25 hours old
2. Add ship data with timestamp 23 hours old
3. Call cleanup function
4. Verify 25-hour-old data removed
5. Verify 23-hour-old data retained

**Pass Criteria:** Correct retention policy

---

#### TC-BE-007: MQTT Client - Connection Retry
**Priority:** HIGH
**Type:** Unit Test
**Component:** `backend/internal/mqtt/client.go`

**Test Steps:**
1. Start MQTT client with invalid broker URL
2. Verify connection fails
3. Fix broker URL
4. Verify automatic reconnection succeeds within 30 seconds

**Pass Criteria:** Resilient reconnection logic

---

#### TC-BE-008: MQTT Client - Message Parsing
**Priority:** HIGH
**Type:** Unit Test

**Test Steps:**
1. Publish valid AIS JSON to topic `tytoalba/ships/563012345/ais`
2. Verify message parsed correctly
3. Verify ship added to store
4. Publish invalid JSON
5. Verify error logged, no crash

**Pass Criteria:** Robust error handling

---

#### TC-BE-009: CORS Headers Present
**Priority:** LOW
**Type:** Unit Test

**Test Steps:**
1. Send OPTIONS request to `/api/mqtt/ships`
2. Verify `Access-Control-Allow-Origin` header present
3. Verify `Access-Control-Allow-Methods` includes GET, POST

**Pass Criteria:** CORS enabled

---

#### TC-BE-010: Health Check Endpoint
**Priority:** LOW
**Type:** Unit Test

**Test Steps:**
1. Send GET request to `/health`
2. Verify status 200
3. Verify response `{"status": "ok"}`

**Pass Criteria:** Health endpoint responsive

---

### CATEGORY 2: ML Service Unit Tests (15 cases)

#### TC-ML-001: remove_outliers - Z-Score Method
**Priority:** CRITICAL
**Type:** Unit Test
**Component:** `ml-service/src/preprocessing/data_pipeline.py`

**Preconditions:**
- Sample DataFrame with 100 rows
- 5 known outliers (z-score > 3)

**Test Steps:**
1. Call `remove_outliers(df, method='z-score', threshold=3)`
2. Verify resulting DataFrame has 95 rows
3. Verify 5 outliers removed
4. Verify valid data retained

**Pass Criteria:** Correct outlier detection and removal

---

#### TC-ML-002: remove_outliers - IQR Method
**Priority:** HIGH
**Type:** Unit Test

**Test Data:**
```python
data = {
    'speed': [10, 12, 11, 13, 10, 100],  # 100 is outlier
    'course': [90, 95, 92, 88, 91, 85]
}
```

**Test Steps:**
1. Call `remove_outliers(df, method='iqr')`
2. Verify row with speed=100 removed
3. Verify 5 valid rows remain

**Pass Criteria:** IQR method works correctly

---

#### TC-ML-003: remove_outliers - Invalid Method
**Priority:** MEDIUM
**Type:** Unit Test

**Test Steps:**
1. Call `remove_outliers(df, method='invalid_method')`
2. Verify raises `ValueError`
3. Verify error message: "Method must be 'z-score' or 'iqr'"

**Pass Criteria:** Input validation works

---

#### TC-ML-004: remove_outliers - Empty DataFrame
**Priority:** MEDIUM
**Type:** Unit Test

**Test Steps:**
1. Create empty DataFrame `pd.DataFrame()`
2. Call `remove_outliers(df)`
3. Verify returns empty DataFrame
4. Verify no errors raised

**Pass Criteria:** Edge case handled

---

#### TC-ML-005: remove_outliers - All Outliers
**Priority:** LOW
**Type:** Unit Test

**Test Steps:**
1. Create DataFrame where all rows are outliers
2. Call `remove_outliers(df, method='z-score', threshold=0.1)`
3. Verify returns empty DataFrame

**Pass Criteria:** Edge case handled

---

#### TC-ML-006: normalize_features - Standard Scaling
**Priority:** HIGH
**Type:** Unit Test

**Test Steps:**
1. Create sample features `[[10], [20], [30]]`
2. Call `normalize_features(data, method='standard')`
3. Verify mean ≈ 0
4. Verify std ≈ 1

**Pass Criteria:** Correct normalization

---

#### TC-ML-007: create_sequences - Correct Shape
**Priority:** HIGH
**Type:** Unit Test

**Test Steps:**
1. Input data shape: (100, 5)
2. Window size: 10
3. Call `create_sequences(data, window_size=10)`
4. Verify output shape: (90, 10, 5)

**Pass Criteria:** Sequence generation correct

---

#### TC-ML-008: LSTM Model - GPU Detection
**Priority:** MEDIUM
**Type:** Unit Test
**Component:** `ml-service/src/models/lstm_arrival_predictor.py`

**Test Steps:**
1. Initialize LSTM model
2. If GPU available, verify model uses GPU
3. If no GPU, verify fallback to CPU
4. Verify no errors in either case

**Pass Criteria:** Auto-detection works

---

#### TC-ML-009: LSTM Model - Training Convergence
**Priority:** HIGH
**Type:** Unit Test

**Test Steps:**
1. Train model on synthetic data for 50 epochs
2. Verify training loss decreases
3. Verify validation loss decreases
4. Verify final MAE < 60 minutes

**Pass Criteria:** Model trains successfully

---

#### TC-ML-010: LSTM Predict - Single Vessel
**Priority:** CRITICAL
**Type:** Unit Test

**Test Data:**
```python
{
    "vessel_mmsi": "563012345",
    "ship_type": "bulk_carrier",
    "destination_lat": 1.2644,
    "destination_lon": 103.8229,
    "current_lat": -5.5,
    "current_lon": 112.5,
    "speed": 12.5
}
```

**Test Steps:**
1. Call prediction endpoint
2. Verify response contains ETA timestamp
3. Verify confidence score 0.0-1.0

**Pass Criteria:** Valid prediction returned

---

#### TC-ML-011: LSTM Predict - Batch 30 Vessels
**Priority:** HIGH
**Type:** Unit Test

**Test Steps:**
1. Create batch request with 30 vessels
2. Send to `/predict/batch` endpoint
3. Verify all 30 predictions returned
4. Verify execution time < 10 seconds

**Pass Criteria:** Batch processing efficient

---

#### TC-ML-012: LSTM Predict - Exceeds Batch Limit
**Priority:** MEDIUM
**Type:** Unit Test

**Test Steps:**
1. Create batch request with 31 vessels
2. Send to `/predict/batch` endpoint
3. Verify response status 400
4. Verify error message: "Maximum 30 vessels allowed"

**Pass Criteria:** Limit enforced

---

#### TC-ML-013: API Health Endpoint
**Priority:** LOW
**Type:** Unit Test

**Test Steps:**
1. GET `/health` endpoint
2. Verify status 200
3. Verify response contains model status

**Pass Criteria:** Health check works

---

#### TC-ML-014: Model Info Endpoint
**Priority:** LOW
**Type:** Unit Test

**Test Steps:**
1. GET `/model/info` endpoint
2. Verify returns model architecture details
3. Verify includes layer count, parameters

**Pass Criteria:** Model introspection works

---

#### TC-ML-015: Prediction Confidence Score
**Priority:** MEDIUM
**Type:** Unit Test

**Test Steps:**
1. Make 10 predictions
2. Verify all confidence scores between 0.0 and 1.0
3. Verify average confidence > 0.75

**Pass Criteria:** Confidence scoring reasonable

---

### CATEGORY 3: Integration Tests (15 cases)

#### TC-INT-001: End-to-End Ship Tracking Flow
**Priority:** CRITICAL
**Type:** Integration Test

**Components:** MQTT Broker + Backend + Storage

**Test Steps:**
1. Start MQTT broker, backend server
2. Publish ship AIS data to MQTT topic
3. Wait 2 seconds
4. Query backend API `/api/mqtt/ship?mmsi=563012345`
5. Verify ship data retrieved correctly

**Pass Criteria:** Full pipeline works

---

#### TC-INT-002: Backend + ML Service Integration
**Priority:** CRITICAL
**Type:** Integration Test

**Components:** Backend + ML Service

**Test Steps:**
1. Start both backend and ML service
2. Backend fetches ship data
3. Backend calls ML service `/predict/arrival`
4. Verify prediction returned
5. Verify prediction stored/logged

**Pass Criteria:** Services communicate correctly

---

#### TC-INT-003: MQTT Reconnection After Broker Restart
**Priority:** HIGH
**Type:** Integration Test

**Test Steps:**
1. Start backend with MQTT connection
2. Stop MQTT broker
3. Wait 5 seconds
4. Restart MQTT broker
5. Verify backend reconnects within 30 seconds
6. Verify messages processed after reconnect

**Pass Criteria:** Resilient connection

---

#### TC-INT-004: Database Persistence (Future)
**Priority:** LOW
**Type:** Integration Test

**Test Steps:**
1. Add ship data to storage
2. Restart backend server
3. Verify ship data persisted
4. Verify history retained

**Pass Criteria:** Data persists (when DB implemented)

---

#### TC-INT-005: Frontend + Backend API Integration
**Priority:** HIGH
**Type:** Integration Test

**Test Steps:**
1. Start backend and frontend
2. Load dashboard page
3. Verify API calls successful
4. Verify ship data displayed
5. Verify no CORS errors

**Pass Criteria:** Frontend communicates with backend

---

#### TC-INT-006: Weather API Integration (Future)
**Priority:** MEDIUM
**Type:** Integration Test

**Test Steps:**
1. Call weather API for specific coordinates
2. Verify weather data retrieved
3. Verify integrated into ML prediction

**Pass Criteria:** External API integrated

---

#### TC-INT-007: Batch Prediction with Real Ship Data
**Priority:** HIGH
**Type:** Integration Test

**Test Steps:**
1. Collect real ship data for 10 vessels
2. Send batch prediction request
3. Verify all predictions returned
4. Compare with actual arrival times

**Pass Criteria:** Predictions reasonable

---

#### TC-INT-008: Concurrent User Sessions
**Priority:** MEDIUM
**Type:** Integration Test

**Test Steps:**
1. Simulate 10 concurrent users
2. Each queries different ships
3. Verify all requests successful
4. Verify no data corruption

**Pass Criteria:** Handles concurrency

---

#### TC-INT-009: Historical Data Cleanup
**Priority:** MEDIUM
**Type:** Integration Test

**Test Steps:**
1. Add ship data over 25 hours
2. Run cleanup job
3. Verify only last 24 hours retained
4. Verify API returns correct time range

**Pass Criteria:** Cleanup works

---

#### TC-INT-010: MQTT Topic Subscription
**Priority:** HIGH
**Type:** Integration Test

**Test Steps:**
1. Backend subscribes to `tytoalba/ships/#`
2. Publish to `tytoalba/ships/563012345/ais`
3. Publish to `tytoalba/ships/563012346/sensors`
4. Verify both messages received
5. Verify correct routing

**Pass Criteria:** Topic wildcards work

---

#### TC-INT-011: Error Handling - ML Service Down
**Priority:** HIGH
**Type:** Integration Test

**Test Steps:**
1. Stop ML service
2. Backend attempts prediction
3. Verify graceful error handling
4. Verify backend doesn't crash

**Pass Criteria:** Graceful degradation

---

#### TC-INT-012: CORS Preflight Requests
**Priority:** MEDIUM
**Type:** Integration Test

**Test Steps:**
1. Frontend sends OPTIONS request
2. Verify CORS headers returned
3. Verify subsequent GET request succeeds

**Pass Criteria:** CORS flow works

---

#### TC-INT-013: Rate Limiting (Future)
**Priority:** LOW
**Type:** Integration Test

**Test Steps:**
1. Send 100 requests in 1 second
2. Verify rate limiter triggers
3. Verify 429 Too Many Requests returned

**Pass Criteria:** Rate limiting works

---

#### TC-INT-014: Logging and Monitoring
**Priority:** MEDIUM
**Type:** Integration Test

**Test Steps:**
1. Trigger various API calls
2. Verify logs written correctly
3. Verify log levels appropriate
4. Verify no sensitive data in logs

**Pass Criteria:** Logging works

---

#### TC-INT-015: Authentication (Future)
**Priority:** LOW
**Type:** Integration Test

**Test Steps:**
1. Access protected endpoint without token
2. Verify 401 Unauthorized
3. Obtain valid token
4. Retry with token
5. Verify 200 Success

**Pass Criteria:** Auth works when implemented

---

### CATEGORY 4: End-to-End Tests (5 cases)

#### TC-E2E-001: Complete User Journey - Track Ship
**Priority:** CRITICAL
**Type:** E2E Test

**Test Steps:**
1. User opens dashboard
2. User searches for ship MMSI
3. Dashboard displays ship location on map
4. Dashboard shows predicted arrival time
5. User views ship history

**Pass Criteria:** Full workflow works

---

#### TC-E2E-002: Real-time Update Flow
**Priority:** HIGH
**Type:** E2E Test

**Test Steps:**
1. User viewing ship on dashboard
2. Ship publishes new AIS position
3. Dashboard updates within 5 seconds
4. Map shows new position

**Pass Criteria:** Real-time updates work

---

#### TC-E2E-003: Prediction Accuracy Test
**Priority:** CRITICAL
**Type:** E2E Test

**Test Steps:**
1. Collect 10 voyages with actual arrival times
2. Make predictions at 50% journey
3. Compare predicted vs actual
4. Verify MAPE < 8%

**Pass Criteria:** Acceptable accuracy

---

#### TC-E2E-004: System Load Test
**Priority:** HIGH
**Type:** E2E Test

**Test Steps:**
1. Simulate 50 ships publishing every 30 seconds
2. Simulate 20 concurrent users
3. Monitor system for 1 hour
4. Verify no crashes
5. Verify API latency < 500ms

**Pass Criteria:** System stable under load

---

#### TC-E2E-005: Disaster Recovery
**Priority:** MEDIUM
**Type:** E2E Test

**Test Steps:**
1. Running system with active ships
2. Force crash backend server
3. Restart backend
4. Verify system recovers
5. Verify no data loss (when DB added)

**Pass Criteria:** System resilient

---

## 📈 Test Execution Results Template

### Test Run Summary
**Date:** [YYYY-MM-DD]
**Environment:** [Development/Staging/Production]
**Tester:** [Name]

| Category | Total | Passed | Failed | Skipped | Pass Rate |
|----------|-------|--------|--------|---------|-----------|
| Backend Unit | 10 | - | - | - | -% |
| ML Service Unit | 15 | - | - | - | -% |
| Integration | 15 | - | - | - | -% |
| E2E | 5 | - | - | - | -% |
| **TOTAL** | **50** | **-** | **-** | **-** | **-%** |

---

## 🐛 Defect Report Template

**Defect ID:** DEF-[NUMBER]
**Severity:** [Critical/High/Medium/Low]
**Priority:** [P0/P1/P2/P3]
**Status:** [Open/In Progress/Fixed/Closed]

**Test Case:** TC-[CATEGORY]-[NUMBER]
**Component:** [Component name]
**Reproducibility:** [Always/Sometimes/Once]

**Steps to Reproduce:**
1. [Step 1]
2. [Step 2]

**Expected Result:**
[What should happen]

**Actual Result:**
[What actually happened]

**Error Logs:**
```
[Error output]
```

**Screenshots:**
[Attach if applicable]

**Environment:**
- OS: [Linux/Windows/macOS]
- Go Version: [1.21.x]
- Python Version: [3.10.x]
- Browser: [Chrome/Firefox] (if frontend)

---

## 📊 Coverage Reports

### Backend Coverage Target
**Minimum:** 80%
**Target File:** `backend/coverage.html`

```bash
# Generate coverage
cd backend
go test ./internal/... -coverprofile=coverage.out
go tool cover -html=coverage.out -o coverage.html
```

### ML Service Coverage Target
**Minimum:** 85%
**Target File:** `ml-service/htmlcov/index.html`

```bash
# Generate coverage
cd ml-service
pytest tests/ --cov=src --cov-report=html
```

---

## 🔄 Continuous Integration

### CI/CD Pipeline Checks
- ✅ All unit tests pass
- ✅ Coverage ≥ 70%
- ✅ No linting errors
- ✅ Cyclomatic complexity ≤ 10
- ✅ Build succeeds

### GitHub Actions Workflow
```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test-backend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Go tests
        run: go test ./internal/... -v -cover

  test-ml-service:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Python tests
        run: pytest tests/ -v --cov=src
```

---

## ⏱️ Test Execution Schedule

### Daily (Automated)
- All unit tests (30 tests)
- Quick smoke tests (5 tests)
- Duration: ~5 minutes

### Weekly (Automated)
- All integration tests (15 tests)
- Performance regression tests
- Duration: ~30 minutes

### Pre-Release (Manual)
- Full test suite (50 tests)
- UAT scenarios
- Load testing
- Duration: ~4 hours

---

## 📝 Submission Checklist (Nov 24, 2025)

### Required Deliverables
- ✅ This test documentation (completed)
- ✅ 50+ test cases documented
- ✅ Test execution results
- ✅ Coverage reports (HTML)
- ✅ Defect log (if any)
- ✅ Test metrics dashboard

### Bonus Items
- ✅ CI/CD pipeline configured
- ✅ Performance test results
- ✅ Load testing report
- ✅ Security test results

---

## 🔗 Related Documents

- **Unit Testing Plan:** `docs/UNIT_TESTING_PLAN.md`
- **TODO Checklist:** `TODO_CHECKLIST_NUMBERED.md`
- **Project README:** `README.md`
- **Cyclomatic Complexity Analysis:** `cyclomatic_complexity_analysis.txt`

---

## 📞 Test Team Contacts

- **Test Lead:** Angga Suryabrata
- **Backend QA:** Angga
- **ML Service QA:** Putri
- **Frontend QA:** Putri
- **Integration QA:** All team

---

**Document Version:** 1.0
**Last Updated:** November 2, 2025
**Next Review:** November 17, 2025
**Final Submission:** November 24, 2025
