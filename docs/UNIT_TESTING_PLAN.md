# TytoAlba Unit Testing Plan

**Created:** November 2, 2025
**Due Date:** November 17, 2025 (University Assignment)
**Owner:** Angga Suryabrata
**Related Tasks:** TODO #4, #5, #14

---

## 📋 Overview

This plan outlines the unit testing strategy for the TytoAlba Maritime Vessel Tracking & Prediction System, focusing on backend (Go) and ML service (Python) components.

## 🎯 Testing Goals

### Coverage Targets
- **Backend (Go):** 80% code coverage
- **ML Service (Python):** 85% code coverage
- **Frontend (Vue):** 70% code coverage (future)

### Quality Metrics
- All tests must pass in CI/CD pipeline
- Maximum test execution time: 30 seconds for unit tests
- Zero flaky tests allowed

---

## 🧪 Backend Unit Tests (Go)

### 1. Handlers Package (`backend/internal/handlers/`)

#### Test File: `mqtt_ships_test.go`
**Functions to Test:**

**a) GetShipHistory Handler (Priority HIGH)**
- Location: `mqtt_ships.go:100-137`
- Cyclomatic Complexity: 7 (needs refactoring first)
- Estimated Time: 3 hours

**Test Cases (7+):**
```go
TestGetShipHistory_ValidMMSI_Success
TestGetShipHistory_InvalidMMSI_BadRequest
TestGetShipHistory_MissingMMSI_BadRequest
TestGetShipHistory_NonExistentShip_NotFound
TestGetShipHistory_ValidTimeRange_FilteredResults
TestGetShipHistory_InvalidTimeRange_BadRequest
TestGetShipHistory_EmptyHistory_EmptyArray
```

**b) GetAllShips Handler**
```go
TestGetAllShips_Success_ReturnsAllShips
TestGetAllShips_EmptyStore_ReturnsEmptyArray
TestGetAllShips_CORSHeaders_Present
```

**c) GetBulkCarriers Handler**
```go
TestGetBulkCarriers_FiltersCorrectly
TestGetBulkCarriers_ExcludesPushers
TestGetBulkCarriers_EmptyResult_NoBC
```

**d) GetSingleShip Handler**
```go
TestGetSingleShip_ValidMMSI_Success
TestGetSingleShip_InvalidMMSI_BadRequest
TestGetSingleShip_NotFound_404
```

**Implementation Steps:**
1. Create `backend/internal/handlers/mqtt_ships_test.go`
2. Set up test fixtures with mock ship data
3. Use httptest.ResponseRecorder for HTTP testing
4. Mock the ship store interface
5. Test all edge cases and error paths

---

### 2. Storage Package (`backend/internal/storage/`)

#### Test File: `ship_store_test.go`

**Functions to Test:**

**a) AddShip**
```go
TestAddShip_NewShip_Success
TestAddShip_UpdateExisting_Success
TestAddShip_ConcurrentWrites_ThreadSafe
```

**b) GetShip**
```go
TestGetShip_ExistingShip_Success
TestGetShip_NonExistent_Nil
```

**c) GetAllShips**
```go
TestGetAllShips_MultipleShips_Sorted
TestGetAllShips_EmptyStore_EmptySlice
```

**d) GetShipHistory**
```go
TestGetShipHistory_ValidTimeRange
TestGetShipHistory_ExpiredData_Cleaned
TestGetShipHistory_24HourRetention
```

---

### 3. MQTT Package (`backend/internal/mqtt/`)

#### Test File: `client_test.go`

**Functions to Test:**

**a) MQTT Connection**
```go
TestConnect_ValidBroker_Success
TestConnect_InvalidBroker_Error
TestConnect_Reconnect_AfterDisconnect
```

**b) Message Processing**
```go
TestProcessMessage_ValidAIS_ParsedCorrectly
TestProcessMessage_InvalidJSON_Logged
TestProcessMessage_MissingFields_Error
```

---

## 🤖 ML Service Unit Tests (Python)

### 1. Data Pipeline (`ml-service/src/preprocessing/data_pipeline.py`)

#### Test File: `tests/test_data_pipeline.py`
**Priority: CRITICAL** (TODO Task #4)

**a) remove_outliers Function (Priority HIGH)**
- Location: Line 164-209
- Cyclomatic Complexity: 10 (needs refactoring to CC=3-4 first)
- Estimated Time: 4 hours

**Test Cases (10+):**
```python
test_remove_outliers_no_outliers_unchanged()
test_remove_outliers_z_score_method()
test_remove_outliers_iqr_method()
test_remove_outliers_invalid_method_raises_error()
test_remove_outliers_empty_dataframe_returns_empty()
test_remove_outliers_all_outliers_returns_empty()
test_remove_outliers_threshold_2_sigma()
test_remove_outliers_threshold_3_sigma()
test_remove_outliers_missing_values_handled()
test_remove_outliers_single_row_unchanged()
test_remove_outliers_performance_large_dataset()
```

**b) normalize_features Function**
```python
test_normalize_features_standard_scaling()
test_normalize_features_minmax_scaling()
test_normalize_features_preserves_shape()
test_normalize_features_handles_constants()
```

**c) create_sequences Function**
```python
test_create_sequences_correct_shape()
test_create_sequences_window_size_validation()
test_create_sequences_insufficient_data()
```

**Implementation Steps:**
1. Create `ml-service/tests/test_data_pipeline.py`
2. Use pytest framework
3. Create fixtures for sample datasets
4. Use pandas.testing for dataframe comparison
5. Add parameterized tests for multiple scenarios

---

### 2. LSTM Model (`ml-service/src/models/lstm_arrival_predictor.py`)

#### Test File: `tests/test_lstm_model.py`

**a) Model Initialization**
```python
test_model_init_correct_architecture()
test_model_init_gpu_detection()
test_model_init_cpu_fallback()
```

**b) Training Functions**
```python
test_train_model_convergence()
test_train_model_early_stopping()
test_train_model_validation_split()
```

**c) Prediction Functions**
```python
test_predict_single_vessel()
test_predict_batch_30_vessels()
test_predict_confidence_scores()
test_predict_invalid_input_raises()
```

---

### 3. API Inference (`ml-service/src/api_inference.py`)

#### Test File: `tests/test_api_inference.py`

**a) FastAPI Endpoints**
```python
test_predict_arrival_endpoint_success()
test_predict_arrival_invalid_data_422()
test_predict_batch_endpoint_max_30()
test_predict_batch_exceeds_limit_400()
test_model_info_endpoint()
test_health_endpoint()
```

**Existing File:** `ml-service/test_api.py` (needs enhancement)

---

## 🧰 Testing Tools & Frameworks

### Backend (Go)
```bash
# Install testing tools
go get -u github.com/stretchr/testify/assert
go get -u github.com/stretchr/testify/mock

# Run tests
go test ./internal/... -v -cover

# Generate coverage report
go test ./internal/... -coverprofile=coverage.out
go tool cover -html=coverage.out -o coverage.html
```

### ML Service (Python)
```bash
# Install testing tools
pip install pytest pytest-cov pytest-mock

# Run tests
pytest tests/ -v --cov=src --cov-report=html

# Run with specific markers
pytest -m "unit" -v
```

---

## 📁 Test Directory Structure

```
TytoAlba/
├── backend/
│   ├── internal/
│   │   ├── handlers/
│   │   │   ├── mqtt_ships.go
│   │   │   └── mqtt_ships_test.go        ← NEW
│   │   ├── storage/
│   │   │   ├── ship_store.go
│   │   │   └── ship_store_test.go        ← NEW
│   │   └── mqtt/
│   │       ├── client.go
│   │       └── client_test.go            ← NEW
│   └── go.mod
│
├── ml-service/
│   ├── tests/                            ← NEW DIRECTORY
│   │   ├── __init__.py
│   │   ├── conftest.py                   ← pytest fixtures
│   │   ├── test_data_pipeline.py         ← NEW (Priority)
│   │   ├── test_lstm_model.py            ← NEW
│   │   └── test_api_inference.py         ← Enhance existing
│   ├── src/
│   └── test_api.py                       ← Existing (migrate to tests/)
│
└── docs/
    ├── UNIT_TESTING_PLAN.md              ← This file
    └── TEST_DOCUMENTATION.md              ← Next file
```

---

## ⏱️ Implementation Timeline

### Week 1 (Nov 2-8, 2025)
**Focus: Backend Tests**

- **Day 1-2:** Set up testing infrastructure
  - Create test files structure
  - Install testing frameworks
  - Configure CI/CD integration

- **Day 3-4:** Implement handler tests
  - `mqtt_ships_test.go` (7+ test cases)
  - Estimated: 3 hours

- **Day 5:** Implement storage tests
  - `ship_store_test.go`
  - Estimated: 2 hours

- **Day 6:** Implement MQTT tests
  - `client_test.go`
  - Estimated: 2 hours

- **Day 7:** Code review and coverage check

### Week 2 (Nov 9-15, 2025)
**Focus: ML Service Tests**

- **Day 1-3:** Implement data pipeline tests
  - `test_data_pipeline.py` (10+ test cases)
  - Priority: `remove_outliers` function
  - Estimated: 4 hours

- **Day 4-5:** Implement model tests
  - `test_lstm_model.py`
  - Estimated: 3 hours

- **Day 6:** Enhance API tests
  - Improve existing `test_api.py`
  - Estimated: 2 hours

- **Day 7:** Final integration and documentation

### Deadline: November 17, 2025 ✅

---

## 🎯 Success Criteria

### Required for Submission (Nov 17)
- ✅ All test files created and committed to Git
- ✅ Minimum 15 backend test cases (Go)
- ✅ Minimum 15 ML service test cases (Python)
- ✅ All tests passing in CI/CD
- ✅ Coverage reports generated (HTML format)
- ✅ Test documentation completed

### Quality Gates
- ❌ **Block merge if:** Coverage < 70%
- ❌ **Block merge if:** Any test fails
- ❌ **Block merge if:** Test execution > 60 seconds

---

## 🔗 Related Documents

- **Test Documentation Plan:** `docs/TEST_DOCUMENTATION.md`
- **TODO Checklist:** `TODO_CHECKLIST_NUMBERED.md` (Tasks #4, #5, #14)
- **Code Quality Analysis:** `cyclomatic_complexity_analysis.txt`

---

## 📞 Contact & Ownership

- **Primary Owner:** Angga Suryabrata
- **Backend Tests:** Angga
- **ML Service Tests:** Putri (with Angga support)
- **Code Review:** All team members

---

**Last Updated:** November 2, 2025
**Next Review:** November 10, 2025
