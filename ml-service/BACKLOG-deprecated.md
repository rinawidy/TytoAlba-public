# TytoAlba ML Service - Development Backlog

## 🔴 High Priority

### Data Integration
- [ ] **Implement AIS Data Source**
  - Connect to AIS database (PostgreSQL/MongoDB)
  - Implement `fetch_ais_history()` in `data_pipeline.py`
  - Add data validation and error handling
  - Test with real vessel MMSI numbers

- [ ] **Implement Weather API Integration**
  - Choose weather service (OpenWeatherMap, NOAA, or Marine Weather API)
  - Implement `fetch_weather_along_route()` in `data_pipeline.py`
  - Add API rate limiting and caching
  - Handle API failures gracefully

- [ ] **Collect Training Data**
  - Gather historical voyage data (minimum 1000 voyages)
  - Clean and validate data quality
  - Create train/validation/test splits
  - Document data schema

### Model Training
- [ ] **Initial Model Training**
  - Train model on collected historical data
  - Evaluate performance metrics (MAE, MAPE)
  - Save best model checkpoint
  - Document training process and results

- [ ] **Hyperparameter Tuning**
  - Experiment with different CNN filters (32, 64, 128)
  - Test LSTM units (32, 64, 128)
  - Adjust dropout rates (0.2, 0.3, 0.4)
  - Try different learning rates
  - Document best configuration

## 🟡 Medium Priority

### Model Improvements
- [ ] **Feature Engineering**
  - Add vessel characteristics (length, draft, type, tonnage)
  - Include port-specific features (approach complexity, traffic patterns)
  - Add seasonal patterns (monsoon, trade winds)
  - Experiment with route encoding (one-hot, embedding)

- [ ] **Model Enhancements**
  - Implement multi-head attention mechanism
  - Try Transformer architecture as alternative
  - Add ensemble predictions (multiple models)
  - Implement model confidence calibration

- [ ] **Uncertainty Quantification**
  - Enhance MC Dropout implementation
  - Add prediction intervals (95% confidence)
  - Implement Bayesian neural network approach
  - Visualize uncertainty in predictions

### API & Service
- [ ] **API Enhancements**
  - Add batch prediction endpoint (multiple vessels)
  - Implement WebSocket for real-time updates
  - Add prediction explanation endpoint (feature importance)
  - Create historical prediction logging

- [ ] **Authentication & Security**
  - Implement API key authentication
  - Add JWT token support
  - Rate limiting per user/key
  - Input validation and sanitization

- [ ] **Monitoring & Logging**
  - Set up structured logging (JSON format)
  - Add performance metrics (latency, throughput)
  - Implement prediction accuracy tracking
  - Create monitoring dashboard (Grafana/Prometheus)

### Testing
- [ ] **Unit Tests**
  - Test preprocessing functions
  - Test model prediction logic
  - Test API endpoints
  - Aim for 80%+ coverage

- [ ] **Integration Tests**
  - Test full pipeline (data → prediction)
  - Test with various edge cases
  - Test error handling
  - Load testing (concurrent requests)

- [ ] **Model Validation**
  - Cross-validation on historical data
  - A/B testing framework
  - Continuous evaluation pipeline
  - Drift detection

## 🟢 Low Priority / Future Enhancements

### Advanced Features
- [ ] **Multi-Task Learning**
  - Predict both arrival time AND fuel consumption
  - Add route optimization suggestions
  - Predict optimal speed for ETA

- [ ] **Real-Time Updates**
  - Stream processing for live AIS data
  - Continuous prediction updates during voyage
  - Alert system for ETA changes

- [ ] **Route Planning**
  - Suggest optimal routes based on weather
  - Calculate alternative routes
  - Compare ETA for different routes

- [ ] **Explainability**
  - SHAP values for feature importance
  - Attention weight visualization
  - Generate natural language explanations
  - Create voyage timeline with key events

### Data & Infrastructure
- [ ] **Data Pipeline**
  - Automated data collection from AIS sources
  - Real-time weather data streaming
  - Data quality monitoring
  - Automated retraining pipeline

- [ ] **Model Versioning**
  - Implement MLflow or DVC
  - Track model experiments
  - A/B test different model versions
  - Rollback capability

- [ ] **Scalability**
  - Containerize with Docker
  - Kubernetes deployment
  - Auto-scaling based on load
  - Model serving optimization (TensorFlow Serving)

- [ ] **Database Optimization**
  - Add caching layer (Redis)
  - Optimize AIS data queries
  - Create materialized views for common queries
  - Implement data partitioning

### Frontend Integration
- [ ] **Web Interface**
  - Create admin dashboard for model management
  - Visualization of predictions
  - Historical accuracy charts
  - Model performance metrics

- [ ] **Mobile Support**
  - Optimize API for mobile clients
  - Reduce response payload size
  - Add offline prediction capability

### Documentation
- [ ] **API Documentation**
  - OpenAPI/Swagger enhancement
  - Add code examples in multiple languages
  - Create Postman collection
  - Video tutorials

- [ ] **Developer Guide**
  - Contributing guidelines
  - Code style guide
  - Architecture decision records (ADR)
  - Deployment runbooks

- [ ] **User Guide**
  - End-user documentation
  - Interpretation of confidence scores
  - When to trust predictions
  - Limitations and edge cases

## 🔬 Research & Exploration

### Alternative Approaches
- [ ] **Graph Neural Networks**
  - Model shipping routes as graphs
  - Incorporate port network topology
  - Vessel interaction modeling

- [ ] **Physics-Informed Neural Networks**
  - Incorporate vessel dynamics equations
  - Use physics constraints in loss function
  - Hybrid model-based + data-driven

- [ ] **Reinforcement Learning**
  - Optimal speed control for target ETA
  - Route optimization under uncertainty
  - Multi-agent coordination

### External Data Sources
- [ ] **Additional Data Integration**
  - Sea current data (ocean models)
  - Tide information for port approach
  - Port congestion real-time data
  - Canal transit schedules (Suez, Panama)
  - Piracy risk zones (route detours)

- [ ] **Satellite Imagery**
  - Use satellite data for vessel detection
  - Weather pattern recognition
  - Sea state estimation

## 📊 Performance Targets

### Model Performance Goals
- [ ] MAE < 25 minutes (currently ~30 min target)
- [ ] MAPE < 5% (currently 5-8% target)
- [ ] Confidence calibration: 90% CI contains 90% of actual values
- [ ] Handle 95%+ of prediction requests successfully

### API Performance Goals
- [ ] Response time < 500ms (p95)
- [ ] Throughput > 100 requests/second
- [ ] Uptime > 99.5%
- [ ] Error rate < 0.1%

## 🐛 Known Issues / Technical Debt

### To Fix
- [ ] `fetch_ais_history()` not implemented (raises NotImplementedError)
- [ ] `fetch_weather_along_route()` not implemented (raises NotImplementedError)
- [ ] Missing error handling for malformed AIS data
- [ ] No retry logic for external API calls
- [ ] Hardcoded normalization constants in preprocessing
- [ ] Missing input validation in API endpoints
- [ ] No graceful shutdown handling

### To Refactor
- [ ] Extract configuration to config classes
- [ ] Separate concerns in `data_pipeline.py` (too large)
- [ ] Move constants to configuration files
- [ ] Improve error messages for users
- [ ] Add type hints consistently across codebase

## 📝 Notes

### Dependencies to Monitor
- TensorFlow version updates
- FastAPI security patches
- NumPy/Pandas compatibility

### External Factors
- AIS data availability and cost
- Weather API rate limits and pricing
- Cloud infrastructure costs
- Model retraining frequency

---

## Priority Legend
- 🔴 High Priority: Critical for MVP/production
- 🟡 Medium Priority: Important for production quality
- 🟢 Low Priority: Nice to have / future enhancements
- 🔬 Research: Experimental / proof of concept

## ✅ Recently Completed (2024-10-21)

### Frontend & Backend Integration
- [x] Created backend ship data storage (`backend/data/ships.txt`)
- [x] Implemented Go backend API for ship data (`/api/ships`)
- [x] Updated frontend to fetch from backend with fallback
- [x] Implemented solid/dotted route visualization
  - Solid green line for traversed path
  - Dotted gray line for remaining route
  - Based on `currentRouteIndex` tracking

### Files Created/Modified
- `backend/data/ships.txt` - Ship data in pipe-delimited format
- `backend/internal/handlers/ships.go` - API handler
- `backend/cmd/api/main.go` - HTTP server
- `backend/README.md` - Backend documentation
- `frontend/src/views/Dashboard.vue` - Route visualization updates

---

## Update Log
- 2024-10-21: Initial backlog created for ML service
- 2024-10-21: Completed frontend route visualization with solid/dotted lines
- 2024-10-21: Backend API for ship data implemented
- [Add your updates here]
