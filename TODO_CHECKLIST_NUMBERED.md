# TytoAlba Project - Numbered Task Checklist

**Last Updated:** October 31, 2025
**Total Tasks:** 35
**In Progress:** 3
**Pending:** 32

---

## 📌 Master Checklist - All Tasks

### 🔴 PRIORITY 1 - Code Quality & Refactoring (Week 1-2)

#### ⚙️ Refactoring Tasks
☐ 1. Refactor `remove_outliers` function from CC=10 to CC=3-4
   - File: `ml-service/src/preprocessing/data_pipeline.py:164-209`
   - Owner: Angga/Putri | Effort: 4h | Impact: HIGH

☐ 2. Extract parameter parsing in `GetShipHistory` (reduce CC from 7 to 5)
   - File: `backend/internal/handlers/mqtt_ships.go:100-137`
   - Owner: Angga | Effort: 2h | Impact: MEDIUM

☐ 3. Create middleware for CORS and error handling standardization
   - File: `backend/internal/middleware/` (new)
   - Owner: Angga | Effort: 3h | Impact: HIGH

#### 🧪 Testing Tasks
☐ 4. Add unit tests for `remove_outliers` (10+ test cases)
   - File: `ml-service/tests/test_data_pipeline.py`
   - Owner: Putri | Effort: 4h | Impact: HIGH

☐ 5. Add unit tests for `GetShipHistory` (7+ test cases)
   - File: `backend/internal/handlers/mqtt_ships_test.go`
   - Owner: Angga | Effort: 3h | Impact: MEDIUM

#### 🔄 CI/CD Integration
☐ 6. Integrate gocyclo and radon into CI/CD pipeline
   - File: `.github/workflows/code-quality.yml`
   - Owner: Angga | Effort: 2h | Impact: MEDIUM

---

### 🟡 PRIORITY 2 - Core Development (Week 3-6)

#### 🎨 Frontend Development (Vue.js 3)
🔄 7. Complete Frontend development (Vue.js 3 + TypeScript + Tailwind) **[IN PROGRESS]**
   - Files: `frontend/src/views/Dashboard.vue`, components
   - Owner: Putri | Effort: 20h | Impact: CRITICAL

☐ 8. Implement route visualization with Leaflet.js (solid/dotted lines)
   - File: `frontend/src/components/RouteMap.vue`
   - Owner: Putri | Effort: 8h | Impact: HIGH

☐ 9. Integrate weather API for real-time weather overlay on map
   - File: `frontend/src/components/WeatherOverlay.vue`
   - Owner: Putri | Effort: 6h | Impact: MEDIUM

#### 🤖 ML Service Development
🔄 10. Complete ML Service integration with Backend API **[IN PROGRESS]**
   - File: `ml-service/src/api_inference.py`
   - Owner: Angga/Putri | Effort: 12h | Impact: CRITICAL

🔄 11. Improve LSTM model accuracy from 92% to 95% target **[IN PROGRESS]**
   - File: `ml-service/src/models/lstm_arrival_predictor.py`
   - Owner: Putri | Effort: 30h | Impact: CRITICAL

#### 💾 Database & Infrastructure
☐ 12. Implement PostgreSQL database with TimescaleDB extension
   - Files: `database/schema.sql`, `backend/internal/database/`
   - Owner: Rina/Angga | Effort: 15h | Impact: HIGH

☐ 13. Deploy MQTT infrastructure to production environment
   - Files: Docker configs, mosquitto setup
   - Owner: Angga | Effort: 6h | Impact: HIGH

---

### 🟢 PRIORITY 3 - Testing & Quality Assurance (Week 7-8)

☐ 14. Create comprehensive testing pyramid (Unit 60%, Integration 30%, E2E 10%)
   - Files: Multiple test files across all services
   - Owner: All | Effort: 25h | Impact: CRITICAL

☐ 15. Write 50+ detailed test cases covering all test categories
   - File: `docs/testing/test_cases.md`
   - Owner: All | Effort: 12h | Impact: HIGH

☐ 16. Set up performance testing (load, stress, ML inference latency)
   - Files: `tests/performance/` with k6 scripts
   - Owner: Angga | Effort: 8h | Impact: MEDIUM

☐ 17. Conduct User Acceptance Testing (UAT) with 2-week pilot
   - Duration: 2 weeks (Dec 16-26, 2025)
   - Owner: All | Effort: 40h | Impact: CRITICAL

---

### 📚 PRIORITY 4 - UAS Report Documentation (Week 9-10)

#### 📄 UAS Report Enhancements
☐ 18. Create Risk Management section with mitigation strategies
   - File: `docs/UAS_Section_RiskManagement.md`
   - Owner: Angga | Effort: 4h | Impact: HIGH

☐ 19. Develop enhanced Stakeholder Analysis matrix with engagement strategy
   - File: `docs/UAS_Section_Stakeholders.md`
   - Owner: Rina | Effort: 3h | Impact: MEDIUM

☐ 20. Document Project Governance Structure with RACI matrix
   - File: `docs/UAS_Section_Governance.md`
   - Owner: Angga | Effort: 3h | Impact: MEDIUM

☐ 21. Create detailed LSTM Architecture documentation with hyperparameter tuning results
   - File: `docs/UAS_Section_MLArchitecture.md`
   - Owner: Putri | Effort: 6h | Impact: HIGH

☐ 22. Document Data Pipeline with quality gates and validation rules
   - File: `docs/UAS_Section_DataPipeline.md`
   - Owner: Rina | Effort: 4h | Impact: MEDIUM

☐ 23. Develop Change Management & User Adoption strategy
   - File: `docs/UAS_Section_ChangeManagement.md`
   - Owner: Angga | Effort: 3h | Impact: MEDIUM

☐ 24. Create training materials and user manual for captains
   - Files: `docs/user_manual/ShipCaptain.pdf`, `FleetManager.pdf`
   - Owner: Putri | Effort: 12h | Impact: HIGH

☐ 25. Define Post-Implementation Review Framework with success metrics
   - File: `docs/UAS_Section_PostImplementation.md`
   - Owner: All | Effort: 2h | Impact: MEDIUM

☐ 26. Document Technical Debt backlog with prioritization
   - File: `docs/UAS_Section_Maintenance.md`
   - Owner: Angga | Effort: 3h | Impact: MEDIUM

☐ 27. Create maintenance schedule and runbooks for common issues
   - Files: `docs/operations/runbooks/`
   - Owner: Angga | Effort: 6h | Impact: HIGH

☐ 28. Write Academic Contribution section addressing research questions
   - File: `docs/UAS_Section_AcademicContribution.md`
   - Owner: Putri | Effort: 8h | Impact: CRITICAL

☐ 29. Prepare comparative analysis (LSTM vs Random Forest vs Linear Regression)
   - File: `ml-service/experiments/model_comparison.py`
   - Owner: Putri | Effort: 10h | Impact: HIGH

#### 🛠️ Technical Documentation
☐ 30. Generate API documentation using OpenAPI/Swagger
   - File: `docs/api/openapi.yaml`
   - Owner: Angga | Effort: 4h | Impact: HIGH

☐ 31. Create Docker Compose and Kubernetes deployment manifests
   - Files: `docker-compose.yml`, `k8s/`
   - Owner: Angga | Effort: 6h | Impact: CRITICAL

☐ 32. Implement monitoring dashboard for system health and model performance
   - Files: Grafana dashboards, Prometheus configs
   - Owner: Angga | Effort: 8h | Impact: HIGH

☐ 33. Set up automated model retraining pipeline
   - File: `ml-service/training/auto_retrain.py`
   - Owner: Putri | Effort: 10h | Impact: MEDIUM

☐ 34. Conduct security testing (OWASP Top 10, penetration testing)
   - Owner: Angga | Effort: 8h | Impact: HIGH

#### 📑 Appendices
☐ 35. Create all UAS appendices (A-J) including Sprint Retrospectives, ERD, etc.
   - Files: `docs/appendices/Appendix_A.md` through `Appendix_J.md`
   - Owner: All | Effort: 20h | Impact: CRITICAL

---

## 📊 Task Summary by Category

### By Priority
- 🔴 **Priority 1 (Code Quality):** Tasks #1-6 (6 tasks)
- 🟡 **Priority 2 (Development):** Tasks #7-13 (7 tasks)
- 🟢 **Priority 3 (Testing):** Tasks #14-17 (4 tasks)
- 📚 **Priority 4 (Documentation):** Tasks #18-35 (18 tasks)

### By Status
- ⏳ **Pending:** 31 tasks
- 🔄 **In Progress:** 3 tasks (#7, #10, #11)
- ✅ **Completed:** 0 tasks

### By Owner
- **Angga (Backend/DevOps):** Tasks #2, #3, #5, #6, #10, #12, #13, #16, #18, #20, #23, #26, #27, #30, #31, #32, #34
- **Putri (Frontend/ML):** Tasks #1, #4, #7, #8, #9, #11, #21, #24, #28, #29, #33
- **Rina (Database/Data):** Tasks #12, #19, #22
- **All Team:** Tasks #10, #14, #15, #17, #25, #35

### By Effort (Estimated Hours)
- 🔥 **High Effort (>10 hours):** Tasks #7, #10, #11, #12, #14, #15, #17, #24, #35
- ⚡ **Medium Effort (5-10 hours):** Tasks #8, #9, #16, #21, #27, #29, #31, #32, #33, #34
- ✅ **Low Effort (<5 hours):** Tasks #1, #2, #3, #4, #5, #6, #18, #19, #20, #22, #23, #25, #26, #30

### By Impact
- 🔥 **Critical Impact:** Tasks #7, #10, #11, #14, #17, #28, #31, #35
- ⭐ **High Impact:** Tasks #1, #3, #4, #8, #12, #13, #15, #18, #21, #24, #27, #30, #32, #34
- ✨ **Medium Impact:** Tasks #2, #5, #6, #9, #16, #19, #20, #22, #23, #25, #26, #29, #33

---

## 🎯 Recommended Task Sequence

### This Week (Nov 1-7, 2025)
**Focus:** Code Quality & Critical Development

**Day 1-2:**
- [ ] Task #1: Refactor `remove_outliers` (4h)
- [ ] Task #2: Extract parameter parsing (2h)
- [ ] Task #3: Create middleware (3h)

**Day 3-4:**
- [ ] Task #4: Unit tests for `remove_outliers` (4h)
- [ ] Task #5: Unit tests for `GetShipHistory` (3h)
- [ ] Task #6: CI/CD integration (2h)

**Day 5-7:**
- [ ] Task #7: Continue frontend development (10h this week)
- [ ] Task #11: LSTM model improvement (10h this week)

**Total Effort This Week:** ~38 hours

---

### Next Week (Nov 8-14, 2025)
**Focus:** Complete Core Features

- [ ] Task #7: Finish frontend dashboard (10h)
- [ ] Task #8: Route visualization (8h)
- [ ] Task #9: Weather API integration (6h)
- [ ] Task #10: ML Service API completion (10h)
- [ ] Task #11: LSTM model tuning continues (10h)

**Total Effort:** ~44 hours

---

### Week 3-4 (Nov 15-28, 2025)
**Focus:** Database & Infrastructure

- [ ] Task #12: PostgreSQL + TimescaleDB (15h)
- [ ] Task #13: MQTT production deployment (6h)
- [ ] Task #30: API documentation (4h)
- [ ] Task #31: Docker/K8s configs (6h)
- [ ] Continue Task #11: Final model tuning (10h)

**Total Effort:** ~41 hours

---

### Week 5-6 (Nov 29 - Dec 12, 2025)
**Focus:** Testing Preparation

- [ ] Task #14: Testing pyramid implementation (25h)
- [ ] Task #15: Write test cases (12h)
- [ ] Task #16: Performance testing setup (8h)
- [ ] Task #32: Monitoring dashboard (8h)

**Total Effort:** ~53 hours

---

### Week 7-8 (Dec 13-26, 2025)
**Focus:** UAT & Testing

- [ ] Task #17: User Acceptance Testing (40h over 2 weeks)
- [ ] Task #34: Security testing (8h)
- Execute all tests from Task #14-15

**Total Effort:** ~48 hours

---

### Week 9-10 (Dec 27 - Jan 9, 2026)
**Focus:** UAS Documentation

**Documentation Tasks:**
- [ ] Task #18: Risk Management (4h)
- [ ] Task #19: Stakeholder Analysis (3h)
- [ ] Task #20: Governance Structure (3h)
- [ ] Task #21: LSTM Architecture (6h)
- [ ] Task #22: Data Pipeline (4h)
- [ ] Task #23: Change Management (3h)
- [ ] Task #24: User manuals (12h)
- [ ] Task #25: Post-Implementation Review (2h)
- [ ] Task #26: Technical Debt (3h)
- [ ] Task #27: Maintenance runbooks (6h)
- [ ] Task #28: Academic Contribution (8h)
- [ ] Task #29: Comparative analysis (10h)
- [ ] Task #35: All appendices (20h)

**Total Effort:** ~84 hours

---

### Week 11 (Jan 10-16, 2026)
**Focus:** Deployment

- Deployment activities (not in numbered list - part of final phase)
- System handover
- Hypercare support

---

### Week 12 (Jan 17-23, 2026)
**Focus:** Final Submission

- Compile final UAS report
- Create presentation
- Final submission

---

## 🚀 Quick Start Guide

### Want to Start Now?

**Option 1: High Impact, Quick Wins**
Start with: Tasks #1, #2, #3 (Code quality improvements, ~9 hours total)

**Option 2: Critical Path First**
Start with: Tasks #7, #10, #11 (Already in progress, keep momentum)

**Option 3: Complete One Category**
Start with: Tasks #1-6 (Finish all code quality tasks, ~18 hours total)

**Option 4: Documentation Sprint**
Start with: Tasks #18-29 (UAS documentation, can work in parallel)

---

## 💬 How to Select Tasks

Just tell me:
- **Single task:** "Let's do task #1"
- **Multiple tasks:** "I want to work on tasks #1, #2, and #3"
- **By category:** "Let's tackle all Priority 1 tasks"
- **By owner:** "Show me all tasks for Putri"
- **By time:** "What can I finish in 4 hours?"

I'll help you with:
- Detailed implementation steps
- Code examples
- Testing strategies
- Documentation templates
- Progress tracking

---

**Ready to start? Just pick your task number(s)!** 🎯
