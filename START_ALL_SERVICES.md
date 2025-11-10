# TytoAlba - Start All Services

## Quick Start Guide

### Option 1: Manual Start (Recommended for Demo)

Open **3 separate terminals** and run:

#### Terminal 1: Backend (Go API)
```bash
cd /mnt/c/Users/angga.suryabrata/VisCode/TytoAlba/backend
go run cmd/api/main.go
```
**Port:** 8080 (default for Go backend)

---

#### Terminal 2: Frontend (Vue.js)
```bash
cd /mnt/c/Users/angga.suryabrata/VisCode/TytoAlba/frontend
npm run dev
```
**Port:** 5173 (Vite default) or 3000
**Access:** http://localhost:5173

---

#### Terminal 3: ML Service (Python Flask)
```bash
cd /mnt/c/Users/angga.suryabrata/VisCode/TytoAlba/ml-service
source venv/bin/activate
python api/ml_service.py
```
**Port:** 5000
**Access:** http://localhost:5000

---

## Service Endpoints

### Backend (Go)
- Base URL: `http://localhost:8080`
- WebSocket: `ws://localhost:8080/ws`
- API: `/api/v1/*`

### Frontend
- Dashboard: `http://localhost:5173`
- Shows 29 ships with real-time tracking

### ML Service
- Health: `GET http://localhost:5000/health`
- Model Info: `GET http://localhost:5000/api/models/info`
- ETA: `POST http://localhost:5000/api/predict/eta`
- Fuel: `POST http://localhost:5000/api/predict/fuel`
- Anomaly: `POST http://localhost:5000/api/detect/anomaly`
- Route: `POST http://localhost:5000/api/optimize/route`

---

## Before First Run

### 1. Install ML Service Dependencies
```bash
cd /mnt/c/Users/angga.suryabrata/VisCode/TytoAlba/ml-service
source venv/bin/activate
pip install torch>=2.5.0 numpy>=1.26.0 flask>=3.0.0 flask-cors>=4.0.0
```

### 2. Install Frontend Dependencies (if not done)
```bash
cd /mnt/c/Users/angga.suryabrata/VisCode/TytoAlba/frontend
npm install
```

### 3. Install Backend Dependencies (if not done)
```bash
cd /mnt/c/Users/angga.suryabrata/VisCode/TytoAlba/backend
go mod download
```

---

## Verify Services Running

### Check Backend
```bash
curl http://localhost:8080/health
# or
curl http://localhost:8080/api/v1/ships
```

### Check ML Service
```bash
curl http://localhost:5000/health
```

### Check Frontend
Open browser: `http://localhost:5173`

---

## Troubleshooting

### Backend won't start
- Check if port 8080 is in use: `lsof -i :8080`
- Make sure Go is installed: `go version`

### Frontend won't start
- Check if port 5173 is in use
- Run `npm install` in frontend directory
- Check Node version: `node -v` (need v16+)

### ML Service won't start
- Make sure venv is activated
- Install dependencies: `pip install torch numpy flask flask-cors`
- Check Python version: `python --version` (need 3.8+)

---

## Demo Day Quick Start

**Run these commands in order:**

```bash
# Terminal 1 - ML Service
cd /mnt/c/Users/angga.suryabrata/VisCode/TytoAlba/ml-service
source venv/bin/activate
python api/ml_service.py

# Terminal 2 - Backend
cd /mnt/c/Users/angga.suryabrata/VisCode/TytoAlba/backend
go run cmd/api/main.go

# Terminal 3 - Frontend
cd /mnt/c/Users/angga.suryabrata/VisCode/TytoAlba/frontend
npm run dev
```

**Access:** http://localhost:5173

---

## Architecture Flow

```
User Browser (localhost:5173)
    ↓
Frontend (Vue.js + Vite)
    ↓
Backend API (Go - localhost:8080)
    ↓
ML Service (Python Flask - localhost:5000)
    ↓
4 LSTM Models (PyTorch)
```

---

**Created:** November 7, 2025
**Project:** TytoAlba Maritime Vessel Tracking & Prediction
