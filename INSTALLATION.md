# TytoAlba Installation Guide

Complete setup instructions for development and production deployment.

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Quick Setup (Development)](#quick-setup-development)
3. [MQTT Broker Setup](#mqtt-broker-setup)
4. [Backend Setup](#backend-setup)
5. [ML Service Setup](#ml-service-setup)
6. [Frontend Setup](#frontend-setup)
7. [Docker Deployment](#docker-deployment)
8. [Production Deployment](#production-deployment)
9. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum
- **OS**: Linux (Ubuntu 20.04+), macOS 11+, Windows 10+ with WSL2
- **CPU**: 4 cores
- **RAM**: 8 GB
- **Storage**: 10 GB free space

### Recommended (for ML Training)
- **CPU**: 8+ cores
- **RAM**: 16 GB
- **GPU**: NVIDIA GPU with CUDA 11.8+ (optional, auto-detected)
- **Storage**: 20 GB SSD

### Software Dependencies
- **Go**: 1.21 or higher
- **Python**: 3.10, 3.11 (TensorFlow 2.15 compatibility)
- **Node.js**: 18 or higher
- **MQTT Broker**: Mosquitto 2.0+ or HiveMQ
- **Git**: 2.30+

---

## Quick Setup (Development)

### 1. Clone Repository
```bash
git clone https://github.com/your-org/tytoalba.git
cd tytoalba
```

### 2. Install All Dependencies
```bash
# Backend (Go)
cd backend
go mod tidy
cd ..

# ML Service (Python)
cd ml-service
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cd ..

# Frontend (Node.js)
cd frontend
npm install
cd ..
```

### 3. Start MQTT Broker
```bash
# Install Mosquitto
sudo apt-get update
sudo apt-get install mosquitto mosquitto-clients

# Start broker
mosquitto -v
```

### 4. Start All Services

**Terminal 1 - Backend**:
```bash
cd backend
go run cmd/api/main.go
```

**Terminal 2 - ML Service**:
```bash
cd ml-service
source venv/bin/activate
python train.py --synthetic --n_samples 1000 --epochs 50
python inference.py
```

**Terminal 3 - Frontend**:
```bash
cd frontend
npm run dev
```

### 5. Verify Installation
Open browser:
- Frontend: http://localhost:3000
- Backend: http://localhost:8080/health
- ML Service: http://localhost:8000/docs

---

## MQTT Broker Setup

### Option 1: Local Mosquitto (Development)

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y mosquitto mosquitto-clients

# Enable and start service
sudo systemctl enable mosquitto
sudo systemctl start mosquitto

# Check status
sudo systemctl status mosquitto
```

#### macOS
```bash
brew install mosquitto

# Start broker
brew services start mosquitto

# Or run in foreground
/opt/homebrew/opt/mosquitto/sbin/mosquitto -c /opt/homebrew/etc/mosquitto/mosquitto.conf
```

#### Windows
1. Download from https://mosquitto.org/download/
2. Install to `C:\Program Files\mosquitto`
3. Run as service or from command line:
```cmd
cd "C:\Program Files\mosquitto"
mosquitto.exe -v
```

### Option 2: Docker Mosquitto

Create `docker-compose.mqtt.yml`:
```yaml
version: '3.8'

services:
  mosquitto:
    image: eclipse-mosquitto:2.0
    container_name: tytoalba-mqtt
    ports:
      - "1883:1883"
      - "9001:9001"
    volumes:
      - ./mqtt/config:/mosquitto/config
      - ./mqtt/data:/mosquitto/data
      - ./mqtt/log:/mosquitto/log
    restart: unless-stopped
```

Create `mqtt/config/mosquitto.conf`:
```
listener 1883
allow_anonymous true
persistence true
persistence_location /mosquitto/data/
log_dest file /mosquitto/log/mosquitto.log
log_dest stdout
```

Start:
```bash
docker-compose -f docker-compose.mqtt.yml up -d
docker logs -f tytoalba-mqtt
```

### Option 3: Cloud MQTT (Production)

**HiveMQ Cloud**:
```bash
# Get credentials from HiveMQ console
export MQTT_BROKER_URL="ssl://your-cluster.hivemq.cloud:8883"
export MQTT_USERNAME="your-username"
export MQTT_PASSWORD="your-password"
```

**AWS IoT Core**:
```bash
export MQTT_BROKER_URL="ssl://your-endpoint.iot.region.amazonaws.com:8883"
# Use certificates for authentication
```

### Testing MQTT
```bash
# Subscribe to all topics
mosquitto_sub -h localhost -t "tytoalba/ships/#" -v

# Publish test message (in another terminal)
mosquitto_pub -h localhost -t "tytoalba/ships/563012345/ais" -m '{
  "vessel_mmsi": "563012345",
  "ship_type": "bulk_carrier",
  "timestamp": "2024-10-28T12:00:00Z",
  "latitude": -5.5,
  "longitude": 112.5,
  "speed": 12.5,
  "course": 145.0
}'
```

---

## Backend Setup

### 1. Install Go
```bash
# Ubuntu/Debian
sudo apt-get install golang-1.21

# macOS
brew install go@1.21

# Verify
go version  # Should show go1.21.x
```

### 2. Install Dependencies
```bash
cd backend
go mod tidy
```

### 3. Configuration

Create `.env` file:
```bash
MQTT_BROKER_URL=tcp://localhost:1883
MQTT_CLIENT_ID=tytoalba-backend
MQTT_USERNAME=
MQTT_PASSWORD=
PORT=:8080
```

### 4. Run Backend
```bash
# Development
go run cmd/api/main.go

# Build binary
go build -o tytoalba-backend cmd/api/main.go

# Run binary
./tytoalba-backend
```

### 5. Verify Backend
```bash
# Health check
curl http://localhost:8080/health

# MQTT status
curl http://localhost:8080/mqtt/status

# Ship statistics
curl http://localhost:8080/api/mqtt/stats
```

Expected output:
```json
{
  "status": "healthy",
  "service": "tytoalba-backend",
  "mqtt_connected": true,
  "ships_tracked": 0
}
```

---

## ML Service Setup

### 1. Install Python
```bash
# Ubuntu/Debian
sudo apt-get install python3.10 python3.10-venv python3-pip

# macOS
brew install python@3.10

# Verify
python3 --version  # Should show Python 3.10.x
```

### 2. Create Virtual Environment
```bash
cd ml-service
python3 -m venv venv

# Activate
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: TensorFlow installation may take 5-10 minutes.

### 4. GPU Setup (Optional)

If you have NVIDIA GPU:
```bash
# Check CUDA version
nvidia-smi

# Install CUDA-enabled TensorFlow
pip install tensorflow[and-cuda]==2.15.0

# Verify GPU detection
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### 5. Train Model

**With Synthetic Data** (for testing):
```bash
python train.py --synthetic --n_samples 1000 --epochs 50
```

**With Your Own Data**:
```bash
# Prepare CSV file in data/ folder
python train.py --data data/historical_voyages.csv --epochs 100 --batch_size 32
```

Training output:
```
======================================================================
  TytoAlba ML Service - LSTM Model Training
  Bulk Carrier Vessel Arrival Time Prediction
======================================================================
✓ GPU detected: 1 GPU(s)
📂 Loading training data from: ...
✓ Training completed!
  Best model: models/vessel_arrival_lstm_20241028_123045.h5
  Final model: models/vessel_arrival_lstm.h5
```

### 6. Start Inference Server
```bash
python inference.py
```

Output:
```
======================================================================
  TytoAlba ML Inference Server Starting...
======================================================================
✓ Model loaded successfully from models/vessel_arrival_lstm.h5
  Device: GPU
  Parameters: 145,729

🚀 Starting inference server on 0.0.0.0:8000
   Device: GPU
   API Docs: http://0.0.0.0:8000/docs
```

### 7. Test ML Service
```bash
# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model/info

# Single prediction (will fail without AIS data implementation)
curl -X POST http://localhost:8000/predict/arrival \
  -H "Content-Type: application/json" \
  -d '{
    "vessel_mmsi": "563012345",
    "ship_type": "bulk_carrier",
    "destination_lat": 1.2644,
    "destination_lon": 103.8229
  }'
```

### 8. Evaluate Model (Optional)
```bash
python evaluate.py --synthetic --n_samples 200
```

Results saved to `evaluation_results/`:
- `prediction_vs_actual_*.png`
- `error_distribution_*.png`
- `accuracy_by_window_*.png`
- `evaluation_report_*.txt`

---

## Frontend Setup

### 1. Install Node.js
```bash
# Ubuntu/Debian
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# macOS
brew install node@18

# Verify
node --version  # Should show v18.x.x
npm --version
```

### 2. Install Dependencies
```bash
cd frontend
npm install
```

### 3. Configuration

Update `.env.local`:
```bash
VITE_API_URL=http://localhost:8080
VITE_ML_API_URL=http://localhost:8000
```

### 4. Run Development Server
```bash
npm run dev
```

Output:
```
VITE v4.x.x  ready in 123 ms

➜  Local:   http://localhost:3000/
➜  Network: use --host to expose
```

### 5. Build for Production
```bash
npm run build

# Preview production build
npm run preview
```

---

## Docker Deployment

### Full Stack Docker Compose

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  mosquitto:
    image: eclipse-mosquitto:2.0
    ports:
      - "1883:1883"
    volumes:
      - ./mqtt/config:/mosquitto/config
      - mqtt-data:/mosquitto/data
    restart: unless-stopped

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8080:8080"
    environment:
      - MQTT_BROKER_URL=tcp://mosquitto:1883
      - MQTT_CLIENT_ID=tytoalba-backend
      - PORT=:8080
    depends_on:
      - mosquitto
    restart: unless-stopped

  ml-service:
    build:
      context: ./ml-service
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=models/vessel_arrival_lstm.h5
      - API_HOST=0.0.0.0
      - API_PORT=8000
    volumes:
      - ./ml-service/models:/app/models
    restart: unless-stopped

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:80"
    environment:
      - VITE_API_URL=http://localhost:8080
      - VITE_ML_API_URL=http://localhost:8000
    restart: unless-stopped

volumes:
  mqtt-data:
```

### Create Dockerfiles

**backend/Dockerfile**:
```dockerfile
FROM golang:1.21-alpine AS builder
WORKDIR /app
COPY . .
RUN go mod download
RUN go build -o tytoalba-backend cmd/api/main.go

FROM alpine:latest
WORKDIR /app
COPY --from=builder /app/tytoalba-backend .
COPY data/ ./data/
EXPOSE 8080
CMD ["./tytoalba-backend"]
```

**ml-service/Dockerfile**:
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "inference.py"]
```

**frontend/Dockerfile**:
```dockerfile
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### Start All Services
```bash
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## Production Deployment

### 1. Environment Configuration

Create production `.env`:
```bash
# MQTT
MQTT_BROKER_URL=ssl://your-broker.cloud:8883
MQTT_USERNAME=production-user
MQTT_PASSWORD=secure-password
MQTT_CLIENT_ID=tytoalba-backend-prod

# Backend
PORT=:8080

# ML Service
MODEL_PATH=models/vessel_arrival_lstm.h5
API_HOST=0.0.0.0
API_PORT=8000
```

### 2. Security Checklist

- [ ] Enable MQTT TLS/SSL encryption
- [ ] Use MQTT authentication (username/password)
- [ ] Configure MQTT ACLs (access control)
- [ ] Enable backend CORS for specific origins only
- [ ] Use HTTPS for all HTTP endpoints
- [ ] Implement API authentication
- [ ] Set secure environment variables
- [ ] Configure firewall rules

### 3. Systemd Services (Linux)

**/etc/systemd/system/tytoalba-backend.service**:
```ini
[Unit]
Description=TytoAlba Backend API
After=network.target mosquitto.service

[Service]
Type=simple
User=tytoalba
WorkingDirectory=/opt/tytoalba/backend
ExecStart=/opt/tytoalba/backend/tytoalba-backend
Restart=always
Environment="MQTT_BROKER_URL=tcp://localhost:1883"
Environment="PORT=:8080"

[Install]
WantedBy=multi-user.target
```

**/etc/systemd/system/tytoalba-ml.service**:
```ini
[Unit]
Description=TytoAlba ML Service
After=network.target

[Service]
Type=simple
User=tytoalba
WorkingDirectory=/opt/tytoalba/ml-service
ExecStart=/opt/tytoalba/ml-service/venv/bin/python inference.py
Restart=always
Environment="MODEL_PATH=models/vessel_arrival_lstm.h5"

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable tytoalba-backend
sudo systemctl enable tytoalba-ml
sudo systemctl start tytoalba-backend
sudo systemctl start tytoalba-ml

# Check status
sudo systemctl status tytoalba-backend
sudo systemctl status tytoalba-ml
```

### 4. Nginx Reverse Proxy

**/etc/nginx/sites-available/tytoalba**:
```nginx
server {
    listen 80;
    server_name tytoalba.example.com;

    # Frontend
    location / {
        root /var/www/tytoalba/frontend/dist;
        try_files $uri $uri/ /index.html;
    }

    # Backend API
    location /api/ {
        proxy_pass http://localhost:8080/api/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # ML Service
    location /ml/ {
        proxy_pass http://localhost:8000/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
    }
}
```

Enable and restart:
```bash
sudo ln -s /etc/nginx/sites-available/tytoalba /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

---

## Troubleshooting

### MQTT Connection Issues

**Problem**: `MQTT connection failed: connection refused`

**Solutions**:
```bash
# Check if broker is running
sudo systemctl status mosquitto
mosquitto -v

# Check port
sudo netstat -tlnp | grep 1883

# Test connection
mosquitto_sub -h localhost -t test
```

### TensorFlow GPU Not Detected

**Problem**: Model trains on CPU despite having GPU

**Solutions**:
```bash
# Check NVIDIA driver
nvidia-smi

# Install CUDA toolkit
sudo apt-get install nvidia-cuda-toolkit

# Reinstall TensorFlow with CUDA
pip uninstall tensorflow
pip install tensorflow[and-cuda]==2.15.0

# Verify
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Backend Port Already in Use

**Problem**: `bind: address already in use`

**Solutions**:
```bash
# Find process using port 8080
sudo lsof -ti:8080

# Kill process
kill $(lsof -ti:8080)

# Or change port
export PORT=:8081
```

### ML Model Not Found

**Problem**: `Model file not found at models/vessel_arrival_lstm.h5`

**Solution**:
```bash
cd ml-service
python train.py --synthetic --n_samples 1000 --epochs 50
```

### Frontend API Connection Failed

**Problem**: Frontend can't connect to backend

**Solutions**:
```bash
# Check backend is running
curl http://localhost:8080/health

# Update frontend .env.local
VITE_API_URL=http://localhost:8080

# Restart frontend
npm run dev
```

### Python Version Incompatibility

**Problem**: `TensorFlow requires Python 3.10 or 3.11`

**Solution**:
```bash
# Install Python 3.10
sudo apt-get install python3.10 python3.10-venv

# Create venv with correct version
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Next Steps

After successful installation:

1. **Test MQTT Data Flow**
   - Publish test ship data
   - Verify backend receives data
   - Check ship appears in API

2. **Train ML Model**
   - Collect historical voyage data
   - Train model with real data
   - Evaluate model performance

3. **Configure Ship Devices**
   - Implement MQTT client on ships
   - Test data transmission
   - Verify data format

4. **Deploy to Production**
   - Set up cloud MQTT broker
   - Deploy services with Docker
   - Configure monitoring

---

**For project overview**, see [README.md](README.md)
**For version history**, see [CHANGELOG.md](CHANGELOG.md)
