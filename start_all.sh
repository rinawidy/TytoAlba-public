#!/bin/bash

# TytoAlba - Start All Services
# This script starts all 3 services in separate background processes

echo "========================================"
echo "TytoAlba - Starting All Services"
echo "========================================"
echo ""

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Log files
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

echo "📁 Project root: $PROJECT_ROOT"
echo "📝 Logs directory: $LOG_DIR"
echo ""

# Function to kill any existing process on port
kill_port() {
    local port=$1
    local service_name=$2

    # Method 1: Kill by port using fuser
    if command -v fuser >/dev/null 2>&1; then
        fuser -k ${port}/tcp 2>/dev/null && echo "   🔄 Cleaning up old $service_name process (Port $port)..." && sleep 1
    fi

    # Method 2: Parse PIDs from ss/netstat and kill
    local pids=$(ss -tlnp 2>/dev/null | grep ":$port" | grep -oP 'pid=\K[0-9]+' | sort -u)
    if [ -z "$pids" ]; then
        pids=$(netstat -tlnp 2>/dev/null | grep ":$port" | awk '{print $7}' | cut -d'/' -f1 | grep -E '^[0-9]+$')
    fi

    if [ -n "$pids" ]; then
        echo "   🔄 Cleaning up old $service_name process (Port $port, PIDs: $pids)..."
        for pid in $pids; do
            kill -9 $pid 2>/dev/null
        done
        sleep 1
    fi

    # Method 3: Kill by process patterns
    if [ "$port" = "8080" ]; then
        pkill -9 -f "cmd/api/main.go" 2>/dev/null
        pkill -9 -f "go run.*main.go" 2>/dev/null
        pkill -9 -f "go-build.*main" 2>/dev/null
    elif [ "$port" = "5000" ]; then
        pkill -9 -f "ml_service.py" 2>/dev/null
    elif [ "$port" = "5173" ]; then
        pkill -9 -f "vite" 2>/dev/null
        pkill -9 -f "node.*vite" 2>/dev/null
    fi

    # Verify port is now free
    sleep 1
    if ss -tln 2>/dev/null | grep -q ":$port " || netstat -tln 2>/dev/null | grep -q ":$port "; then
        echo "   ⚠️  Warning: Port $port still in use after cleanup"
        return 1
    fi
    return 0
}

# Check MQTT Service
echo "📡 Checking MQTT Broker (Port 1883)..."
if systemctl is-active --quiet mosquitto 2>/dev/null; then
    echo "   ✓ MQTT Broker (mosquitto) is running"
elif pgrep mosquitto > /dev/null 2>&1; then
    echo "   ✓ MQTT Broker (mosquitto) is running"
else
    echo "   ⚠️  MQTT Broker not detected (optional for basic functionality)"
    echo "   💡 To install: sudo apt-get install mosquitto mosquitto-clients"
fi
echo ""

# Start Backend (depends on MQTT)
echo "🔧 Starting Backend (Port 8080)..."
kill_port 8080 "Backend"
cd "$PROJECT_ROOT/backend"
nohup go run cmd/api/main.go > "$LOG_DIR/backend.log" 2>&1 &
BACKEND_PID=$!
echo "   ✓ Backend started (PID: $BACKEND_PID)"
echo "   📄 Log: $LOG_DIR/backend.log"
sleep 2
echo ""

# Start ML Service (depends on Backend)
echo "🐍 Starting ML Service (Port 5000)..."
kill_port 5000 "ML Service"
cd "$PROJECT_ROOT/ml-service"
source venv/bin/activate
nohup python api/ml_service.py > "$LOG_DIR/ml-service.log" 2>&1 &
ML_PID=$!
echo "   ✓ ML Service started (PID: $ML_PID)"
echo "   📄 Log: $LOG_DIR/ml-service.log"
sleep 2
echo ""

# Start Frontend (depends on Backend + ML Service)
echo "🎨 Starting Frontend (Port 5173)..."
kill_port 5173 "Frontend"
cd "$PROJECT_ROOT/frontend"
nohup npm run dev > "$LOG_DIR/frontend.log" 2>&1 &
FRONTEND_PID=$!
echo "   ✓ Frontend started (PID: $FRONTEND_PID)"
echo "   📄 Log: $LOG_DIR/frontend.log"
echo ""

# Wait for services to start
echo "⏳ Waiting for services to initialize..."
sleep 5
echo ""

# Check service health
echo "========================================"
echo "Service Status"
echo "========================================"

# Check ML Service
if curl -s http://localhost:5000/health > /dev/null 2>&1; then
    echo "✅ ML Service: Running (http://localhost:5000)"
else
    echo "❌ ML Service: Not responding"
fi

# Check Backend
if curl -s http://localhost:8080/health > /dev/null 2>&1; then
    echo "✅ Backend: Running (http://localhost:8080)"
else
    echo "❌ Backend: Not responding (may still be starting)"
fi

# Check Frontend
if curl -s http://localhost:5173 > /dev/null 2>&1; then
    echo "✅ Frontend: Running (http://localhost:5173)"
else
    echo "❌ Frontend: Not responding (may still be starting)"
fi

echo ""
echo "========================================"
echo "Access Points"
echo "========================================"
echo "🌐 Frontend Dashboard: http://localhost:5173"
echo "🔌 Backend API: http://localhost:8080"
echo "🤖 ML Service: http://localhost:5000"
echo ""
echo "📝 View logs:"
echo "   tail -f $LOG_DIR/ml-service.log"
echo "   tail -f $LOG_DIR/backend.log"
echo "   tail -f $LOG_DIR/frontend.log"
echo ""
echo "🛑 Stop all services:"
echo "   ./stop_all.sh"
echo ""
echo "========================================"
