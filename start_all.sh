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

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        echo "⚠️  Port $port is already in use"
        return 1
    fi
    return 0
}

# Start ML Service
echo "🐍 Starting ML Service (Port 5000)..."
if check_port 5000; then
    cd "$PROJECT_ROOT/ml-service"
    source venv/bin/activate
    nohup python api/ml_service.py > "$LOG_DIR/ml-service.log" 2>&1 &
    ML_PID=$!
    echo "   ✓ ML Service started (PID: $ML_PID)"
    echo "   📄 Log: $LOG_DIR/ml-service.log"
else
    echo "   ✗ ML Service not started (port conflict)"
fi
echo ""

# Start Backend
echo "🔧 Starting Backend (Port 8080)..."
if check_port 8080; then
    cd "$PROJECT_ROOT/backend"
    nohup go run cmd/api/main.go > "$LOG_DIR/backend.log" 2>&1 &
    BACKEND_PID=$!
    echo "   ✓ Backend started (PID: $BACKEND_PID)"
    echo "   📄 Log: $LOG_DIR/backend.log"
else
    echo "   ✗ Backend not started (port conflict)"
fi
echo ""

# Start Frontend
echo "🎨 Starting Frontend (Port 5173)..."
if check_port 5173; then
    cd "$PROJECT_ROOT/frontend"
    nohup npm run dev > "$LOG_DIR/frontend.log" 2>&1 &
    FRONTEND_PID=$!
    echo "   ✓ Frontend started (PID: $FRONTEND_PID)"
    echo "   📄 Log: $LOG_DIR/frontend.log"
else
    echo "   ✗ Frontend not started (port conflict)"
fi
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
