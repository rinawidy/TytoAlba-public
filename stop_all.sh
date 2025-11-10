#!/bin/bash

# TytoAlba - Stop All Services
# This script stops all running TytoAlba services

echo "========================================"
echo "TytoAlba - Stopping All Services"
echo "========================================"
echo ""

# Function to kill process on port
kill_port() {
    local port=$1
    local service_name=$2

    local pid=$(lsof -ti :$port)
    if [ -n "$pid" ]; then
        echo "🛑 Stopping $service_name (Port $port, PID: $pid)..."
        kill -9 $pid 2>/dev/null
        echo "   ✓ $service_name stopped"
    else
        echo "ℹ️  $service_name not running on port $port"
    fi
}

# Stop services by port
kill_port 5000 "ML Service"
kill_port 8080 "Backend"
kill_port 5173 "Frontend"

echo ""
echo "✅ All services stopped"
echo "========================================"
