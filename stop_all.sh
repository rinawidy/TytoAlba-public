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

    # Method 1: Kill by port using fuser (more reliable than lsof)
    if command -v fuser >/dev/null 2>&1; then
        fuser -k ${port}/tcp 2>/dev/null && echo "🛑 Stopping $service_name (Port $port)..." && sleep 1
    fi

    # Method 2: Parse PIDs from ss/netstat and kill
    local pids=$(ss -tlnp 2>/dev/null | grep ":$port" | grep -oP 'pid=\K[0-9]+' | sort -u)
    if [ -z "$pids" ]; then
        pids=$(netstat -tlnp 2>/dev/null | grep ":$port" | awk '{print $7}' | cut -d'/' -f1 | grep -E '^[0-9]+$')
    fi

    if [ -n "$pids" ]; then
        echo "🛑 Stopping $service_name (Port $port, PIDs: $pids)..."
        for pid in $pids; do
            kill -9 $pid 2>/dev/null
        done
        sleep 1
    fi

    # Method 3: Kill by process patterns
    if [ "$port" = "8080" ]; then
        # Kill Go backend processes
        pkill -9 -f "cmd/api/main.go" 2>/dev/null
        pkill -9 -f "go run.*main.go" 2>/dev/null
        # Kill compiled binary
        pkill -9 -f "go-build.*main" 2>/dev/null
    elif [ "$port" = "5000" ]; then
        # Kill Python ML service
        pkill -9 -f "ml_service.py" 2>/dev/null
    elif [ "$port" = "5173" ]; then
        # Kill Vite dev server
        pkill -9 -f "vite" 2>/dev/null
        pkill -9 -f "node.*vite" 2>/dev/null
    fi

    # Verify port is free
    sleep 1
    if ss -tln 2>/dev/null | grep -q ":$port " || netstat -tln 2>/dev/null | grep -q ":$port "; then
        echo "   ⚠️  Warning: Port $port still in use"
    else
        echo "   ✓ $service_name stopped, port $port freed"
    fi
}

# Stop services by port
kill_port 5000 "ML Service"
kill_port 8080 "Backend"
kill_port 5173 "Frontend"

echo ""
echo "✅ All services stopped"
echo "========================================"
