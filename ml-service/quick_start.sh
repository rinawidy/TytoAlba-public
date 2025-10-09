#!/bin/bash

# Quick Start Script for TytoAlba ML Service
# This script sets up and runs the ML service

echo "========================================"
echo "TytoAlba ML Service - Quick Start"
echo "========================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cp .env.example .env
fi

# Train models if they don't exist
if [ ! -f "models/fuel_prediction_model.pkl" ]; then
    echo "Training fuel prediction model..."
    python train_fuel_model.py
fi

if [ ! -f "models/arrival_prediction_model.pkl" ]; then
    echo "Training arrival prediction model..."
    python train_arrival_model.py
fi

# Start the server
echo "========================================"
echo "Starting ML Service..."
echo "========================================"
python run_server.py
