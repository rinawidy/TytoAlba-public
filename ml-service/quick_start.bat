@echo off
REM Quick Start Script for TytoAlba ML Service (Windows)
REM This script sets up and runs the ML service

echo ========================================
echo TytoAlba ML Service - Quick Start
echo ========================================

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Create .env if it doesn't exist
if not exist ".env" (
    echo Creating .env file...
    copy .env.example .env
)

REM Train models if they don't exist
if not exist "models\fuel_prediction_model.pkl" (
    echo Training fuel prediction model...
    python train_fuel_model.py
)

if not exist "models\arrival_prediction_model.pkl" (
    echo Training arrival prediction model...
    python train_arrival_model.py
)

REM Start the server
echo ========================================
echo Starting ML Service...
echo ========================================
python run_server.py
