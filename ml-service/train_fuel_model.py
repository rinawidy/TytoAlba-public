"""
Training script for Fuel Prediction Model

This script demonstrates how to train the fuel prediction model.
Replace the sample data with your actual dataset.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

from src.models.fuel_predictor import FuelPredictor


def generate_sample_data(n_samples=1000):
    """
    Generate sample data for demonstration

    In production, replace this with actual data loading from CSV/database
    """
    np.random.seed(42)

    # Generate features
    distance = np.random.uniform(10, 500, n_samples)  # km
    vehicle_weight = np.random.uniform(2000, 15000, n_samples)  # kg
    avg_speed = np.random.uniform(40, 120, n_samples)  # km/h
    vehicle_type = np.random.randint(0, 3, n_samples)  # 0, 1, 2
    terrain_type = np.random.randint(0, 3, n_samples)  # 0, 1, 2

    # Create realistic fuel consumption formula
    # Base consumption + distance factor + weight factor + speed factor + terrain/vehicle adjustments
    base_consumption = 5
    distance_factor = distance * 0.08
    weight_factor = vehicle_weight * 0.001
    speed_factor = (avg_speed / 80) * 2  # Penalty for speed != 80
    terrain_adjustment = terrain_type * 3
    vehicle_adjustment = vehicle_type * 2

    fuel_consumption = (
        base_consumption +
        distance_factor +
        weight_factor +
        speed_factor +
        terrain_adjustment +
        vehicle_adjustment +
        np.random.normal(0, 2, n_samples)  # Add noise
    )

    # Create DataFrame
    df = pd.DataFrame({
        'distance': distance,
        'vehicle_weight': vehicle_weight,
        'avg_speed': avg_speed,
        'vehicle_type': vehicle_type,
        'terrain_type': terrain_type,
        'fuel_consumption': fuel_consumption
    })

    return df


def load_real_data(file_path):
    """
    Load real data from CSV file

    Expected columns:
    - distance
    - vehicle_weight
    - avg_speed
    - vehicle_type
    - terrain_type
    - fuel_consumption (target)
    """
    df = pd.read_csv(file_path)
    return df


def train_model(data_source='sample', data_path=None):
    """
    Train the fuel prediction model

    Args:
        data_source: 'sample' or 'file'
        data_path: Path to CSV file if data_source='file'
    """
    print("=" * 50)
    print("FUEL PREDICTION MODEL TRAINING")
    print("=" * 50)

    # Load data
    if data_source == 'sample':
        print("\n📊 Generating sample data...")
        df = generate_sample_data(n_samples=1000)
    else:
        print(f"\n📊 Loading data from {data_path}...")
        df = load_real_data(data_path)

    print(f"Dataset shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())

    # Prepare features and target
    feature_columns = ['distance', 'vehicle_weight', 'avg_speed', 'vehicle_type', 'terrain_type']
    X = df[feature_columns].values
    y = df['fuel_consumption'].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\n📈 Training set: {X_train.shape[0]} samples")
    print(f"📉 Test set: {X_test.shape[0]} samples")

    # Train model
    print("\n🔧 Training Random Forest model...")
    predictor = FuelPredictor()
    predictor.train(X_train, y_train)

    # Evaluate model
    print("\n📊 Evaluating model...")
    y_pred_train = predictor.predict(X_train)
    y_pred_test = predictor.predict(X_test)

    # Calculate metrics
    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_r2 = r2_score(y_train, y_pred_train)

    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_r2 = r2_score(y_test, y_pred_test)

    print("\n" + "=" * 50)
    print("TRAINING RESULTS")
    print("=" * 50)
    print("\nTraining Set:")
    print(f"  MAE:  {train_mae:.2f} liters")
    print(f"  RMSE: {train_rmse:.2f} liters")
    print(f"  R²:   {train_r2:.4f}")

    print("\nTest Set:")
    print(f"  MAE:  {test_mae:.2f} liters")
    print(f"  RMSE: {test_rmse:.2f} liters")
    print(f"  R²:   {test_r2:.4f}")

    # Feature importance
    print("\n" + "=" * 50)
    print("FEATURE IMPORTANCE")
    print("=" * 50)
    importance = predictor.get_feature_importance()
    for feature, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature:20s}: {imp:.4f}")

    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/fuel_prediction_model.pkl'
    predictor.save_model(model_path)

    print("\n" + "=" * 50)
    print(f"✅ Model saved to {model_path}")
    print("=" * 50)

    # Test prediction
    print("\n" + "=" * 50)
    print("SAMPLE PREDICTION")
    print("=" * 50)
    sample_features = {
        'distance': 150.0,
        'vehicle_weight': 8000.0,
        'avg_speed': 70.0,
        'vehicle_type': 1,
        'terrain_type': 1
    }
    prediction = predictor.predict(sample_features)
    print(f"\nInput: {sample_features}")
    print(f"Predicted fuel consumption: {prediction:.2f} liters")
    print("=" * 50)


if __name__ == "__main__":
    # Train with sample data
    train_model(data_source='sample')

    # To train with real data, uncomment below:
    # train_model(data_source='file', data_path='data/fuel_data.csv')
