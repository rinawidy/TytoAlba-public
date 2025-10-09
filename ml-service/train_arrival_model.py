"""
Training script for Arrival Time Prediction Model

This script demonstrates how to train the arrival time prediction model.
Replace the sample data with your actual dataset.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

from src.models.arrival_predictor import ArrivalPredictor


def generate_sample_data(n_samples=1000):
    """
    Generate sample data for demonstration

    In production, replace this with actual data loading from CSV/database
    """
    np.random.seed(42)

    # Generate features
    distance = np.random.uniform(10, 300, n_samples)  # km
    departure_hour = np.random.randint(0, 24, n_samples)  # 0-23
    day_of_week = np.random.randint(0, 7, n_samples)  # 0-6
    route_id = np.random.randint(0, 20, n_samples)  # 0-19
    avg_traffic_level = np.random.randint(0, 3, n_samples)  # 0, 1, 2

    # Create realistic arrival time formula
    # Base time + distance factor + traffic factor + rush hour penalty
    base_time = 10  # minutes
    distance_factor = distance * 0.8  # ~0.8 min per km
    traffic_factor = avg_traffic_level * 15  # Low: 0, Medium: 15, High: 30 min

    # Rush hour penalty (7-9 AM and 5-7 PM)
    rush_hour_penalty = np.where(
        ((departure_hour >= 7) & (departure_hour <= 9)) |
        ((departure_hour >= 17) & (departure_hour <= 19)),
        20, 0
    )

    # Weekend bonus (faster on weekends)
    weekend_bonus = np.where(day_of_week >= 5, -10, 0)

    # Route variation
    route_variation = np.random.uniform(-5, 5, n_samples)

    arrival_time = (
        base_time +
        distance_factor +
        traffic_factor +
        rush_hour_penalty +
        weekend_bonus +
        route_variation +
        np.random.normal(0, 5, n_samples)  # Add noise
    )

    # Ensure positive values
    arrival_time = np.maximum(arrival_time, 5)

    # Historical average time (similar to actual but with some variation)
    historical_avg_time = arrival_time + np.random.normal(0, 10, n_samples)

    # Create DataFrame
    df = pd.DataFrame({
        'distance': distance,
        'departure_hour': departure_hour,
        'day_of_week': day_of_week,
        'route_id': route_id,
        'avg_traffic_level': avg_traffic_level,
        'historical_avg_time': historical_avg_time,
        'arrival_time': arrival_time
    })

    return df


def load_real_data(file_path):
    """
    Load real data from CSV file

    Expected columns:
    - distance
    - departure_hour
    - day_of_week
    - route_id
    - avg_traffic_level
    - historical_avg_time
    - arrival_time (target)
    """
    df = pd.read_csv(file_path)
    return df


def train_model(data_source='sample', data_path=None):
    """
    Train the arrival time prediction model

    Args:
        data_source: 'sample' or 'file'
        data_path: Path to CSV file if data_source='file'
    """
    print("=" * 50)
    print("ARRIVAL TIME PREDICTION MODEL TRAINING")
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
    feature_columns = [
        'distance',
        'departure_hour',
        'day_of_week',
        'route_id',
        'avg_traffic_level',
        'historical_avg_time'
    ]
    X = df[feature_columns].values
    y = df['arrival_time'].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\n📈 Training set: {X_train.shape[0]} samples")
    print(f"📉 Test set: {X_test.shape[0]} samples")

    # Train model
    print("\n🔧 Training Random Forest model...")
    predictor = ArrivalPredictor()
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
    print(f"  MAE:  {train_mae:.2f} minutes")
    print(f"  RMSE: {train_rmse:.2f} minutes")
    print(f"  R²:   {train_r2:.4f}")

    print("\nTest Set:")
    print(f"  MAE:  {test_mae:.2f} minutes")
    print(f"  RMSE: {test_rmse:.2f} minutes")
    print(f"  R²:   {test_r2:.4f}")

    # Feature importance
    print("\n" + "=" * 50)
    print("FEATURE IMPORTANCE")
    print("=" * 50)
    importance = predictor.get_feature_importance()
    for feature, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature:25s}: {imp:.4f}")

    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/arrival_prediction_model.pkl'
    predictor.save_model(model_path)

    print("\n" + "=" * 50)
    print(f"✅ Model saved to {model_path}")
    print("=" * 50)

    # Test prediction
    print("\n" + "=" * 50)
    print("SAMPLE PREDICTION")
    print("=" * 50)
    sample_features = {
        'distance': 120.0,
        'departure_time': '2024-10-09T14:30:00',
        'route_id': 5,
        'avg_traffic_level': 1,
        'historical_avg_time': 95.5
    }
    result = predictor.predict_with_datetime(sample_features)
    print(f"\nInput: {sample_features}")
    print(f"Predicted travel time: {result['travel_time_minutes']:.2f} minutes")
    if 'estimated_arrival_time' in result:
        print(f"Estimated arrival: {result['estimated_arrival_time']}")
    print("=" * 50)


if __name__ == "__main__":
    # Train with sample data
    train_model(data_source='sample')

    # To train with real data, uncomment below:
    # train_model(data_source='file', data_path='data/arrival_data.csv')
