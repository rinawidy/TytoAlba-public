"""
Training script for Vessel Arrival LSTM model

This script should be run separately from the inference service.
It trains the model on historical voyage data and saves it for deployment.

Usage:
    python train_lstm_model.py --data data/historical_voyages.csv --epochs 100
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models.lstm_arrival_predictor import VesselArrivalLSTM
from preprocessing.data_pipeline import VoyageDataPreprocessor


def load_training_data(data_path: str) -> pd.DataFrame:
    """
    Load historical voyage data from CSV

    Expected columns:
        - vessel_mmsi: Vessel identifier
        - voyage_id: Unique voyage identifier
        - ais_data: JSON string of AIS position history
        - weather_data: JSON string of weather data
        - destination_lat: Destination latitude
        - destination_lon: Destination longitude
        - actual_arrival_time: Actual arrival time (minutes from start)

    Args:
        data_path: Path to CSV file

    Returns:
        DataFrame with voyage data
    """
    print(f"[INFO] Loading training data from {data_path}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)

    print(f"[INFO] Loaded {len(df)} voyage records")
    print(f"[INFO] Columns: {df.columns.tolist()}")

    return df


def prepare_training_data(df: pd.DataFrame, preprocessor: VoyageDataPreprocessor):
    """
    Convert raw voyage data to training format

    Args:
        df: DataFrame with voyage data
        preprocessor: Data preprocessor instance

    Returns:
        Tuple of (X_seq, X_static, y)
    """
    print("[INFO] Preparing training data...")

    X_sequences = []
    X_statics = []
    y_targets = []

    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"[INFO] Processing record {idx}/{len(df)}")

        try:
            # Extract voyage data
            # NOTE: This is a placeholder. In practice, you would:
            # 1. Parse AIS data from JSON/database
            # 2. Parse weather data from JSON/API
            # 3. Preprocess using the pipeline

            # For now, assuming we have methods to extract this data
            # You would implement these based on your data format

            # Example (pseudo-code):
            # ais_data = json.loads(row['ais_data'])
            # weather_data = json.loads(row['weather_data'])
            #
            # sequence, static = preprocessor.create_sequence(ais_data, weather_data)
            #
            # X_sequences.append(sequence)
            # X_statics.append(static)
            # y_targets.append(row['actual_arrival_time'])

            pass  # Replace with actual implementation

        except Exception as e:
            print(f"[WARNING] Failed to process record {idx}: {e}")
            continue

    print(f"[INFO] Prepared {len(X_sequences)} training samples")

    return (
        np.array(X_sequences),
        np.array(X_statics),
        np.array(y_targets)
    )


def generate_synthetic_data(n_samples: int = 1000):
    """
    Generate synthetic training data for testing

    This is a placeholder for demonstration. Replace with real data.

    Args:
        n_samples: Number of samples to generate

    Returns:
        Tuple of (X_seq, X_static, y)
    """
    print(f"[INFO] Generating {n_samples} synthetic samples...")

    np.random.seed(42)

    # Generate synthetic sequences [n_samples, 48, 8]
    X_seq = np.random.randn(n_samples, 48, 8)

    # Generate synthetic static features [n_samples, 10]
    X_static = np.random.rand(n_samples, 10)

    # Generate synthetic targets (arrival times in minutes: 100-1000)
    y = np.random.uniform(100, 1000, size=n_samples)

    print("[INFO] Synthetic data generated")

    return X_seq, X_static, y


def plot_training_history(history: dict, save_path: str = 'training_history.png'):
    """
    Plot training and validation metrics

    Args:
        history: Training history dict
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss
    axes[0, 0].plot(history['loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # MAE
    axes[0, 1].plot(history['mae'], label='Train MAE')
    axes[0, 1].plot(history['val_mae'], label='Val MAE')
    axes[0, 1].set_title('Mean Absolute Error')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE (minutes)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # MSE
    axes[1, 0].plot(history['mse'], label='Train MSE')
    axes[1, 0].plot(history['val_mse'], label='Val MSE')
    axes[1, 0].set_title('Mean Squared Error')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MSE')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # MAPE
    axes[1, 1].plot(history['mape'], label='Train MAPE')
    axes[1, 1].plot(history['val_mape'], label='Val MAPE')
    axes[1, 1].set_title('Mean Absolute Percentage Error')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MAPE (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[INFO] Training history plot saved to {save_path}")


def main():
    """Main training function"""

    parser = argparse.ArgumentParser(description='Train Vessel Arrival LSTM model')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to training data CSV')
    parser.add_argument('--synthetic', action='store_true',
                       help='Use synthetic data for testing')
    parser.add_argument('--n_samples', type=int, default=1000,
                       help='Number of synthetic samples (if --synthetic)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio')
    parser.add_argument('--output', type=str, default='models/vessel_arrival_lstm.h5',
                       help='Output model path')

    args = parser.parse_args()

    print("=" * 70)
    print("VESSEL ARRIVAL TIME PREDICTION - MODEL TRAINING")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Initialize preprocessor
    preprocessor = VoyageDataPreprocessor()

    # Load or generate data
    if args.synthetic:
        print("[INFO] Using synthetic data for testing")
        X_seq, X_static, y = generate_synthetic_data(args.n_samples)
    else:
        if args.data is None:
            print("[ERROR] Please provide --data path or use --synthetic")
            sys.exit(1)

        df = load_training_data(args.data)
        X_seq, X_static, y = prepare_training_data(df, preprocessor)

    print(f"\n[INFO] Data shapes:")
    print(f"  - Sequences: {X_seq.shape}")
    print(f"  - Static features: {X_static.shape}")
    print(f"  - Targets: {y.shape}")

    # Split data
    print(f"\n[INFO] Splitting data (val_split={args.val_split})")

    indices = np.arange(len(y))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=args.val_split,
        random_state=42
    )

    X_seq_train, X_seq_val = X_seq[train_idx], X_seq[val_idx]
    X_static_train, X_static_val = X_static[train_idx], X_static[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    print(f"  - Training samples: {len(train_idx)}")
    print(f"  - Validation samples: {len(val_idx)}")

    # Initialize model
    print("\n[INFO] Initializing LSTM model...")
    model = VesselArrivalLSTM()
    model.summary()

    # Train model
    print(f"\n[INFO] Training model for {args.epochs} epochs...")
    print(f"[INFO] Batch size: {args.batch_size}")

    history = model.train(
        X_train=(X_seq_train, X_static_train),
        y_train=y_train,
        X_val=(X_seq_val, X_static_val),
        y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_save_path=args.output
    )

    # Evaluate
    print("\n[INFO] Evaluating on validation set...")
    metrics = model.evaluate(
        X_test=(X_seq_val, X_static_val),
        y_test=y_val
    )

    print("\n" + "=" * 70)
    print("FINAL METRICS")
    print("=" * 70)
    for metric, value in metrics.items():
        print(f"{metric.upper():15s}: {value:.4f}")

    # Save model
    print(f"\n[INFO] Model saved to: {args.output}")

    # Plot history
    plot_path = args.output.replace('.h5', '_history.png')
    plot_training_history(history, save_path=plot_path)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nModel ready for deployment!")
    print(f"Use this model in the inference service by setting:")
    print(f"  MODEL_PATH={args.output}")


if __name__ == "__main__":
    main()
