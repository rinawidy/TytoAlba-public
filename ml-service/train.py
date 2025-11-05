"""
TytoAlba ML Service - Training Script
LSTM-based Vessel Arrival Time Prediction for Bulk Carrier Ships

This script handles model training with GPU/CPU auto-detection.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# GPU/CPU Detection and Configuration
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

from src.models.lstm_arrival_predictor import VesselArrivalLSTM
from src.preprocessing.data_pipeline import VoyageDataPreprocessor

# Configure TensorFlow to use GPU if available, otherwise CPU
def configure_device():
    """
    Auto-detect and configure GPU/CPU for TensorFlow
    """
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        try:
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"✓ GPU detected: {len(gpus)} Physical GPU(s), {len(logical_gpus)} Logical GPU(s)")
            print(f"  GPU Name: {tf.test.gpu_device_name()}")
            return 'GPU'
        except RuntimeError as e:
            print(f"⚠ GPU configuration error: {e}")
            print("  Falling back to CPU")
            return 'CPU'
    else:
        print("ℹ No GPU detected. Using CPU for training.")
        # Optimize CPU performance
        tf.config.threading.set_intra_op_parallelism_threads(0)  # Auto-tune
        tf.config.threading.set_inter_op_parallelism_threads(0)  # Auto-tune
        return 'CPU'


def load_training_data(data_path, ship_type_filter='bulk_carrier'):
    """
    Load and preprocess training data

    Args:
        data_path: Path to CSV file with historical voyage data
        ship_type_filter: Ship type to filter ('bulk_carrier' only)

    Returns:
        DataFrame with filtered training data
    """
    print(f"\n📂 Loading training data from: {data_path}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found: {data_path}")

    # Load data
    df = pd.read_csv(data_path)
    print(f"  Total records loaded: {len(df)}")

    # Filter by ship type (bulk carriers only)
    if 'ship_type' in df.columns:
        df = df[df['ship_type'].str.lower() == ship_type_filter.lower()]
        print(f"  Filtered to bulk carriers: {len(df)} records")
    else:
        print(f"  ⚠ Warning: 'ship_type' column not found. Processing all ships.")

    # Validate required columns
    required_cols = ['vessel_mmsi', 'voyage_id', 'ais_data', 'weather_data',
                     'destination_lat', 'destination_lon', 'actual_arrival_time']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print(f"  ✓ Data validation passed")
    return df


def generate_synthetic_data(n_samples=1000):
    """
    Generate synthetic training data for testing

    Args:
        n_samples: Number of synthetic voyage samples to generate

    Returns:
        DataFrame with synthetic data
    """
    print(f"\n🔬 Generating {n_samples} synthetic voyage samples...")

    np.random.seed(42)

    data = []

    for i in range(n_samples):
        # Random start position (Singapore region)
        start_lat = np.random.uniform(-10, 10)
        start_lon = np.random.uniform(95, 125)

        # Random destination (within 1000km)
        dest_lat = start_lat + np.random.uniform(-10, 10)
        dest_lon = start_lon + np.random.uniform(-10, 10)

        # Generate 48 timesteps of AIS data (24 hours)
        ais_sequence = []
        current_lat, current_lon = start_lat, start_lon

        for t in range(48):
            # Interpolate position towards destination
            progress = (t + 1) / 48
            current_lat = start_lat + (dest_lat - start_lat) * progress + np.random.normal(0, 0.01)
            current_lon = start_lon + (dest_lon - start_lon) * progress + np.random.normal(0, 0.01)

            ais_sequence.append({
                'timestamp': f'2024-10-{20 + t // 24}T{t % 24:02d}:00:00Z',
                'latitude': current_lat,
                'longitude': current_lon,
                'speed': np.random.uniform(8, 15),  # knots
                'course': np.random.uniform(0, 360)
            })

        # Generate weather data
        weather_sequence = []
        for t in range(48):
            weather_sequence.append({
                'timestamp': ais_sequence[t]['timestamp'],
                'wind_speed': np.random.uniform(2, 12),  # m/s
                'wave_height': np.random.uniform(0.5, 3.0),  # meters
                'current_speed': np.random.uniform(0.1, 1.0),  # m/s
                'temperature': np.random.uniform(25, 32)  # celsius
            })

        # Calculate actual arrival time (in minutes)
        distance = np.sqrt((dest_lat - start_lat)**2 + (dest_lon - start_lon)**2) * 111  # km
        avg_speed = np.mean([p['speed'] for p in ais_sequence])
        actual_time = (distance / (avg_speed * 1.852)) * 60  # minutes

        data.append({
            'vessel_mmsi': f'56301{1000 + i}',
            'voyage_id': f'V{i+1:04d}',
            'ship_type': 'bulk_carrier',
            'ais_data': str(ais_sequence),
            'weather_data': str(weather_sequence),
            'destination_lat': dest_lat,
            'destination_lon': dest_lon,
            'actual_arrival_time': actual_time
        })

    df = pd.DataFrame(data)
    print(f"  ✓ Generated {len(df)} synthetic voyages")
    return df


def prepare_training_data(df, preprocessor):
    """
    Prepare sequences and labels for training

    Args:
        df: DataFrame with voyage data
        preprocessor: VoyageDataPreprocessor instance

    Returns:
        X_seq, X_static, y: Training data arrays
    """
    print(f"\n⚙️  Preprocessing {len(df)} voyages...")

    X_seq_list = []
    X_static_list = []
    y_list = []

    skipped = 0

    for idx, row in df.iterrows():
        try:
            # Parse AIS and weather data
            import ast
            ais_data = ast.literal_eval(row['ais_data']) if isinstance(row['ais_data'], str) else row['ais_data']
            weather_data = ast.literal_eval(row['weather_data']) if isinstance(row['weather_data'], str) else row['weather_data']

            # Create sequences
            sequence = preprocessor.create_sequence(ais_data, weather_data)

            # Extract features
            features = preprocessor.extract_features(
                ais_data,
                weather_data,
                row['destination_lat'],
                row['destination_lon']
            )

            # Normalize sequence
            sequence_norm = preprocessor.normalize_sequence(sequence)

            # Create static features
            static = preprocessor.create_static_features(features)

            X_seq_list.append(sequence_norm)
            X_static_list.append(static)
            y_list.append(row['actual_arrival_time'])

        except Exception as e:
            skipped += 1
            if skipped <= 5:  # Show first 5 errors
                print(f"  ⚠ Skipped voyage {idx}: {e}")

    X_seq = np.array(X_seq_list)
    X_static = np.array(X_static_list)
    y = np.array(y_list)

    print(f"  ✓ Processed: {len(X_seq)} voyages")
    if skipped > 0:
        print(f"  ⚠ Skipped: {skipped} voyages due to errors")

    print(f"  Sequence shape: {X_seq.shape}")
    print(f"  Static shape: {X_static.shape}")
    print(f"  Labels shape: {y.shape}")

    return X_seq, X_static, y


def train_model(data_path=None, synthetic=False, n_samples=1000,
                epochs=100, batch_size=32, validation_split=0.2):
    """
    Main training function

    Args:
        data_path: Path to training data CSV
        synthetic: Generate synthetic data if True
        n_samples: Number of synthetic samples
        epochs: Training epochs
        batch_size: Batch size
        validation_split: Validation data fraction
    """
    print("=" * 70)
    print("  TytoAlba ML Service - LSTM Model Training")
    print("  Bulk Carrier Vessel Arrival Time Prediction")
    print("=" * 70)

    # Configure device
    device = configure_device()

    # Initialize preprocessor and model
    preprocessor = VoyageDataPreprocessor()
    model = VesselArrivalLSTM()

    # Load or generate data
    if synthetic:
        df = generate_synthetic_data(n_samples)
    elif data_path:
        df = load_training_data(data_path, ship_type_filter='bulk_carrier')
    else:
        raise ValueError("Must provide either data_path or synthetic=True")

    # Prepare training data
    X_seq, X_static, y = prepare_training_data(df, preprocessor)

    # Split train/validation
    split_idx = int(len(X_seq) * (1 - validation_split))

    X_seq_train, X_seq_val = X_seq[:split_idx], X_seq[split_idx:]
    X_static_train, X_static_val = X_static[:split_idx], X_static[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"\n📊 Dataset Split:")
    print(f"  Training samples: {len(X_seq_train)}")
    print(f"  Validation samples: {len(X_seq_val)}")

    # Setup callbacks
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = model_dir / f'vessel_arrival_lstm_{timestamp}.h5'

    callbacks = [
        ModelCheckpoint(
            str(model_path),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            verbose=1,
            min_lr=1e-6
        ),
        TensorBoard(
            log_dir=f'logs/{timestamp}',
            histogram_freq=1
        )
    ]

    # Train model
    print(f"\n🚀 Starting training on {device}...")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")

    history = model.model.fit(
        x=[X_seq_train, X_static_train],
        y=y_train,
        validation_data=([X_seq_val, X_static_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Save final model
    final_model_path = model_dir / 'vessel_arrival_lstm.h5'
    model.save_model(str(final_model_path))

    print(f"\n✓ Training completed!")
    print(f"  Best model: {model_path}")
    print(f"  Final model: {final_model_path}")

    # Print final metrics
    final_val_loss = min(history.history['val_loss'])
    final_val_mae = min(history.history['val_mae'])

    print(f"\n📈 Final Metrics:")
    print(f"  Validation Loss: {final_val_loss:.4f}")
    print(f"  Validation MAE: {final_val_mae:.2f} minutes")
    print(f"  Device: {device}")

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train LSTM vessel arrival prediction model')

    parser.add_argument('--data', type=str, help='Path to training data CSV')
    parser.add_argument('--synthetic', action='store_true', help='Generate synthetic training data')
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of synthetic samples')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--validation_split', type=float, default=0.2, help='Validation split ratio')

    args = parser.parse_args()

    try:
        train_model(
            data_path=args.data,
            synthetic=args.synthetic,
            n_samples=args.n_samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_split=args.validation_split
        )
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
