"""
TytoAlba ML Service - Model Evaluation
Evaluate trained LSTM model performance on test data

Metrics:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- R² Score
- Prediction distribution analysis
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.lstm_arrival_predictor import VesselArrivalLSTM
from src.preprocessing.data_pipeline import VoyageDataPreprocessor


# ============================================================================
# GPU/CPU Configuration
# ============================================================================

def configure_device():
    """Auto-detect and configure GPU/CPU for TensorFlow"""
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU detected: {len(gpus)} GPU(s)")
            return 'GPU'
        except RuntimeError as e:
            print(f"⚠ GPU error: {e}. Using CPU.")
            return 'CPU'
    else:
        print("ℹ Using CPU for evaluation")
        return 'CPU'


# ============================================================================
# Evaluation Functions
# ============================================================================

def load_test_data(data_path, ship_type_filter='bulk_carrier'):
    """
    Load test/validation data

    Args:
        data_path: Path to test data CSV
        ship_type_filter: Ship type to filter

    Returns:
        DataFrame with test data
    """
    print(f"\n📂 Loading test data from: {data_path}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Test data not found: {data_path}")

    df = pd.read_csv(data_path)
    print(f"  Total records: {len(df)}")

    # Filter by ship type
    if 'ship_type' in df.columns:
        df = df[df['ship_type'].str.lower() == ship_type_filter.lower()]
        print(f"  Filtered to bulk carriers: {len(df)}")

    print(f"  ✓ Data loaded")
    return df


def prepare_test_data(df, preprocessor):
    """
    Prepare test sequences and labels

    Args:
        df: DataFrame with voyage data
        preprocessor: VoyageDataPreprocessor instance

    Returns:
        X_seq, X_static, y, metadata: Test data and voyage metadata
    """
    print(f"\n⚙️  Preprocessing {len(df)} test voyages...")

    X_seq_list = []
    X_static_list = []
    y_list = []
    metadata_list = []

    skipped = 0

    for idx, row in df.iterrows():
        try:
            # Parse data
            import ast
            ais_data = ast.literal_eval(row['ais_data']) if isinstance(row['ais_data'], str) else row['ais_data']
            weather_data = ast.literal_eval(row['weather_data']) if isinstance(row['weather_data'], str) else row['weather_data']

            # Create sequences
            sequence = preprocessor.create_sequence(ais_data, weather_data)
            features = preprocessor.extract_features(
                ais_data, weather_data,
                row['destination_lat'], row['destination_lon']
            )

            sequence_norm = preprocessor.normalize_sequence(sequence)
            static = preprocessor.create_static_features(features)

            X_seq_list.append(sequence_norm)
            X_static_list.append(static)
            y_list.append(row['actual_arrival_time'])

            # Store metadata
            metadata_list.append({
                'vessel_mmsi': row['vessel_mmsi'],
                'voyage_id': row['voyage_id'],
                'distance_km': features['distance_remaining'],
                'avg_speed': features['avg_speed'],
                'weather_severity': features['weather_severity']
            })

        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"  ⚠ Skipped voyage {idx}: {e}")

    X_seq = np.array(X_seq_list)
    X_static = np.array(X_static_list)
    y = np.array(y_list)

    print(f"  ✓ Processed: {len(X_seq)} voyages")
    if skipped > 0:
        print(f"  ⚠ Skipped: {skipped} voyages")

    return X_seq, X_static, y, metadata_list


def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics

    Args:
        y_true: Actual arrival times
        y_pred: Predicted arrival times

    Returns:
        Dictionary of metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)

    # Percentage within X minutes
    within_15min = np.mean(np.abs(y_true - y_pred) <= 15) * 100
    within_30min = np.mean(np.abs(y_true - y_pred) <= 30) * 100
    within_60min = np.mean(np.abs(y_true - y_pred) <= 60) * 100

    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'within_15min': within_15min,
        'within_30min': within_30min,
        'within_60min': within_60min
    }


def plot_results(y_true, y_pred, metrics, save_dir='evaluation_results'):
    """
    Create evaluation plots

    Args:
        y_true: Actual arrival times
        y_pred: Predicted arrival times
        metrics: Calculated metrics dictionary
        save_dir: Directory to save plots
    """
    Path(save_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)

    # 1. Prediction vs Actual scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.5, s=30)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
             'r--', lw=2, label='Perfect Prediction')

    plt.xlabel('Actual Arrival Time (minutes)', fontsize=12)
    plt.ylabel('Predicted Arrival Time (minutes)', fontsize=12)
    plt.title(f'Prediction vs Actual\nMAE: {metrics["mae"]:.2f} min, R²: {metrics["r2"]:.3f}',
              fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/prediction_vs_actual_{timestamp}.png', dpi=300)
    print(f"  ✓ Saved: prediction_vs_actual_{timestamp}.png")
    plt.close()

    # 2. Error distribution
    errors = y_pred - y_true

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero Error')
    plt.xlabel('Prediction Error (minutes)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Error Distribution\nMean: {np.mean(errors):.2f} min, Std: {np.std(errors):.2f} min',
              fontsize=13)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    abs_errors = np.abs(errors)
    plt.hist(abs_errors, bins=50, edgecolor='black', alpha=0.7, color='orange')
    plt.axvline(x=metrics['mae'], color='r', linestyle='--', lw=2,
                label=f'MAE: {metrics["mae"]:.2f} min')
    plt.xlabel('Absolute Error (minutes)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Absolute Error Distribution', fontsize=13)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/error_distribution_{timestamp}.png', dpi=300)
    print(f"  ✓ Saved: error_distribution_{timestamp}.png")
    plt.close()

    # 3. Accuracy within time windows
    plt.figure(figsize=(10, 6))
    time_windows = [15, 30, 45, 60, 90, 120]
    accuracies = [np.mean(np.abs(errors) <= t) * 100 for t in time_windows]

    plt.bar(range(len(time_windows)), accuracies, color='steelblue', alpha=0.8)
    plt.xticks(range(len(time_windows)), [f'±{t} min' for t in time_windows])
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xlabel('Time Window', fontsize=12)
    plt.title('Prediction Accuracy by Time Window', fontsize=14)
    plt.ylim(0, 100)

    for i, (t, acc) in enumerate(zip(time_windows, accuracies)):
        plt.text(i, acc + 2, f'{acc:.1f}%', ha='center', fontsize=10)

    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/accuracy_by_window_{timestamp}.png', dpi=300)
    print(f"  ✓ Saved: accuracy_by_window_{timestamp}.png")
    plt.close()


def generate_report(metrics, y_true, y_pred, metadata, save_dir='evaluation_results'):
    """
    Generate text evaluation report

    Args:
        metrics: Metrics dictionary
        y_true: Actual arrival times
        y_pred: Predicted arrival times
        metadata: Voyage metadata
        save_dir: Directory to save report
    """
    Path(save_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = f'{save_dir}/evaluation_report_{timestamp}.txt'

    errors = y_pred - y_true

    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("  TytoAlba LSTM Model Evaluation Report\n")
        f.write("  Bulk Carrier Vessel Arrival Time Prediction\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Test Samples: {len(y_true)}\n\n")

        f.write("-" * 70 + "\n")
        f.write("Performance Metrics\n")
        f.write("-" * 70 + "\n\n")

        f.write(f"Mean Absolute Error (MAE):       {metrics['mae']:.2f} minutes\n")
        f.write(f"Root Mean Squared Error (RMSE):  {metrics['rmse']:.2f} minutes\n")
        f.write(f"Mean Absolute % Error (MAPE):    {metrics['mape']:.2f}%\n")
        f.write(f"R² Score:                        {metrics['r2']:.4f}\n\n")

        f.write("-" * 70 + "\n")
        f.write("Accuracy Within Time Windows\n")
        f.write("-" * 70 + "\n\n")

        f.write(f"Within ±15 minutes:  {metrics['within_15min']:.1f}%\n")
        f.write(f"Within ±30 minutes:  {metrics['within_30min']:.1f}%\n")
        f.write(f"Within ±60 minutes:  {metrics['within_60min']:.1f}%\n\n")

        f.write("-" * 70 + "\n")
        f.write("Error Statistics\n")
        f.write("-" * 70 + "\n\n")

        f.write(f"Mean Error:          {np.mean(errors):.2f} minutes\n")
        f.write(f"Std Dev Error:       {np.std(errors):.2f} minutes\n")
        f.write(f"Min Error:           {np.min(errors):.2f} minutes\n")
        f.write(f"Max Error:           {np.max(errors):.2f} minutes\n")
        f.write(f"Median Abs Error:    {np.median(np.abs(errors)):.2f} minutes\n\n")

        f.write("-" * 70 + "\n")
        f.write("Worst Predictions (Top 10)\n")
        f.write("-" * 70 + "\n\n")

        abs_errors = np.abs(errors)
        worst_indices = np.argsort(abs_errors)[-10:][::-1]

        f.write(f"{'MMSI':<12} {'Voyage':<10} {'Actual':<8} {'Pred':<8} {'Error':<8} {'Speed':<7}\n")
        f.write("-" * 70 + "\n")

        for idx in worst_indices:
            mmsi = metadata[idx]['vessel_mmsi']
            voyage = metadata[idx]['voyage_id']
            actual = y_true[idx]
            pred = y_pred[idx]
            error = errors[idx]
            speed = metadata[idx]['avg_speed']

            f.write(f"{mmsi:<12} {voyage:<10} {actual:>7.1f} {pred:>7.1f} "
                   f"{error:>7.1f} {speed:>6.1f}kn\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("End of Report\n")
        f.write("=" * 70 + "\n")

    print(f"  ✓ Saved: evaluation_report_{timestamp}.txt")


def evaluate_model(model_path, test_data_path=None, synthetic=False, n_samples=200):
    """
    Main evaluation function

    Args:
        model_path: Path to trained model
        test_data_path: Path to test data CSV
        synthetic: Generate synthetic test data if True
        n_samples: Number of synthetic samples
    """
    print("=" * 70)
    print("  TytoAlba LSTM Model Evaluation")
    print("=" * 70)

    # Configure device
    device = configure_device()

    # Load model
    print(f"\n📦 Loading model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = VesselArrivalLSTM(model_path=model_path)
    preprocessor = VoyageDataPreprocessor()

    print(f"  ✓ Model loaded")
    print(f"  Parameters: {model.model.count_params():,}")

    # Load test data
    if synthetic:
        from train import generate_synthetic_data
        df = generate_synthetic_data(n_samples)
    elif test_data_path:
        df = load_test_data(test_data_path)
    else:
        raise ValueError("Must provide either test_data_path or synthetic=True")

    # Prepare test data
    X_seq, X_static, y_true, metadata = prepare_test_data(df, preprocessor)

    # Make predictions
    print(f"\n🔮 Making predictions on {len(X_seq)} voyages...")
    y_pred = model.model.predict([X_seq, X_static], verbose=1)
    y_pred = y_pred.flatten()

    # Calculate metrics
    print(f"\n📊 Calculating metrics...")
    metrics = calculate_metrics(y_true, y_pred)

    # Print results
    print(f"\n" + "=" * 70)
    print("  Evaluation Results")
    print("=" * 70)
    print(f"\n  MAE:   {metrics['mae']:.2f} minutes")
    print(f"  RMSE:  {metrics['rmse']:.2f} minutes")
    print(f"  MAPE:  {metrics['mape']:.2f}%")
    print(f"  R²:    {metrics['r2']:.4f}")
    print(f"\n  Accuracy within ±15 min: {metrics['within_15min']:.1f}%")
    print(f"  Accuracy within ±30 min: {metrics['within_30min']:.1f}%")
    print(f"  Accuracy within ±60 min: {metrics['within_60min']:.1f}%")

    # Generate plots and report
    print(f"\n📈 Generating visualizations...")
    plot_results(y_true, y_pred, metrics)

    print(f"\n📄 Generating evaluation report...")
    generate_report(metrics, y_true, y_pred, metadata)

    print(f"\n✓ Evaluation completed!")
    print(f"  Results saved to: evaluation_results/")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate LSTM vessel arrival prediction model')

    parser.add_argument('--model', type=str, default='models/vessel_arrival_lstm.h5',
                       help='Path to trained model')
    parser.add_argument('--test_data', type=str, help='Path to test data CSV')
    parser.add_argument('--synthetic', action='store_true', help='Generate synthetic test data')
    parser.add_argument('--n_samples', type=int, default=200, help='Number of synthetic samples')

    args = parser.parse_args()

    try:
        evaluate_model(
            model_path=args.model,
            test_data_path=args.test_data,
            synthetic=args.synthetic,
            n_samples=args.n_samples
        )
    except KeyboardInterrupt:
        print("\n\n⚠ Evaluation interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
