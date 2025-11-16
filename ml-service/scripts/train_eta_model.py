"""
Train ETA Prediction Model (LSTM)
Predicts estimated time of arrival based on current position, speed, and course
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

class ETAPredictor(nn.Module):
    """LSTM-based ETA prediction model"""
    def __init__(self, input_size=5, hidden_size=64, num_layers=2):
        super(ETAPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=0.2)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        # For single timestep: (batch_size, 1, input_size)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Take the last output
        last_output = lstm_out[:, -1, :]

        # Fully connected layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out

def load_data(data_path):
    """Load and preprocess training data"""
    df = pd.read_csv(data_path)

    # Features
    X = df[['latitude', 'longitude', 'speed_knots', 'course', 'distance_nm']].values

    # Labels (ETA in hours)
    y = df['eta_hours'].values

    return X, y

def train_model(X_train, y_train, X_val, y_val, epochs=100):
    """Train the ETA prediction model"""
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Convert to tensors and add sequence dimension
    X_train_tensor = torch.FloatTensor(X_train_scaled).unsqueeze(1)  # Add seq_len dimension
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_tensor = torch.FloatTensor(X_val_scaled).unsqueeze(1)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)

    # Initialize model
    model = ETAPredictor(input_size=5, hidden_size=64, num_layers=2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    print("\n🚂 Training ETA Prediction Model...")
    print("="*60)

    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⚠️  Early stopping at epoch {epoch+1}")
                model.load_state_dict(best_model_state)
                break

    print("="*60)
    print(f"✅ Training complete! Best validation loss: {best_val_loss:.4f}")

    return model, scaler

def evaluate_model(model, scaler, X_test, y_test):
    """Evaluate model performance"""
    X_test_scaled = scaler.transform(X_test)
    X_test_tensor = torch.FloatTensor(X_test_scaled).unsqueeze(1)

    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor).numpy()

    # Calculate metrics
    mse = np.mean((predictions.flatten() - y_test) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions.flatten() - y_test))

    print("\n📊 Model Performance Metrics:")
    print("="*60)
    print(f"  • Mean Squared Error (MSE): {mse:.4f}")
    print(f"  • Root Mean Squared Error (RMSE): {rmse:.4f} hours")
    print(f"  • Mean Absolute Error (MAE): {mae:.4f} hours")
    print(f"  • Average prediction error: ±{mae*60:.1f} minutes")
    print("="*60)

    return predictions

def save_model(model, scaler, output_dir):
    """Save trained model and scaler"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_dir / 'eta_model.pth'
    torch.save(model.state_dict(), model_path)

    # Save scaler
    scaler_path = output_dir / 'eta_scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"\n💾 Model saved:")
    print(f"  • {model_path}")
    print(f"  • {scaler_path}")

def main():
    """Main training function"""
    print("="*60)
    print("🎯 ETA PREDICTION MODEL TRAINING")
    print("="*60)

    # Load data
    data_path = Path(__file__).parent.parent / 'data' / 'processed' / 'eta_training_data.csv'
    print(f"\n📂 Loading data from: {data_path}")
    X, y = load_data(data_path)
    print(f"  ✓ Loaded {len(X)} training samples")
    print(f"  ✓ Features shape: {X.shape}")
    print(f"  ✓ Labels shape: {y.shape}")

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"\n📊 Data split:")
    print(f"  • Training: {len(X_train)} samples")
    print(f"  • Validation: {len(X_val)} samples")
    print(f"  • Test: {len(X_test)} samples")

    # Train model
    model, scaler = train_model(X_train, y_train, X_val, y_val, epochs=100)

    # Evaluate model
    predictions = evaluate_model(model, scaler, X_test, y_test)

    # Save model
    output_dir = Path(__file__).parent.parent / 'models'
    save_model(model, scaler, output_dir)

    print("\n" + "="*60)
    print("✅ ETA MODEL TRAINING COMPLETE")
    print("="*60)
    print("\n🎯 Next: Train fuel consumption model")
    print("   Run: python scripts/train_fuel_model.py")

if __name__ == '__main__':
    main()
