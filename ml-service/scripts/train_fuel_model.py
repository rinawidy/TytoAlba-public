"""
Train fuel consumption prediction model
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pickle

class FuelPredictor(nn.Module):
    """Neural network for fuel consumption prediction"""
    def __init__(self, input_size=4, hidden_size=32):
        super(FuelPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, 16)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

def load_data(file_path):
    """Load training data"""
    df = pd.read_csv(file_path)
    X = df[['avg_speed', 'distance_nm', 'time_hours', 'course_change']].values
    y = df['fuel_liters'].values
    return X, y

def train_model(X_train, y_train, X_val, y_val, epochs=100):
    """Train the fuel prediction model"""
    model = FuelPredictor(input_size=4, hidden_size=32)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).view(-1, 1)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val).view(-1, 1)

    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0

    print("\n🚂 Training Fuel Consumption Model...")
    print("=" * 60)

    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t)

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⚠️  Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(best_model_state)
    print("=" * 60)
    print(f"✅ Training complete! Best validation loss: {best_val_loss:.4f}")

    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    model.eval()
    X_test_t = torch.FloatTensor(X_test)

    with torch.no_grad():
        predictions = model(X_test_t).numpy().flatten()

    mse = np.mean((predictions - y_test) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - y_test))

    print("\n📊 Model Performance Metrics:")
    print("=" * 60)
    print(f"  • Mean Squared Error (MSE): {mse:.2f}")
    print(f"  • Root Mean Squared Error (RMSE): {rmse:.2f} liters")
    print(f"  • Mean Absolute Error (MAE): {mae:.2f} liters")
    print(f"  • Average prediction error: ±{mae:.1f} liters")
    print("=" * 60)

def main():
    """Main training function"""
    print("=" * 60)
    print("⛽ FUEL CONSUMPTION MODEL TRAINING")
    print("=" * 60)

    # Load data
    data_path = Path(__file__).parent.parent / 'data' / 'processed' / 'fuel_training_data.csv'
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

    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Train model
    model = train_model(X_train, y_train, X_val, y_val, epochs=100)

    # Evaluate
    evaluate_model(model, X_test, y_test)

    # Save model
    model_dir = Path(__file__).parent.parent / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), model_dir / 'fuel_model.pth')
    with open(model_dir / 'fuel_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print("\n💾 Model saved:")
    print(f"  • {model_dir / 'fuel_model.pth'}")
    print(f"  • {model_dir / 'fuel_scaler.pkl'}")

    print("\n" + "=" * 60)
    print("✅ FUEL MODEL TRAINING COMPLETE")
    print("=" * 60)
    print("\n🎯 Next: Train anomaly detection model")
    print("   Run: python scripts/train_anomaly_model.py")

if __name__ == '__main__':
    main()
