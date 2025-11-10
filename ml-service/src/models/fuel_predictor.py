"""
PyTorch-based fuel consumption prediction model using LSTM
Predicts fuel usage based on temporal patterns in speed, load, and weather

LSTM Advantage over Random Forest:
- Captures sequential dependencies in speed/load changes over time
- Models fuel consumption patterns across voyage stages
- Learns temporal correlations RF cannot capture
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional, Dict
import os


class FuelConsumptionLSTM(nn.Module):
    """
    LSTM model for predicting fuel consumption

    Architecture:
        Input → Bidirectional LSTM → Attention → Dense layers → Output

    Why LSTM beats Random Forest:
    - RF treats each speed/load measurement independently
    - LSTM captures how fuel consumption changes over voyage stages
    - LSTM learns acceleration patterns (speed changes over time)
    - LSTM models cumulative effects that RF misses

    Input:
        - Sequence data: [batch, 48 timesteps, 10 features]
          Features: [speed, rpm, load, wave_height, wind_speed, current,
                     latitude, longitude, heading, draft]
        - Static features: [batch, 8 features]
          Features: [dwt, engine_power, loa, beam, build_year, fuel_capacity,
                     distance_to_destination, cargo_weight]

    Output:
        - Fuel consumption in liters/hour: [batch, 1]
    """

    def __init__(self, sequence_length: int = 48,
                 sequence_features: int = 10,
                 static_features: int = 8):
        """
        Initialize fuel consumption model

        Args:
            sequence_length: Number of timesteps (default: 48 = 24 hours at 30min intervals)
            sequence_features: Number of temporal features
            static_features: Number of vessel specifications
        """
        super(FuelConsumptionLSTM, self).__init__()

        self.sequence_length = sequence_length
        self.sequence_features = sequence_features
        self.static_features = static_features

        # BIDIRECTIONAL LSTM FOR TEMPORAL PATTERNS
        # -----------------------------------------
        # Two layers to capture both immediate and long-term patterns
        self.lstm1 = nn.LSTM(
            input_size=sequence_features,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        # Output: [batch, 48, 256] (128 forward + 128 backward)

        self.dropout1 = nn.Dropout(p=0.3)

        # Second LSTM layer for higher-level patterns
        self.lstm2 = nn.LSTM(
            input_size=256,  # From bidirectional first layer
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        # Output: [batch, 48, 128]

        # ATTENTION MECHANISM
        # -------------------
        # Focus on high fuel consumption periods
        self.attention_W = nn.Parameter(torch.randn(128, 1))
        self.attention_b = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.attention_W)

        self.dropout2 = nn.Dropout(p=0.3)

        # FEED-FORWARD NETWORK
        # --------------------
        # Combine LSTM output (128) with static features (8) = 136
        self.dense1 = nn.Linear(128 + static_features, 96)
        self.relu1 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.2)

        self.dense2 = nn.Linear(96, 48)
        self.relu2 = nn.ReLU()
        self.dropout4 = nn.Dropout(p=0.2)

        self.dense3 = nn.Linear(48, 24)
        self.relu3 = nn.ReLU()

        # OUTPUT LAYER
        # ------------
        self.output = nn.Linear(24, 1)

    def forward(self, sequence_input, static_input):
        """
        Forward pass

        Args:
            sequence_input: [batch, 48, 10] - temporal features
            static_input: [batch, 8] - vessel specs

        Returns:
            [batch, 1] - fuel consumption in liters/hour
        """
        # First LSTM layer
        lstm1_out, _ = self.lstm1(sequence_input)  # [batch, 48, 256]
        lstm1_out = self.dropout1(lstm1_out)

        # Second LSTM layer
        lstm2_out, _ = self.lstm2(lstm1_out)  # [batch, 48, 128]

        # Attention mechanism - focus on critical periods
        attention_scores = torch.tanh(
            torch.matmul(lstm2_out, self.attention_W) + self.attention_b
        )  # [batch, 48, 1]

        attention_weights = torch.softmax(attention_scores, dim=1)

        # Weighted sum across timesteps
        context = torch.sum(lstm2_out * attention_weights, dim=1)  # [batch, 128]

        context = self.dropout2(context)

        # Combine with static features
        combined = torch.cat([context, static_input], dim=1)  # [batch, 136]

        # Dense layers
        x = self.dense1(combined)
        x = self.relu1(x)
        x = self.dropout3(x)

        x = self.dense2(x)
        x = self.relu2(x)
        x = self.dropout4(x)

        x = self.dense3(x)
        x = self.relu3(x)

        # Output fuel consumption (always positive)
        output = torch.relu(self.output(x))  # [batch, 1]

        return output


class FuelConsumptionPredictor:
    """
    Wrapper class for fuel consumption prediction
    """

    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        """
        Initialize predictor

        Args:
            model_path: Path to pre-trained model (.pth)
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = FuelConsumptionLSTM().to(self.device)

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        elif model_path:
            print(f"[WARNING] Model path {model_path} not found. Using new model.")

    def predict(self, sequence_data: np.ndarray, static_features: np.ndarray) -> float:
        """
        Predict fuel consumption

        Args:
            sequence_data: [48, 10] - temporal voyage data
            static_features: [8] - vessel specifications

        Returns:
            Fuel consumption in liters/hour
        """
        self.model.eval()

        with torch.no_grad():
            seq_tensor = torch.FloatTensor(sequence_data).unsqueeze(0).to(self.device)
            static_tensor = torch.FloatTensor(static_features).unsqueeze(0).to(self.device)

            prediction = self.model(seq_tensor, static_tensor)

            return float(prediction[0][0].cpu().numpy())

    def predict_with_confidence(self, sequence_data: np.ndarray,
                                static_features: np.ndarray,
                                n_samples: int = 10) -> Tuple[float, float]:
        """
        Predict with uncertainty estimation using Monte Carlo Dropout

        Returns:
            (mean_fuel_consumption, confidence_score)
        """
        self.model.train()  # Enable dropout

        predictions = []

        with torch.no_grad():
            seq_tensor = torch.FloatTensor(sequence_data).unsqueeze(0).to(self.device)
            static_tensor = torch.FloatTensor(static_features).unsqueeze(0).to(self.device)

            for _ in range(n_samples):
                pred = self.model(seq_tensor, static_tensor)
                predictions.append(float(pred[0][0].cpu().numpy()))

        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)

        # Confidence score (higher = more confident)
        confidence = 1 / (1 + std_pred / (mean_pred + 1e-6))

        return mean_pred, confidence

    def train(self, X_train: Tuple[np.ndarray, np.ndarray], y_train: np.ndarray,
              X_val: Tuple[np.ndarray, np.ndarray], y_val: np.ndarray,
              epochs: int = 100, batch_size: int = 32,
              learning_rate: float = 0.001,
              model_save_path: str = 'models/fuel_consumption_lstm.pth') -> Dict:
        """
        Train the fuel consumption model

        Args:
            X_train: (sequence_train, static_train)
            y_train: Fuel consumption targets (liters/hour)
            X_val: (sequence_val, static_val)
            y_val: Validation targets
            epochs: Max epochs
            batch_size: Batch size
            learning_rate: Learning rate
            model_save_path: Where to save best model

        Returns:
            Training history dict
        """
        sequence_train, static_train = X_train
        sequence_val, static_val = X_val

        # Convert to tensors
        seq_train_tensor = torch.FloatTensor(sequence_train).to(self.device)
        static_train_tensor = torch.FloatTensor(static_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)

        seq_val_tensor = torch.FloatTensor(sequence_val).to(self.device)
        static_val_tensor = torch.FloatTensor(static_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)

        # Data loader
        train_dataset = torch.utils.data.TensorDataset(
            seq_train_tensor, static_train_tensor, y_train_tensor
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        # Optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Training history
        history = {'train_loss': [], 'train_mae': [], 'val_loss': [], 'val_mae': []}

        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0

        print(f"[INFO] Training Fuel Consumption model on {self.device}")

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_losses = []
            train_maes = []

            for seq_batch, static_batch, y_batch in train_loader:
                optimizer.zero_grad()
                predictions = self.model(seq_batch, static_batch)
                loss = criterion(predictions, y_batch)

                loss.backward()
                optimizer.step()

                mae = torch.mean(torch.abs(predictions - y_batch))
                train_losses.append(loss.item())
                train_maes.append(mae.item())

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_predictions = self.model(seq_val_tensor, static_val_tensor)
                val_loss = criterion(val_predictions, y_val_tensor)
                val_mae = torch.mean(torch.abs(val_predictions - y_val_tensor))

            # Record history
            epoch_train_loss = np.mean(train_losses)
            epoch_train_mae = np.mean(train_maes)
            epoch_val_loss = val_loss.item()
            epoch_val_mae = val_mae.item()

            history['train_loss'].append(epoch_train_loss)
            history['train_mae'].append(epoch_train_mae)
            history['val_loss'].append(epoch_val_loss)
            history['val_mae'].append(epoch_val_mae)

            # Progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Loss: {epoch_train_loss:.4f} - MAE: {epoch_train_mae:.2f} L/h - "
                      f"Val Loss: {epoch_val_loss:.4f} - Val MAE: {epoch_val_mae:.2f} L/h")

            # Early stopping
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                patience_counter = 0
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                }, model_save_path)
                print(f"[INFO] Model saved to {model_save_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[INFO] Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        self.load_model(model_save_path)

        return history

    def evaluate(self, X_test: Tuple[np.ndarray, np.ndarray],
                y_test: np.ndarray) -> Dict:
        """
        Evaluate model

        Returns:
            Dict with MSE, MAE, MAPE metrics
        """
        sequence_test, static_test = X_test

        self.model.eval()

        with torch.no_grad():
            seq_tensor = torch.FloatTensor(sequence_test).to(self.device)
            static_tensor = torch.FloatTensor(static_test).to(self.device)
            y_tensor = torch.FloatTensor(y_test).unsqueeze(1).to(self.device)

            predictions = self.model(seq_tensor, static_tensor)

            mse = nn.MSELoss()(predictions, y_tensor)
            mae = torch.mean(torch.abs(predictions - y_tensor))
            mape = torch.mean(torch.abs((y_tensor - predictions) / (y_tensor + 1e-8))) * 100

        return {
            'mse': mse.item(),
            'mae': mae.item(),
            'mape': mape.item()
        }

    def load_model(self, path: str):
        """Load model from file"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        print(f"[INFO] Fuel model loaded from {path}")

    def save_model(self, path: str):
        """Save model to file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({'model_state_dict': self.model.state_dict()}, path)
        print(f"[INFO] Fuel model saved to {path}")

    def summary(self):
        """Print model summary"""
        print(self.model)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")


# Test the model
if __name__ == "__main__":
    print("[INFO] Testing Fuel Consumption LSTM model...")

    predictor = FuelConsumptionPredictor(device='cpu')
    predictor.summary()

    # Test with dummy data
    print("\n[INFO] Testing prediction...")
    dummy_sequence = np.random.randn(48, 10).astype(np.float32)
    dummy_static = np.random.randn(8).astype(np.float32)

    fuel_consumption = predictor.predict(dummy_sequence, dummy_static)
    print(f"Predicted fuel consumption: {fuel_consumption:.2f} L/h")

    mean_fuel, confidence = predictor.predict_with_confidence(dummy_sequence, dummy_static)
    print(f"Mean: {mean_fuel:.2f} L/h, Confidence: {confidence:.4f}")

    print("\n[SUCCESS] Fuel consumption model test completed!")
