"""
PyTorch-based vessel arrival time prediction model
Converted from TensorFlow/Keras implementation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional, Dict
import os


class AttentionLayer(nn.Module):
    """
    Custom attention layer for focusing on important timesteps

    This layer learns to assign importance weights to different timesteps
    in the sequence, allowing the model to focus on critical voyage segments.
    """

    def __init__(self, hidden_size: int):
        """
        Initialize attention layer

        Args:
            hidden_size: Dimension of the input features
        """
        super(AttentionLayer, self).__init__()

        # Learnable parameters
        self.W = nn.Parameter(torch.randn(hidden_size, 1))
        self.b = nn.Parameter(torch.zeros(1))

        # Initialize weights using Xavier/Glorot uniform
        nn.init.xavier_uniform_(self.W)

    def forward(self, inputs):
        """
        Forward pass of attention layer

        Args:
            inputs: Tensor of shape [batch, seq_len, hidden_size]

        Returns:
            Weighted input of shape [batch, seq_len, hidden_size]
        """
        # Calculate attention scores: [batch, seq_len, hidden_size] @ [hidden_size, 1] = [batch, seq_len, 1]
        e = torch.tanh(torch.matmul(inputs, self.W) + self.b)

        # Apply softmax to get attention weights: [batch, seq_len, 1]
        attention_weights = torch.softmax(e, dim=1)

        # Apply attention weights element-wise
        weighted_input = inputs * attention_weights

        return weighted_input


class VesselArrivalLSTM(nn.Module):
    """
    LSTM-based arrival time prediction model

    Architecture:
        Input → CNN → Attention → Dropout → Bidirectional LSTM
        → Concatenate with static features → Dense layers → Output

    Input:
        - Sequence data: [batch, 48 timesteps, 8 features]
        - Static features: [batch, 10 features]

    Output:
        - Arrival time in minutes: [batch, 1]
    """

    def __init__(self, sequence_length: int = 48,
                 sequence_features: int = 8,
                 static_features: int = 10):
        """
        Initialize the model architecture

        Args:
            sequence_length: Number of timesteps in sequence (default: 48)
            sequence_features: Number of features per timestep (default: 8)
            static_features: Number of static features (default: 10)
        """
        super(VesselArrivalLSTM, self).__init__()

        self.sequence_length = sequence_length
        self.sequence_features = sequence_features
        self.static_features = static_features

        # CNN FEATURE EXTRACTION
        # ----------------------
        # PyTorch Conv1d expects input shape: [batch, channels, length]
        # We'll need to permute from [batch, length, channels]

        # First convolutional block
        self.conv1 = nn.Conv1d(
            in_channels=sequence_features,
            out_channels=64,
            kernel_size=3,
            padding=1  # 'same' padding
        )
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        # After maxpool: [batch, 64, 24]

        # Second convolutional block
        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1  # 'same' padding
        )
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        # After maxpool: [batch, 128, 12]

        # ATTENTION MECHANISM
        # -------------------
        self.attention = AttentionLayer(hidden_size=128)

        # DROPOUT REGULARIZATION
        # ----------------------
        self.dropout1 = nn.Dropout(p=0.3)

        # BIDIRECTIONAL LSTM
        # ------------------
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if 1 > 1 else 0  # Only applies between layers
        )
        # Output: [batch, seq_len, 128] (64 forward + 64 backward)

        # FEED-FORWARD NETWORK
        # --------------------
        # Input dimension: 128 (LSTM) + 10 (static) = 138
        self.dense1 = nn.Linear(128 + static_features, 128)
        self.relu3 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.3)

        self.dense2 = nn.Linear(128, 64)
        self.relu4 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.2)

        self.dense3 = nn.Linear(64, 32)
        self.relu5 = nn.ReLU()

        # OUTPUT LAYER
        # ------------
        self.output = nn.Linear(32, 1)

    def forward(self, sequence_input, static_input):
        """
        Forward pass through the network

        Args:
            sequence_input: Tensor of shape [batch, 48, 8]
            static_input: Tensor of shape [batch, 10]

        Returns:
            Output tensor of shape [batch, 1]
        """
        # CNN expects [batch, channels, length], so permute from [batch, length, channels]
        x = sequence_input.permute(0, 2, 1)  # [batch, 8, 48]

        # First conv block
        x = self.conv1(x)           # [batch, 64, 48]
        x = self.relu1(x)
        x = self.maxpool1(x)        # [batch, 64, 24]

        # Second conv block
        x = self.conv2(x)           # [batch, 128, 24]
        x = self.relu2(x)
        x = self.maxpool2(x)        # [batch, 128, 12]

        # Permute back to [batch, length, channels] for LSTM
        x = x.permute(0, 2, 1)      # [batch, 12, 128]

        # Attention mechanism
        x = self.attention(x)       # [batch, 12, 128]

        # Dropout
        x = self.dropout1(x)

        # Bidirectional LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: [batch, 12, 128]

        # Take the last timestep output
        lstm_out = lstm_out[:, -1, :]  # [batch, 128]

        # Concatenate with static features
        combined = torch.cat([lstm_out, static_input], dim=1)  # [batch, 138]

        # Feed-forward network
        x = self.dense1(combined)
        x = self.relu3(x)
        x = self.dropout2(x)

        x = self.dense2(x)
        x = self.relu4(x)
        x = self.dropout3(x)

        x = self.dense3(x)
        x = self.relu5(x)

        # Output layer
        output = self.output(x)     # [batch, 1]

        return output


class VesselArrivalPredictor:
    """
    Wrapper class for VesselArrivalLSTM model
    Provides high-level interface for training, prediction, and model management
    """

    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        """
        Initialize the predictor

        Args:
            model_path: Path to pre-trained model file (.pth or .pt)
            device: Device to run model on ('cpu' or 'cuda')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = VesselArrivalLSTM().to(self.device)
        self.sequence_length = 48
        self.sequence_features = 8
        self.static_features = 10

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        elif model_path:
            print(f"[WARNING] Model path {model_path} not found. Using new model.")

    def predict(self, sequence_data: np.ndarray, static_features: np.ndarray) -> float:
        """
        Make inference prediction

        Args:
            sequence_data: Numpy array of shape [48, 8]
            static_features: Numpy array of shape [10]

        Returns:
            Predicted travel time in minutes
        """
        self.model.eval()

        with torch.no_grad():
            # Convert to tensors and add batch dimension
            seq_tensor = torch.FloatTensor(sequence_data).unsqueeze(0).to(self.device)
            static_tensor = torch.FloatTensor(static_features).unsqueeze(0).to(self.device)

            # Predict
            prediction = self.model(seq_tensor, static_tensor)

            return float(prediction[0][0].cpu().numpy())

    def predict_with_confidence(self, sequence_data: np.ndarray,
                                static_features: np.ndarray,
                                n_samples: int = 10) -> Tuple[float, float]:
        """
        Make prediction with confidence estimate using Monte Carlo Dropout

        Args:
            sequence_data: Numpy array of shape [48, 8]
            static_features: Numpy array of shape [10]
            n_samples: Number of forward passes for uncertainty estimation

        Returns:
            Tuple of (mean_prediction, confidence_score)
            - mean_prediction: Average predicted minutes
            - confidence_score: Confidence (0-1, higher is more confident)
        """
        self.model.train()  # Enable dropout

        predictions = []

        with torch.no_grad():
            # Convert to tensors and add batch dimension
            seq_tensor = torch.FloatTensor(sequence_data).unsqueeze(0).to(self.device)
            static_tensor = torch.FloatTensor(static_features).unsqueeze(0).to(self.device)

            # Multiple forward passes with dropout enabled
            for _ in range(n_samples):
                pred = self.model(seq_tensor, static_tensor)
                predictions.append(float(pred[0][0].cpu().numpy()))

        # Calculate statistics
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)

        # Convert std to confidence score (0-1)
        # Lower std = higher confidence
        confidence = 1 / (1 + std_pred / (mean_pred + 1e-6))

        return mean_pred, confidence

    def train(self, X_train: Tuple[np.ndarray, np.ndarray], y_train: np.ndarray,
              X_val: Tuple[np.ndarray, np.ndarray], y_val: np.ndarray,
              epochs: int = 100, batch_size: int = 32,
              learning_rate: float = 0.001,
              model_save_path: str = 'models/vessel_arrival_lstm.pth') -> Dict:
        """
        Train the LSTM model

        Args:
            X_train: Tuple of (sequence_train, static_train)
            y_train: Training targets (arrival times in minutes)
            X_val: Tuple of (sequence_val, static_val)
            y_val: Validation targets
            epochs: Maximum number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            model_save_path: Path to save best model

        Returns:
            Training history dict
        """
        sequence_train, static_train = X_train
        sequence_val, static_val = X_val

        # Convert to PyTorch tensors
        seq_train_tensor = torch.FloatTensor(sequence_train).to(self.device)
        static_train_tensor = torch.FloatTensor(static_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)

        seq_val_tensor = torch.FloatTensor(sequence_val).to(self.device)
        static_val_tensor = torch.FloatTensor(static_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)

        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(
            seq_train_tensor, static_train_tensor, y_train_tensor
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        # Optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Training history
        history = {
            'train_loss': [],
            'train_mae': [],
            'val_loss': [],
            'val_mae': []
        }

        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0

        print(f"[INFO] Training on {self.device}")

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_losses = []
            train_maes = []

            for seq_batch, static_batch, y_batch in train_loader:
                # Forward pass
                optimizer.zero_grad()
                predictions = self.model(seq_batch, static_batch)
                loss = criterion(predictions, y_batch)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Calculate MAE
                mae = torch.mean(torch.abs(predictions - y_batch))

                train_losses.append(loss.item())
                train_maes.append(mae.item())

            # Validation phase
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

            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Loss: {epoch_train_loss:.4f} - MAE: {epoch_train_mae:.4f} - "
                      f"Val Loss: {epoch_val_loss:.4f} - Val MAE: {epoch_val_mae:.4f}")

            # Early stopping and model saving
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                patience_counter = 0
                # Save best model
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
        Evaluate model performance

        Args:
            X_test: Tuple of (sequence_test, static_test)
            y_test: Test targets

        Returns:
            Dict of evaluation metrics
        """
        sequence_test, static_test = X_test

        self.model.eval()

        with torch.no_grad():
            # Convert to tensors
            seq_tensor = torch.FloatTensor(sequence_test).to(self.device)
            static_tensor = torch.FloatTensor(static_test).to(self.device)
            y_tensor = torch.FloatTensor(y_test).unsqueeze(1).to(self.device)

            # Predict
            predictions = self.model(seq_tensor, static_tensor)

            # Calculate metrics
            mse = nn.MSELoss()(predictions, y_tensor)
            mae = torch.mean(torch.abs(predictions - y_tensor))
            mape = torch.mean(torch.abs((y_tensor - predictions) / (y_tensor + 1e-8))) * 100

        metrics = {
            'loss': mse.item(),
            'mae': mae.item(),
            'mse': mse.item(),
            'mape': mape.item()
        }

        return metrics

    def load_model(self, path: str):
        """
        Load pre-trained model from file

        Args:
            path: Path to model file (.pth or .pt)
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        print(f"[INFO] Model loaded from {path}")

    def save_model(self, path: str):
        """
        Save trained model to file

        Args:
            path: Path to save model (.pth or .pt)
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
        }, path)
        print(f"[INFO] Model saved to {path}")

    def summary(self):
        """Print model architecture summary"""
        print(self.model)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    def get_model_info(self) -> Dict:
        """
        Get model architecture information

        Returns:
            Dict with model details
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            "model_name": "VesselArrivalLSTM",
            "framework": "PyTorch",
            "device": str(self.device),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "sequence_length": self.sequence_length,
            "sequence_features": self.sequence_features,
            "static_features": self.static_features,
        }


# Test the model
if __name__ == "__main__":
    print("[INFO] Testing PyTorch VesselArrivalLSTM model...")

    # Create model
    predictor = VesselArrivalPredictor(device='cpu')

    # Print summary
    predictor.summary()

    # Test with dummy data
    print("\n[INFO] Testing forward pass with dummy data...")
    dummy_sequence = np.random.randn(48, 8).astype(np.float32)
    dummy_static = np.random.randn(10).astype(np.float32)

    prediction = predictor.predict(dummy_sequence, dummy_static)
    print(f"Prediction: {prediction:.2f} minutes")

    # Test confidence prediction
    mean_pred, confidence = predictor.predict_with_confidence(dummy_sequence, dummy_static, n_samples=10)
    print(f"Mean prediction: {mean_pred:.2f} minutes, Confidence: {confidence:.4f}")

    # Model info
    info = predictor.get_model_info()
    print("\n[INFO] Model Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    print("\n[SUCCESS] PyTorch model test completed!")
