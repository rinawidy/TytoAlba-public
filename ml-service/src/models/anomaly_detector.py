"""
PyTorch-based anomaly detection model using LSTM Autoencoder
Detects unusual vessel behavior patterns from sequential data

LSTM Advantage over Random Forest:
- RF cannot model normal sequential behavior patterns
- LSTM learns what "normal" voyage patterns look like over time
- LSTM detects sequential anomalies (e.g., unusual maneuvering sequences)
- LSTM captures context-dependent anomalies RF would miss
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional, Dict, List
import os


class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder for anomaly detection

    Architecture:
        Encoder: Input → LSTM → Compressed representation
        Decoder: Compressed → LSTM → Reconstructed input

    Anomaly detection approach:
        - Train on normal voyage data
        - High reconstruction error = anomaly
        - LSTM learns temporal patterns of normal behavior

    Why LSTM beats Random Forest:
    - RF treats each timestep independently
    - LSTM learns sequential dependencies (e.g., normal acceleration patterns)
    - LSTM detects unusual sequences even if individual points are normal
    - Example: Sudden course change + speed drop + unusual heading
              RF: 3 independent features
              LSTM: Abnormal sequence pattern

    Input:
        - Sequence data: [batch, 48 timesteps, 12 features]
          Features: [lat, lon, speed, heading, course, rate_of_turn,
                     draft, rpm, wave_height, wind_speed, current_speed,
                     distance_to_port]

    Output:
        - Reconstructed sequence: [batch, 48, 12]
        - Reconstruction error (MSE) used for anomaly score
    """

    def __init__(self, sequence_length: int = 48,
                 sequence_features: int = 12,
                 latent_dim: int = 32):
        """
        Initialize LSTM Autoencoder

        Args:
            sequence_length: Number of timesteps
            sequence_features: Number of features per timestep
            latent_dim: Size of compressed representation
        """
        super(LSTMAutoencoder, self).__init__()

        self.sequence_length = sequence_length
        self.sequence_features = sequence_features
        self.latent_dim = latent_dim

        # ENCODER
        # -------
        # Compress temporal patterns into latent representation
        self.encoder_lstm1 = nn.LSTM(
            input_size=sequence_features,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        self.encoder_dropout1 = nn.Dropout(p=0.2)

        self.encoder_lstm2 = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        self.encoder_dropout2 = nn.Dropout(p=0.2)

        # Compress to latent space
        self.encoder_dense = nn.Linear(64, latent_dim)

        # DECODER
        # -------
        # Reconstruct from latent representation
        self.decoder_dense = nn.Linear(latent_dim, 64)

        self.decoder_lstm1 = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        self.decoder_dropout1 = nn.Dropout(p=0.2)

        self.decoder_lstm2 = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        self.decoder_dropout2 = nn.Dropout(p=0.2)

        # Output layer - reconstruct original features
        self.decoder_output = nn.Linear(128, sequence_features)

    def encode(self, x):
        """
        Encode sequence to latent representation

        Args:
            x: [batch, seq_len, features]

        Returns:
            latent: [batch, latent_dim]
        """
        # First encoder LSTM
        lstm1_out, _ = self.encoder_lstm1(x)
        lstm1_out = self.encoder_dropout1(lstm1_out)

        # Second encoder LSTM
        lstm2_out, (h_n, c_n) = self.encoder_lstm2(lstm1_out)
        lstm2_out = self.encoder_dropout2(lstm2_out)

        # Take last timestep and compress to latent
        last_output = lstm2_out[:, -1, :]  # [batch, 64]
        latent = self.encoder_dense(last_output)  # [batch, latent_dim]

        return latent

    def decode(self, latent):
        """
        Decode latent representation back to sequence

        Args:
            latent: [batch, latent_dim]

        Returns:
            reconstructed: [batch, seq_len, features]
        """
        # Expand latent to sequence
        x = self.decoder_dense(latent)  # [batch, 64]

        # Repeat across timesteps
        x = x.unsqueeze(1).repeat(1, self.sequence_length, 1)  # [batch, seq_len, 64]

        # First decoder LSTM
        lstm1_out, _ = self.decoder_lstm1(x)
        lstm1_out = self.decoder_dropout1(lstm1_out)

        # Second decoder LSTM
        lstm2_out, _ = self.decoder_lstm2(lstm1_out)
        lstm2_out = self.decoder_dropout2(lstm2_out)

        # Reconstruct features
        reconstructed = self.decoder_output(lstm2_out)  # [batch, seq_len, features]

        return reconstructed

    def forward(self, x):
        """
        Forward pass: encode then decode

        Args:
            x: [batch, seq_len, features]

        Returns:
            reconstructed: [batch, seq_len, features]
        """
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed


class VesselAnomalyDetector:
    """
    Wrapper class for LSTM-based anomaly detection
    """

    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        """
        Initialize anomaly detector

        Args:
            model_path: Path to pre-trained model (.pth)
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = LSTMAutoencoder().to(self.device)

        # Threshold for anomaly detection (set after training)
        self.threshold = None

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        elif model_path:
            print(f"[WARNING] Model path {model_path} not found. Using new model.")

    def compute_reconstruction_error(self, sequence_data: np.ndarray) -> float:
        """
        Compute reconstruction error for a sequence

        Args:
            sequence_data: [48, 12] - vessel behavior sequence

        Returns:
            Reconstruction error (MSE)
        """
        self.model.eval()

        with torch.no_grad():
            seq_tensor = torch.FloatTensor(sequence_data).unsqueeze(0).to(self.device)
            reconstructed = self.model(seq_tensor)

            # Calculate MSE
            mse = torch.mean((seq_tensor - reconstructed) ** 2).item()

        return mse

    def detect_anomaly(self, sequence_data: np.ndarray,
                      threshold_percentile: float = 95) -> Dict:
        """
        Detect if sequence is anomalous

        Args:
            sequence_data: [48, 12] - vessel behavior sequence
            threshold_percentile: Use this percentile of training errors as threshold

        Returns:
            Dict with {
                'is_anomaly': bool,
                'anomaly_score': float,
                'threshold': float,
                'severity': str  # 'normal', 'mild', 'moderate', 'severe'
            }
        """
        error = self.compute_reconstruction_error(sequence_data)

        # If no threshold set, use a default or set from training
        if self.threshold is None:
            self.threshold = 0.1  # Default threshold
            print("[WARNING] Using default threshold. Train model first for better threshold.")

        is_anomaly = error > self.threshold

        # Determine severity
        if error < self.threshold:
            severity = 'normal'
        elif error < self.threshold * 1.5:
            severity = 'mild'
        elif error < self.threshold * 2.0:
            severity = 'moderate'
        else:
            severity = 'severe'

        return {
            'is_anomaly': bool(is_anomaly),
            'anomaly_score': float(error),
            'threshold': float(self.threshold),
            'severity': severity,
            'confidence': min(1.0, error / (self.threshold + 1e-6)) if is_anomaly else 1.0
        }

    def detect_anomalies_batch(self, sequence_batch: np.ndarray) -> List[Dict]:
        """
        Detect anomalies in batch of sequences

        Args:
            sequence_batch: [N, 48, 12] - batch of sequences

        Returns:
            List of anomaly detection results
        """
        results = []
        for seq in sequence_batch:
            result = self.detect_anomaly(seq)
            results.append(result)
        return results

    def train(self, X_train: np.ndarray, X_val: np.ndarray,
              epochs: int = 100, batch_size: int = 32,
              learning_rate: float = 0.001,
              model_save_path: str = 'models/anomaly_detector_lstm.pth',
              threshold_percentile: float = 95) -> Dict:
        """
        Train the autoencoder on NORMAL vessel behavior

        Args:
            X_train: [N, 48, 12] - normal behavior sequences only
            X_val: [M, 48, 12] - validation sequences (normal)
            epochs: Max epochs
            batch_size: Batch size
            learning_rate: Learning rate
            model_save_path: Where to save best model
            threshold_percentile: Percentile of reconstruction errors to use as threshold

        Returns:
            Training history dict
        """
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)

        # Data loader
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, X_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        # Optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Training history
        history = {'train_loss': [], 'val_loss': []}

        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0

        print(f"[INFO] Training Anomaly Detector on {self.device}")

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_losses = []

            for X_batch, _ in train_loader:
                optimizer.zero_grad()
                reconstructed = self.model(X_batch)
                loss = criterion(reconstructed, X_batch)

                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_reconstructed = self.model(X_val_tensor)
                val_loss = criterion(val_reconstructed, X_val_tensor)

            # Record history
            epoch_train_loss = np.mean(train_losses)
            epoch_val_loss = val_loss.item()

            history['train_loss'].append(epoch_train_loss)
            history['val_loss'].append(epoch_val_loss)

            # Progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Loss: {epoch_train_loss:.6f} - "
                      f"Val Loss: {epoch_val_loss:.6f}")

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

        # Set anomaly threshold based on training data
        self.model.eval()
        train_errors = []
        with torch.no_grad():
            for i in range(len(X_train)):
                seq_tensor = torch.FloatTensor(X_train[i]).unsqueeze(0).to(self.device)
                reconstructed = self.model(seq_tensor)
                error = torch.mean((seq_tensor - reconstructed) ** 2).item()
                train_errors.append(error)

        self.threshold = np.percentile(train_errors, threshold_percentile)
        print(f"[INFO] Anomaly threshold set to {self.threshold:.6f} ({threshold_percentile}th percentile)")

        return history

    def evaluate(self, X_normal: np.ndarray, X_anomaly: np.ndarray) -> Dict:
        """
        Evaluate model on normal and anomalous data

        Args:
            X_normal: [N, 48, 12] - normal sequences
            X_anomaly: [M, 48, 12] - anomalous sequences

        Returns:
            Dict with precision, recall, f1_score, accuracy
        """
        # Get predictions
        normal_results = self.detect_anomalies_batch(X_normal)
        anomaly_results = self.detect_anomalies_batch(X_anomaly)

        # True negatives (correctly identified as normal)
        tn = sum(1 for r in normal_results if not r['is_anomaly'])
        # False positives (normal flagged as anomaly)
        fp = sum(1 for r in normal_results if r['is_anomaly'])
        # True positives (correctly identified anomalies)
        tp = sum(1 for r in anomaly_results if r['is_anomaly'])
        # False negatives (anomalies missed)
        fn = sum(1 for r in anomaly_results if not r['is_anomaly'])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        }

    def load_model(self, path: str):
        """Load model from file"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        print(f"[INFO] Anomaly detector loaded from {path}")

    def save_model(self, path: str):
        """Save model to file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'threshold': self.threshold
        }, path)
        print(f"[INFO] Anomaly detector saved to {path}")

    def summary(self):
        """Print model summary"""
        print(self.model)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        if self.threshold:
            print(f"Anomaly threshold: {self.threshold:.6f}")


# Test the model
if __name__ == "__main__":
    print("[INFO] Testing Anomaly Detection LSTM Autoencoder...")

    detector = VesselAnomalyDetector(device='cpu')
    detector.summary()

    # Test with dummy data
    print("\n[INFO] Testing anomaly detection...")

    # Normal sequence
    normal_sequence = np.random.randn(48, 12).astype(np.float32) * 0.1

    # Anomalous sequence (larger variance)
    anomaly_sequence = np.random.randn(48, 12).astype(np.float32) * 2.0

    detector.threshold = 0.5  # Set a test threshold

    normal_result = detector.detect_anomaly(normal_sequence)
    print(f"\nNormal sequence:")
    print(f"  Anomaly: {normal_result['is_anomaly']}")
    print(f"  Score: {normal_result['anomaly_score']:.4f}")
    print(f"  Severity: {normal_result['severity']}")

    anomaly_result = detector.detect_anomaly(anomaly_sequence)
    print(f"\nAnomalous sequence:")
    print(f"  Anomaly: {anomaly_result['is_anomaly']}")
    print(f"  Score: {anomaly_result['anomaly_score']:.4f}")
    print(f"  Severity: {anomaly_result['severity']}")

    print("\n[SUCCESS] Anomaly detection model test completed!")
