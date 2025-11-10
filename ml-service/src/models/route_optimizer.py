"""
PyTorch-based route optimization model using LSTM
Predicts optimal waypoints and trajectory for fuel-efficient routing

LSTM Advantage over Random Forest:
- RF cannot predict future trajectories from past movement patterns
- LSTM learns vessel dynamics and momentum over time
- LSTM captures sequential decision-making (each waypoint affects next)
- LSTM models cumulative effects of route choices
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional, Dict, List
import os


class RouteOptimizationLSTM(nn.Module):
    """
    LSTM model for route optimization and trajectory prediction

    Architecture:
        Encoder: Historical trajectory → LSTM → Trajectory embedding
        Predictor: Embedding + Environment → LSTM → Future waypoints

    Why LSTM beats Random Forest:
    - RF predicts each waypoint independently
    - LSTM predicts smooth, connected trajectory considering:
        * Vessel momentum and inertia
        * Sequential decision impact (current turn affects next position)
        * Cumulative fuel consumption along route
        * Time-dependent weather/current changes
    - LSTM learns realistic vessel movement physics

    Input:
        - Historical trajectory: [batch, 24 timesteps, 4 features]
          Features: [lat, lon, speed, heading]
        - Environmental conditions: [batch, 24 timesteps, 6 features]
          Features: [wave_height, wind_speed, wind_direction,
                     current_speed, current_direction, sea_state]
        - Vessel specs: [batch, 5 features]
          Features: [loa, beam, draft, max_speed, fuel_capacity]
        - Destination: [batch, 2 features]
          Features: [dest_lat, dest_lon]

    Output:
        - Optimal waypoints: [batch, 12 future timesteps, 2]
          Features: [lat, lon]
        - Expected fuel consumption: [batch, 1]
        - ETA hours: [batch, 1]
    """

    def __init__(self, history_length: int = 24,
                 prediction_length: int = 12,
                 trajectory_features: int = 4,
                 environment_features: int = 6,
                 vessel_features: int = 5,
                 destination_features: int = 2):
        """
        Initialize route optimization model

        Args:
            history_length: Historical timesteps to consider
            prediction_length: Future timesteps to predict
            trajectory_features: Features in trajectory data
            environment_features: Features in environment data
            vessel_features: Vessel specification features
            destination_features: Destination coordinates
        """
        super(RouteOptimizationLSTM, self).__init__()

        self.history_length = history_length
        self.prediction_length = prediction_length
        self.trajectory_features = trajectory_features
        self.environment_features = environment_features
        self.vessel_features = vessel_features
        self.destination_features = destination_features

        # TRAJECTORY ENCODER
        # ------------------
        # Process historical movement patterns
        self.trajectory_lstm = nn.LSTM(
            input_size=trajectory_features,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        # Output: [batch, history_len, 256]

        # ENVIRONMENT ENCODER
        # -------------------
        # Process weather/sea conditions
        self.environment_lstm = nn.LSTM(
            input_size=environment_features,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        # Output: [batch, history_len, 128]

        # ATTENTION MECHANISM
        # -------------------
        # Focus on critical trajectory segments
        self.attention_W = nn.Parameter(torch.randn(256, 1))
        self.attention_b = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.attention_W)

        # FUSION LAYER
        # ------------
        # Combine trajectory, environment, vessel specs, and destination
        # 256 (trajectory) + 128 (environment) + 5 (vessel) + 2 (destination) = 391
        fusion_input_size = 256 + 128 + vessel_features + destination_features

        self.fusion_dense1 = nn.Linear(fusion_input_size, 256)
        self.fusion_relu1 = nn.ReLU()
        self.fusion_dropout1 = nn.Dropout(p=0.3)

        self.fusion_dense2 = nn.Linear(256, 128)
        self.fusion_relu2 = nn.ReLU()
        self.fusion_dropout2 = nn.Dropout(p=0.2)

        # ROUTE PREDICTOR
        # ---------------
        # Generate optimal waypoints sequence
        self.route_lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        # WAYPOINT GENERATOR
        # ------------------
        # Predict (lat, lon) for each future timestep
        self.waypoint_output = nn.Linear(64, 2)  # [lat, lon]

        # AUXILIARY PREDICTIONS
        # ---------------------
        # Predict fuel consumption and ETA
        self.fuel_predictor = nn.Linear(128, 1)
        self.eta_predictor = nn.Linear(128, 1)

    def forward(self, trajectory_history, environment_history,
                vessel_specs, destination):
        """
        Forward pass

        Args:
            trajectory_history: [batch, 24, 4] - past positions/headings
            environment_history: [batch, 24, 6] - weather/sea conditions
            vessel_specs: [batch, 5] - vessel specifications
            destination: [batch, 2] - target coordinates

        Returns:
            Dict with:
                'waypoints': [batch, 12, 2] - predicted route
                'fuel_consumption': [batch, 1] - expected fuel use
                'eta_hours': [batch, 1] - estimated time
        """
        batch_size = trajectory_history.size(0)

        # Encode trajectory
        traj_out, _ = self.trajectory_lstm(trajectory_history)  # [batch, 24, 256]

        # Encode environment
        env_out, _ = self.environment_lstm(environment_history)  # [batch, 24, 128]

        # Apply attention to trajectory
        attention_scores = torch.tanh(
            torch.matmul(traj_out, self.attention_W) + self.attention_b
        )  # [batch, 24, 1]
        attention_weights = torch.softmax(attention_scores, dim=1)
        traj_context = torch.sum(traj_out * attention_weights, dim=1)  # [batch, 256]

        # Pool environment features (take last timestep)
        env_context = env_out[:, -1, :]  # [batch, 128]

        # Concatenate all contexts
        combined = torch.cat([
            traj_context,
            env_context,
            vessel_specs,
            destination
        ], dim=1)  # [batch, 391]

        # Fusion layers
        fused = self.fusion_dense1(combined)
        fused = self.fusion_relu1(fused)
        fused = self.fusion_dropout1(fused)

        fused = self.fusion_dense2(fused)
        fused = self.fusion_relu2(fused)
        fused = self.fusion_dropout2(fused)  # [batch, 128]

        # Predict auxiliary outputs
        fuel_consumption = self.fuel_predictor(fused)  # [batch, 1]
        eta_hours = torch.relu(self.eta_predictor(fused))  # [batch, 1] (positive)

        # Generate waypoint sequence
        # Repeat fused representation for each prediction timestep
        route_input = fused.unsqueeze(1).repeat(1, self.prediction_length, 1)  # [batch, 12, 128]

        # Route LSTM
        route_out, _ = self.route_lstm(route_input)  # [batch, 12, 64]

        # Generate waypoints
        waypoints = self.waypoint_output(route_out)  # [batch, 12, 2]

        return {
            'waypoints': waypoints,
            'fuel_consumption': fuel_consumption,
            'eta_hours': eta_hours
        }


class RouteOptimizer:
    """
    Wrapper class for route optimization
    """

    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        """
        Initialize route optimizer

        Args:
            model_path: Path to pre-trained model (.pth)
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = RouteOptimizationLSTM().to(self.device)

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        elif model_path:
            print(f"[WARNING] Model path {model_path} not found. Using new model.")

    def predict_route(self, trajectory_history: np.ndarray,
                     environment_history: np.ndarray,
                     vessel_specs: np.ndarray,
                     destination: np.ndarray) -> Dict:
        """
        Predict optimal route

        Args:
            trajectory_history: [24, 4] - historical trajectory
            environment_history: [24, 6] - weather/sea conditions
            vessel_specs: [5] - vessel specifications
            destination: [2] - destination coordinates

        Returns:
            Dict with waypoints, fuel, and ETA
        """
        self.model.eval()

        with torch.no_grad():
            traj_tensor = torch.FloatTensor(trajectory_history).unsqueeze(0).to(self.device)
            env_tensor = torch.FloatTensor(environment_history).unsqueeze(0).to(self.device)
            vessel_tensor = torch.FloatTensor(vessel_specs).unsqueeze(0).to(self.device)
            dest_tensor = torch.FloatTensor(destination).unsqueeze(0).to(self.device)

            prediction = self.model(traj_tensor, env_tensor, vessel_tensor, dest_tensor)

            return {
                'waypoints': prediction['waypoints'][0].cpu().numpy(),  # [12, 2]
                'fuel_consumption': float(prediction['fuel_consumption'][0][0].cpu().numpy()),
                'eta_hours': float(prediction['eta_hours'][0][0].cpu().numpy())
            }

    def predict_multiple_routes(self, trajectory_history: np.ndarray,
                               environment_history: np.ndarray,
                               vessel_specs: np.ndarray,
                               destination: np.ndarray,
                               n_samples: int = 5) -> Dict:
        """
        Generate multiple route options using Monte Carlo Dropout

        Returns:
            Dict with multiple route options and best route
        """
        self.model.train()  # Enable dropout

        routes = []
        fuels = []
        etas = []

        with torch.no_grad():
            traj_tensor = torch.FloatTensor(trajectory_history).unsqueeze(0).to(self.device)
            env_tensor = torch.FloatTensor(environment_history).unsqueeze(0).to(self.device)
            vessel_tensor = torch.FloatTensor(vessel_specs).unsqueeze(0).to(self.device)
            dest_tensor = torch.FloatTensor(destination).unsqueeze(0).to(self.device)

            for _ in range(n_samples):
                prediction = self.model(traj_tensor, env_tensor, vessel_tensor, dest_tensor)

                routes.append(prediction['waypoints'][0].cpu().numpy())
                fuels.append(float(prediction['fuel_consumption'][0][0].cpu().numpy()))
                etas.append(float(prediction['eta_hours'][0][0].cpu().numpy()))

        # Find best route (minimum fuel consumption)
        best_idx = np.argmin(fuels)

        return {
            'routes': routes,
            'fuel_consumptions': fuels,
            'etas': etas,
            'best_route': routes[best_idx],
            'best_fuel': fuels[best_idx],
            'best_eta': etas[best_idx]
        }

    def train(self, X_train: Tuple, y_train: Tuple,
              X_val: Tuple, y_val: Tuple,
              epochs: int = 100, batch_size: int = 32,
              learning_rate: float = 0.001,
              model_save_path: str = 'models/route_optimizer_lstm.pth') -> Dict:
        """
        Train the route optimization model

        Args:
            X_train: (traj_train, env_train, vessel_train, dest_train)
            y_train: (waypoints_train, fuel_train, eta_train)
            X_val: (traj_val, env_val, vessel_val, dest_val)
            y_val: (waypoints_val, fuel_val, eta_val)
            epochs: Max epochs
            batch_size: Batch size
            learning_rate: Learning rate
            model_save_path: Where to save best model

        Returns:
            Training history dict
        """
        traj_train, env_train, vessel_train, dest_train = X_train
        waypoints_train, fuel_train, eta_train = y_train

        traj_val, env_val, vessel_val, dest_val = X_val
        waypoints_val, fuel_val, eta_val = y_val

        # Convert to tensors
        traj_train_t = torch.FloatTensor(traj_train).to(self.device)
        env_train_t = torch.FloatTensor(env_train).to(self.device)
        vessel_train_t = torch.FloatTensor(vessel_train).to(self.device)
        dest_train_t = torch.FloatTensor(dest_train).to(self.device)
        waypoints_train_t = torch.FloatTensor(waypoints_train).to(self.device)
        fuel_train_t = torch.FloatTensor(fuel_train).unsqueeze(1).to(self.device)
        eta_train_t = torch.FloatTensor(eta_train).unsqueeze(1).to(self.device)

        traj_val_t = torch.FloatTensor(traj_val).to(self.device)
        env_val_t = torch.FloatTensor(env_val).to(self.device)
        vessel_val_t = torch.FloatTensor(vessel_val).to(self.device)
        dest_val_t = torch.FloatTensor(dest_val).to(self.device)
        waypoints_val_t = torch.FloatTensor(waypoints_val).to(self.device)
        fuel_val_t = torch.FloatTensor(fuel_val).unsqueeze(1).to(self.device)
        eta_val_t = torch.FloatTensor(eta_val).unsqueeze(1).to(self.device)

        # Data loader
        train_dataset = torch.utils.data.TensorDataset(
            traj_train_t, env_train_t, vessel_train_t, dest_train_t,
            waypoints_train_t, fuel_train_t, eta_train_t
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Loss functions
        waypoint_criterion = nn.MSELoss()
        fuel_criterion = nn.MSELoss()
        eta_criterion = nn.MSELoss()

        # Training history
        history = {'train_loss': [], 'val_loss': []}

        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0

        print(f"[INFO] Training Route Optimizer on {self.device}")

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_losses = []

            for traj_b, env_b, vessel_b, dest_b, waypoints_b, fuel_b, eta_b in train_loader:
                optimizer.zero_grad()

                predictions = self.model(traj_b, env_b, vessel_b, dest_b)

                # Multi-task loss
                waypoint_loss = waypoint_criterion(predictions['waypoints'], waypoints_b)
                fuel_loss = fuel_criterion(predictions['fuel_consumption'], fuel_b)
                eta_loss = eta_criterion(predictions['eta_hours'], eta_b)

                # Weighted combination
                total_loss = waypoint_loss + 0.5 * fuel_loss + 0.5 * eta_loss

                total_loss.backward()
                optimizer.step()

                train_losses.append(total_loss.item())

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_predictions = self.model(traj_val_t, env_val_t, vessel_val_t, dest_val_t)

                val_waypoint_loss = waypoint_criterion(val_predictions['waypoints'], waypoints_val_t)
                val_fuel_loss = fuel_criterion(val_predictions['fuel_consumption'], fuel_val_t)
                val_eta_loss = eta_criterion(val_predictions['eta_hours'], eta_val_t)

                val_loss = val_waypoint_loss + 0.5 * val_fuel_loss + 0.5 * val_eta_loss

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

        return history

    def evaluate(self, X_test: Tuple, y_test: Tuple) -> Dict:
        """
        Evaluate model

        Returns:
            Dict with waypoint_mae, fuel_mae, eta_mae
        """
        traj_test, env_test, vessel_test, dest_test = X_test
        waypoints_test, fuel_test, eta_test = y_test

        self.model.eval()

        with torch.no_grad():
            traj_t = torch.FloatTensor(traj_test).to(self.device)
            env_t = torch.FloatTensor(env_test).to(self.device)
            vessel_t = torch.FloatTensor(vessel_test).to(self.device)
            dest_t = torch.FloatTensor(dest_test).to(self.device)

            waypoints_t = torch.FloatTensor(waypoints_test).to(self.device)
            fuel_t = torch.FloatTensor(fuel_test).unsqueeze(1).to(self.device)
            eta_t = torch.FloatTensor(eta_test).unsqueeze(1).to(self.device)

            predictions = self.model(traj_t, env_t, vessel_t, dest_t)

            waypoint_mae = torch.mean(torch.abs(predictions['waypoints'] - waypoints_t)).item()
            fuel_mae = torch.mean(torch.abs(predictions['fuel_consumption'] - fuel_t)).item()
            eta_mae = torch.mean(torch.abs(predictions['eta_hours'] - eta_t)).item()

        return {
            'waypoint_mae': waypoint_mae,
            'fuel_mae': fuel_mae,
            'eta_mae': eta_mae
        }

    def load_model(self, path: str):
        """Load model from file"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        print(f"[INFO] Route optimizer loaded from {path}")

    def save_model(self, path: str):
        """Save model to file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({'model_state_dict': self.model.state_dict()}, path)
        print(f"[INFO] Route optimizer saved to {path}")

    def summary(self):
        """Print model summary"""
        print(self.model)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")


# Test the model
if __name__ == "__main__":
    print("[INFO] Testing Route Optimization LSTM...")

    optimizer = RouteOptimizer(device='cpu')
    optimizer.summary()

    # Test with dummy data
    print("\n[INFO] Testing route prediction...")
    dummy_trajectory = np.random.randn(24, 4).astype(np.float32)
    dummy_environment = np.random.randn(24, 6).astype(np.float32)
    dummy_vessel = np.random.randn(5).astype(np.float32)
    dummy_destination = np.array([1.5, 103.8], dtype=np.float32)  # Singapore coords

    route = optimizer.predict_route(
        dummy_trajectory, dummy_environment, dummy_vessel, dummy_destination
    )

    print(f"Predicted waypoints shape: {route['waypoints'].shape}")
    print(f"Fuel consumption: {route['fuel_consumption']:.2f} L")
    print(f"ETA: {route['eta_hours']:.2f} hours")

    print("\n[INFO] Testing multiple route generation...")
    routes = optimizer.predict_multiple_routes(
        dummy_trajectory, dummy_environment, dummy_vessel, dummy_destination,
        n_samples=5
    )

    print(f"Generated {len(routes['routes'])} route options")
    print(f"Best fuel consumption: {routes['best_fuel']:.2f} L")
    print(f"Best ETA: {routes['best_eta']:.2f} hours")

    print("\n[SUCCESS] Route optimization model test completed!")
