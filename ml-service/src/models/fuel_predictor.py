"""
Fuel Prediction Model using Random Forest Regressor

Features expected:
- distance (km)
- vehicle_weight (kg)
- avg_speed (km/h)
- vehicle_type (encoded: 0=small, 1=medium, 2=large)
- terrain_type (encoded: 0=flat, 1=hilly, 2=mountain)
"""

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, List, Union


class FuelPredictor:
    def __init__(self, model_path: str = None):
        """
        Initialize Fuel Predictor

        Args:
            model_path: Path to saved model file
        """
        self.model = None
        self.feature_names = [
            'distance',
            'vehicle_weight',
            'avg_speed',
            'vehicle_type',
            'terrain_type'
        ]

        if model_path:
            self.load_model(model_path)

    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs):
        """
        Train the Random Forest model

        Args:
            X_train: Training features
            y_train: Training target (fuel consumption)
            **kwargs: Additional parameters for RandomForestRegressor
        """
        # Default parameters for Random Forest
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }

        # Update with any user-provided parameters
        params.update(kwargs)

        self.model = RandomForestRegressor(**params)
        self.model.fit(X_train, y_train)

        return self

    def predict(self, features: Union[Dict, List[Dict], np.ndarray]) -> Union[float, List[float]]:
        """
        Predict fuel consumption

        Args:
            features: Can be:
                - Dict with feature names as keys
                - List of dicts
                - numpy array

        Returns:
            Predicted fuel consumption (liters)
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train() or load_model() first.")

        # Convert dict to array
        if isinstance(features, dict):
            X = self._dict_to_array([features])
            prediction = self.model.predict(X)
            return float(prediction[0])

        # Convert list of dicts to array
        elif isinstance(features, list) and all(isinstance(f, dict) for f in features):
            X = self._dict_to_array(features)
            predictions = self.model.predict(X)
            return predictions.tolist()

        # Direct numpy array
        elif isinstance(features, np.ndarray):
            predictions = self.model.predict(features)
            return predictions.tolist() if len(predictions) > 1 else float(predictions[0])

        else:
            raise ValueError("Features must be dict, list of dicts, or numpy array")

    def _dict_to_array(self, feature_dicts: List[Dict]) -> np.ndarray:
        """Convert list of feature dicts to numpy array"""
        X = []
        for features in feature_dicts:
            row = [features.get(name, 0) for name in self.feature_names]
            X.append(row)
        return np.array(X)

    def save_model(self, path: str):
        """Save trained model to file"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names
        }, path)

        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load trained model from file"""
        data = joblib.load(path)
        self.model = data['model']
        self.feature_names = data['feature_names']

        print(f"Model loaded from {path}")

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        if self.model is None:
            raise ValueError("Model not trained or loaded.")

        importances = self.model.feature_importances_
        return {
            name: float(importance)
            for name, importance in zip(self.feature_names, importances)
        }
