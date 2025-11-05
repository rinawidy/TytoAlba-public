"""
LSTM-based vessel arrival time prediction model
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Bidirectional, Dropout,
    Conv1D, MaxPooling1D, Concatenate, Multiply, Softmax, Layer
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from typing import Tuple, Optional
import os


class AttentionLayer(Layer):
    """
    Custom attention layer for focusing on important timesteps
    """

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], 1),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(1,),
            initializer='zeros',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # Calculate attention scores
        e = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        # Apply softmax
        attention_weights = tf.nn.softmax(e, axis=1)
        # Apply attention weights
        weighted_input = inputs * attention_weights
        return weighted_input

    def compute_output_shape(self, input_shape):
        return input_shape


class VesselArrivalLSTM:
    """
    LSTM-based arrival time prediction model

    Architecture:
        Input → CNN → Attention → Dropout → Bidirectional LSTM
        → Concatenate with static features → Dense layers → Output
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the model

        Args:
            model_path: Path to pre-trained model file (.h5 or SavedModel format)
        """
        self.model = None
        self.sequence_length = 48
        self.sequence_features = 8
        self.static_features = 10

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            if model_path:
                print(f"[WARNING] Model path {model_path} not found. Building new model.")
            self.model = self.build_model()


    def build_model(self) -> Model:
        """
        Build the neural network architecture

        Returns:
            Compiled Keras model
        """
        # INPUT LAYERS
        # ------------

        # Sequence input: [batch, 48 timesteps, 8 features]
        sequence_input = Input(
            shape=(self.sequence_length, self.sequence_features),
            name='sequence_input'
        )

        # Static features: [batch, 10 features]
        static_input = Input(
            shape=(self.static_features,),
            name='static_input'
        )

        # CNN FEATURE EXTRACTION
        # ----------------------

        # First convolutional block
        x = Conv1D(
            filters=64,
            kernel_size=3,
            activation='relu',
            padding='same',
            name='conv1d_1'
        )(sequence_input)
        x = MaxPooling1D(pool_size=2, name='maxpool_1')(x)
        # Shape: [batch, 24, 64]

        # Second convolutional block
        x = Conv1D(
            filters=128,
            kernel_size=3,
            activation='relu',
            padding='same',
            name='conv1d_2'
        )(x)
        x = MaxPooling1D(pool_size=2, name='maxpool_2')(x)
        # Shape: [batch, 12, 128]

        # ATTENTION MECHANISM
        # -------------------

        x = AttentionLayer(name='attention')(x)
        # Learns to focus on critical voyage segments

        # DROPOUT REGULARIZATION
        # ----------------------

        x = Dropout(rate=0.3, name='dropout_1')(x)

        # BIDIRECTIONAL LSTM
        # ------------------

        lstm_out = Bidirectional(
            LSTM(
                units=64,
                return_sequences=False,
                dropout=0.2,
                recurrent_dropout=0.2,
                name='lstm'
            ),
            name='bidirectional_lstm'
        )(x)
        # Shape: [batch, 128] (64 forward + 64 backward)

        # CONCATENATE WITH STATIC FEATURES
        # --------------------------------

        combined = Concatenate(name='concatenate')([lstm_out, static_input])
        # Shape: [batch, 138]

        # FEED-FORWARD NETWORK
        # --------------------

        dense1 = Dense(128, activation='relu', name='dense_1')(combined)
        dense1 = Dropout(0.3, name='dropout_2')(dense1)

        dense2 = Dense(64, activation='relu', name='dense_2')(dense1)
        dense2 = Dropout(0.2, name='dropout_3')(dense2)

        dense3 = Dense(32, activation='relu', name='dense_3')(dense2)

        # OUTPUT LAYER
        # ------------

        output = Dense(1, activation='linear', name='arrival_minutes')(dense3)

        # BUILD MODEL
        # -----------

        model = Model(
            inputs=[sequence_input, static_input],
            outputs=output,
            name='VesselArrivalLSTM'
        )

        # COMPILE
        # -------

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mae', 'mse', 'mape']
        )

        return model


    def predict(self, sequence_data: np.ndarray, static_features: np.ndarray) -> float:
        """
        Make inference prediction

        Args:
            sequence_data: Numpy array of shape [48, 8]
            static_features: Numpy array of shape [10]

        Returns:
            Predicted travel time in minutes
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load or build a model first.")

        # Add batch dimension
        seq_batch = np.expand_dims(sequence_data, axis=0)
        static_batch = np.expand_dims(static_features, axis=0)

        # Predict
        prediction = self.model.predict(
            [seq_batch, static_batch],
            verbose=0
        )

        return float(prediction[0][0])


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
        if self.model is None:
            raise ValueError("Model not loaded. Please load or build a model first.")

        # Enable dropout during inference
        predictions = []

        # Add batch dimension
        seq_batch = np.expand_dims(sequence_data, axis=0)
        static_batch = np.expand_dims(static_features, axis=0)

        for _ in range(n_samples):
            pred = self.model([seq_batch, static_batch], training=True)
            predictions.append(float(pred[0][0]))

        # Calculate statistics
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)

        # Convert std to confidence score (0-1)
        # Lower std = higher confidence
        # Using a sigmoid-like transformation
        confidence = 1 / (1 + std_pred / (mean_pred + 1e-6))

        return mean_pred, confidence


    def train(self, X_train: Tuple[np.ndarray, np.ndarray], y_train: np.ndarray,
              X_val: Tuple[np.ndarray, np.ndarray], y_val: np.ndarray,
              epochs: int = 100, batch_size: int = 32,
              model_save_path: str = 'models/vessel_arrival_lstm.h5') -> dict:
        """
        Train the LSTM model

        Args:
            X_train: Tuple of (sequence_train, static_train)
            y_train: Training targets (arrival times in minutes)
            X_val: Tuple of (sequence_val, static_val)
            y_val: Validation targets
            epochs: Maximum number of training epochs
            batch_size: Batch size for training
            model_save_path: Path to save best model

        Returns:
            Training history dict
        """
        if self.model is None:
            self.model = self.build_model()

        sequence_train, static_train = X_train
        sequence_val, static_val = X_val

        # Callbacks
        callbacks = [
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
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]

        # Train
        history = self.model.fit(
            x=[sequence_train, static_train],
            y=y_train,
            validation_data=([sequence_val, static_val], y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        return history.history


    def evaluate(self, X_test: Tuple[np.ndarray, np.ndarray],
                y_test: np.ndarray) -> dict:
        """
        Evaluate model performance

        Args:
            X_test: Tuple of (sequence_test, static_test)
            y_test: Test targets

        Returns:
            Dict of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load or build a model first.")

        sequence_test, static_test = X_test

        results = self.model.evaluate(
            [sequence_test, static_test],
            y_test,
            verbose=0
        )

        metrics = {
            'loss': results[0],
            'mae': results[1],
            'mse': results[2],
            'mape': results[3]
        }

        return metrics


    def load_model(self, path: str):
        """
        Load pre-trained model from file

        Args:
            path: Path to model file (.h5 or SavedModel directory)
        """
        # Register custom layers
        custom_objects = {'AttentionLayer': AttentionLayer}

        self.model = keras.models.load_model(path, custom_objects=custom_objects)
        print(f"[INFO] Model loaded from {path}")


    def save_model(self, path: str):
        """
        Save trained model to file

        Args:
            path: Path to save model (.h5 or directory for SavedModel)
        """
        if self.model is None:
            raise ValueError("No model to save. Please build or load a model first.")

        self.model.save(path)
        print(f"[INFO] Model saved to {path}")


    def summary(self):
        """Print model architecture summary"""
        if self.model is None:
            raise ValueError("No model to summarize. Please build or load a model first.")

        self.model.summary()


    def get_model_info(self) -> dict:
        """
        Get model architecture information

        Returns:
            Dict with model details
        """
        if self.model is None:
            return {"status": "No model loaded"}

        return {
            "model_name": self.model.name,
            "total_parameters": self.model.count_params(),
            "trainable_parameters": sum([np.prod(v.shape) for v in self.model.trainable_weights]),
            "sequence_length": self.sequence_length,
            "sequence_features": self.sequence_features,
            "static_features": self.static_features,
            "layers": len(self.model.layers)
        }
