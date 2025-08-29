"""Model management for doodle recognition."""

import numpy as np
from typing import Optional, Dict, List, Tuple
from config import config
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from database import Base

try:
    import tensorflow as tf  # type: ignore
except Exception:
    tf = None

class DoodleModel:
    """Encapsulates the doodle recognition model."""
    
    def __init__(self):
        self.model: Optional[tf.keras.Model] = None
        self.class_names = config.class_names
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the Keras model if TensorFlow is available."""
        if tf is None:
            print("TensorFlow not installed; prediction endpoints will be unavailable.")
            return
        
        if not config.model_exists:
            print(f"Error: Model file not found at {config.model_path}")
            print("Please ensure the model file is in the backend directory.")
            return
        
        try:
            print("Loading Keras model ...")
            self.model = tf.keras.models.load_model(config.model_path)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Warning: Failed to load Keras model: {e}")
            self.model = None
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready."""
        return self.model is not None
    
    def predict(self, processed_image: np.ndarray) -> Tuple[str, float, List[Dict[str, float]], Dict[str, float]]:
        """
        Make prediction on processed image.
        
        Returns:
            - label: Best prediction label
            - confidence: Confidence score for best prediction
            - top_predictions: Top 3 predictions with scores
            - all_predictions: All class predictions
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        # Make prediction
        preds = self.model.predict(processed_image, verbose=0)
        
        # Get best prediction
        best_idx = int(np.argmax(preds[0]))
        label = self.class_names[best_idx]
        confidence = float(preds[0][best_idx])
        
        # Get top 3 predictions
        top_indices = np.argsort(preds[0])[-3:][::-1]
        top_predictions = [
            {"class": self.class_names[i], "confidence": float(preds[0][i])} 
            for i in top_indices
        ]
        
        # Get all predictions
        all_predictions = {
            self.class_names[i]: float(preds[0][i]) 
            for i in range(len(self.class_names))
        }
        
        return label, confidence, top_predictions, all_predictions


# Global model instance
doodle_model = DoodleModel()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)