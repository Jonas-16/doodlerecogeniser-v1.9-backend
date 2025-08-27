"""Configuration module for the Doodle Recognition API."""

import os
from typing import List

# Load .env if present for easier local configuration
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


class Config:
    """Application configuration class."""
    
    def __init__(self):
        self.model_path = self._get_model_path()
        self.class_names = [
            'banana', 'apple', 'tree', 'car', 'smiley face', 
            'snake', 'ice cream', 'eye', 'star', 'envelope'
        ]
        self.allowed_origins = self._get_allowed_origins()
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY")
        
    def _get_model_path(self) -> str:
        """Get the path to the model file."""
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(here, 'doodle_recognizer_10classes_96x96.h5')
    
    def _get_allowed_origins(self) -> List[str]:
        """Get allowed CORS origins."""
        origins_str = os.getenv(
            "ALLOWED_ORIGINS", 
            "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001"
        )
        return origins_str.split(",")
    
    @property
    def model_exists(self) -> bool:
        """Check if model file exists."""
        return os.path.exists(self.model_path)


# Global config instance
config = Config()


SECRET_KEY = os.getenv("SECRET_KEY", "fallback_secret")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./users.db")