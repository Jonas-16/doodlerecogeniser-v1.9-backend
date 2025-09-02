"""Pydantic schemas for API requests and responses."""

from pydantic import BaseModel, EmailStr
from typing import List, Dict, Any, Optional

class PredictionRequest(BaseModel):
    """Request schema for prediction endpoint."""
    image: List[float]
    width: int = 28
    height: int = 28


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint."""
    label: str
    confidence: float
    top_predictions: List[Dict[str, Any]]
    all_predictions: Dict[str, float]


class TestResponse(BaseModel):
    """Response schema for test endpoint."""
    message: str
    model_loaded: bool
    classes: List[str]


class InterpretRequest(BaseModel):
    """Request schema for interpretation endpoint."""
    image: Any
    prediction: str
    confidence: float


class InterpretResponse(BaseModel):
    """Response schema for interpretation endpoint."""
    interpretation: str


class GenAIGuessRequest(BaseModel):
    """Request schema for GenAI guess endpoint."""
    image: str  # data URL (e.g., "data:image/png;base64,...")
    prompt: Optional[str] = None


class GenAIGuessResponse(BaseModel):
    """Response schema for GenAI guess endpoint."""
    guess: str


class HealthResponse(BaseModel):
    """Response schema for health endpoint."""
    status: str
    model_loaded: bool
    num_classes: int


class GenAIStatusResponse(BaseModel):
    """Response schema for GenAI status endpoint."""
    available: bool
    reason: Optional[str] = None


# --- Stability AI image generation ---
class StabilityGenerateRequest(BaseModel):
    """Request schema for Stability AI image generation (image-to-image)."""
    image: str  # data URL (e.g., "data:image/png;base64,...")
    prompt: Optional[str] = None
    strength: Optional[float] = 0.6  # how strongly to follow the prompt vs original sketch
    output_format: Optional[str] = "png"  # png or jpeg


class StabilityGenerateResponse(BaseModel):
    """Response schema for Stability AI image generation."""
    image_base64: str  # base64-encoded image without data URL prefix
    format: str  # e.g., "png"

class UserCreate(BaseModel):
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: int
    email: EmailStr

    class Config:
        orm_mode = True

class ImageInput(BaseModel):
    image: List[float]
    width: int = 28
    height: int = 28