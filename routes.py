"""API routes for the Doodle Recognition API."""

import base64
import numpy as np
from io import BytesIO
from PIL import Image, ImageFilter, ImageEnhance
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import cv2
from pydantic import BaseModel

from schemas import (
    PredictionRequest, PredictionResponse, TestResponse, 
    InterpretRequest, InterpretResponse, GenAIGuessRequest, 
    GenAIGuessResponse, HealthResponse, GenAIStatusResponse,
    StabilityGenerateRequest, StabilityGenerateResponse,
)
from schemas import UserCreate, UserLogin, UserResponse, SavePredictionRequest
from models import doodle_model, User, PredictionHistory
from services import hash_password, verify_password, create_token
from preprocessing import ImagePreprocessor
from services import (
    genai_service,
    interpretation_service,
    generate_stability_from_data_url,
    get_stability_status,
)
from config import config
from database import SessionLocal
from datetime import datetime
from schemas import ImageInput


# Create router
router = APIRouter()

# Initialize preprocessor
preprocessor = ImagePreprocessor()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- AUTH ROUTES ---

@router.get("/test", response_model=TestResponse)
async def test():
    """Test endpoint to check if the API is working."""
    return TestResponse(
        message="Backend is working!", 
        model_loaded=doodle_model.is_loaded, 
        classes=config.class_names
    )


@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok", 
        model_loaded=doodle_model.is_loaded, 
        num_classes=len(config.class_names)
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest, db: Session = Depends(get_db)):
    try:
        if not doodle_model.is_loaded:
            raise HTTPException(
                status_code=503, 
                detail="Model not available on server (TensorFlow not installed or model failed to load)"
            )
        
        expected = req.width * req.height
        if len(req.image) != expected:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid data length. Expected {expected}, got {len(req.image)}"
            )

        pixel_array = np.array(req.image, dtype='float32')
        processed = preprocessor.preprocess_from_flat(pixel_array, req.width, req.height)
        label, confidence, top_predictions, all_predictions = doodle_model.predict(processed)
        user_id = req.user_id  # Add user_id to PredictionRequest schema

        if user_id:
            history = PredictionHistory(
                user_id=user_id,
                predicted_class=label,
                created_at=datetime.utcnow()
            )
            db.add(history)
            db.commit()
            db.refresh(history)

        return PredictionResponse(
            label=label,
            confidence=confidence,
            top_predictions=top_predictions,
            all_predictions=all_predictions,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/download_processed")
async def download_processed(req: PredictionRequest):
    """Download the processed image for debugging purposes."""
    try:
        expected = req.width * req.height
        if len(req.image) != expected:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid data length. Expected {expected}, got {len(req.image)}"
            )

        processed = preprocessor.preprocess_from_flat(
            np.array(req.image, dtype='float32'), req.width, req.height
        )
        image_2d = processed[0, :, :, 0]

        img_uint8 = (image_2d * 255.0).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8, mode='L')
        buf = BytesIO()
        pil_img.save(buf, format='PNG')
        buf.seek(0)

        return StreamingResponse(
            BytesIO(buf.getvalue()), 
            media_type="image/png", 
            headers={"Content-Disposition": "attachment; filename=processed.png"}
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/interpret", response_model=InterpretResponse)
async def interpret(req: InterpretRequest):
    """Provide interpretation of a prediction result."""
    try:
        interpretation = interpretation_service.interpret_prediction(
            req.prediction, req.confidence
        )
        return InterpretResponse(interpretation=interpretation)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/genai_guess", response_model=GenAIGuessResponse)
async def genai_guess(req: GenAIGuessRequest):
    """Use Google Gemini to guess what the doodle represents."""
    try:
        if not genai_service.is_available:
            available, reason = genai_service.get_status()
            raise HTTPException(
                status_code=503, 
                detail=f"GenAI service unavailable: {reason}"
            )

        # Parse the data URL
        try:
            mime_type, image_data = genai_service.parse_data_url(req.image)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Generate guess
        try:
            guess = genai_service.generate_guess(image_data, mime_type, req.prompt)
            if not guess:
                guess = "Unable to determine what this represents"
            
            return GenAIGuessResponse(guess=guess)
        
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


@router.get("/genai_status", response_model=GenAIStatusResponse)
async def genai_status():
    """Get the status of the GenAI service."""
    available, reason = genai_service.get_status()
    return GenAIStatusResponse(available=available, reason=reason if not available else None)


@router.post("/stability_generate", response_model=StabilityGenerateResponse)
async def stability_generate(req: StabilityGenerateRequest):
    """Generate an image from a doodle using Stability AI (image-to-image)."""
    try:
        available, reason = get_stability_status()
        if not available:
            raise HTTPException(status_code=503, detail=f"Stability AI unavailable: {reason}")

        try:
            b64, fmt = generate_stability_from_data_url(
                data_url=req.image,
                prompt=req.prompt,
                strength=req.strength or 0.6,
                output_format=req.output_format or "png",
            )
            # Enhance image before returning
            try:
                raw = base64.b64decode(b64)
                with BytesIO(raw) as bio:
                    img = Image.open(bio).convert("RGB")
                # Upscale 2x with high-quality resampling
                new_size = (img.width * 2, img.height * 2)
                img = img.resize(new_size, Image.LANCZOS)
                # Light denoise, then sharpen
                img = img.filter(ImageFilter.MedianFilter(size=3))
                img = img.filter(ImageFilter.UnsharpMask(radius=1.4, percent=160, threshold=3))
                # Slight color/contrast boost
                img = ImageEnhance.Color(img).enhance(1.06)
                img = ImageEnhance.Contrast(img).enhance(1.05)
                # Re-encode
                out = BytesIO()
                target_fmt = (fmt or "png").upper()
                if target_fmt == "JPG":
                    target_fmt = "JPEG"
                img.save(out, format=target_fmt)
                b64 = base64.b64encode(out.getvalue()).decode("utf-8")
            except Exception:
                # If enhancement fails, fall back to the original b64
                pass
        except ValueError as e:
            # data URL parsing error
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            # API or environment error
            raise HTTPException(status_code=502, detail=str(e))

        return StabilityGenerateResponse(image_base64=b64, format=fmt)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


@router.post("/stability_generate_download")
async def stability_generate_download(req: StabilityGenerateRequest):
    """Generate an image via Stability AI and return it as a file download."""
    try:
        available, reason = get_stability_status()
        if not available:
            raise HTTPException(status_code=503, detail=f"Stability AI unavailable: {reason}")

        try:
            b64, fmt = generate_stability_from_data_url(
                data_url=req.image,
                prompt=req.prompt,
                strength=req.strength or 0.6,
                output_format=req.output_format or "png",
            )
            # Enhance image before streaming
            try:
                raw = base64.b64decode(b64)
                with BytesIO(raw) as bio:
                    img = Image.open(bio).convert("RGB")
                new_size = (img.width * 2, img.height * 2)
                img = img.resize(new_size, Image.LANCZOS)
                img = img.filter(ImageFilter.MedianFilter(size=3))
                img = img.filter(ImageFilter.UnsharpMask(radius=1.4, percent=160, threshold=3))
                img = ImageEnhance.Color(img).enhance(1.06)
                img = ImageEnhance.Contrast(img).enhance(1.05)
                out = BytesIO()
                target_fmt = (fmt or "png").upper()
                if target_fmt == "JPG":
                    target_fmt = "JPEG"
                img.save(out, format=target_fmt)
                enhanced_raw = out.getvalue()
                raw = enhanced_raw
            except Exception:
                pass
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=502, detail=str(e))

        raw = base64.b64decode(b64) if isinstance(b64, str) else raw
        media_type = f"image/{fmt.lower()}"
        filename = f"stability_output.{fmt.lower()}"
        return StreamingResponse(
            BytesIO(raw),
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

@router.post("/save_prediction")
def save_prediction(req: SavePredictionRequest, db: Session = Depends(get_db)):
    history = PredictionHistory(
        user_id=req.user_id,
        predicted_class=req.predicted_class,
        created_at=datetime.utcnow()
    )
    db.add(history)
    db.commit()
    db.refresh(history)
    return {"message": "Prediction saved", "history_id": history.id}

@router.get("/get_history/{user_id}")
def get_history(user_id: int, db: Session = Depends(get_db)):
    history = db.query(PredictionHistory).filter(PredictionHistory.user_id == user_id).all()
    # Always return a list, even if empty
    return [
        {
            "predicted_class": h.predicted_class,
            "created_at": h.created_at.isoformat()
        }
        for h in history
    ]

@router.post("/login")
def login(req: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == req.username).first()
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_token(user.id, user.username)

    return {
        "access_token": token,
        "token_type": "bearer",
        "user_id": user.id,
        "username": user.username
    }


@router.post("/signin")
def signin(req: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.username == req.username).first()
    if existing:
        raise HTTPException(status_code=409, detail="Username already exists")

    user = User(username=req.username, password_hash=hash_password(req.password))
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_token(user.id, user.username)

    return {
        "access_token": token,
        "token_type": "bearer",
        "user_id": user.id,
        "username": user.username
    }