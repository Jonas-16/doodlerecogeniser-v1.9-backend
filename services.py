"""Complete Gemini (Generative AI) Setup and services for the Doodle API."""

import os
import random
import base64
from typing import Any, Tuple, Optional
from pathlib import Path
import bcrypt
import jwt
import datetime
from config import SECRET_KEY


try:
    # Try to load environment variables from a .env file next to this file if present
    from dotenv import load_dotenv  # type: ignore
    env_path = Path(__file__).with_name('.env')
    # Load from explicit path and also allow fallback to default discovery
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()
except Exception:
    pass

try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None


# ---------------------- Configuration ----------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
gemini_model = None
STABILITY_API_KEY = os.environ.get("STABILITY_API_KEY")


def initialize_gemini():
    """Initialize Gemini model with proper error handling and model selection."""
    global gemini_model
    
    if not GEMINI_API_KEY:
        print("GEMINI_API_KEY not set in environment variables.")
        return False
        
    if genai is None:
        print("google-generativeai package not found. Please install it with: pip install google-generativeai")
        return False
        
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        
        # List available models to verify API key works
        models = [m for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        if not models:
            print("No suitable Gemini models found.")
            return False
            
        # Use the first available model (usually gemini-1.5-flash or similar)
        model_name = models[0].name.split('/')[-1]
        gemini_model = genai.GenerativeModel(model_name)
        print(f"Successfully initialized Gemini model: {model_name}")
        
        # Test the connection with a simple prompt
        try:
            response = gemini_model.generate_content("Hello, Gemini!")
            print("Gemini connection test successful!")
            return True
        except Exception as e:
            print(f"Gemini connection test failed: {e}")
            gemini_model = None
            return False
            
    except Exception as e:
        print(f"Failed to initialize Gemini: {e}")
        gemini_model = None
        return False


# ---------------------- Utility Functions ----------------------
def is_gemini_available():
    """Check if Gemini model is available."""
    return gemini_model is not None


def get_gemini_status():
    """Get the status of Gemini service."""
    if is_gemini_available():
        return True, "Gemini service is available"
    
    if genai is None:
        return False, "google-generativeai library not installed"
    elif not GEMINI_API_KEY:
        return False, "GEMINI_API_KEY environment variable not set"
    else:
        return False, "Failed to initialize Gemini model"


def get_stability_status():
    """Get the status of Stability AI service."""
    if STABILITY_API_KEY:
        return True, "Stability AI key is set"
    return False, "STABILITY_API_KEY environment variable not set"


def extract_text_from_gemini_result(result: Any) -> str:
    """Extract text from Gemini result object."""
    # Try the high-level convenience property first
    try:
        text = getattr(result, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()
    except Exception:
        pass
    
    # Try candidates structure
    try:
        candidates = getattr(result, "candidates", []) or []
        if candidates:
            for cand in candidates:
                content = getattr(cand, "content", None)
                if not content:
                    continue
                parts = getattr(content, "parts", []) or []
                for part in parts:
                    maybe_text = getattr(part, "text", None)
                    if isinstance(maybe_text, str) and maybe_text.strip():
                        return maybe_text.strip()
    except Exception:
        pass
    
    return ""


def parse_data_url(data_url: str) -> Tuple[str, bytes]:
    """Parse a data URL and extract MIME type and binary data."""
    if not isinstance(data_url, str) or not data_url.startswith("data:"):
        raise ValueError("Expected a data URL string starting with 'data:'")
    
    try:
        header, b64data = data_url.split(",", 1)
        # header example: data:image/png;base64
        mime = header.split(";")[0][5:] if ";" in header else header[5:]
        return mime, base64.b64decode(b64data)
    except Exception as exc:
        raise ValueError(f"Invalid data URL: {exc}")


# ---------------------- Stability AI Generation ----------------------
def generate_stability_image(
    image_data: bytes,
    mime_type: str,
    prompt: Optional[str] = None,
    strength: float = 0.3,
    output_format: str = "png",
) -> Tuple[str, str]:
    """Use Stability AI image-to-image to generate an image from a sketch.

    Returns a tuple of (base64_image, format)
    """
    if not STABILITY_API_KEY:
        raise RuntimeError("Stability AI not available: STABILITY_API_KEY not set")

    try:
        import requests  # lazy import
    except Exception as e:
        raise RuntimeError(f"requests library not available: {e}")

    # Choose an engine; SDXL image-to-image endpoint
    engine_id = "stable-diffusion-xl-1024-v1-0"
    url = f"https://api.stability.ai/v1/generation/{engine_id}/image-to-image"

    # Default prompt if none provided
    # Encourage a polished AI recreation of the sketch
    base_prompt = (
        "Vibrant, stylized, artistic reinterpretation of the subject sketched; "
        "preserve overall composition and silhouette, correct proportions; painterly brushwork, dynamic lighting, rich colors, high quality."
    )
    final_prompt = (prompt + ", " + base_prompt) if (prompt and prompt.strip()) else base_prompt

    # SDXL requires specific dimensions; prepare image to 1024x1024 with padding while preserving aspect ratio
    prepared_image_bytes: bytes = image_data
    prepared_mime: str = "image/png"
    try:
        from io import BytesIO
        from PIL import Image, ImageFilter, ImageEnhance  # type: ignore

        def prepare_image_for_sdxl(data: bytes) -> bytes:
            with BytesIO(data) as bio:
                img = Image.open(bio).convert("RGB")
            # target canvas
            target_w, target_h = 1024, 1024
            # scale to fit within target while preserving aspect
            img.thumbnail((target_w, target_h), Image.LANCZOS)
            # paste centered on opaque white canvas
            canvas = Image.new("RGB", (target_w, target_h), (255, 255, 255))
            x = (target_w - img.width) // 2
            y = (target_h - img.height) // 2
            canvas.paste(img, (x, y))

            # Lightly blur and reduce contrast to prevent exact line copying
            canvas = canvas.filter(ImageFilter.GaussianBlur(radius=1.5))
            enhancer = ImageEnhance.Contrast(canvas)
            canvas = enhancer.enhance(0.75)
            enhancer_b = ImageEnhance.Brightness(canvas)
            canvas = enhancer_b.enhance(1.05)
            out = BytesIO()
            # Always send PNG
            canvas.save(out, format="PNG")
            return out.getvalue()

        prepared_image_bytes = prepare_image_for_sdxl(image_data)
        prepared_mime = "image/png"
    except Exception as _prep_err:
        # If Pillow is not available or processing fails, fall back to original bytes (may 400 on Stability)
        pass

    files = {
        "init_image": ("input.png", prepared_image_bytes, prepared_mime)
    }
    data = {
        "image_strength": str(max(0.0, min(strength, 1.0))),
        "cfg_scale": "10",
        "samples": "1",
        "steps": "40",
        "style_preset": "fantasy-art",
        "seed": str(random.randint(1, 2_147_483_647)),
    }
    # text_prompts[0][text] syntax expected by Stability v1 API
    data["text_prompts[0][text]"] = final_prompt
    data["text_prompts[0][weight]"] = "1"
    # Negative prompt to avoid defects
    data["text_prompts[1][text]"] = (
        "blurry, low quality, distorted, extra limbs, duplicate, noisy, messy, artifacts, jpeg artifacts, watermark"
    )
    data["text_prompts[1][weight]"] = "-1"

    headers = {
        "Authorization": f"Bearer {STABILITY_API_KEY}",
        "Accept": "application/json",
    }

    try:
        resp = requests.post(url, headers=headers, files=files, data=data, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"Stability API error {resp.status_code}: {resp.text[:500]}")

        payload = resp.json()
        artifacts = payload.get("artifacts") or []
        if not artifacts:
            raise RuntimeError("Stability API returned no artifacts")
        b64 = artifacts[0].get("base64")
        if not b64:
            raise RuntimeError("Stability API response missing base64 image")
        fmt = output_format or "png"
        return b64, fmt
    except Exception as e:
        raise RuntimeError(f"Error generating with Stability AI: {e}")


def generate_stability_from_data_url(
    data_url: str,
    prompt: Optional[str] = None,
    strength: float = 0.6,
    output_format: str = "png",
) -> Tuple[str, str]:
    """Convenience wrapper accepting a data URL."""
    mime_type, image_data = parse_data_url(data_url)
    return generate_stability_image(image_data, mime_type, prompt, strength, output_format)


# ---------------------- Main Generation Function ----------------------
def generate_gemini_guess(image_data: bytes, mime_type: str, custom_prompt: str = None) -> str:
    """Generate a guess for what the doodle represents using Gemini.
    
    Args:
        image_data: Binary image data
        mime_type: MIME type of the image (e.g., 'image/png')
        custom_prompt: Optional custom prompt for the AI
        
    Returns:
        str: The AI's guess about the doodle
    """
    if not is_gemini_available():
        return "Error: Gemini AI service is not available. Please check your API key and internet connection."
        
    try:
        # Create a base64 encoded string of the image
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # Prepare the prompt
        prompt = custom_prompt or """
        Analyze this doodle and describe what it represents in one or two words. 
        Choose from these categories if possible: banana, apple, tree, car, smiley face, 
        snake, ice cream, eye, star, envelope. Just return the best matching word or a short phrase.
        """.strip()
        
        # Create the message parts
        image_part = {
            "mime_type": mime_type,
            "data": base64_image
        }
        
        # Generate content
        response = gemini_model.generate_content(
            [prompt, image_part],
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": 100,
            }
        )
        
        # Extract and return the response text
        result_text = extract_text_from_gemini_result(response)
        if result_text:
            return result_text
        else:
            return "Error: Could not interpret the AI's response."
                
    except Exception as e:
        error_msg = f"Error generating guess: {str(e)}"
        print(error_msg)
        return error_msg


# ---------------------- Alternative function for data URLs ----------------------
def generate_guess_from_data_url(data_url: str, custom_prompt: str = None) -> str:
    """Generate a guess from a data URL (convenience function)."""
    try:
        mime_type, image_data = parse_data_url(data_url)
        return generate_gemini_guess(image_data, mime_type, custom_prompt)
    except Exception as e:
        return f"Error processing data URL: {str(e)}"


# ---------------------- Service Class (for compatibility) ----------------------
class GenAIService:
    """Service class wrapper for compatibility with existing imports."""
    
    def __init__(self):
        self.model = gemini_model
    
    @property
    def is_available(self):
        return is_gemini_available()
    
    def get_status(self):
        return get_gemini_status()
    
    def generate_guess(self, image_data: bytes, mime_type: str, custom_prompt: str = None) -> str:
        return generate_gemini_guess(image_data, mime_type, custom_prompt)
    
    def parse_data_url(self, data_url: str):
        return parse_data_url(data_url)


# ---------------------- Interpretation Service ----------------------
class InterpretationService:
    """Service for providing interpretation of predictions."""
    
    def __init__(self):
        # Default class names - you can modify these based on your model
        self.class_names = [
            'banana', 'apple', 'tree', 'car', 'smiley face', 
            'snake', 'ice cream', 'eye', 'star', 'envelope',
            'house', 'cat', 'dog', 'flower', 'sun'
        ]
    
    def interpret_prediction(self, prediction: str, confidence: float) -> str:
        """Generate an interpretation of the prediction result."""
        import difflib
        
        if confidence > 0.8:
            confidence_desc = "very confident"
        elif confidence > 0.6:
            confidence_desc = "confident"
        elif confidence > 0.4:
            confidence_desc = "somewhat confident"
        else:
            confidence_desc = "not very confident"
        
        # Find similar class names for suggestions
        similar_classes = difflib.get_close_matches(
            prediction, self.class_names, n=3, cutoff=0.3
        )
        
        interpretation = f"The model is {confidence_desc} (confidence: {confidence:.2f}) that this is a {prediction}."
        
        if len(similar_classes) > 1:
            others = [cls for cls in similar_classes if cls != prediction]
            if others:
                interpretation += f" Other possibilities the model considered: {', '.join(others)}."
        
        return interpretation


# ---------------------- Service Instances ----------------------
genai_service = GenAIService()
interpretation_service = InterpretationService()


# ---------------------- Initialize on Import ----------------------
def safe_initialize_gemini():
    """Initialize Gemini with better quota handling."""
    global gemini_model
    
    if not GEMINI_API_KEY:
        print("GEMINI_API_KEY not set in environment variables.")
        return False
        
    if genai is None:
        print("google-generativeai package not found. Please install it with: pip install google-generativeai")
        return False
        
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        
        # List available models to verify API key works
        models = [m for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        if not models:
            print("No suitable Gemini models found.")
            return False
        
        # Prefer flash model for free tier (lower quota usage)
        preferred_models = ['gemini-1.5-flash', 'gemini-1.5-flash-latest']
        model_name: Optional[str] = None
        
        for preferred in preferred_models:
            for model in models:
                if preferred in model.name:
                    model_name = preferred
                    break
            if model_name:
                break
        
        # Fallback to first available model
        if not model_name:
            model_name = models[0].name.split('/')[-1]
            
        gemini_model = genai.GenerativeModel(model_name)
        print(f"Successfully initialized Gemini model: {model_name}")
        
        # Skip connection test to avoid quota usage during initialization
        print("Gemini model ready (skipping test to preserve quota)")
        return True
            
    except Exception as e:
        print(f"Failed to initialize Gemini: {e}")
        gemini_model = None
        return False


if GEMINI_API_KEY and genai is not None:
    safe_initialize_gemini()
else:
    if genai is None:
        print("google-generativeai not installed; Gemini functionality will be unavailable.")
    else:
        print("GEMINI_API_KEY not set; Gemini functionality will be unavailable.")


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))

def create_token(user_id: int, username: str) -> str:
    payload = {
        "sub": user_id,   # subject = user id
        "username": username,  # store username in token
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24)  # expiration
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def decode_token(token: str):
    return jwt.decode(token, SECRET_KEY, algorithms=["HS256"])

# ---------------------- Example Usage ----------------------
"""
# Example of how to use in your Flask/FastAPI endpoints:


@app.route('/genai_guess', methods=['POST'])
def genai_guess():
    try:
        data = request.get_json()
        image_data_url = data.get('image')
        custom_prompt = data.get('prompt')
        
        if not image_data_url:
            return jsonify({'error': 'No image provided'}), 400
            
        guess = generate_guess_from_data_url(image_data_url, custom_prompt)
        
        return jsonify({
            'guess': guess,
            'available': is_gemini_available(),
            'status': get_gemini_status()[1]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/genai_status', methods=['GET'])
def genai_status():
    available, status = get_gemini_status()
    return jsonify({
        'available': available,
        'status': status,
        'model': gemini_model.model_name if gemini_model else None
    })
"""
