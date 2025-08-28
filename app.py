"""Main FastAPI application for Doodle Recognition API."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import uvicorn
from sqlalchemy import create_engine

from models import Base
from services import create_token 
from config import config, DATABASE_URL
from routes import router

origins = [
    "https://doodlerecogeniser-v1-9-frontend.vercel.app",  # your live frontend
    "http://localhost:3000",  # keep for local dev
]

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Doodle Recognition API",
        description="FastAPI backend for 28x28 doodle prediction",
        version="1.0.0",
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,  # allow cookies / auth headers
        allow_methods=["*"],     # allow all HTTP methods
        allow_headers=["*"],     # allow all headers
    )


    # Include routes
    app.include_router(router)

    # Root route for deployment check
    @app.get("/")
    def root():
        return {"status": "Backend is running!"}

    
    # Add exception handler
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request, exc: RequestValidationError):
        return JSONResponse(status_code=400, content={"detail": exc.errors()})
    
    return app


# Create the app instance
app = create_app()
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5001, reload=True)
