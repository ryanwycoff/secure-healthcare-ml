# secure-healthcare-ml/api/main.py

"""
Main module to initialize the FastAPI application and include all routes
for model prediction, explanation, and user authentication.
"""

from fastapi import FastAPI
from .auth import router as auth_router
from .explain import router as explain_router
from .predict import router as predict_router
from .config import API_TITLE, API_DESCRIPTION, API_VERSION

# Initialize the FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
)

# Include routers for different functionality
app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(explain_router, prefix="/explain", tags=["explainability"])
app.include_router(predict_router, prefix="/predict", tags=["prediction"])

@app.get("/")
async def root():
    """
    Root endpoint to test if the API is running.
    """
    return {"message": "Welcome to the Secure Healthcare ML API!"}
