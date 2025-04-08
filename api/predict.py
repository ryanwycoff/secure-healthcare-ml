# secure-healthcare-ml/api/predict.py

"""
This module handles the prediction of healthcare data using the trained
machine learning model. It provides an endpoint for users to send input
data and receive predictions.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import pandas as pd
from typing import List
from .auth import get_current_user
from .models import Model
from .schemas import PredictionRequest, PredictionResponse
from .utils import load_model

# Initialize router for prediction endpoints
router = APIRouter()

# Load the model for prediction
model = load_model('model_v1.pkl')

# Endpoint for model prediction
@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, current_user: dict = Depends(get_current_user)):
    """
    Provide a prediction from the trained model based on the given input features.
    """
    # Preprocess the input data
    input_data = pd.DataFrame([request.features])
    
    # Get model prediction
    prediction = model.predict(input_data)
    
    if prediction is None:
        raise HTTPException(status_code=400, detail="Prediction failed.")
    
    return PredictionResponse(prediction=prediction[0])

