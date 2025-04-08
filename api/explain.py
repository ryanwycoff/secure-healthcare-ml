# secure-healthcare-ml/api/explain.py

"""
This module handles the explanation of machine learning model predictions
using various interpretability techniques.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import List
import shap
import pandas as pd
from .auth import get_current_user
from .models import Model
from .schemas import PredictionRequest, PredictionResponse
from .utils import load_model

# Initialize router for model explanation endpoints
router = APIRouter()

# Load the model for explanation
model = load_model('model_v1.pkl')

# Endpoint for explaining model predictions using SHAP (SHapley Additive exPlanations)
@router.post("/explain", response_model=PredictionResponse)
async def explain_prediction(request: PredictionRequest, current_user: dict = Depends(get_current_user)):
    """
    Provide model predictions and SHAP-based explanations for the given input.
    """
    # Preprocess the input data
    input_data = pd.DataFrame([request.features])
    
    # Get model prediction
    prediction = model.predict(input_data)
    
    # Generate SHAP values for explanation
    explainer = shap.KernelExplainer(model.predict, input_data)
    shap_values = explainer.shap_values(input_data)
    
    # Generate SHAP summary plot (you could also return this as an image, for example)
    shap_summary = shap.summary_plot(shap_values, input_data, show=False)
    
    # Convert the SHAP values into a suitable format for the response
    shap_values_as_dict = {f"feature_{i}": shap_values[0][i].tolist() for i in range(len(shap_values[0]))}

    return PredictionResponse(prediction=prediction[0], shap_values=shap_values_as_dict)

