# secure-healthcare-ml/api/utils.py

"""
This module contains utility functions for loading models, data preprocessing,
and other common functions used throughout the API.
"""

import pickle
import pandas as pd
from typing import Any
from sklearn.preprocessing import StandardScaler
from fastapi import HTTPException


def load_model(model_path: str) -> Any:
    """
    Load a machine learning model from a pickle file.
    
    Args:
        model_path (str): The file path to the model file.
    
    Returns:
        Any: The loaded machine learning model.
    """
    try:
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)
        return model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")


def preprocess_data(input_data: dict) -> pd.DataFrame:
    """
    Preprocess input data to ensure it is in the correct format for the model.
    
    Args:
        input_data (dict): The raw input data from the user request.
    
    Returns:
        pd.DataFrame: The preprocessed data in the form of a DataFrame.
    """
    try:
        # Convert input data to DataFrame
        data_df = pd.DataFrame([input_data])
        
        # Example preprocessing: scale the data using StandardScaler (if necessary)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_df)
        
        # Return the scaled DataFrame
        return pd.DataFrame(scaled_data, columns=data_df.columns)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data preprocessing failed: {str(e)}")


def validate_input(input_data: dict, expected_columns: list) -> bool:
    """
    Validate that the input data contains all the expected columns.
    
    Args:
        input_data (dict): The raw input data from the user request.
        expected_columns (list): A list of columns that are expected in the input data.
    
    Returns:
        bool: True if the input data contains all expected columns, False otherwise.
    """
    missing_columns = [col for col in expected_columns if col not in input_data]
    if missing_columns:
        raise HTTPException(status_code=400, detail=f"Missing columns: {', '.join(missing_columns)}")
    return True
