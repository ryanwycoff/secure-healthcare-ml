# secure-healthcare-ml/api/__init__.py

"""
API module for the Secure Healthcare ML service.
This module sets up the FastAPI application and handles the 
requests for prediction and other endpoints.
"""

from fastapi import FastAPI
from .main import app as api_app

__all__ = ["api_app"]
