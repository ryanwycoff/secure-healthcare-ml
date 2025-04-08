# secure-healthcare-ml/tests/test_api.py

import unittest
import json
from fastapi.testclient import TestClient
from api.main import app  # Assuming the FastAPI app is defined in api.main

class TestAPI(unittest.TestCase):
    
    def setUp(self):
        """Setup the test client for FastAPI app."""
        self.client = TestClient(app)
    
    def test_health_check(self):
        """Test the /health endpoint to ensure the API is running."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "OK"})
    
    def test_predict(self):
        """Test the /predict endpoint to ensure prediction functionality."""
        # Prepare a sample payload for prediction
        sample_data = {
            "age": 45,
            "sex": "M",
            "bmi": 30.5,
            "children": 2,
            "smoker": "yes",
            "region": "southeast"
        }
        
        response = self.client.post("/predict", json=sample_data)
        self.assertEqual(response.status_code, 200)
        
        # Check if the response contains the prediction
        prediction = response.json()
        self.assertIn("prediction", prediction)
        self.assertIsInstance(prediction["prediction"], float)
    
    def test_authentication(self):
        """Test the authentication functionality."""
        # Assuming authentication requires a POST request to /auth
        auth_data = {
            "username": "testuser",
            "password": "testpassword"
        }
        
        response = self.client.post("/auth", json=auth_data)
        self.assertEqual(response.status_code, 200)
        
        auth_response = response.json()
        self.assertIn("access_token", auth_response)
    
    def test_explainability(self):
        """Test the explainability endpoint (SHAP values)."""
        sample_data = {
            "age": 45,
            "sex": "M",
            "bmi": 30.5,
            "children": 2,
            "smoker": "yes",
            "region": "southeast"
        }
        
        response = self.client.post("/explain", json=sample_data)
        self.assertEqual(response.status_code, 200)
        
        explainability = response.json()
        self.assertIn("shap_values", explainability)
        self.assertIsInstance(explainability["shap_values"], list)
    
    def test_invalid_predict(self):
        """Test the /predict endpoint with invalid data."""
        invalid_data = {
            "age": "invalid_value",  # Invalid type for age
            "sex": "M",
            "bmi": 30.5,
            "children": 2,
            "smoker": "yes",
            "region": "southeast"
        }
        
        response = self.client.post("/predict", json=invalid_data)
        self.assertEqual(response.status_code, 400)
        self.assertIn("detail", response.json())
    
    def test_model_status(self):
        """Test if the model is loaded and ready for predictions."""
        response = self.client.get("/model-status")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"model_status": "loaded"})

if __name__ == "__main__":
    unittest.main()
