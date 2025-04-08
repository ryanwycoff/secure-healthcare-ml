# secure-healthcare-ml/tests/test_explain.py

import unittest
import json
from fastapi.testclient import TestClient
from api.main import app  # Assuming the FastAPI app is defined in api.main

class TestExplainability(unittest.TestCase):
    
    def setUp(self):
        """Setup the test client for FastAPI app."""
        self.client = TestClient(app)
    
    def test_explain_shap(self):
        """Test the /explain endpoint to ensure explainability is working with SHAP."""
        # Sample input data to explain the prediction
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
        
        # Check if the response contains SHAP values
        explainability = response.json()
        self.assertIn("shap_values", explainability)
        self.assertIsInstance(explainability["shap_values"], list)
        
        # Optionally, validate the structure of SHAP values (e.g., length, content)
        self.assertGreater(len(explainability["shap_values"]), 0)
        self.assertIsInstance(explainability["shap_values"][0], float)
    
    def test_explain_invalid_data(self):
        """Test the /explain endpoint with invalid input data."""
        invalid_data = {
            "age": "invalid_value",  # Invalid type for age
            "sex": "M",
            "bmi": 30.5,
            "children": 2,
            "smoker": "yes",
            "region": "southeast"
        }
        
        response = self.client.post("/explain", json=invalid_data)
        self.assertEqual(response.status_code, 400)
        self.assertIn("detail", response.json())
    
    def test_explain_no_input(self):
        """Test the /explain endpoint with no input data."""
        response = self.client.post("/explain")
        self.assertEqual(response.status_code, 400)
        self.assertIn("detail", response.json())
    
    def test_explain_feature_importance(self):
        """Test the /explain endpoint to ensure feature importance is returned."""
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
        
        # Check if feature importance is returned
        explainability = response.json()
        self.assertIn("feature_importance", explainability)
        self.assertIsInstance(explainability["feature_importance"], dict)
        
        # Check if the feature importance contains valid keys (e.g., the features in the input data)
        feature_importance = explainability["feature_importance"]
        self.assertIn("age", feature_importance)
        self.assertIn("sex", feature_importance)
    
    def test_explain_model_status(self):
        """Test if the model is available for explainability."""
        response = self.client.get("/model-status")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"model_status": "loaded"})

if __name__ == "__main__":
    unittest.main()
