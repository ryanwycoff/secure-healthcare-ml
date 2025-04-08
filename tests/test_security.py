# secure-healthcare-ml/tests/test_security.py

import unittest
from fastapi.testclient import TestClient
from api.main import app  # Assuming the FastAPI app is defined in api.main

class TestSecurity(unittest.TestCase):
    
    def setUp(self):
        """Setup the test client for FastAPI app."""
        self.client = TestClient(app)

    def test_api_authentication_required(self):
        """Test if authentication is required to access protected API endpoints."""
        # Try accessing a protected endpoint without authentication
        response = self.client.get("/predict")
        self.assertEqual(response.status_code, 401)  # Unauthorized access
        self.assertIn("detail", response.json())  # Check for authentication failure message

    def test_valid_api_key(self):
        """Test if valid API keys are accepted."""
        headers = {"Authorization": "Bearer valid-api-key"}
        response = self.client.get("/predict", headers=headers)
        self.assertEqual(response.status_code, 200)  # Should be authorized
        
    def test_invalid_api_key(self):
        """Test if invalid API keys are rejected."""
        headers = {"Authorization": "Bearer invalid-api-key"}
        response = self.client.get("/predict", headers=headers)
        self.assertEqual(response.status_code, 401)  # Unauthorized access
        self.assertIn("detail", response.json())  # Check for error message

    def test_sql_injection_protection(self):
        """Test for SQL injection protection in the application."""
        # Attempt an SQL injection attack through a query parameter
        payload = {"user_input": "' OR 1=1 --"}
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 400)  # Should be rejected, not execute SQL
        self.assertIn("detail", response.json())  # Check for validation error message

    def test_data_encryption(self):
        """Test if sensitive data is being encrypted or properly handled."""
        # Sending a request with sensitive data
        sensitive_data = {"patient_id": "12345", "medical_history": "Sensitive info"}
        response = self.client.post("/predict", json=sensitive_data)
        
        # Ensure that the sensitive data is not returned in the response in plaintext
        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        
        # Assuming the sensitive data should not be in the response
        self.assertNotIn("medical_history", response_data)  # Ensure it isn't returned as plaintext

    def test_model_security(self):
        """Test that the model is not exposed to unauthorized users."""
        # Try to access model prediction without proper authorization
        headers = {"Authorization": "Bearer invalid-api-key"}
        response = self.client.get("/predict", headers=headers)
        self.assertEqual(response.status_code, 401)  # Unauthorized access

    def test_sensitive_data_access_control(self):
        """Test that sensitive healthcare data is restricted and cannot be accessed by unauthorized users."""
        # Try accessing sensitive healthcare data without proper authorization
        headers = {"Authorization": "Bearer invalid-api-key"}
        response = self.client.get("/data/sensitive", headers=headers)
        self.assertEqual(response.status_code, 403)  # Forbidden access
        self.assertIn("detail", response.json())  # Should contain a message about restricted access

    def test_cors_security(self):
        """Test CORS headers to ensure cross-origin requests are controlled."""
        response = self.client.options("/predict")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Access-Control-Allow-Origin", response.headers)
        self.assertEqual(response.headers["Access-Control-Allow-Origin"], "*")  # Replace "*" with your allowed domain if needed

    def test_ensure_data_privacy_compliance(self):
        """Test if the application is compliant with data privacy regulations (e.g., HIPAA)."""
        sensitive_data = {"patient_id": "12345", "medical_history": "Sensitive info"}
        
        # Send a request with sensitive data
        response = self.client.post("/predict", json=sensitive_data)
        
        # Check for secure handling of sensitive data (response should not include sensitive data in plaintext)
        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        
        # Ensure that sensitive data is not exposed in the response
        self.assertNotIn("medical_history", response_data)

if __name__ == "__main__":
    unittest.main()
