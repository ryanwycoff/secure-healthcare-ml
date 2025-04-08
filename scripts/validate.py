# secure-healthcare-ml/scripts/validate.py

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import sys
import os

# Adding the path to the src folder (or where your modules are located)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Import preprocessing functions
from scripts.preprocess import load_data, preprocess_data, split_data

def load_trained_model(model_path):
    """
    Load a trained model from the specified path.
    
    Args:
    - model_path (str): Path to the saved model.
    
    Returns:
    - model (sklearn.ensemble.RandomForestClassifier): Trained model.
    """
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model's performance using accuracy, confusion matrix, and classification report.
    
    Args:
    - model (sklearn.ensemble.RandomForestClassifier): Trained model.
    - X_test (np.ndarray): Testing features.
    - y_test (pd.Series): Testing target.
    
    Returns:
    - accuracy (float): Model's accuracy.
    - cm (np.ndarray): Confusion matrix.
    - report (str): Classification report.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Model accuracy: {accuracy:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Classification Report:\n{report}")
    
    return accuracy, cm, report

if __name__ == "__main__":
    # Load and preprocess data
    data_path = "../data/processed/processed_data.csv"
    df = load_data(data_path)
    X, y = preprocess_data(df)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Load trained model
    model_path = "../models/model_v1.pkl"
    model = load_trained_model(model_path)
    
    # Evaluate model
    accuracy, cm, report = evaluate_model(model, X_test, y_test)
