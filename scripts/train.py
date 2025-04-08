# secure-healthcare-ml/scripts/train.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
import sys
import os

# Adding the path to the src folder (or where your modules are located)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Import preprocessing functions
from scripts.preprocess import load_data, preprocess_data, split_data

def train_model(X_train, y_train):
    """
    Train a RandomForest model on the provided data.
    
    Args:
    - X_train (np.ndarray): Training features.
    - y_train (pd.Series): Training target.
    
    Returns:
    - model (sklearn.ensemble.RandomForestClassifier): Trained model.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print(f"Model trained with {X_train.shape[0]} samples.")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model's performance using accuracy and confusion matrix.
    
    Args:
    - model (sklearn.ensemble.RandomForestClassifier): Trained model.
    - X_test (np.ndarray): Testing features.
    - y_test (pd.Series): Testing target.
    
    Returns:
    - accuracy (float): Model's accuracy.
    - cm (np.ndarray): Confusion matrix.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Model accuracy: {accuracy:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    return accuracy, cm

def save_model(model, model_path):
    """
    Save the trained model to a specified path using joblib.
    
    Args:
    - model (sklearn.ensemble.RandomForestClassifier): Trained model.
    - model_path (str): Path to save the model.
    """
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    # Load and preprocess data
    data_path = "../data/processed/processed_data.csv"
    df = load_data(data_path)
    X, y = preprocess_data(df)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    accuracy, cm = evaluate_model(model, X_test, y_test)
    
    # Save trained model
    model_path = "../models/model_v1.pkl"
    save_model(model, model_path)
