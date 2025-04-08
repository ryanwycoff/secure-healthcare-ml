# secure-healthcare-ml/scripts/preprocess.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(data_path):
    """
    Load the healthcare dataset from a specified path.
    Args:
    - data_path (str): Path to the dataset (CSV file).

    Returns:
    - pd.DataFrame: Loaded dataset.
    """
    df = pd.read_csv(data_path)
    print(f"Data loaded from {data_path}. Shape: {df.shape}")
    return df

def preprocess_data(df):
    """
    Preprocess the data by handling missing values, encoding categorical variables,
    and scaling the features.
    
    Args:
    - df (pd.DataFrame): The raw dataset to preprocess.
    
    Returns:
    - X (pd.DataFrame): Preprocessed feature set.
    - y (pd.Series): Target variable.
    """
    # Drop non-numeric columns (assuming 'target' is the target variable)
    df = df.select_dtypes(include=[np.number])

    # Handle missing values by filling with the mean (or you could choose another strategy)
    df.fillna(df.mean(), inplace=True)

    # Separate features and target variable
    X = df.drop(columns=['target'])
    y = df['target']

    # Feature scaling (StandardScaler standardizes the features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def split_data(X, y, test_size=0.3, random_state=42):
    """
    Split the dataset into training and test sets.
    
    Args:
    - X (pd.DataFrame or np.ndarray): Feature set.
    - y (pd.Series or np.ndarray): Target variable.
    - test_size (float): Proportion of the data to include in the test split.
    - random_state (int): Seed for reproducibility.
    
    Returns:
    - X_train (np.ndarray): Training features.
    - X_test (np.ndarray): Testing features.
    - y_train (pd.Series): Training target.
    - y_test (pd.Series): Testing target.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f"Data split into training and test sets with {X_train.shape[0]} training samples and {X_test.shape[0]} test samples.")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Define data path
    data_path = "../data/synthetic_fhir_data.csv"
    
    # Load and preprocess data
    df = load_data(data_path)
    X, y = preprocess_data(df)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Save the processed data for future use
    processed_data_path = "../data/processed/processed_data.csv"
    processed_df = pd.DataFrame(X_train, columns=df.drop(columns=['target']).columns)
    processed_df['target'] = y_train
    processed_df.to_csv(processed_data_path, index=False)
    print(f"Processed data saved to {processed_data_path}.")
