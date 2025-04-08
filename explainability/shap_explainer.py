# secure-healthcare-ml/explainability/shap_explainer.py

import shap
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from typing import Any, Union

class SHAPExplainer:
    def __init__(self, model: BaseEstimator, feature_names: Union[list, np.ndarray]):
        """
        Initializes the SHAP explainer with a model and feature names.

        Args:
            model (BaseEstimator): The trained model to explain (e.g., sklearn model).
            feature_names (list or np.ndarray): List of feature names.
        """
        self.model = model
        self.feature_names = feature_names

    def explain(self, X: pd.DataFrame, num_features: int = 10) -> shap.Explanation:
        """
        Generates SHAP explanations for the model predictions on the given dataset.

        Args:
            X (pd.DataFrame): Input features for which SHAP values need to be computed.
            num_features (int): Number of top features to visualize in the explanation.

        Returns:
            shap.Explanation: SHAP explanation object containing the SHAP values.
        """
        # SHAP Explainer based on the model type
        if isinstance(self.model, BaseEstimator):
            explainer = shap.Explainer(self.model)
        else:
            raise ValueError("Model must be a scikit-learn estimator")

        # Get SHAP values for the input data
        shap_values = explainer(X)

        # Visualize the top 'num_features' features
        shap.summary_plot(shap_values, X, feature_names=self.feature_names, max_display=num_features)

        return shap_values

    def local_explanation(self, X: pd.DataFrame, instance_idx: int = 0) -> shap.Explanation:
        """
        Explains a single instance prediction using SHAP.

        Args:
            X (pd.DataFrame): The input dataset.
            instance_idx (int): The index of the instance to explain.

        Returns:
            shap.Explanation: SHAP explanation for the instance.
        """
        # Get SHAP values for the dataset
        shap_values = self.explain(X)

        # Visualize the SHAP values for the single instance
        shap.initjs()
        shap.force_plot(shap_values[instance_idx].values, shap_values[instance_idx].base_values, X.iloc[instance_idx])

        return shap_values[instance_idx]

# Example Usage:
# Assuming you have a trained model `model` and a dataset `X`
# model = ...  # A trained model (e.g., RandomForest, XGBoost)
# feature_names = X.columns.tolist()

# explainer = SHAPExplainer(model, feature_names)
# shap_values = explainer.explain(X)
# shap_values_local = explainer.local_explanation(X, instance_idx=0)
