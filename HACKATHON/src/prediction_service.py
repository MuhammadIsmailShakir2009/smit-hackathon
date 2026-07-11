"""
prediction_service.py
----------------------
Yeh module ANN model aur scaler ko load karta hai (ya agar wo exist nahi
karte to automatically train kar deta hai) aur ek clean predict() function
expose karta hai jo poori app mein use hota hai. SHAP explainability module
bhi isi predict function ko wrapper ke taur par use karta hai.
"""

import numpy as np
import torch
import joblib

from config import MODEL_WEIGHTS_PATH, SCALER_PATH, FEATURE_NAMES
from models.ann_model import ResumeFitANN, load_model
from models.train_model import train_and_save_model

_model = None
_scaler = None


def ensure_model_ready():
    """Agar trained model/scaler files exist nahi karte to unhe train kar deta hai."""
    global _model, _scaler

    if MODEL_WEIGHTS_PATH.exists() and SCALER_PATH.exists():
        _model = load_model(MODEL_WEIGHTS_PATH, input_dim=len(FEATURE_NAMES))
        _scaler = joblib.load(SCALER_PATH)
    else:
        _model, _scaler = train_and_save_model(verbose=False)
        _model.eval()

    return _model, _scaler


def get_model_and_scaler():
    """Model aur scaler ko lazy-load karta hai (sirf ek dafa load hoga)."""
    global _model, _scaler
    if _model is None or _scaler is None:
        ensure_model_ready()
    return _model, _scaler


def predict_fit_probability(feature_array: np.ndarray) -> float:
    """
    Ek single candidate ke feature array (shape: 1 x num_features) se
    fit probability predict karta hai. Yeh ANN (primary predictive engine)
    ka direct output hai.
    """
    model, scaler = get_model_and_scaler()

    scaled_features = scaler.transform(feature_array)
    tensor_input = torch.tensor(scaled_features, dtype=torch.float32)

    with torch.no_grad():
        probability = model(tensor_input).item()

    return float(probability)


def predict_proba_batch(feature_matrix: np.ndarray) -> np.ndarray:
    """
    SHAP KernelExplainer ke liye batch prediction function. Yeh multiple
    rows ek sath le kar probabilities ka array return karta hai.
    """
    model, scaler = get_model_and_scaler()

    scaled = scaler.transform(feature_matrix)
    tensor_input = torch.tensor(scaled, dtype=torch.float32)

    with torch.no_grad():
        probabilities = model(tensor_input).numpy().flatten()

    return probabilities
