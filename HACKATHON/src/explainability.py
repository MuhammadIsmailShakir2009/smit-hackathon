"""
explainability.py
------------------
Yeh module "Explainable AI" requirement ko poora karta hai. Hum SHAP
(SHapley Additive exPlanations) use kar rahe hain taake har prediction
ke peeche kaun se features (aur kitna) contribute kar rahe hain, yeh
pata chal sake. Sath hi confidence score bhi calculate karte hain.
"""

import numpy as np
import shap

from config import FEATURE_NAMES
from src.prediction_service import predict_proba_batch

# Background dataset SHAP KernelExplainer ke liye chahiye hota hai
# (yeh "average/baseline" candidate ko represent karta hai)
_BACKGROUND_DATA = np.array(
    [
        [0.3, 0.3, 0.5, 0.3, 0.5],
        [0.5, 0.5, 0.6, 0.5, 0.5],
        [0.7, 0.7, 0.8, 0.7, 0.7],
        [0.2, 0.1, 0.4, 0.2, 0.4],
        [0.6, 0.4, 0.6, 0.6, 0.6],
    ],
    dtype=np.float32,
)


def compute_confidence(probability: float) -> float:
    """
    Model ki probability output (0-1) ko ek 0-100% confidence score mein
    convert karta hai. Jitni probability 0.5 se door hogi utni zyada confidence hai.
    """
    distance_from_midpoint = abs(probability - 0.5) * 2  # 0 to 1 range
    confidence_percent = round(distance_from_midpoint * 100, 1)
    return confidence_percent


def explain_prediction(feature_array: np.ndarray) -> dict:
    """
    SHAP KernelExplainer use kar ke ek single candidate ki prediction
    explain karta hai. Return: feature-wise contribution values aur
    unko human-readable ranking mein sorted list.
    """
    explainer = shap.KernelExplainer(predict_proba_batch, _BACKGROUND_DATA)

    # nsamples kam rakhte hain taake explanation fast ho (real-time UI ke liye)
    shap_values = explainer.shap_values(feature_array, nsamples=100, silent=True)

    # shap_values ka shape (1, num_features) hota hai regression/binary output ke liye
    shap_row = np.array(shap_values).flatten()

    contributions = []
    for name, value, shap_val in zip(FEATURE_NAMES, feature_array.flatten(), shap_row):
        contributions.append(
            {
                "feature": name,
                "value": round(float(value), 3),
                "shap_contribution": round(float(shap_val), 4),
            }
        )

    # Sabse zyada impact wale features pehle dikhao (absolute value ke hisab se)
    contributions_sorted = sorted(
        contributions, key=lambda x: abs(x["shap_contribution"]), reverse=True
    )

    base_value = float(np.array(explainer.expected_value).flatten()[0])

    return {
        "base_value": round(base_value, 4),
        "contributions": contributions_sorted,
    }


def readable_feature_name(feature_key: str) -> str:
    """Feature ke internal naam ko UI ke liye readable label mein convert karta hai."""
    mapping = {
        "skill_match_score": "Skill Match with Job Description",
        "experience_years_norm": "Years of Experience",
        "education_score": "Education Level",
        "text_similarity_score": "Resume-JD Text Similarity",
        "resume_quality_score": "Resume Quality / Completeness",
    }
    return mapping.get(feature_key, feature_key)
