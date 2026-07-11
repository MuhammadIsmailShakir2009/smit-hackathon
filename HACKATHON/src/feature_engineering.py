"""
feature_engineering.py
-----------------------
Yeh module "Tabular modality" ko represent karta hai. Text modality (NLP)
aur PDF modality (parsing) se jo raw signals nikalte hain, unko yahan
combine kar ke ek structured, numerical feature vector banaya jata hai
jo ANN model ko predictive input ke taur par diya jata hai.
"""

import numpy as np

from config import FEATURE_NAMES
from src.text_processor import (
    compute_skill_match_score,
    extract_experience_years,
    compute_education_score,
    compute_text_similarity,
    compute_resume_quality_score,
)


def build_feature_vector(resume_text: str, jd_text: str) -> dict:
    """
    Resume aur JD text se poora tabular feature set banata hai.
    Return: dictionary jisme har feature apni raw aur normalized value ke sath hai.
    """
    skill_score, matched_skills, missing_skills = compute_skill_match_score(resume_text, jd_text)
    experience_years = extract_experience_years(resume_text)
    education_score = compute_education_score(resume_text)
    similarity_score = compute_text_similarity(resume_text, jd_text)
    quality_score = compute_resume_quality_score(resume_text)

    # Experience years ko 0-1 range mein normalize karte hain (cap 15 years par)
    experience_norm = min(experience_years / 15.0, 1.0)

    features = {
        "skill_match_score": round(float(skill_score), 3),
        "experience_years_norm": round(float(experience_norm), 3),
        "education_score": round(float(education_score), 3),
        "text_similarity_score": round(float(similarity_score), 3),
        "resume_quality_score": round(float(quality_score), 3),
    }

    meta = {
        "experience_years_raw": experience_years,
        "matched_skills": sorted(matched_skills),
        "missing_skills": sorted(missing_skills),
    }

    return {"features": features, "meta": meta}


def feature_dict_to_array(features: dict) -> np.ndarray:
    """Feature dictionary ko ANN model ke expected order mein numpy array banata hai."""
    return np.array([[features[name] for name in FEATURE_NAMES]], dtype=np.float32)
