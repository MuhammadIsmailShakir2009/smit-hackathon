"""
text_processor.py
------------------
Yeh module "Text modality" ko handle karta hai. Resume aur Job Description
ke raw text par NLP style processing hoti hai:
    1. Skill extraction (master skill list se matching)
    2. Experience years extraction (regex based)
    3. Education level scoring
    4. TF-IDF based text similarity (resume vs job description)

Note: Hum yahan lightweight approach use kar rahe hain (regex + TF-IDF)
taake bhaari deep learning language models download kiye baghair bhi
project turant chal jaye.
"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import MASTER_SKILLS, EDUCATION_SCORES
from src.utils import clean_text


def extract_skills(text: str) -> set:
    """Text mein se master skill list ke against matching skills nikalta hai."""
    cleaned = clean_text(text)
    found_skills = set()
    for skill in MASTER_SKILLS:
        # Word boundary use karte hain taake partial match na ho (e.g. "r" har jaga match na ho)
        pattern = r"(?<![a-zA-Z0-9])" + re.escape(skill) + r"(?![a-zA-Z0-9])"
        if re.search(pattern, cleaned):
            found_skills.add(skill)
    return found_skills


def compute_skill_match_score(resume_text: str, jd_text: str) -> tuple:
    """
    Resume aur JD dono se skills nikal kar overlap ratio calculate karta hai.
    Return: (score between 0-1, matched_skills set, missing_skills set)
    """
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)

    if not jd_skills:
        # Agar JD mein koi master skill match nahi hui to resume skill count par base karo
        score = min(len(resume_skills) / 10.0, 1.0)
        return score, resume_skills, set()

    matched = resume_skills.intersection(jd_skills)
    missing = jd_skills.difference(resume_skills)

    score = len(matched) / len(jd_skills)
    return round(score, 3), matched, missing


def extract_experience_years(text: str) -> float:
    """
    Resume text se total years of experience nikalne ki koshish karta hai.
    Do tareeqay try karte hain:
        1. Direct pattern jaise "5 years of experience"
        2. Date range pattern jaise "2018 - 2023" (year difference)
    """
    cleaned = clean_text(text)

    # Pattern 1: "X years" ya "X+ years"
    direct_matches = re.findall(r"(\d{1,2})\+?\s*(?:years|yrs)", cleaned)
    if direct_matches:
        years = [int(y) for y in direct_matches]
        return float(max(years))

    # Pattern 2: Date ranges jaise 2018-2023 ya 2018 to present
    year_ranges = re.findall(r"(19|20)\d{2}\s*(?:-|to|–)\s*(?:(19|20)\d{2}|present|current)", cleaned)
    all_years = re.findall(r"\b(19|20)\d{2}\b", cleaned)

    if len(all_years) >= 2:
        # Sab years nikal kar sabse purana aur sabse naya year lo
        full_years = [int(m) for m in re.findall(r"\b((?:19|20)\d{2})\b", cleaned)]
        if full_years:
            span = max(full_years) - min(full_years)
            return float(max(span, 0))

    return 0.0


def compute_education_score(text: str) -> float:
    """Resume text mein sabse high degree dhoond kar uska score deta hai."""
    cleaned = clean_text(text)
    best_score = 0.2  # default agar kuch match na ho (baseline)

    for keyword, score in EDUCATION_SCORES.items():
        if keyword in cleaned:
            best_score = max(best_score, score)

    return best_score


def compute_text_similarity(resume_text: str, jd_text: str) -> float:
    """TF-IDF vectorization ke zariye resume aur JD ke darmiyan cosine similarity nikalta hai."""
    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(jd_text)

    if not resume_clean.strip() or not jd_clean.strip():
        return 0.0

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        tfidf_matrix = vectorizer.fit_transform([resume_clean, jd_clean])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(float(similarity), 3)
    except ValueError:
        # Agar vocabulary empty ho jaye (bohot chhota text) to 0 return karo
        return 0.0


def compute_resume_quality_score(text: str) -> float:
    """
    Resume ki basic quality/completeness score - length, structure aur
    keyword density par base karte hue ek rough heuristic score deta hai.
    """
    if not text.strip():
        return 0.0

    word_count = len(text.split())
    length_score = min(word_count / 400.0, 1.0)  # ~400 words ek achi resume length

    # Zaroori sections ki presence check karo
    sections = ["experience", "education", "skills", "summary", "objective", "project"]
    cleaned = clean_text(text)
    section_hits = sum(1 for s in sections if s in cleaned)
    section_score = section_hits / len(sections)

    quality_score = (length_score * 0.5) + (section_score * 0.5)
    return round(quality_score, 3)
