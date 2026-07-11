"""
config.py
---------
Yeh file poore project ki central configuration file hai.
Yahan par saari paths, constants, environment variables aur skill taxonomy
define ki gayi hai taake har module isay import kar ke use kar sakay.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# .env file ko load karna taake API keys environment variables mein aa jayen
load_dotenv()

# ---------------------------------------------------------------------------
# BASE PATHS
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
SAMPLE_RESUMES_DIR = DATA_DIR / "sample_resumes"
TRAINING_DATA_PATH = DATA_DIR / "training_data.csv"

MODELS_DIR = BASE_DIR / "models"
SAVED_MODEL_DIR = MODELS_DIR / "saved_model"
MODEL_WEIGHTS_PATH = SAVED_MODEL_DIR / "ann_model.pt"
SCALER_PATH = SAVED_MODEL_DIR / "feature_scaler.joblib"

DATABASE_DIR = BASE_DIR / "database"
DATABASE_PATH = DATABASE_DIR / "hr_copilot.db"

OUTPUTS_DIR = BASE_DIR / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Zaroori folders agar exist nahi karte to bana dete hain
for folder in [DATA_DIR, SAMPLE_RESUMES_DIR, SAVED_MODEL_DIR, DATABASE_DIR, REPORTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# ENVIRONMENT VARIABLES (API KEYS)
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # future ke liye extensible

# ---------------------------------------------------------------------------
# APP CONSTANTS
# ---------------------------------------------------------------------------
APP_TITLE = "👩🧑‍💼 Ismail HR Copilot"
RECOMMENDATION_THRESHOLD = 0.5  # is se upar probability ho to "Recommended"

# Feature vector ke names - ANN model isi order mein features expect karta hai
FEATURE_NAMES = [
    "skill_match_score",
    "experience_years_norm",
    "education_score",
    "text_similarity_score",
    "resume_quality_score",
]

# ---------------------------------------------------------------------------
# SKILL TAXONOMY
# ---------------------------------------------------------------------------
# Yeh master skill list hai jis se resume aur job description dono ka
# skill overlap nikala jata hai. Yeh generic rakhi gayi hai taake har
# industry/role ke liye kaam kare.
MASTER_SKILLS = [
    # Programming / Tech
    "python", "java", "c++", "c#", "javascript", "typescript", "sql", "r programming",
    "html", "css", "react", "angular", "node.js", "django", "flask", "fastapi",
    "machine learning", "deep learning", "data science", "artificial intelligence",
    "nlp", "computer vision", "tensorflow", "pytorch", "keras", "scikit-learn",
    "aws", "azure", "gcp", "docker", "kubernetes", "git", "linux", "devops",
    "power bi", "tableau", "excel", "spark", "hadoop", "etl", "api development",
    # Business / Management
    "project management", "agile", "scrum", "leadership", "team management",
    "stakeholder management", "budgeting", "strategic planning", "negotiation",
    "risk management", "business analysis", "operations management",
    # Finance / Accounting
    "financial analysis", "accounting", "auditing", "taxation", "bookkeeping",
    "financial modeling", "investment analysis", "risk assessment",
    # Marketing / Sales
    "digital marketing", "seo", "content marketing", "social media marketing",
    "sales strategy", "crm", "market research", "brand management",
    # HR
    "recruitment", "talent acquisition", "employee relations", "payroll",
    "performance management", "hr policies", "onboarding", "compensation",
    # Soft Skills
    "communication", "problem solving", "critical thinking", "teamwork",
    "time management", "adaptability", "creativity", "decision making",
    # Engineering
    "autocad", "solidworks", "mechanical design", "electrical engineering",
    "civil engineering", "quality control", "manufacturing", "supply chain",
    # Healthcare
    "patient care", "clinical research", "medical coding", "healthcare management",
    # Education
    "curriculum development", "teaching", "training", "instructional design",
    # Legal
    "contract law", "compliance", "legal research", "corporate law",
]

# Education degree keywords -> score mapping
EDUCATION_SCORES = {
    "phd": 1.0,
    "doctorate": 1.0,
    "master": 0.8,
    "mba": 0.8,
    "m.sc": 0.8,
    "m.tech": 0.8,
    "ms": 0.75,
    "bachelor": 0.6,
    "b.sc": 0.6,
    "b.tech": 0.6,
    "be": 0.6,
    "bs": 0.6,
    "diploma": 0.4,
    "associate": 0.35,
    "high school": 0.2,
    "intermediate": 0.2,
}

# ---------------------------------------------------------------------------
# BUSINESS MODEL / COMMERCIALIZATION STRATEGY (handbook requirement)
# ---------------------------------------------------------------------------
BUSINESS_MODEL_TEXT = """Target Market: Mid-to-large enterprises with high-volume recruitment needs \
(500+ applications/month) across IT, BPO, banking, and manufacturing sectors, where manual \
resume screening currently consumes significant recruiter hours per hire.

Revenue Model (SaaS - Tiered Subscription):
- Starter Tier ($99/month): Up to 200 resume screenings/month, 1 HR seat, standard explainability.
- Growth Tier ($349/month): Up to 1,500 screenings/month, 5 HR seats, SHAP dashboards, PDF reports.
- Enterprise Tier (Custom pricing): Unlimited screenings, ATS/HRIS integration, dedicated support, \
custom skill taxonomies, on-premise deployment option.
- Additional revenue stream: Pay-per-report API access for smaller agencies and staffing firms.

Value Proposition: Reduces average time-to-shortlist by an estimated 70-80% by automating the \
first-pass screening while keeping a human recruiter firmly in control via the Human-in-the-Loop \
approval workflow. Explainable AI builds trust and supports compliance with fair-hiring regulations \
by making every AI recommendation auditable.

Go-to-Market Strategy: Direct sales to HR/Talent Acquisition leaders, partnerships with existing \
Applicant Tracking System (ATS) vendors for plug-in integration, and a free trial tier (50 resumes) \
to drive product-led growth.

Competitive Advantage: Unlike black-box screening tools, this Co-Pilot combines a transparent, \
explainable predictive engine (ANN + SHAP) with an LLM-based reasoning layer for natural-language \
justifications, plus a mandatory human approval gate — reducing legal/compliance risk around \
automated hiring decisions (a growing regulatory concern globally).

Cost Structure: Cloud compute (model inference), LLM API usage (metered per report), engineering \
and compliance maintenance, and customer success/support staff.

Key Metrics for Scaling: Cost-per-screened-resume, recruiter hours saved per month, HR approval \
rate vs AI recommendation (model trust indicator), and net revenue retention across tiers."""

