# AI Co-Pilot for HR Resume Screening (👩🧑‍💼 Ismail HR Copilot)

A complete, hackathon-ready **AI Co-Pilot** built for the "AI Innovation Hackathon" theme
*"Build an AI Co-Pilot for Industry"* — solving the **HR Resume Screening** challenge.

## What This Project Does

Upload a Job Description and multiple candidate resumes. The AI Co-Pilot will:

1. Parse resumes (PDF / DOCX / TXT).
2. Extract skills, experience, education, and text similarity via NLP.
3. Predict a **job-fit probability** using a custom **PyTorch ANN** (the primary predictive engine).
4. Explain every prediction using **SHAP** (Explainable AI) + a confidence score.
5. Use an **LLM (gemini ai)** only for summarization, natural-language explanation, and report writing.
6. Let an HR reviewer **Approve / Reject / Modify** each AI recommendation (Human-in-the-Loop).
7. Generate a **downloadable PDF report** with an executive summary and business model.

## Data Modalities Used (Handbook Requirement: 3+)

| # | Modality | How it's used |
|---|----------|----------------|
| 1 | PDF / DOCX documents | Resume files parsed with `pdfplumber` and `python-docx` |
| 2 | Text (NLP) | Skill extraction, experience/education parsing, TF-IDF similarity |
| 3 | Tabular data | Engineered numeric features fed into the ANN model |

## Project Structure

```
hr_resume_screening_copilot/
├── app.py                     # Main Streamlit app (entry point)
├── config.py                  # Central configuration & constants
├── requirements.txt
├── .env.example
├── .env                       # Paste your OpenAI API key here
├── data/
│   ├── sample_job_description.txt
│   ├── sample_resumes/        # Sample resumes for quick testing
│   └── training_data.csv      # Auto-generated synthetic training data
├── models/
│   ├── ann_model.py           # PyTorch ANN architecture
│   ├── train_model.py         # Training script (auto-runs if no saved model found)
│   └── saved_model/           # Saved model weights + scaler (auto-created)
├── src/
│   ├── resume_parser.py       # PDF/DOCX/TXT parsing
│   ├── text_processor.py      # NLP: skills, experience, education, similarity
│   ├── feature_engineering.py # Tabular feature vector builder
│   ├── prediction_service.py  # Loads/auto-trains ANN, exposes predict()
│   ├── explainability.py      # SHAP explanations + confidence score
│   ├── llm_service.py         # OpenAI LLM calls (summary/explanation/report)
│   ├── database.py            # SQLite storage for candidates + HITL decisions
│   ├── report_generator.py    # PDF report generation (reportlab)
│   └── utils.py                # Helper functions
├── database/
│   └── hr_copilot.db          # Auto-created SQLite database
└── outputs/
    └── reports/                # (optional) local copies of generated reports
```


```bash
python3.11 -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```


> The app will still run without a key — the LLM-based text sections will show a
> placeholder message instead of AI-generated text. The ANN prediction and SHAP
> explainability work fully offline without any API key.

### 4. Run the app

```bash
streamlit run app.py
```

The very first time you run the app, it will automatically generate a synthetic
training dataset and train the ANN model (takes a few seconds). This is expected —
you do not need to run anything manually.

### 5. (Optional) Re-train the model manually

```bash
python models/train_model.py
```

## Quick Test

1. Go to **"1. Upload & Screen"**.
2. Paste the contents of `data/sample_job_description.txt` as the Job Description.
3. Upload both files from `data/sample_resumes/` (`strong_candidate.txt` and `weak_candidate.txt`).
4. Click **Run AI Screening**.
5. Go to **"2. Human-in-the-Loop Review"** to see AI recommendations, SHAP explanations,
   and approve/reject/modify them.
6. Go to **"4. Generate Report"** to build and download the PDF report.

## Notes on the ANN Model

Since no labeled "hired vs not-hired" dataset is publicly required by the handbook, the
ANN is trained on a **rule-based synthetic dataset** (`models/train_model.py`) that mimics
realistic scoring patterns with added noise. This is a standard, legitimate approach when
real labeled hiring data is unavailable, and keeps the deep learning model as the genuine
predictive engine (not the LLM), exactly as required by the hackathon rules.
