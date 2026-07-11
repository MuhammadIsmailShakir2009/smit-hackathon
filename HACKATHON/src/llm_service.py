"""
llm_service.py
---------------
IMPORTANT (handbook compliance): Is project mein LLM sirf REASONING,
SUMMARIZATION, EXPLANATION aur REPORT TEXT generate karne ke liye use
hota hai. Candidate ka "fit score" predict karne ka kaam sirf ANN
(models/ann_model.py) karta hai. LLM kabhi bhi primary predictive
engine ke taur par use nahi hota.

Hum yahan OpenAI API use kar rahe hain. Agar API key configure nahi
hai to graceful fallback (template-based text) provide karte hain
taake app crash na ho.
"""

from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_MODEL

_client = None


def _get_client():
    """OpenAI client ko lazy initialize karta hai."""
    global _client
    if _client is None and OPENAI_API_KEY:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def _call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 400) -> str:
    """Generic LLM call wrapper with error handling."""
    client = _get_client()

    if client is None:
        return (
            "[LLM not configured] OPENAI_API_KEY .env file mein set nahi hai. "
            "Yeh sirf ek placeholder text hai — real AI-generated text ke liye "
            "apni API key .env file mein add karein."
        )

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:  # noqa: BLE001
        return f"[LLM Error] Response generate nahi ho saka: {exc}"


def summarize_resume(resume_text: str) -> str:
    """Resume ka concise professional summary generate karta hai (LLM reasoning task)."""
    system_prompt = (
        "You are an expert HR analyst. Summarize the candidate's resume in 3-4 concise "
        "bullet points covering their core expertise, experience level, and standout strengths. "
        "Be factual and professional. Do not invent information not present in the resume."
    )
    user_prompt = f"Resume text:\n\n{resume_text[:4000]}"
    return _call_llm(system_prompt, user_prompt, max_tokens=300)


def generate_explanation_narrative(
    candidate_name: str,
    fit_probability: float,
    confidence: float,
    top_contributions: list,
    matched_skills: list,
    missing_skills: list,
) -> str:
    """
    SHAP explainability data ko natural language explanation mein convert karta hai.
    Yeh LLM ka "explanation generation" reasoning task hai — prediction khud
    ANN model ne pehle hi kar li hoti hai, LLM sirf isko explain karta hai.
    """
    system_prompt = (
        "You are an HR AI Co-Pilot assistant. Explain, in plain professional English, "
        "why the AI model gave this recommendation for a candidate. Reference the top "
        "contributing factors provided. Keep it to 4-5 sentences. Be balanced and objective, "
        "mentioning both strengths and gaps if present."
    )

    contributions_text = "\n".join(
        f"- {c['feature']}: value={c['value']}, impact={c['shap_contribution']}"
        for c in top_contributions[:5]
    )

    user_prompt = (
        f"Candidate: {candidate_name}\n"
        f"AI Fit Probability: {fit_probability:.2%}\n"
        f"Confidence Score: {confidence}%\n"
        f"Top contributing factors (SHAP values):\n{contributions_text}\n"
        f"Matched skills: {', '.join(matched_skills) if matched_skills else 'None found'}\n"
        f"Missing required skills: {', '.join(missing_skills) if missing_skills else 'None'}\n\n"
        "Write a short, clear explanation for the HR reviewer."
    )

    return _call_llm(system_prompt, user_prompt, max_tokens=250)


def generate_report_narrative(summary_stats: dict) -> str:
    """Poori batch screening ke liye ek executive summary narrative generate karta hai."""
    system_prompt = (
        "You are an HR analytics assistant writing an executive summary for a hiring "
        "manager. Be concise (5-6 sentences), professional, and data-driven."
    )
    user_prompt = (
        f"Total candidates screened: {summary_stats.get('total')}\n"
        f"AI Recommended: {summary_stats.get('recommended')}\n"
        f"AI Not Recommended: {summary_stats.get('not_recommended')}\n"
        f"HR Approved: {summary_stats.get('approved')}\n"
        f"HR Rejected: {summary_stats.get('rejected')}\n"
        f"HR Modified: {summary_stats.get('modified')}\n"
        f"Average AI confidence: {summary_stats.get('avg_confidence')}%\n\n"
        "Write an executive summary for this hiring batch."
    )
    return _call_llm(system_prompt, user_prompt, max_tokens=350)
