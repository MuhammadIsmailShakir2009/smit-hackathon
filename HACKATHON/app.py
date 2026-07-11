"""
app.py
------
Main Streamlit application. Yeh file poori HR Resume Screening AI Co-Pilot
web app ka entry point hai. Handbook ke tamam mandatory requirements yahan
integrate kiye gaye hain:
    - Multimodal data (PDF/DOCX resumes, Text/NLP, Tabular features)
    - ANN predictive engine
    - LLM for reasoning/summarization/explanation
    - Human-in-the-Loop workflow
    - Explainable AI (SHAP + confidence)
    - Downloadable PDF report
    - Business model page
"""

import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from config import APP_TITLE, RECOMMENDATION_THRESHOLD, BUSINESS_MODEL_TEXT
from src.resume_parser import parse_resume
from src.feature_engineering import build_feature_vector, feature_dict_to_array
from src.prediction_service import predict_fit_probability, ensure_model_ready
from src.explainability import explain_prediction, compute_confidence, readable_feature_name
from src.llm_service import summarize_resume, generate_explanation_narrative, generate_report_narrative
from src.database import (
    init_db,
    insert_candidate,
    update_hitl_decision,
    get_all_candidates,
    clear_all_candidates,
)
from src.report_generator import generate_pdf_report
from src.utils import generate_id

# ---------------------------------------------------------------------------
# PAGE CONFIG & INITIAL SETUP
# ---------------------------------------------------------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="👩🧑‍💼", layout="wide")

# Database aur ANN model ko app start hote hi ready kar dete hain
init_db()

if "model_ready" not in st.session_state:
    with st.spinner("Preparing AI model (first-time setup, training ANN on synthetic data)..."):
        ensure_model_ready()
    st.session_state["model_ready"] = True

if "jd_text" not in st.session_state:
    st.session_state["jd_text"] = ""


# ---------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------------------------
st.sidebar.title("👩🧑‍💼 Ismail HR  Copilot")
page = st.sidebar.radio(
    "Navigate",
    [
        "1. Upload & Screen",
        "2. Human-in-the-Loop Review",
        "3. Analytics Dashboard",
        "4. Generate Report",
        
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("Theme: AI Co-Pilot for Industry — HR Resume Screening")

if st.sidebar.button("🗑️ Clear All Screened Candidates"):
    clear_all_candidates()
    st.sidebar.success("All candidate data cleared.")


# ---------------------------------------------------------------------------
# PAGE 1: UPLOAD & SCREEN
# ---------------------------------------------------------------------------
if page == "1. Upload & Screen":
    st.title(APP_TITLE)
    st.write(
        "Upload a job description and one or more candidate resumes. "
        "The AI Co-Pilot will parse, analyze, and score each candidate automatically."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Step 1: Job Description")
        jd_input_method = st.radio("Provide Job Description via:", ["Paste Text", "Upload File"], horizontal=True)

        if jd_input_method == "Paste Text":
            jd_text = st.text_area("Paste the Job Description here", height=220, value=st.session_state["jd_text"])
        else:
            jd_file = st.file_uploader("Upload Job Description (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"])
            jd_text = ""
            if jd_file is not None:
                jd_text = parse_resume(jd_file.name, jd_file.read())
                st.text_area("Extracted Job Description Text", value=jd_text, height=220, disabled=True)

        st.session_state["jd_text"] = jd_text

    with col2:
        st.subheader("Step 2: Upload Resumes")
        resume_files = st.file_uploader(
            "Upload candidate resumes (PDF/DOCX/TXT) — multiple files allowed",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
        )

    st.markdown("---")

    if st.button("🚀 Run AI Screening", type="primary", use_container_width=True):
        if not st.session_state["jd_text"].strip():
            st.error("Please provide a Job Description first.")
        elif not resume_files:
            st.error("Please upload at least one resume.")
        else:
            jd_text = st.session_state["jd_text"]
            progress = st.progress(0, text="Starting screening...")

            for idx, resume_file in enumerate(resume_files):
                progress.progress(
                    (idx) / len(resume_files),
                    text=f"Processing {resume_file.name} ({idx + 1}/{len(resume_files)})...",
                )

                # ---- PDF / DOCX / TXT Modality: parsing ----
                resume_bytes = resume_file.read()
                resume_text = parse_resume(resume_file.name, resume_bytes)

                if not resume_text.strip():
                    st.warning(f"Could not extract text from {resume_file.name}. Skipping.")
                    continue

                # ---- Text + Tabular Modality: feature engineering ----
                feature_result = build_feature_vector(resume_text, jd_text)
                features = feature_result["features"]
                meta = feature_result["meta"]
                feature_array = feature_dict_to_array(features)

                # ---- ANN Predictive Engine ----
                probability = predict_fit_probability(feature_array)
                confidence = compute_confidence(probability)
                recommendation = "Recommended" if probability >= RECOMMENDATION_THRESHOLD else "Not Recommended"

                # ---- Explainable AI (SHAP) ----
                explanation_data = explain_prediction(feature_array)

                # ---- LLM Reasoning: summary + explanation narrative ----
                candidate_name = resume_file.name.rsplit(".", 1)[0].replace("_", " ").title()
                llm_summary = summarize_resume(resume_text)
                llm_explanation = generate_explanation_narrative(
                    candidate_name=candidate_name,
                    fit_probability=probability,
                    confidence=confidence,
                    top_contributions=explanation_data["contributions"],
                    matched_skills=meta["matched_skills"],
                    missing_skills=meta["missing_skills"],
                )

                # ---- Save to Database ----
                candidate_id = generate_id("CAND")
                record = {
                    "candidate_id": candidate_id,
                    "candidate_name": candidate_name,
                    "file_name": resume_file.name,
                    "resume_text": resume_text,
                    "jd_snapshot": jd_text[:2000],
                    "features": features,
                    "matched_skills": meta["matched_skills"],
                    "missing_skills": meta["missing_skills"],
                    "ai_probability": probability,
                    "ai_confidence": confidence,
                    "ai_recommendation": recommendation,
                    "llm_summary": llm_summary,
                    "llm_explanation": llm_explanation,
                    "hitl_decision": "Pending",
                }
                insert_candidate(record)

            progress.progress(1.0, text="Screening complete!")
            st.success(f"Successfully screened {len(resume_files)} candidate(s). Go to 'Human-in-the-Loop Review' to see results.")


# ---------------------------------------------------------------------------
# PAGE 2: HUMAN IN THE LOOP REVIEW
# ---------------------------------------------------------------------------
elif page == "2. Human-in-the-Loop Review":
    st.title("Human-in-the-Loop Review")
    st.write("Review each AI recommendation and Approve, Reject, or Modify the decision.")

    candidates = get_all_candidates()

    if not candidates:
        st.info("No candidates screened yet. Go to 'Upload & Screen' first.")
    else:
        for cand in candidates:
            with st.expander(
                f"{'✅' if cand['ai_recommendation']=='Recommended' else '⚠️'} "
                f"{cand['candidate_name']}  —  Fit: {cand['ai_probability']*100:.1f}%  "
                f"| Confidence: {cand['ai_confidence']}%  | Status: {cand['hitl_decision']}"
            ):
                col1, col2 = st.columns([1.3, 1])

                with col1:
                    st.markdown("**🧠 LLM Resume Summary**")
                    st.write(cand["llm_summary"])

                    st.markdown("**💡 AI Explanation**")
                    st.write(cand["llm_explanation"])

                    matched = json.loads(cand["matched_skills_json"])
                    missing = json.loads(cand["missing_skills_json"])
                    st.markdown(f"**Matched Skills:** {', '.join(matched) if matched else 'None'}")
                    st.markdown(f"**Missing Skills:** {', '.join(missing) if missing else 'None'}")

                with col2:
                    st.markdown("**📊 Feature Contribution (SHAP)**")
                    features = json.loads(cand["features_json"])
                    feature_array = feature_dict_to_array(features)
                    explanation_data = explain_prediction(feature_array)

                    fig = go.Figure(
                        go.Bar(
                            x=[c["shap_contribution"] for c in explanation_data["contributions"]],
                            y=[readable_feature_name(c["feature"]) for c in explanation_data["contributions"]],
                            orientation="h",
                            marker_color=[
                                "#16A34A" if c["shap_contribution"] >= 0 else "#DC2626"
                                for c in explanation_data["contributions"]
                            ],
                        )
                    )
                    fig.update_layout(
                        height=260,
                        margin=dict(l=10, r=10, t=10, b=10),
                        xaxis_title="Impact on Fit Score",
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_{cand['candidate_id']}")

                st.markdown("---")
                st.markdown("**✅ Human-in-the-Loop Decision**")

                hitl_col1, hitl_col2, hitl_col3 = st.columns([1, 1, 2])
                with hitl_col1:
                    if st.button("✅ Approve", key=f"approve_{cand['candidate_id']}", use_container_width=True):
                        update_hitl_decision(cand["candidate_id"], "Approved")
                        st.rerun()
                with hitl_col2:
                    if st.button("❌ Reject", key=f"reject_{cand['candidate_id']}", use_container_width=True):
                        update_hitl_decision(cand["candidate_id"], "Rejected")
                        st.rerun()
                with hitl_col3:
                    modified_score = st.slider(
                        "Modify Fit Score Manually",
                        0.0, 100.0, float(cand["ai_probability"]) * 100,
                        key=f"slider_{cand['candidate_id']}",
                    )

                notes = st.text_input("HR Notes / Reason", value=cand.get("hitl_notes") or "", key=f"notes_{cand['candidate_id']}")

                if st.button("💾 Save Modification & Notes", key=f"modify_{cand['candidate_id']}"):
                    update_hitl_decision(
                        cand["candidate_id"], "Modified", notes=notes, modified_score=modified_score / 100.0
                    )
                    st.success("Modification saved.")
                    st.rerun()


# ---------------------------------------------------------------------------
# PAGE 3: ANALYTICS DASHBOARD
# ---------------------------------------------------------------------------
elif page == "3. Analytics Dashboard":
    st.title("Analytics Dashboard")

    candidates = get_all_candidates()

    if not candidates:
        st.info("No candidates screened yet.")
    else:
        df = pd.DataFrame(candidates)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Candidates", len(df))
        c2.metric("AI Recommended", int((df["ai_recommendation"] == "Recommended").sum()))
        c3.metric("HR Approved", int((df["hitl_decision"] == "Approved").sum()))
        c4.metric("Avg. Confidence", f"{df['ai_confidence'].mean():.1f}%")

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("AI Recommendation Breakdown")
            rec_counts = df["ai_recommendation"].value_counts()
            fig1 = go.Figure(go.Pie(labels=rec_counts.index, values=rec_counts.values, hole=0.4))
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.subheader("HR Decision Breakdown")
            dec_counts = df["hitl_decision"].value_counts()
            fig2 = go.Figure(go.Pie(labels=dec_counts.index, values=dec_counts.values, hole=0.4))
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Candidate Fit Scores")
        fig3 = go.Figure(
            go.Bar(x=df["candidate_name"], y=df["ai_probability"] * 100, marker_color="#2563EB")
        )
        fig3.update_layout(yaxis_title="Fit Probability (%)", xaxis_title="Candidate")
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Raw Data Table")
        display_cols = [
            "candidate_name", "file_name", "ai_recommendation",
            "ai_probability", "ai_confidence", "hitl_decision", "hitl_notes",
        ]
        st.dataframe(df[display_cols], use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE 4: GENERATE REPORT
# ---------------------------------------------------------------------------
elif page == "4. Generate Report":
    st.title("Generate Downloadable Report")

    candidates = get_all_candidates()

    if not candidates:
        st.info("No candidates screened yet. Nothing to report.")
    else:
        total = len(candidates)
        recommended = sum(1 for c in candidates if c["ai_recommendation"] == "Recommended")
        approved = sum(1 for c in candidates if c["hitl_decision"] == "Approved")
        rejected = sum(1 for c in candidates if c["hitl_decision"] == "Rejected")
        modified = sum(1 for c in candidates if c["hitl_decision"] == "Modified")
        avg_confidence = round(sum(c["ai_confidence"] for c in candidates) / total, 1)

        summary_stats = {
            "total": total,
            "recommended": recommended,
            "not_recommended": total - recommended,
            "approved": approved,
            "rejected": rejected,
            "modified": modified,
            "avg_confidence": avg_confidence,
        }

        st.write("Summary that will be included in the report:")
        st.json(summary_stats)

        if st.button("📝 Generate Executive Summary (LLM)", type="primary"):
            with st.spinner("Generating executive summary using LLM..."):
                narrative = generate_report_narrative(summary_stats)
            st.session_state["executive_summary"] = narrative

        executive_summary = st.session_state.get(
            "executive_summary", "Click the button above to generate an AI-written executive summary."
        )
        st.text_area("Executive Summary (editable)", value=executive_summary, height=150, key="exec_summary_box")

        if st.button("📄 Build & Download PDF Report", use_container_width=True):
            with st.spinner("Building PDF report..."):
                pdf_bytes = generate_pdf_report(
                    candidates, st.session_state.get("exec_summary_box", executive_summary), summary_stats
                )
            st.download_button(
                label="⬇️ Download PDF Report",
                data=pdf_bytes,
                file_name="HR_Resume_Screening_Report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )


