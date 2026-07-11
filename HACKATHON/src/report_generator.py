"""
report_generator.py
--------------------
Yeh module downloadable PDF report generate karta hai jisme:
    - Executive summary (LLM generated)
    - Har candidate ka table (AI recommendation, confidence, HITL decision)
    - Business model / commercialization strategy section
Handbook requirement: "Generate a downloadable PDF or DOCX report."
"""

from io import BytesIO
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)

from config import BUSINESS_MODEL_TEXT


def _build_styles():
    """Custom paragraph styles taiyaar karta hai."""
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading2"],
            spaceBefore=14,
            spaceAfter=8,
            textColor=colors.HexColor("#1F2937"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodySmall",
            parent=styles["BodyText"],
            fontSize=9.5,
            leading=13,
        )
    )
    return styles


def generate_pdf_report(candidates: list, executive_summary: str, summary_stats: dict) -> bytes:
    """
    Poori screening batch ka PDF report generate karta hai aur bytes return
    karta hai (Streamlit download button ke liye).
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
    )
    styles = _build_styles()
    elements = []

    # ---- Title Page ----
    elements.append(Paragraph("AI Co-Pilot for HR Resume Screening", styles["Title"]))
    elements.append(
        Paragraph(f"Screening Report — Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"])
    )
    elements.append(Spacer(1, 16))

    # ---- Executive Summary ----
    elements.append(Paragraph("Executive Summary", styles["SectionHeading"]))
    elements.append(Paragraph(executive_summary.replace("\n", "<br/>"), styles["BodySmall"]))
    elements.append(Spacer(1, 10))

    # ---- Summary Stats Table ----
    stats_data = [
        ["Metric", "Value"],
        ["Total Candidates Screened", str(summary_stats.get("total", 0))],
        ["AI Recommended", str(summary_stats.get("recommended", 0))],
        ["AI Not Recommended", str(summary_stats.get("not_recommended", 0))],
        ["HR Approved", str(summary_stats.get("approved", 0))],
        ["HR Rejected", str(summary_stats.get("rejected", 0))],
        ["HR Modified", str(summary_stats.get("modified", 0))],
        ["Average AI Confidence", f"{summary_stats.get('avg_confidence', 0)}%"],
    ]
    stats_table = Table(stats_data, colWidths=[9 * cm, 6 * cm])
    stats_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F2937")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
            ]
        )
    )
    elements.append(stats_table)
    elements.append(Spacer(1, 14))

    # ---- Candidate Details Table ----
    elements.append(Paragraph("Candidate-wise Screening Results", styles["SectionHeading"]))

    table_data = [["Candidate", "AI Rec.", "Fit %", "Confidence", "HR Decision"]]
    for cand in candidates:
        table_data.append(
            [
                Paragraph(cand.get("candidate_name", "N/A"), styles["BodySmall"]),
                cand.get("ai_recommendation", "N/A"),
                f"{cand.get('ai_probability', 0) * 100:.1f}%",
                f"{cand.get('ai_confidence', 0)}%",
                cand.get("hitl_decision", "Pending"),
            ]
        )

    candidate_table = Table(table_data, colWidths=[5.5 * cm, 3 * cm, 2.3 * cm, 2.7 * cm, 3 * cm])
    candidate_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563EB")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
            ]
        )
    )
    elements.append(candidate_table)
    elements.append(Spacer(1, 10))

    # ---- Individual candidate explanations ----
    elements.append(PageBreak())
    elements.append(Paragraph("Detailed AI Explanations", styles["SectionHeading"]))
    for cand in candidates:
        elements.append(Paragraph(f"<b>{cand.get('candidate_name', 'N/A')}</b> — {cand.get('file_name', '')}", styles["Heading4"]))
        elements.append(Paragraph(cand.get("llm_explanation", "No explanation available."), styles["BodySmall"]))
        if cand.get("hitl_notes"):
            elements.append(Paragraph(f"<i>HR Notes: {cand['hitl_notes']}</i>", styles["BodySmall"]))
        elements.append(Spacer(1, 8))

    # ---- Business Model Section ----
    elements.append(PageBreak())
    elements.append(Paragraph("Business Model & Commercialization Strategy", styles["SectionHeading"]))
    for para in BUSINESS_MODEL_TEXT.split("\n\n"):
        elements.append(Paragraph(para.replace("\n", "<br/>"), styles["BodySmall"]))
        elements.append(Spacer(1, 6))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()
