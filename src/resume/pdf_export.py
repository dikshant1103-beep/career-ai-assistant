"""Render a TailoredResume to a clean, ATS-friendly PDF using ReportLab.

The layout is deliberately simple (single column, system fonts) so ATS systems
can parse it. Sections: name/headline, summary, core skills, experience &
projects (bullets grouped by section), cover-letter hook, learning list.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

from src.resume.tailor import TailoredResume
from src.utils.logger import get_logger

log = get_logger(__name__)


def export_resume_to_pdf(
    resume: TailoredResume,
    output_path: Path | str,
    candidate_name: str = "Your Name",
    contact_line: str = "email@example.com  |  +XX XXXXXXXXXX  |  linkedin.com/in/...",
) -> Path:
    """Write the tailored resume to ``output_path`` and return the path."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(out),
        pagesize=LETTER,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
        title=f"{candidate_name} - Tailored Resume",
    )

    styles = getSampleStyleSheet()
    name_style = ParagraphStyle(
        "Name", parent=styles["Title"], fontSize=20, leading=24,
        spaceAfter=2, alignment=TA_LEFT,
    )
    headline_style = ParagraphStyle(
        "Headline", parent=styles["Normal"], fontSize=11, leading=14,
        textColor="#444444", spaceAfter=2,
    )
    contact_style = ParagraphStyle(
        "Contact", parent=styles["Normal"], fontSize=9, leading=12,
        textColor="#666666", spaceAfter=10,
    )
    section_style = ParagraphStyle(
        "Section", parent=styles["Heading2"], fontSize=12, leading=15,
        textColor="#1a1a1a", spaceBefore=8, spaceAfter=4,
        borderPadding=2,
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"], fontSize=10, leading=13, spaceAfter=4,
    )
    bullet_style = ParagraphStyle(
        "Bullet", parent=body_style, leftIndent=12, bulletIndent=2, spaceAfter=2,
    )

    story: list = []

    # Header
    story.append(Paragraph(_e(candidate_name), name_style))
    if resume.headline:
        story.append(Paragraph(_e(resume.headline), headline_style))
    story.append(Paragraph(_e(contact_line), contact_style))

    # Summary
    if resume.summary:
        story.append(Paragraph("PROFESSIONAL SUMMARY", section_style))
        story.append(Paragraph(_e(resume.summary), body_style))

    # Skills
    if resume.core_skills:
        story.append(Paragraph("CORE SKILLS", section_style))
        story.append(Paragraph(_e(" • ".join(resume.core_skills)), body_style))

    # Bullets grouped by section
    if resume.bullets:
        grouped: dict[str, list] = {}
        for b in resume.bullets:
            grouped.setdefault(b.section or "Experience", []).append(b)
        # Order: Experience, Project, Research, others
        order = ["Experience", "Project", "Research"]
        keys = [k for k in order if k in grouped] + [
            k for k in grouped.keys() if k not in order
        ]
        for section in keys:
            heading = "PROJECTS" if section.lower().startswith("project") else (
                "RESEARCH" if section.lower().startswith("research") else "EXPERIENCE"
            )
            story.append(Paragraph(heading, section_style))
            for b in grouped[section]:
                title_line = f"<b>{_e(b.title)}</b>" if b.title else ""
                if title_line:
                    story.append(Paragraph(title_line, body_style))
                story.append(Paragraph(_e(b.rewritten_bullet), bullet_style, bulletText="•"))

    # Learning list
    if resume.missing_tech_to_learn:
        story.append(Paragraph("CURRENTLY LEARNING / FAMILIAR WITH", section_style))
        story.append(Paragraph(_e(", ".join(resume.missing_tech_to_learn)), body_style))

    # Cover letter hook
    if resume.cover_letter_hook:
        story.append(Spacer(1, 6))
        story.append(Paragraph("COVER LETTER HOOK (use in applications)", section_style))
        story.append(Paragraph(_e(resume.cover_letter_hook), body_style))

    doc.build(story)
    log.info("Tailored resume PDF written to %s", out)
    return out


def _e(s: str) -> str:
    """Escape XML/Paragraph-unsafe chars."""
    return (
        (s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
