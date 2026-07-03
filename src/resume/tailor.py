"""Tailor the candidate's resume content to a specific JD.

Workflow:
  1. (Optional) Pre-run JDAnalyzer to get ATS keywords + matched / missing skills.
  2. Retrieve resume + projects + research from the vector store.
  3. Ask Claude (via prompts/resume_tailor.txt) to rewrite bullets + summary.
  4. Return a structured TailoredResume that can be exported to PDF.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.llm.claude_client import ClaudeClient
from src.llm.prompt_manager import get_prompt_manager
from src.rag.retriever import Retriever
from src.scoring.jd_analyzer import JDAnalyzer, JDAnalysis
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class TailoredBullet:
    section: str = ""
    title: str = ""
    rewritten_bullet: str = ""


@dataclass
class TailoredResume:
    summary: str = ""
    headline: str = ""
    core_skills: List[str] = field(default_factory=list)
    bullets: List[TailoredBullet] = field(default_factory=list)
    missing_tech_to_learn: List[str] = field(default_factory=list)
    ats_keywords_embedded: List[str] = field(default_factory=list)
    cover_letter_hook: str = ""
    jd_analysis: Optional[JDAnalysis] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "summary": self.summary,
            "headline": self.headline,
            "core_skills": self.core_skills,
            "bullets": [b.__dict__ for b in self.bullets],
            "missing_tech_to_learn": self.missing_tech_to_learn,
            "ats_keywords_embedded": self.ats_keywords_embedded,
            "cover_letter_hook": self.cover_letter_hook,
        }
        if self.jd_analysis:
            d["jd_analysis"] = self.jd_analysis.to_dict()
        return d


class ResumeTailor:
    def __init__(
        self,
        client: Optional[ClaudeClient] = None,
        retriever: Optional[Retriever] = None,
        jd_analyzer: Optional[JDAnalyzer] = None,
    ) -> None:
        self.client = client or ClaudeClient()
        self.retriever = retriever or Retriever()
        self.jd_analyzer = jd_analyzer or JDAnalyzer(client=self.client, retriever=self.retriever)
        self.prompts = get_prompt_manager()
        self.system = self.prompts.load("system")

    # ------------------------------------------------------------------ #
    def tailor(
        self,
        jd_text: str,
        precomputed_analysis: Optional[JDAnalysis] = None,
        max_tokens: int = 4096,
    ) -> TailoredResume:
        analysis = precomputed_analysis or self.jd_analyzer.analyze(jd_text)

        log.info("Retrieving resume / project / research chunks…")
        # Pull both resume-ish and project-ish content
        ctx_resume = self.retriever.retrieve(jd_text, k=4, category="resume")
        ctx_projects = self.retriever.retrieve(jd_text, k=3, category="paper")
        ctx_general = self.retriever.retrieve(jd_text, k=5)
        merged = {id(d): d for d in (ctx_resume + ctx_projects + ctx_general)}
        context = self.retriever.format_context(list(merged.values()), max_chars=9000)

        prompt = self.prompts.render(
            "resume_tailor",
            context=context,
            jd=jd_text,
            ats_keywords=", ".join(analysis.ats_keywords),
            matched_skills=", ".join(analysis.candidate_matched_skills),
            missing_skills=", ".join(analysis.candidate_missing_skills),
        )
        data = self.client.complete_json(prompt, system=self.system, max_tokens=max_tokens)
        tailored = self._coerce(data)
        tailored.jd_analysis = analysis
        return tailored

    # ------------------------------------------------------------------ #
    @staticmethod
    def _coerce(data: Dict[str, Any]) -> TailoredResume:
        def _list_str(v) -> List[str]:
            if not v:
                return []
            return [str(x) for x in v]

        bullets_raw = data.get("bullets", []) or []
        bullets: List[TailoredBullet] = []
        for b in bullets_raw:
            if isinstance(b, dict):
                bullets.append(
                    TailoredBullet(
                        section=str(b.get("section", "")),
                        title=str(b.get("title", "")),
                        rewritten_bullet=str(b.get("rewritten_bullet", "")),
                    )
                )
        return TailoredResume(
            summary=str(data.get("summary", "") or ""),
            headline=str(data.get("headline", "") or ""),
            core_skills=_list_str(data.get("core_skills")),
            bullets=bullets,
            missing_tech_to_learn=_list_str(data.get("missing_tech_to_learn")),
            ats_keywords_embedded=_list_str(data.get("ats_keywords_embedded")),
            cover_letter_hook=str(data.get("cover_letter_hook", "") or ""),
        )
