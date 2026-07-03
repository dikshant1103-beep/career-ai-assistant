"""Analyze a job description against the candidate's profile.

Pipeline:
  1. Retrieve top-k profile chunks relevant to the JD.
  2. Call Claude with the JD analyzer prompt -> structured JSON.
  3. Validate and wrap into a JDAnalysis dataclass.
  4. Optionally compute a local TF-IDF cosine match between JD and profile
     (a cheap sanity check independent of the LLM score).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.llm.claude_client import ClaudeClient
from src.llm.prompt_manager import get_prompt_manager
from src.rag.retriever import Retriever
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class JDAnalysis:
    job_title: str = ""
    company: str = ""
    seniority: str = "unknown"
    required_skills: List[str] = field(default_factory=list)
    preferred_skills: List[str] = field(default_factory=list)
    responsibilities: List[str] = field(default_factory=list)
    ats_keywords: List[str] = field(default_factory=list)
    candidate_matched_skills: List[str] = field(default_factory=list)
    candidate_missing_skills: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    compatibility_score: int = 0
    ats_score: int = 0
    improvement_suggestions: List[str] = field(default_factory=list)
    summary: str = ""
    local_cosine_match: float = 0.0  # local TF-IDF sanity score

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class JDAnalyzer:
    def __init__(
        self,
        client: Optional[ClaudeClient] = None,
        retriever: Optional[Retriever] = None,
    ) -> None:
        self.client = client or ClaudeClient()
        self.retriever = retriever or Retriever()
        self.prompts = get_prompt_manager()
        self.system = self.prompts.load("system")

    # ------------------------------------------------------------------ #
    def analyze(self, jd_text: str) -> JDAnalysis:
        """Run the full analysis."""
        log.info("Retrieving profile context for JD…")
        context = self.retriever.context_for(jd_text, k=8)

        log.info("Calling Claude for JD analysis…")
        prompt = self.prompts.render("jd_analyzer", context=context, jd=jd_text)
        data = self.client.complete_json(prompt, system=self.system)

        analysis = self._coerce(data)

        # local sanity score
        analysis.local_cosine_match = _tfidf_cosine(jd_text, context)
        return analysis

    # ------------------------------------------------------------------ #
    @staticmethod
    def _coerce(data: Dict[str, Any]) -> JDAnalysis:
        def _as_list(v) -> List:
            if isinstance(v, list):
                return v
            if v is None:
                return []
            return [v]

        def _as_int(v, default=0) -> int:
            try:
                return int(round(float(v)))
            except (TypeError, ValueError):
                return default

        return JDAnalysis(
            job_title=str(data.get("job_title", "") or ""),
            company=str(data.get("company", "") or ""),
            seniority=str(data.get("seniority", "unknown") or "unknown"),
            required_skills=[str(s) for s in _as_list(data.get("required_skills"))],
            preferred_skills=[str(s) for s in _as_list(data.get("preferred_skills"))],
            responsibilities=[str(s) for s in _as_list(data.get("responsibilities"))],
            ats_keywords=[str(s) for s in _as_list(data.get("ats_keywords"))],
            candidate_matched_skills=[str(s) for s in _as_list(data.get("candidate_matched_skills"))],
            candidate_missing_skills=[str(s) for s in _as_list(data.get("candidate_missing_skills"))],
            strengths=[str(s) for s in _as_list(data.get("strengths"))],
            weaknesses=[str(s) for s in _as_list(data.get("weaknesses"))],
            compatibility_score=_as_int(data.get("compatibility_score")),
            ats_score=_as_int(data.get("ats_score")),
            improvement_suggestions=[str(s) for s in _as_list(data.get("improvement_suggestions"))],
            summary=str(data.get("summary", "") or ""),
        )


# --------------------------------------------------------------------------- #
def _tfidf_cosine(jd: str, profile_context: str) -> float:
    """Cheap TF-IDF cosine similarity between JD and the retrieved profile.

    This is *independent* of the LLM and gives a second opinion on overall fit.
    Returns 0.0..1.0.
    """
    if not jd.strip() or not profile_context.strip():
        return 0.0
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=4000)
    try:
        m = vec.fit_transform([jd, profile_context])
    except ValueError:
        return 0.0
    sim = cosine_similarity(m[0:1], m[1:2])[0, 0]
    return float(np.clip(sim, 0.0, 1.0))
