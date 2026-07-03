"""Rank multiple jobs against the candidate's profile.

Two scoring sources are blended:
  - LLM verdicts via prompts/job_matcher.txt (Claude returns rich JSON)
  - Local TF-IDF cosine between each JD and the retrieved profile context
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
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
class JobInput:
    title: str = ""
    company: str = ""
    jd_text: str = ""

    def header(self, idx: int) -> str:
        return f"[JOB {idx}] title={self.title} | company={self.company}"


@dataclass
class JobRanking:
    job_index: int = 0
    job_title: str = ""
    company: str = ""
    overall_score: float = 0.0
    skill_overlap_score: float = 0.0
    domain_alignment_score: float = 0.0
    ml_relevance_score: float = 0.0
    battery_relevance_score: float = 0.0
    research_alignment_score: float = 0.0
    growth_potential_score: float = 0.0
    verdict: str = ""
    one_line_reason: str = ""
    local_cosine: float = 0.0


@dataclass
class MatchResult:
    rankings: List[JobRanking] = field(default_factory=list)
    top_pick_index: int = 0
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rankings": [r.__dict__ for r in self.rankings],
            "top_pick_index": self.top_pick_index,
            "summary": self.summary,
        }


class JobMatcher:
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
    def rank(self, jobs: List[JobInput]) -> MatchResult:
        if not jobs:
            return MatchResult()

        # Build profile context using a concatenation of all JD texts as query
        combined_query = "\n\n".join(j.jd_text[:600] for j in jobs)
        context = self.retriever.context_for(combined_query, k=10)

        jobs_block = "\n\n".join(
            f"{j.header(i+1)}\n{j.jd_text}" for i, j in enumerate(jobs)
        )
        prompt = self.prompts.render(
            "job_matcher", context=context, jobs_block=jobs_block
        )
        data = self.client.complete_json(prompt, system=self.system)

        rankings = self._coerce_rankings(data, jobs)

        # local TF-IDF cosines
        for r in rankings:
            jd = jobs[r.job_index - 1].jd_text
            r.local_cosine = _cosine(jd, context)

        # If LLM left rankings empty (rare), fall back to local cosine only.
        if not rankings:
            for i, j in enumerate(jobs, 1):
                rankings.append(
                    JobRanking(
                        job_index=i,
                        job_title=j.title,
                        company=j.company,
                        overall_score=_cosine(j.jd_text, context) * 100,
                        local_cosine=_cosine(j.jd_text, context),
                        verdict="Apply if time",
                        one_line_reason="LLM ranking unavailable; fell back to local cosine.",
                    )
                )

        rankings.sort(key=lambda r: r.overall_score, reverse=True)
        top = rankings[0].job_index if rankings else 0
        return MatchResult(
            rankings=rankings,
            top_pick_index=int(data.get("top_pick_index", top) or top),
            summary=str(data.get("summary", "") or ""),
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _coerce_rankings(data: Dict[str, Any], jobs: List[JobInput]) -> List[JobRanking]:
        out: List[JobRanking] = []
        for row in data.get("ranked_jobs", []) or []:
            try:
                idx = int(row.get("job_index", 0))
            except (TypeError, ValueError):
                continue
            if idx < 1 or idx > len(jobs):
                continue

            def _num(key, default=0.0):
                try:
                    return float(row.get(key, default))
                except (TypeError, ValueError):
                    return default

            out.append(
                JobRanking(
                    job_index=idx,
                    job_title=str(row.get("job_title", jobs[idx - 1].title)),
                    company=str(row.get("company", jobs[idx - 1].company)),
                    overall_score=_num("overall_score"),
                    skill_overlap_score=_num("skill_overlap_score"),
                    domain_alignment_score=_num("domain_alignment_score"),
                    ml_relevance_score=_num("ml_relevance_score"),
                    battery_relevance_score=_num("battery_relevance_score"),
                    research_alignment_score=_num("research_alignment_score"),
                    growth_potential_score=_num("growth_potential_score"),
                    verdict=str(row.get("verdict", "")),
                    one_line_reason=str(row.get("one_line_reason", "")),
                )
            )
        return out


# --------------------------------------------------------------------------- #
def _cosine(a: str, b: str) -> float:
    if not a.strip() or not b.strip():
        return 0.0
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=4000)
    try:
        m = vec.fit_transform([a, b])
    except ValueError:
        return 0.0
    return float(np.clip(cosine_similarity(m[0:1], m[1:2])[0, 0], 0.0, 1.0))
