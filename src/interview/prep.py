"""Interview prep, cover letter, recruiter outreach, skill-gap, free-form Q&A.

Each class is a thin orchestrator: retrieve context -> render prompt -> call Claude.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.llm.claude_client import ClaudeClient
from src.llm.prompt_manager import get_prompt_manager
from src.rag.retriever import Retriever
from src.utils.database import JobTracker
from src.utils.logger import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Interview prep
# --------------------------------------------------------------------------- #
@dataclass
class InterviewPack:
    technical_questions: List[Dict[str, Any]] = field(default_factory=list)
    hr_questions: List[Dict[str, Any]] = field(default_factory=list)
    project_questions: List[Dict[str, Any]] = field(default_factory=list)
    domain_specific_questions: List[Dict[str, Any]] = field(default_factory=list)
    behavioural_STAR_examples: List[Dict[str, Any]] = field(default_factory=list)
    questions_for_interviewer: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


class InterviewPrep:
    def __init__(
        self,
        client: Optional[ClaudeClient] = None,
        retriever: Optional[Retriever] = None,
    ) -> None:
        self.client = client or ClaudeClient()
        self.retriever = retriever or Retriever()
        self.prompts = get_prompt_manager()
        self.system = self.prompts.load("system")

    def generate(self, jd_text: str) -> InterviewPack:
        context = self.retriever.context_for(jd_text, k=10)
        prompt = self.prompts.render("interview_prep", context=context, jd=jd_text)
        data = self.client.complete_json(prompt, system=self.system, max_tokens=6000)
        return InterviewPack(
            technical_questions=list(data.get("technical_questions") or []),
            hr_questions=list(data.get("hr_questions") or []),
            project_questions=list(data.get("project_questions") or []),
            domain_specific_questions=list(data.get("domain_specific_questions") or []),
            behavioural_STAR_examples=list(data.get("behavioural_STAR_examples") or []),
            questions_for_interviewer=[str(s) for s in (data.get("questions_for_interviewer") or [])],
        )

    # ----- Mock interview (multi-turn) -----
    def mock_interview(
        self,
        jd_text: str,
        turns: int = 5,
        category: str = "mixed",
        on_question=None,
        on_feedback=None,
    ) -> List[Dict[str, str]]:
        """Run an interactive mock interview loop.

        ``on_question(q)`` is called with each generated question and must
        return the candidate's answer (string). ``on_feedback(fb)`` receives
        Claude's feedback for each answer.

        If the callbacks are ``None``, a transcript is still produced but
        the user's answers default to an empty string (useful for tests).
        """
        ctx = self.retriever.context_for(jd_text, k=8)
        history: List[Dict[str, str]] = []
        for i in range(1, turns + 1):
            q_prompt = (
                f"# CANDIDATE PROFILE CONTEXT\n{ctx}\n\n"
                f"# JOB DESCRIPTION\n{jd_text}\n\n"
                f"# CONVERSATION SO FAR\n{_format_history(history)}\n\n"
                f"Ask ONE next interview question (category={category}). "
                f"Return just the question text, no preamble."
            )
            question = self.client.complete(q_prompt, system=self.system, max_tokens=400).strip()
            answer = on_question(question) if on_question else ""

            fb_prompt = (
                f"# CANDIDATE PROFILE CONTEXT\n{ctx}\n\n"
                f"# JOB DESCRIPTION\n{jd_text}\n\n"
                f"Question: {question}\n"
                f"Candidate's answer: {answer if answer else '(no answer given)'}\n\n"
                "Give concise feedback (3-5 bullets): what was strong, what was weak, "
                "and a 1-line ideal-answer outline. Use the candidate's profile context."
            )
            feedback = self.client.complete(fb_prompt, system=self.system, max_tokens=600).strip()
            if on_feedback:
                on_feedback(feedback)
            history.append({"q": question, "a": answer, "feedback": feedback})
        return history


def _format_history(history: List[Dict[str, str]]) -> str:
    if not history:
        return "(none yet)"
    return "\n\n".join(
        f"Q{i+1}: {h['q']}\nA{i+1}: {h.get('a','')}\nFeedback{i+1}: {h.get('feedback','')}"
        for i, h in enumerate(history)
    )


# --------------------------------------------------------------------------- #
# Cover letter
# --------------------------------------------------------------------------- #
@dataclass
class CoverLetter:
    recipient_line: str = ""
    opening: str = ""
    body: str = ""
    closing: str = ""
    tone: str = ""
    word_count: int = 0

    def as_text(self) -> str:
        return f"{self.recipient_line}\n\n{self.opening}\n\n{self.body}\n\n{self.closing}"


class CoverLetterGenerator:
    def __init__(self, client: Optional[ClaudeClient] = None, retriever: Optional[Retriever] = None):
        self.client = client or ClaudeClient()
        self.retriever = retriever or Retriever()
        self.prompts = get_prompt_manager()
        self.system = self.prompts.load("system")

    def generate(self, jd_text: str) -> CoverLetter:
        ctx = self.retriever.context_for(jd_text, k=8)
        prompt = self.prompts.render("cover_letter", context=ctx, jd=jd_text)
        data = self.client.complete_json(prompt, system=self.system, max_tokens=2500)
        return CoverLetter(
            recipient_line=str(data.get("recipient_line", "Dear Hiring Manager,")),
            opening=str(data.get("opening", "")),
            body=str(data.get("body", "")),
            closing=str(data.get("closing", "")),
            tone=str(data.get("tone", "")),
            word_count=int(data.get("word_count", 0) or 0),
        )


# --------------------------------------------------------------------------- #
# Recruiter outreach
# --------------------------------------------------------------------------- #
@dataclass
class RecruiterMessages:
    short_inmail: str = ""
    standard_inmail: str = ""
    long_followup_email: str = ""
    subject_lines: List[str] = field(default_factory=list)


class RecruiterOutreach:
    def __init__(self, client: Optional[ClaudeClient] = None, retriever: Optional[Retriever] = None):
        self.client = client or ClaudeClient()
        self.retriever = retriever or Retriever()
        self.prompts = get_prompt_manager()
        self.system = self.prompts.load("system")

    def generate(self, jd_text: str) -> RecruiterMessages:
        ctx = self.retriever.context_for(jd_text, k=6)
        prompt = self.prompts.render("recruiter_message", context=ctx, jd=jd_text)
        data = self.client.complete_json(prompt, system=self.system, max_tokens=1500)
        return RecruiterMessages(
            short_inmail=str(data.get("short_inmail", "")),
            standard_inmail=str(data.get("standard_inmail", "")),
            long_followup_email=str(data.get("long_followup_email", "")),
            subject_lines=[str(s) for s in (data.get("subject_lines") or [])],
        )


# --------------------------------------------------------------------------- #
# Skill gap
# --------------------------------------------------------------------------- #
class SkillGapAnalyzer:
    def __init__(self, client: Optional[ClaudeClient] = None, retriever: Optional[Retriever] = None):
        self.client = client or ClaudeClient()
        self.retriever = retriever or Retriever()
        self.prompts = get_prompt_manager()
        self.system = self.prompts.load("system")

    def analyze(self, target: str) -> Dict[str, Any]:
        ctx = self.retriever.context_for(target, k=10)
        prompt = self.prompts.render("skill_gap", context=ctx, jd=target)
        return self.client.complete_json(prompt, system=self.system, max_tokens=3000)


# --------------------------------------------------------------------------- #
# Career Q&A (memory-aware free-form chat over the profile)
# --------------------------------------------------------------------------- #
class CareerChat:
    """RAG-style Q&A over the candidate's profile with optional memory."""

    FEATURE_TAG = "career_qa"

    def __init__(
        self,
        client: Optional[ClaudeClient] = None,
        retriever: Optional[Retriever] = None,
        tracker: Optional[JobTracker] = None,
    ) -> None:
        self.client = client or ClaudeClient()
        self.retriever = retriever or Retriever()
        self.tracker = tracker or JobTracker()
        self.prompts = get_prompt_manager()
        self.system = self.prompts.load("system")

    def ask(self, question: str, remember: bool = True) -> str:
        ctx = self.retriever.context_for(question, k=8)
        history = self.tracker.recent_messages(self.FEATURE_TAG, limit=4) if remember else []
        hist_block = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history)
        full_prompt = self.prompts.render("career_qa", context=ctx, question=question)
        if hist_block:
            full_prompt = f"# RECENT CONVERSATION\n{hist_block}\n\n" + full_prompt

        answer = self.client.complete(full_prompt, system=self.system, max_tokens=2000)
        if remember:
            self.tracker.log_message("user", question, self.FEATURE_TAG)
            self.tracker.log_message("assistant", answer, self.FEATURE_TAG)
        return answer
