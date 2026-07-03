"""Interactive Rich-based menu for the Career AI Assistant."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from src.ingestion.ingest import Ingestor
from src.interview.prep import (
    CareerChat,
    CoverLetterGenerator,
    InterviewPrep,
    RecruiterOutreach,
    SkillGapAnalyzer,
)
from src.rag.retriever import Retriever
from src.rag.vectorstore import VectorStore
from src.resume.pdf_export import export_resume_to_pdf
from src.resume.tailor import ResumeTailor
from src.scoring.jd_analyzer import JDAnalyzer
from src.scoring.matcher import JobInput, JobMatcher
from src.utils.config import get_settings
from src.utils.database import JobRecord, JobTracker
from src.utils.logger import get_logger

log = get_logger(__name__)


BANNER = r"""
[bold cyan] ____                           _    ___      _    ____ ____ ___ ____ _____ [/]
[bold cyan]/ ___|__ _ _ __ ___  ___ _ __  / \  |_ _|    / \  / ___/ ___|_ _/ ___|_   _|[/]
[bold cyan]| |   / _` | '__/ _ \/ _ \ '__|/ _ \  | |    / _ \ \___ \___ \| |\___ \ | |  [/]
[bold cyan]| |__| (_| | | |  __/  __/ |  / ___ \ | |   / ___ \ ___) |__) | | ___) || |  [/]
[bold cyan]\____\__,_|_|  \___|\___|_| /_/   \_\___| /_/   \_\____/____/___|____/ |_|  [/]

[dim]Your local AI-powered job-search assistant — RAG + Claude[/]
"""


class App:
    def __init__(self) -> None:
        self.console = Console()
        self.settings = get_settings()
        self.tracker = JobTracker()
        self.exports_dir = self.settings.project_root / "exports"
        self.exports_dir.mkdir(exist_ok=True)
        self._vectorstore: Optional[VectorStore] = None
        self._retriever: Optional[Retriever] = None

    # -- lazy singletons so we don't load the embedder before we need it -- #
    @property
    def vectorstore(self) -> VectorStore:
        if self._vectorstore is None:
            self._vectorstore = VectorStore()
        return self._vectorstore

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            self._retriever = Retriever(self.vectorstore)
        return self._retriever

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        self.console.print(BANNER)
        while True:
            self._show_menu()
            choice = Prompt.ask(
                "[bold green]Choose[/]",
                choices=[str(i) for i in range(0, 11)],
                default="0",
            )
            try:
                if choice == "1":
                    self._ingest()
                elif choice == "2":
                    self._chat_qa()
                elif choice == "3":
                    self._analyze_jd()
                elif choice == "4":
                    self._tailor_resume()
                elif choice == "5":
                    self._interview_prep()
                elif choice == "6":
                    self._compare_jobs()
                elif choice == "7":
                    self._cover_letter()
                elif choice == "8":
                    self._recruiter_msg()
                elif choice == "9":
                    self._skill_gap()
                elif choice == "10":
                    self._tracker_view()
                elif choice == "0":
                    self.console.print("[yellow]Goodbye![/]")
                    break
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Interrupted.[/]")
            except Exception as exc:
                log.exception("Action failed")
                self.console.print(f"[red]Error:[/] {exc}")

    # ------------------------------------------------------------------ #
    def _show_menu(self) -> None:
        try:
            count = self.vectorstore.count()
        except Exception:
            count = -1
        meta = (
            f"[dim]vector store chunks: {count if count >= 0 else 'n/a'} | "
            f"model: {self.settings.CLAUDE_MODEL}[/]"
        )
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_row("[bold] 1.[/]", "Ingest documents (resume, thesis, papers, JDs...)")
        table.add_row("[bold] 2.[/]", "Ask a career question (RAG over your profile)")
        table.add_row("[bold] 3.[/]", "Analyze a job description")
        table.add_row("[bold] 4.[/]", "Tailor your resume to a JD (+ PDF export)")
        table.add_row("[bold] 5.[/]", "Interview prep / mock interview")
        table.add_row("[bold] 6.[/]", "Compare & rank multiple jobs")
        table.add_row("[bold] 7.[/]", "Generate a cover letter")
        table.add_row("[bold] 8.[/]", "Recruiter outreach messages")
        table.add_row("[bold] 9.[/]", "Skill-gap analysis")
        table.add_row("[bold]10.[/]", "Job application tracker")
        table.add_row("[bold] 0.[/]", "Exit")
        self.console.print(Panel(table, title="Career AI Assistant", subtitle=meta))

    # ------------------------------------------------------------------ #
    # Feature handlers
    # ------------------------------------------------------------------ #
    def _ingest(self) -> None:
        self.console.print(Panel.fit("Ingest documents", style="cyan"))
        mode = Prompt.ask(
            "Ingest: [a]ll data/ folder, [f]ile, [d]irectory, [r]eset store",
            choices=["a", "f", "d", "r"],
            default="a",
        )
        ing = Ingestor(self.vectorstore)
        if mode == "a":
            results = ing.ingest_all()
            tbl = Table(title="Ingestion summary")
            tbl.add_column("Category")
            tbl.add_column("Chunks", justify="right")
            for k, v in results.items():
                tbl.add_row(k, str(v))
            self.console.print(tbl)
        elif mode == "f":
            path = Prompt.ask("Path to file")
            cat = Prompt.ask("Category tag (e.g. resume, thesis, paper)", default="other")
            n = ing.ingest_path(path, category=cat)
            self.console.print(f"[green]Stored {n} chunks.[/]")
        elif mode == "d":
            path = Prompt.ask("Directory path")
            cat = Prompt.ask("Category tag", default="other")
            n = ing.ingest_directory(path, category=cat)
            self.console.print(f"[green]Stored {n} chunks.[/]")
        elif mode == "r":
            if Confirm.ask("[red]Really wipe the vector store?[/]", default=False):
                ing.reset()
                self._vectorstore = None
                self._retriever = None
                self.console.print("[yellow]Vector store cleared.[/]")

    # ------------------------------------------------------------------ #
    def _chat_qa(self) -> None:
        self.console.print(Panel.fit("Career Q&A (type 'exit' to quit)", style="cyan"))
        chat = CareerChat(retriever=self.retriever, tracker=self.tracker)
        while True:
            q = Prompt.ask("[bold blue]You[/]")
            if q.strip().lower() in {"exit", "quit", "q"}:
                break
            with self.console.status("[cyan]Thinking…[/]"):
                ans = chat.ask(q)
            self.console.print(Panel(Markdown(ans), title="Claude", border_style="green"))

    # ------------------------------------------------------------------ #
    def _read_jd(self) -> str:
        src = Prompt.ask("Job description: [p]aste, [f]ile", choices=["p", "f"], default="p")
        if src == "f":
            path = Prompt.ask("Path to JD file")
            return Path(path).read_text(encoding="utf-8", errors="ignore")
        self.console.print("[dim]Paste JD, then a blank line + Ctrl-D (or just type END on its own line):[/]")
        lines: list[str] = []
        try:
            while True:
                ln = input()
                if ln.strip() == "END":
                    break
                lines.append(ln)
        except EOFError:
            pass
        return "\n".join(lines).strip()

    def _analyze_jd(self) -> None:
        jd = self._read_jd()
        if not jd:
            self.console.print("[red]Empty JD.[/]")
            return
        with self.console.status("[cyan]Analyzing JD…[/]"):
            result = JDAnalyzer(retriever=self.retriever).analyze(jd)

        self._render_jd_analysis(result.to_dict(), include_cosine=True)

        if Confirm.ask("Save this job to the tracker?", default=True):
            job = JobRecord(
                title=result.job_title,
                company=result.company,
                jd_text=jd,
                status="analyzed",
                score=float(result.compatibility_score),
                ats_score=float(result.ats_score),
                keywords=result.ats_keywords,
                missing_skills=result.candidate_missing_skills,
            )
            jid = self.tracker.add_job(job)
            self.console.print(f"[green]Saved as job #{jid}.[/]")

    def _render_jd_analysis(self, d: dict, include_cosine: bool = False) -> None:
        meta = Table(show_header=False, box=None)
        meta.add_row("Title", d.get("job_title", ""))
        meta.add_row("Company", d.get("company", ""))
        meta.add_row("Seniority", d.get("seniority", ""))
        meta.add_row("Compatibility", f"{d.get('compatibility_score', 0)} / 100")
        meta.add_row("ATS score", f"{d.get('ats_score', 0)} / 100")
        if include_cosine:
            meta.add_row("Local TF-IDF cosine", f"{d.get('local_cosine_match', 0):.3f}")
        self.console.print(Panel(meta, title="Overview", border_style="cyan"))

        for label, key in [
            ("Required skills", "required_skills"),
            ("Preferred skills", "preferred_skills"),
            ("Matched skills", "candidate_matched_skills"),
            ("Missing skills", "candidate_missing_skills"),
            ("ATS keywords", "ats_keywords"),
        ]:
            vals = d.get(key) or []
            if vals:
                self.console.print(Panel(", ".join(vals), title=label, border_style="magenta"))

        for label, key in [
            ("Strengths", "strengths"),
            ("Weaknesses", "weaknesses"),
            ("Improvement suggestions", "improvement_suggestions"),
            ("Responsibilities", "responsibilities"),
        ]:
            vals = d.get(key) or []
            if vals:
                body = "\n".join(f"• {v}" for v in vals)
                self.console.print(Panel(body, title=label, border_style="yellow"))

        if d.get("summary"):
            self.console.print(Panel(d["summary"], title="Summary", border_style="green"))

    # ------------------------------------------------------------------ #
    def _tailor_resume(self) -> None:
        jd = self._read_jd()
        if not jd:
            self.console.print("[red]Empty JD.[/]")
            return
        with self.console.status("[cyan]Analyzing JD and tailoring resume…[/]"):
            tailor = ResumeTailor(retriever=self.retriever)
            tailored = tailor.tailor(jd)

        self.console.print(Panel(tailored.summary, title="Summary", border_style="green"))
        self.console.print(Panel(tailored.headline, title="Headline", border_style="green"))
        self.console.print(Panel(", ".join(tailored.core_skills), title="Core skills", border_style="cyan"))

        tbl = Table(title="Tailored bullets")
        tbl.add_column("#", justify="right")
        tbl.add_column("Section")
        tbl.add_column("Title")
        tbl.add_column("Bullet")
        for i, b in enumerate(tailored.bullets, 1):
            tbl.add_row(str(i), b.section, b.title, b.rewritten_bullet)
        self.console.print(tbl)

        if Confirm.ask("Export tailored resume to PDF?", default=True):
            name = Prompt.ask("Your name (as it should appear on the resume)", default="Your Name")
            contact = Prompt.ask(
                "Contact line",
                default="you@example.com  |  linkedin.com/in/you  |  github.com/you",
            )
            out = self.exports_dir / f"resume_tailored_{tailored.headline[:30].replace(' ', '_') or 'job'}.pdf"
            export_resume_to_pdf(tailored, out, candidate_name=name, contact_line=contact)
            self.console.print(f"[green]PDF written:[/] {out}")

        if Confirm.ask("Save full tailored JSON to exports/?", default=False):
            out = self.exports_dir / "resume_tailored.json"
            out.write_text(json.dumps(tailored.to_dict(), indent=2), encoding="utf-8")
            self.console.print(f"[green]Saved:[/] {out}")

    # ------------------------------------------------------------------ #
    def _interview_prep(self) -> None:
        jd = self._read_jd()
        if not jd:
            return
        mode = Prompt.ask("Mode: [g]enerate pack, [m]ock interview", choices=["g", "m"], default="g")
        prep = InterviewPrep(retriever=self.retriever)
        if mode == "g":
            with self.console.status("[cyan]Generating interview prep…[/]"):
                pack = prep.generate(jd)
            self._render_interview_pack(pack)
            if Confirm.ask("Save pack JSON?", default=False):
                out = self.exports_dir / "interview_pack.json"
                out.write_text(json.dumps(pack.to_dict(), indent=2), encoding="utf-8")
                self.console.print(f"[green]Saved:[/] {out}")
        else:
            turns = IntPrompt.ask("How many questions?", default=5)

            def ask_user(q: str) -> str:
                self.console.print(Panel(q, title="Interviewer", border_style="cyan"))
                return Prompt.ask("[bold]Your answer[/]")

            def show_feedback(fb: str) -> None:
                self.console.print(Panel(Markdown(fb), title="Feedback", border_style="green"))

            prep.mock_interview(jd, turns=turns, on_question=ask_user, on_feedback=show_feedback)

    def _render_interview_pack(self, pack) -> None:
        def _block(title, items, fmt):
            if not items:
                return
            body = "\n\n".join(fmt(x) for x in items)
            self.console.print(Panel(body, title=title, border_style="cyan"))

        _block(
            "Technical Questions",
            pack.technical_questions,
            lambda x: f"Q: {x.get('question','')}\n• why: {x.get('why_asked','')}\n• outline: {x.get('ideal_answer_outline','')}",
        )
        _block("HR Questions", pack.hr_questions, lambda x: f"Q: {x.get('question','')}\n• outline: {x.get('ideal_answer_outline','')}")
        _block("Project Questions", pack.project_questions, lambda x: f"Q ({x.get('project_referenced','')}): {x.get('question','')}\n• outline: {x.get('ideal_answer_outline','')}")
        _block("Domain Questions", pack.domain_specific_questions, lambda x: f"[{x.get('category','')}] Q: {x.get('question','')}\n• outline: {x.get('ideal_answer_outline','')}")
        _block("STAR Stories", pack.behavioural_STAR_examples,
               lambda x: f"[{x.get('competency','')}] grounded in {x.get('grounded_in','')}\n  S: {x.get('situation','')}\n  T: {x.get('task','')}\n  A: {x.get('action','')}\n  R: {x.get('result','')}")
        if pack.questions_for_interviewer:
            self.console.print(Panel("\n".join(f"• {q}" for q in pack.questions_for_interviewer), title="Questions for interviewer", border_style="magenta"))

    # ------------------------------------------------------------------ #
    def _compare_jobs(self) -> None:
        self.console.print(Panel.fit("Compare jobs", style="cyan"))
        source = Prompt.ask("Use [t]racker saved jobs or enter [n]ew ones?", choices=["t", "n"], default="t")
        jobs: List[JobInput] = []
        if source == "t":
            saved = self.tracker.list_jobs()
            if not saved:
                self.console.print("[red]No saved jobs.[/]")
                return
            tbl = Table(title="Saved jobs")
            tbl.add_column("#", justify="right")
            tbl.add_column("Title")
            tbl.add_column("Company")
            tbl.add_column("Status")
            for j in saved:
                tbl.add_row(str(j.id), j.title, j.company, j.status)
            self.console.print(tbl)
            ids_raw = Prompt.ask("Enter the IDs to compare, comma separated")
            try:
                ids = [int(x.strip()) for x in ids_raw.split(",") if x.strip()]
            except ValueError:
                self.console.print("[red]Invalid IDs.[/]")
                return
            for jid in ids:
                rec = self.tracker.get_job(jid)
                if rec:
                    jobs.append(JobInput(title=rec.title, company=rec.company, jd_text=rec.jd_text))
        else:
            n = IntPrompt.ask("How many jobs to enter?", default=2)
            for i in range(n):
                self.console.print(f"[bold]Job {i+1}[/]")
                title = Prompt.ask("  Title", default="")
                company = Prompt.ask("  Company", default="")
                jd = self._read_jd()
                jobs.append(JobInput(title=title, company=company, jd_text=jd))

        if not jobs:
            return

        with self.console.status("[cyan]Ranking jobs…[/]"):
            result = JobMatcher(retriever=self.retriever).rank(jobs)

        tbl = Table(title="Job ranking")
        tbl.add_column("Idx", justify="right")
        tbl.add_column("Title")
        tbl.add_column("Company")
        tbl.add_column("Overall", justify="right")
        tbl.add_column("Skill", justify="right")
        tbl.add_column("Domain", justify="right")
        tbl.add_column("Verdict")
        tbl.add_column("Reason")
        for r in result.rankings:
            tbl.add_row(
                str(r.job_index), r.job_title, r.company,
                f"{r.overall_score:.0f}", f"{r.skill_overlap_score:.0f}", f"{r.domain_alignment_score:.0f}",
                r.verdict, r.one_line_reason,
            )
        self.console.print(tbl)
        if result.summary:
            self.console.print(Panel(result.summary, title="Summary", border_style="green"))

    # ------------------------------------------------------------------ #
    def _cover_letter(self) -> None:
        jd = self._read_jd()
        if not jd:
            return
        with self.console.status("[cyan]Drafting cover letter…[/]"):
            letter = CoverLetterGenerator(retriever=self.retriever).generate(jd)
        self.console.print(Panel(letter.as_text(), title="Cover letter", border_style="green"))
        if Confirm.ask("Save to exports/cover_letter.txt?", default=True):
            out = self.exports_dir / "cover_letter.txt"
            out.write_text(letter.as_text(), encoding="utf-8")
            self.console.print(f"[green]Saved:[/] {out}")

    # ------------------------------------------------------------------ #
    def _recruiter_msg(self) -> None:
        jd = self._read_jd()
        if not jd:
            return
        with self.console.status("[cyan]Generating outreach messages…[/]"):
            msgs = RecruiterOutreach(retriever=self.retriever).generate(jd)
        self.console.print(Panel(msgs.short_inmail, title="Short InMail (≤300c)", border_style="cyan"))
        self.console.print(Panel(msgs.standard_inmail, title="Standard InMail (≤600c)", border_style="cyan"))
        self.console.print(Panel(msgs.long_followup_email, title="Long follow-up email", border_style="cyan"))
        if msgs.subject_lines:
            self.console.print(Panel("\n".join(f"• {s}" for s in msgs.subject_lines),
                                     title="Subject lines", border_style="magenta"))

    # ------------------------------------------------------------------ #
    def _skill_gap(self) -> None:
        self.console.print(Panel.fit("Skill-gap analysis", style="cyan"))
        target = Prompt.ask("Target role / JD (paste short description)")
        with self.console.status("[cyan]Analyzing gaps…[/]"):
            data = SkillGapAnalyzer(retriever=self.retriever).analyze(target)
        self.console.print(Panel(json.dumps(data, indent=2), title="Skill gap (JSON)", border_style="yellow"))

    # ------------------------------------------------------------------ #
    def _tracker_view(self) -> None:
        while True:
            saved = self.tracker.list_jobs()
            tbl = Table(title=f"Tracked jobs ({len(saved)})")
            tbl.add_column("#")
            tbl.add_column("Title")
            tbl.add_column("Company")
            tbl.add_column("Status")
            tbl.add_column("Score")
            tbl.add_column("ATS")
            for j in saved:
                tbl.add_row(
                    str(j.id), j.title, j.company, j.status,
                    f"{j.score:.0f}" if j.score is not None else "-",
                    f"{j.ats_score:.0f}" if j.ats_score is not None else "-",
                )
            self.console.print(tbl)
            op = Prompt.ask(
                "Action: [v]iew, [u]pdate status, [d]elete, [b]ack",
                choices=["v", "u", "d", "b"],
                default="b",
            )
            if op == "b":
                return
            if op == "v":
                jid = IntPrompt.ask("Job id")
                rec = self.tracker.get_job(jid)
                if not rec:
                    self.console.print("[red]Not found.[/]")
                    continue
                self.console.print(Panel(rec.jd_text[:4000], title=f"{rec.title} @ {rec.company}", border_style="cyan"))
            elif op == "u":
                jid = IntPrompt.ask("Job id")
                rec = self.tracker.get_job(jid)
                if not rec:
                    self.console.print("[red]Not found.[/]")
                    continue
                new_status = Prompt.ask(
                    "New status",
                    choices=["saved", "analyzed", "applied", "interview", "offer", "rejected"],
                    default=rec.status,
                )
                rec.status = new_status
                self.tracker.update_job(rec)
                self.console.print("[green]Updated.[/]")
            elif op == "d":
                jid = IntPrompt.ask("Job id")
                if Confirm.ask(f"Delete job #{jid}?", default=False):
                    self.tracker.delete_job(jid)


def main() -> None:
    try:
        App().run()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)


if __name__ == "__main__":
    main()
