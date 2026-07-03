"""Career AI Assistant - entry point.

Usage:
    python main.py                 # interactive Rich menu
    python main.py ingest          # ingest everything under data/
    python main.py analyze <file>  # analyze a JD file
    python main.py tailor <file>   # tailor resume against a JD file
    python main.py ask "<q>"       # one-shot Q&A
    python main.py status          # show vector-store + config status
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from rich.console import Console

from src.cli.menu import App
from src.ingestion.ingest import Ingestor
from src.interview.prep import CareerChat
from src.rag.vectorstore import VectorStore
from src.resume.pdf_export import export_resume_to_pdf
from src.resume.tailor import ResumeTailor
from src.scoring.jd_analyzer import JDAnalyzer
from src.utils.config import get_settings
from src.utils.logger import get_logger

console = Console()
log = get_logger("main")


def cmd_ingest() -> None:
    res = Ingestor().ingest_all()
    console.print(f"[green]Ingest complete:[/] {res}")


def cmd_status() -> None:
    s = get_settings()
    vs = VectorStore()
    console.print({
        "model": s.CLAUDE_MODEL,
        "embedding": s.EMBEDDING_MODEL,
        "chroma_dir": str(s.chroma_path),
        "chunks_stored": vs.count(),
        "data_dir": str(s.data_path),
        "db_path": str(s.db_full_path),
        "api_key_set": bool(s.ANTHROPIC_API_KEY and not s.ANTHROPIC_API_KEY.startswith("sk-ant-replace")),
    })


def cmd_analyze(jd_path: str) -> None:
    jd = Path(jd_path).read_text(encoding="utf-8")
    result = JDAnalyzer().analyze(jd)
    console.print_json(json.dumps(result.to_dict(), indent=2))


def cmd_tailor(jd_path: str) -> None:
    jd = Path(jd_path).read_text(encoding="utf-8")
    tailored = ResumeTailor().tailor(jd)
    out_dir = get_settings().project_root / "exports"
    out_dir.mkdir(exist_ok=True)
    json_path = out_dir / "resume_tailored.json"
    pdf_path = out_dir / "resume_tailored.pdf"
    json_path.write_text(json.dumps(tailored.to_dict(), indent=2), encoding="utf-8")
    export_resume_to_pdf(tailored, pdf_path)
    console.print(f"[green]JSON:[/] {json_path}")
    console.print(f"[green]PDF :[/] {pdf_path}")


def cmd_ask(question: str) -> None:
    chat = CareerChat()
    console.print(chat.ask(question))


def main() -> int:
    args = sys.argv[1:]
    if not args:
        App().run()
        return 0
    cmd = args[0]
    try:
        if cmd == "ingest":
            cmd_ingest()
        elif cmd == "status":
            cmd_status()
        elif cmd == "analyze" and len(args) >= 2:
            cmd_analyze(args[1])
        elif cmd == "tailor" and len(args) >= 2:
            cmd_tailor(args[1])
        elif cmd == "ask" and len(args) >= 2:
            cmd_ask(" ".join(args[1:]))
        else:
            console.print(__doc__)
            return 1
    except RuntimeError as exc:
        console.print(f"[red]Configuration error:[/] {exc}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
