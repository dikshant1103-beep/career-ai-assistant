# Career AI Assistant

A **local**, AI-powered job-search assistant for engineers / researchers.
Stores your personal documents (resume, thesis, papers, SOP, LinkedIn,
projects) in a local vector database and uses **Claude** + **RAG** to:

- Analyze job descriptions (ATS score, compatibility, missing skills)
- Tailor your resume per job (with PDF export)
- Generate cover letters and recruiter outreach messages
- Run interview prep (technical, HR, project, STAR)
- Run interactive mock interviews
- Rank multiple jobs against your profile
- Skill-gap analysis with a 30/60/90-day plan
- Answer free-form career questions over your own documents
- Track applications in a local SQLite database

Everything runs **locally on Ubuntu/Linux** — only the Claude API call goes
to the network. Your documents and embeddings stay on your machine.

---

## 1. Quickstart (Ubuntu / Linux)

```bash
# 1) Clone or copy the project
cd ~/Desktop/career_ai_assistant

# 2) Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3) Install dependencies (first run downloads PyTorch + a small embedding model)
pip install --upgrade pip
pip install -r requirements.txt

# 4) Configure your environment
cp .env.example .env
# now open .env in your editor and paste your ANTHROPIC_API_KEY (see below)

# 5) (Optional) verify everything is wired up
python main.py status

# 6) Ingest the bundled sample documents
python main.py ingest

# 7) Launch the interactive terminal app
python main.py
```

---

## 2. How to get a Claude API key

1. Go to <https://console.anthropic.com/>
2. Sign in (or sign up — Google works).
3. In the left sidebar choose **API Keys** → click **Create Key**.
4. Name it `career-ai-assistant` and copy the key (starts with `sk-ant-...`).
5. Add credit at **Billing → Plans** (you usually need a small balance to call the API).
6. Open `.env` in this project and replace the placeholder line:
   ```
   ANTHROPIC_API_KEY=sk-ant-replace-me
   ```
   with your real key. Save the file.
7. Run `python main.py status` to confirm `"api_key_set": true`.

### Choosing a model
Edit `CLAUDE_MODEL` in `.env`. Reasonable defaults:
- `claude-haiku-4-5-20251001` — cheapest + fastest, fine for ingest + Q&A
- `claude-sonnet-4-6`   — **recommended default** (balanced cost/quality)
- `claude-opus-4-7`     — best quality, slower + costlier

---

## 3. Folder layout

```
career_ai_assistant/
├── data/                       # YOUR documents (gitignored by default)
│   ├── resumes/                # PDF / DOCX / MD resumes
│   ├── thesis/
│   ├── papers/
│   ├── linkedin/               # exported LinkedIn / profile pages
│   ├── sop/                    # statement of purpose drafts
│   └── jobs/                   # saved JDs (PDF, TXT, MD…)
├── embeddings/                 # ChromaDB persistent store (auto-created)
├── prompts/                    # editable .txt prompt templates
├── src/
│   ├── ingestion/              # loaders, chunker, ingest pipeline
│   ├── rag/                    # embedder + vector store + retriever
│   ├── llm/                    # Claude client + prompt manager
│   ├── scoring/                # JD analyzer + multi-job matcher
│   ├── resume/                 # tailoring + PDF export
│   ├── interview/              # interview prep, cover letter, outreach…
│   ├── utils/                  # config, logging, SQLite tracker
│   └── cli/                    # Rich-based interactive menu
├── tests/                      # pytest tests
├── exports/                    # outputs (PDF, JSON, .txt) — created on demand
├── requirements.txt
├── .env.example
├── README.md
└── main.py                     # entry point
```

---

## 4. Putting your real documents in

Drop your files in the appropriate folder under `data/`:

| Folder           | Put here                                            |
|------------------|-----------------------------------------------------|
| `data/resumes/`  | `MyResume.pdf`, `MyResume.docx`                     |
| `data/thesis/`   | `Thesis.pdf` or chapter `.md` files                 |
| `data/papers/`   | research papers (PDF / MD)                          |
| `data/linkedin/` | exported LinkedIn profile (you can paste as `.md`)  |
| `data/sop/`      | SOP drafts (MD / TXT / DOCX)                        |
| `data/jobs/`     | reference JDs you want indexed                      |

Supported formats: **PDF**, **DOCX**, **MD**, **TXT**.

Then run:

```bash
python main.py ingest
```

Each top-level folder name becomes the chunk's `category` metadata (so the
retriever can filter, e.g. only-resume chunks).

If you ever want to **re-ingest cleanly**, launch the menu and pick
`1. Ingest documents → r. reset store`.

---

## 5. CLI usage

Interactive menu (recommended):

```bash
python main.py
```

You will get a menu like:

```
 1. Ingest documents
 2. Ask a career question
 3. Analyze a job description
 4. Tailor your resume to a JD (+ PDF)
 5. Interview prep / mock interview
 6. Compare & rank multiple jobs
 7. Generate a cover letter
 8. Recruiter outreach messages
 9. Skill-gap analysis
10. Job application tracker
 0. Exit
```

One-shot commands (no menu):

```bash
python main.py ingest
python main.py status
python main.py analyze data/jobs/sample_jd_battery_ml.txt
python main.py tailor  data/jobs/sample_jd_battery_ml.txt
python main.py ask "How does my Mamba RUL project map to a BMS engineer role?"
```

When the CLI asks for a job description you can either:
- paste it (end with a line that just says `END`), or
- point at a file.

---

## 6. Example end-to-end run (sanity check)

After installation:

```bash
python main.py status            # API key set? chunks=0
python main.py ingest            # ingest sample documents
python main.py status            # chunks > 0 now
python main.py analyze data/jobs/sample_jd_battery_ml.txt
python main.py tailor  data/jobs/sample_jd_battery_ml.txt
# -> exports/resume_tailored.pdf and exports/resume_tailored.json
```

---

## 7. Configuration cheat-sheet (`.env`)

| Variable                 | Default                                     | Notes                                    |
|--------------------------|---------------------------------------------|------------------------------------------|
| `ANTHROPIC_API_KEY`      | _(required)_                                | from console.anthropic.com               |
| `CLAUDE_MODEL`           | `claude-sonnet-4-6`                         | see model list above                     |
| `CLAUDE_MAX_TOKENS`      | `4096`                                      | per Claude response                      |
| `CLAUDE_TEMPERATURE`     | `0.3`                                       | 0 = deterministic, 1 = creative          |
| `EMBEDDING_MODEL`        | `sentence-transformers/all-MiniLM-L6-v2`    | 22 MB, CPU-friendly                      |
| `EMBEDDING_DEVICE`       | `cpu`                                       | `cuda` if you have a GPU                 |
| `CHROMA_PERSIST_DIR`     | `./embeddings`                              | local vector store path                  |
| `CHROMA_COLLECTION`      | `career_profile`                            | rename for multiple profiles             |
| `CHUNK_SIZE`             | `900`                                       | characters                               |
| `CHUNK_OVERLAP`          | `120`                                       | characters                               |
| `RAG_TOP_K`              | `6`                                         | chunks retrieved per query               |
| `DB_PATH`                | `./career_assistant.db`                     | SQLite tracker DB                        |
| `LOG_LEVEL`              | `INFO`                                      | `DEBUG` for verbose                      |

---

## 8. Tests

```bash
pytest -q
```

Tests **do not call the Claude API** — the client is mocked.

---

## 9. Debugging guide

| Symptom                                                     | Likely cause                                          | Fix                                                                 |
|------------------------------------------------------------ |-------------------------------------------------------|---------------------------------------------------------------------|
| `RuntimeError: ANTHROPIC_API_KEY is missing.`               | `.env` not created / key still placeholder            | Copy `.env.example` → `.env`, paste real key                        |
| First run hangs at "Loading embedding model"                | Downloading the 22 MB MiniLM model from HuggingFace   | wait once; cached afterward at `~/.cache/huggingface`               |
| `anthropic.AuthenticationError`                             | Bad / expired API key                                 | regenerate at console.anthropic.com                                 |
| `anthropic.RateLimitError`                                  | Too many calls or low credit                          | wait a few seconds; add billing credit; switch to Haiku             |
| Empty PDF extraction                                        | Scanned PDF (image-only)                              | OCR it first or paste the text into a `.md` file                    |
| Vector store has 0 chunks after ingest                      | All your files live outside `data/`                   | move them to the right `data/<category>/` folder                    |
| "ValueError: Claude did not return valid JSON"              | model produced prose instead of JSON                  | lower `CLAUDE_TEMPERATURE` to 0.1 or retry                          |
| Slow first JD analysis                                      | Embedder loading + Chroma opening for the first time  | second call is fast (everything is cached)                          |
| `chromadb` complains about an existing collection           | Mid-upgrade incompatibility                           | menu → `1. Ingest → r. reset store`                                 |

Logs live at `./career_assistant.log` (rotated, max ~5 MB × 3 files).

---

## 10. Architecture (1-screen mental model)

```
   data/*  ──► loaders ──► chunker ──► embedder ──► ChromaDB  (./embeddings)
                                                       │
                                                       ▼
   user input  ──► CLI menu ──► retriever ──► prompt template ──► Claude API
                                                                      │
                                                                      ▼
                                                          JSON / text response
                                                                      │
                                                                      ▼
                                          render in terminal / export PDF / save SQLite
```

Each module is a single-purpose class, so swapping in (for example) a local
LLM, a Streamlit UI, or a different vector store later is straightforward.

---

## 11. Roadmap / extensibility

Designed so each future feature is a drop-in:

- **Streamlit frontend** — point a new `streamlit_app.py` at the same
  `Retriever`, `JDAnalyzer`, `ResumeTailor` classes.
- **LinkedIn scraping** — write `src/ingestion/linkedin_scraper.py` that
  returns `Document` objects and feed them into `Ingestor`.
- **Automatic job tracking** — drop new loaders for JSON job feeds + a
  daily cron that calls `JDAnalyzer` and stores `JobRecord`s.
- **Local LLM support** — implement a `LocalLLMClient` matching
  `ClaudeClient`'s `complete` / `complete_json` interface; the rest of the
  codebase doesn't need to change.
- **Multi-agent workflows** — chain analyzer → tailor → cover-letter →
  outreach in `src/agents/` (each step already returns clean dataclasses).

---

## 12. Privacy

- **No telemetry**. Nothing leaves your machine except the prompt and
  response sent to Claude's API.
- The vector store, SQLite DB, logs, and exports are all local files.
- `.gitignore` excludes `data/`, `embeddings/`, `*.db`, `*.log`, `exports/`,
  and `.env`. Review it before pushing.

---

## 13. License

For your personal use. Treat it as a starter you can extend. No warranty.
