# Electric Circuits Study Assistant

Offline document Q&A assistant built with [Microsoft Foundry Local](https://learn.microsoft.com/en-us/azure/foundry-local/what-is-foundry-local),
following the plan's Phase 2 (Weeks 3–4): data ingestion, a SQLite-backed
retrieval pipeline, and LLM integration. Answers questions grounded in
*Electric Circuits*, 10th Edition (Nilsson & Riedel).

## How it works

1. **`ingest.py`** — reads every `.txt` and `.pdf` file in `documents/`,
   splits each into fixed-size character chunks (with overlap), embeds
   chunks in batches with a local embedding model, and stores chunk text
   + embedding vectors in `knowledge_base.db` (SQLite).
2. **`retrieve.py`** — given a query, embeds it and computes cosine
   similarity against every stored chunk to find the most relevant ones,
   dropping anything below a relevance threshold.
3. **`main.py`** — CLI front end. Loads the embedding and chat models,
   takes a question, retrieves the top matching chunks, and streams a
   grounded answer from the local LLM.
4. **`app.py`** — Streamlit web UI front end, using the same retrieval
   and grounding logic as `main.py`. See "Web UI" below.

## Setup

```powershell
# from the project folder
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -r requirements.txt
```

## Run

```powershell
# 1. Build the knowledge base (run once, or again after editing documents/)
python ingest.py

# 2. Ask questions via the CLI
python main.py

# 2b. Or see retrieval scores for every question (useful for tuning
#     MIN_SCORE or diagnosing bad answers):
python main.py --debug

# 2c. Or use the web UI instead:
streamlit run app.py
```

Run these as scripts (`python main.py`), not by pasting code into a bare
`python` REPL — a REPL starts with an empty namespace, so functions
defined in the file (like `main()`) aren't available unless you `import`
the module or run the file directly.

## Web UI (`app.py`)

`streamlit run app.py` opens a chat-style web page instead of the
terminal loop. It reuses the exact same `db.py`/`retrieve.py` retrieval
and the same grounding rules as `main.py` — this is just a different
front end, not a different pipeline.

- Models load once per server process (`st.cache_resource`), not on
  every question — first load is slow (downloads + loads both models),
  subsequent questions are fast.
- The sidebar has a "Show retrieval debug info" toggle — equivalent to
  `main.py --debug`, but per-answer and collapsible, so you can inspect
  scores and the full context sent to the model without cluttering the
  chat.
- Conversation history persists for the browser session (Streamlit
  session state) but isn't saved to disk — refreshing the page clears it.

## Adding your own documents

Drop `.txt` or `.pdf` files into `documents/`, then re-run `python ingest.py`
to rebuild the knowledge base (it clears and re-populates the table each
time). PDFs are read via their text layer (`pypdf`) — a scanned PDF with
no embedded text will yield no text for that file, since that would need
OCR first.

A full textbook (e.g. Nilsson & Riedel, *Electric Circuits*, 10th ed.) is
hundreds of pages, so ingestion will produce a lot of chunks and take a
while to embed — that's expected, not a bug. Two things worth knowing:

- **Chunking is character-based, not paragraph-aware.** `ingest.py` cuts
  every `CHUNK_SIZE` (500) characters with `CHUNK_OVERLAP` (100)
  characters of overlap, regardless of sentence or paragraph boundaries.
  For a technical textbook with equations and worked examples, a chunk
  can end mid-formula or mid-derivation. If retrieval quality on
  technical passages seems off, that's the first thing to revisit.
- **`retrieve.py`'s `min_score` threshold** was tuned by observation
  against real queries — see TESTING.md. With a different knowledge base
  the right threshold may differ.

## Known failure mode: repetition loops

Without any generation limits, small local chat models can finish a
normal answer and then continue generating anyway, degrading into a loop
that regurgitates (a garbled, re-hallucinated version of) the context
text over and over until cut off. Both `main.py` and `app.py` cap this
via `chat_client.settings.max_tokens = 500` (set once, right after the
chat model loads — this is how the Python SDK exposes generation
settings, not as call-time keyword arguments). This bounds the damage; it
doesn't fix why the model loops in the first place.

## Verifying answers are actually grounded

Small local models don't always obey "answer only from context" — even
with a strengthened system prompt, `phi-3.5-mini` was observed answering
partly from its own training data on a question the book only partially
covered, while explicitly saying out loud that it was doing so. Switching
to `phi-4-mini` measurably reduced (but didn't eliminate) this.

There's no way to make a small model guarantee compliance through
prompting alone, so the reliable check is to read what it actually saw.
In the CLI, run with `--debug`; in the web UI, use the sidebar toggle.
Both print/show the full context text handed to the model for that
question. Compare each specific claim in the answer against that block —
if a claim isn't in there, it didn't come from the book, regardless of
what the system prompt asked for. This is a manual check, not automatic,
because pattern-matching the answer against the context to auto-flag
inferences would be unreliable and give false confidence.

See `TESTING.md` for a documented set of test queries and results.

## Models used

- Embeddings: `qwen3-embedding-0.6b`
- Chat: `phi-4-mini`

Both run comfortably on a laptop CPU. The chat model went `qwen2.5-0.5b`
→ `phi-3.5-mini` → `phi-4-mini` over the course of testing, each swap
chasing better instruction-following.

## Next steps (Plan Phase 3)

- ✅ Test results documented — see `TESTING.md`.
- Code cleanup pass before final submission (remove any leftover
  scratch code, double-check comments are accurate).
- Final presentation prep: problem statement, live demo (including one
  answerable and one "I don't know" question), lessons learned.
