# Dungeon Master Agent

An AI-powered Dungeon Master that runs tabletop RPG sessions using a Retrieval-Augmented Generation (RAG) pipeline. The agent draws from two separate knowledge bases — **World Lore** and **Campaign History** — plus verbatim **short-term memory** of the last few turns, to narrate vivid, consistent, and contextually grounded sessions powered by Google Gemini. Chance-based outcomes are decided by real dice rolls the DM makes through a tool call, and responses stream to the UI token by token.

---

## Architecture Overview

```
Player Action (query)
        │
        ▼
  FastAPI Backend
        │
   ┌────┴────┐
   │         │
Lore KB  Campaign KB        ← ChromaDB Vector Stores
   │         │
   └────┬────┘
        │  Concurrent Retrieval
        ▼
   Cross-Encoder Reranker   ← sentence-transformers
        │
        ▼
   Recent Turns (verbatim) ← short-term memory, last N turns
        │  (overlapping retrieved campaign chunks de-duplicated)
        ▼
   Prompt Builder
        │
        ▼
   Google Gemini LLM        ← via LangChain ChatGoogleGenerativeAI
        │  ↺ roll_dice tool for attacks / checks / saves
        ▼
   DM Response (SSE stream) + Auto-save turn to Campaign KB
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML5 / JavaScript, served by nginx (Docker) |
| Styling | CSS3 |
| API Framework | FastAPI + Uvicorn |
| LLM | Google Gemini (`gemini-3.8-flash`) via `langchain-google-genai` |
| Dice | `roll_dice` LangChain tool bound to the DM (`NdM+K` notation) |
| Embeddings | HuggingFace Sentence Transformers (CPU torch) |
| Vector Store | ChromaDB (via `langchain-chroma`) |
| Reranker | CrossEncoder (`sentence-transformers`) |
| Orchestration | LangChain LCEL + tool-call loop |
| Document Loaders | PDF / Markdown / Text chunking |
| Evaluation | Direct Gemini judge (1–5 rubrics, concurrent, `dm_evaluation` span in LangSmith) |
| Config | Pydantic Settings + `.env` |
| Deployment | Docker + Compose (backend :8000, frontend :5000) |

---

## Project Structure

```
backend/
├── app/
│   ├── api/
│   │   ├── chat.py          # POST /chat/ (JSON or SSE stream), turn count/history/recent, export/import
│   │   └── documents.py     # Lore & campaign ingestion: upload / list / delete / clear
│   ├── core/
│   │   ├── config.py        # Environment & settings (documented env aliases)
│   │   └── vectorstore.py   # ChromaDB setup, chunking, retrievers
│   └── services/
│       ├── rag_chain.py     # RAG pipeline: retrieve → rerank → recent turns → dice tools → generate → save
│       ├── reranker.py      # CrossEncoder reranking logic
│       ├── dice.py          # roll_dice tool (NdM+K parser + LangChain tool)
│       └── evaluator.py     # RAGAS evaluation pipeline (lazy-loaded)
├── Dockerfile               # CPU-torch image, HF models baked in
├── requirements-docker.txt  # Frozen deps matching the tested env
└── requirements.txt

frontend/
├── css/
│   └── styles.css           # Layout & chat UI styling
├── js/
│   ├── api.js               # Fetch + SSE wrapper (tokens, dice, error events)
│   ├── chat.js              # Message rendering, streaming bubbles, sources panel
│   ├── main.js              # Entry point, turn flow, /roll command, stream/eval toggles
│   └── sidebar.js           # Lore list + delete, save/load campaign, reset
├── index.html               # Main HTML entry point
├── Dockerfile + nginx.conf  # Static serving on :5000
docker-compose.yml           # Backend + frontend stack, volumes, healthcheck

lore/
├── emberfall_lore.md        # Starter setting: village, factions, NPCs, hooks
└── README.md                # How to add your own lore

PROJECT_STATUS.md            # Current state, changelog, and remaining work
```

---

## How It Works

1. **Document Ingestion** — Lore PDFs, Markdown, and text files are uploaded via `/documents/lore/upload`, chunked, embedded, and stored in the lore ChromaDB collection (10 MB cap, 409 on duplicates, per-file delete).

2. **Turn Execution** — When a player submits an action via `/chat/`, the system concurrently retrieves relevant chunks from both the lore and campaign history stores using a LangChain `RunnableParallel`. The last `SHORT_TERM_TURNS` turns are also loaded verbatim as short-term memory (oldest-first, per-turn char cap); retrieved campaign chunks already in that window are de-duplicated.

3. **Reranking** — Both retrieved sets are independently reranked using a CrossEncoder to surface the most contextually relevant chunks.

4. **Dice (tool calls)** — If the action has a chance-based outcome (attack, check, save…), the DM must call the `roll_dice` tool first and narrate strictly from the real result instead of inventing one. Rolls are logged server-side and streamed to the UI as 🎲 events. Players can also roll manually with `/roll 2d6+3`.

5. **Generation** — A structured prompt combining system instructions, lore context, campaign context, recent turns, and the player action is sent to Gemini. Responses stream back as SSE (`sources → dice? → token* → done`) and render live; plain JSON is returned when `stream: false`. Both paths report which recent turns were in context (`recent_turns`).

6. **Auto-Persistence** — Each completed turn (player action + DM response) is automatically saved back into the campaign vectorstore, building a living, searchable campaign history. Campaigns can be exported/imported as JSON from the sidebar.

7. **Evaluation** — Responses can be optionally evaluated by passing `"evaluate": true` in the request (or ticking *evaluate* in the UI). Three custom DM-appropriate metrics scored 1–5 by direct Gemini-judge calls (run concurrently) are computed: `dm_relevance`, `lore_consistency`, and `narrative_quality`. When a `reference` answer is provided, `context_precision` and `context_recall` are also computed. The judging is `@traceable` as `dm_evaluation`, so scores show up in LangSmith. (RAGAS was removed: even 0.4.3 hard-imports a `langchain_community` module deleted in 0.4.x.)

---

## Setup

### Option A — Docker (recommended)

1. Copy and fill in the environment file (never commit it):
   ```bash
   cp backend/.env.example backend/.env
   # set GEMINI_API_KEY in backend/.env
   ```

2. Build and start the stack:
   ```bash
   docker compose up -d --build
   docker compose ps   # backend should be (healthy)
   ```

3. Open `http://localhost:5000`, drop `lore/emberfall_lore.md` into **World Lore**, and play.

### Option B — Local dev

1. Install dependencies (a venv with the versions in `backend/requirements-docker.txt` is known-good):
   ```bash
   cd backend
   cp .env.example .env   # set GEMINI_API_KEY
   uvicorn app.main:app --reload   # :8000
   ```

2. Serve the frontend:
   ```bash
   cd ../frontend
   python -m http.server 5000   # :5000
   ```

---

## Environment Variables

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Google Gemini API key (free tier ≈ 20 req/day — swap manually when exhausted) |
| `LLM_MODEL` | Gemini model name (default `gemini-3.8-flash`) |
| `EMBED_MODEL` | HuggingFace embedding model name |
| `RERANK_MODEL` | CrossEncoder model name |
| `LORE_DB_PERSIST_DIR` | ChromaDB persistence path for lore |
| `CAMPAIGN_DB_PERSIST_DIR` | ChromaDB persistence path for campaign |
| `TOP_K_RETRIEVE` | Number of chunks to retrieve per store |
| `TOP_K_RERANK` | Number of chunks to keep after reranking |
| `SHORT_TERM_TURNS` | Number of recent turns injected verbatim (default `5`) |
| `SHORT_TERM_MAX_CHARS` | Per-turn char cap in short-term memory (default `1500`) |
| `LANGSMITH_TRACING` | Set `true` to trace every turn in LangSmith (default `false`) |
| `LANGSMITH_API_KEY` | LangSmith API key (required when tracing is on) |
| `LANGSMITH_PROJECT` | Project name traces are grouped under (default `dungeon-master-agent`) |
| `LANGSMITH_ENDPOINT` | LangSmith API endpoint (default `https://api.smith.langchain.com`) |
| `CHUNK_SIZE` | Document chunk size (characters) |
| `CHUNK_OVERLAP` | Chunk overlap (characters) |

---

## API Quick Reference

| Method & Path | Purpose |
|---|---|
| `POST /chat/` | Player turn (`{query, evaluate?, reference?, stream?}` → JSON or SSE) |
| `GET /chat/turns/count` | Number of turns played |
| `GET /chat/turns/history` | Recent turn logs |
| `GET /chat/turns/recent` | Short-term memory window (`?limit=N`, default `SHORT_TERM_TURNS`) |
| `GET /chat/export` | Campaign JSON export |
| `POST /chat/import` | Campaign JSON import |
| `POST /documents/{lore,campaign}/upload` | Ingest `.pdf`/`.md`/`.txt` |
| `GET /documents/{lore,campaign}/list` | Files with per-file chunk counts |
| `DELETE /documents/{lore,campaign}/{filename}` | Remove one file |
| `DELETE /documents/{lore,campaign}/clear` | Wipe a collection |
| `GET /health` | Health check |

---

## Observability (LangSmith)

Every turn (retrieval → rerank → dice tool calls → generation) is a LangChain runnable, so LangSmith tracing needs no code changes — just opt in:

```bash
# backend/.env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_your_key_here
LANGSMITH_PROJECT=dungeon-master-agent
```

Restart the backend and each `/chat/` turn appears as a trace in the project: prompt inputs, retrieved/reranked context, `roll_dice` tool calls with results, token usage, and latency. Leave `LANGSMITH_TRACING=false` (default) to run fully offline.

---

## Next Steps

- 🐳 **Docker rebuild** — the running backend image predates short-term memory, LangSmith, and the evaluator rewrite. Rebuild + restart to pick them up (note: flaky PyPI caused one failed build; retry if `pip` errors).
- 🎲 **Import fidelity** — preserve original `turn`/`timestamp` metadata on campaign import instead of re-chunking.
- 👥 **Multi-session support** — `session_id`-scoped campaign collections + a session picker in the UI.
- 🧪 **Tests** — `pytest` coverage for chunking, the documents API, and a mocked RAG turn.
- 🔒 **Production hardening** — restricted CORS, auth + rate limiting on ingestion, responsive sidebar, full Markdown rendering.
