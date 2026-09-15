# Dungeon Master Agent

An AI-powered Dungeon Master that runs tabletop RPG sessions using a Retrieval-Augmented Generation (RAG) pipeline. The agent draws from two separate knowledge bases — **World Lore** and **Campaign History** — to narrate vivid, consistent, and contextually grounded sessions powered by Google Gemini. Chance-based outcomes are decided by real dice rolls the DM makes through a tool call, and responses stream to the UI token by token.

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
| LLM | Google Gemini (`gemini-2.5-flash`) via `langchain-google-genai` |
| Dice | `roll_dice` LangChain tool bound to the DM (`NdM+K` notation) |
| Embeddings | HuggingFace Sentence Transformers (CPU torch) |
| Vector Store | ChromaDB (via `langchain-chroma`) |
| Reranker | CrossEncoder (`sentence-transformers`) |
| Orchestration | LangChain LCEL + tool-call loop |
| Document Loaders | PDF / Markdown / Text chunking |
| Evaluation | RAGAS (`dm_relevance`, `lore_consistency`, `narrative_quality`, + `context_precision`/`recall` with reference) |
| Config | Pydantic Settings + `.env` |
| Deployment | Docker + Compose (backend :8000, frontend :5000) |

---

## Project Structure

```
backend/
├── app/
│   ├── api/
│   │   ├── chat.py          # POST /api/chat/ (JSON or SSE stream), turn count/history, export/import
│   │   └── documents.py     # Lore & campaign ingestion: upload / list / delete / clear
│   ├── core/
│   │   ├── config.py        # Environment & settings (documented env aliases)
│   │   └── vectorstore.py   # ChromaDB setup, chunking, retrievers
│   └── services/
│       ├── rag_chain.py     # RAG pipeline: retrieve → rerank → dice tools → generate → save
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

1. **Document Ingestion** — Lore PDFs, Markdown, and text files are uploaded via `/api/documents/lore/upload`, chunked, embedded, and stored in the lore ChromaDB collection (10 MB cap, 409 on duplicates, per-file delete).

2. **Turn Execution** — When a player submits an action via `/api/chat/`, the system concurrently retrieves relevant chunks from both the lore and campaign history stores using a LangChain `RunnableParallel`.

3. **Reranking** — Both retrieved sets are independently reranked using a CrossEncoder to surface the most contextually relevant chunks.

4. **Dice (tool calls)** — If the action has a chance-based outcome (attack, check, save…), the DM must call the `roll_dice` tool first and narrate strictly from the real result instead of inventing one. Rolls are logged server-side and streamed to the UI as 🎲 events. Players can also roll manually with `/roll 2d6+3`.

5. **Generation** — A structured prompt combining system instructions, lore context, campaign context, and the player action is sent to Gemini. Responses stream back as SSE (`sources → dice? → token* → done`) and render live; plain JSON is returned when `stream: false`.

6. **Auto-Persistence** — Each completed turn (player action + DM response) is automatically saved back into the campaign vectorstore, building a living, searchable campaign history. Campaigns can be exported/imported as JSON from the sidebar.

7. **Evaluation** — Responses can be optionally evaluated by passing `"evaluate": true` in the request (or ticking *evaluate* in the UI). Three custom DM-appropriate metrics scored 1–5 via an LLM judge are computed: `dm_relevance`, `lore_consistency`, and `narrative_quality`. When a `reference` answer is provided, `context_precision` and `context_recall` are also computed.

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
| `LLM_MODEL` | Gemini model name (default `gemini-2.5-flash`) |
| `EMBED_MODEL` | HuggingFace embedding model name |
| `RERANK_MODEL` | CrossEncoder model name |
| `LORE_DB_PERSIST_DIR` | ChromaDB persistence path for lore |
| `CAMPAIGN_DB_PERSIST_DIR` | ChromaDB persistence path for campaign |
| `TOP_K_RETRIEVE` | Number of chunks to retrieve per store |
| `TOP_K_RERANK` | Number of chunks to keep after reranking |
| `CHUNK_SIZE` | Document chunk size (characters) |
| `CHUNK_OVERLAP` | Chunk overlap (characters) |

---

## API Quick Reference

| Method & Path | Purpose |
|---|---|
| `POST /api/chat/` | Player turn (`{query, evaluate?, reference?, stream?}` → JSON or SSE) |
| `GET /api/chat/turns/count` | Number of turns played |
| `GET /api/chat/turns/history` | Recent turn logs |
| `GET /api/chat/export` | Campaign JSON export |
| `POST /api/chat/import` | Campaign JSON import |
| `POST /api/documents/{lore,campaign}/upload` | Ingest `.pdf`/`.md`/`.txt` |
| `GET /api/documents/{lore,campaign}/list` | Files with per-file chunk counts |
| `DELETE /api/documents/{lore,campaign}/{filename}` | Remove one file |
| `DELETE /api/documents/{lore,campaign}/clear` | Wipe a collection |
| `GET /health` | Health check |

---

## Next Steps

- 🔧 **Evaluator repair** — make RAGAS robust against langchain 1.x (or replace with direct Gemini-judge calls) and verify `{"evaluate": true}` end to end.
- 🎲 **Import fidelity** — preserve original `turn`/`timestamp` metadata on campaign import instead of re-chunking.
- 👥 **Multi-session support** — `session_id`-scoped campaign collections + a session picker in the UI.
- 🧪 **Tests** — `pytest` coverage for chunking, the documents API, and a mocked RAG turn.
- 🔒 **Production hardening** — restricted CORS, auth + rate limiting on ingestion, responsive sidebar, full Markdown rendering.
