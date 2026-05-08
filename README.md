# Dungeon Master Agent

An AI-powered Dungeon Master backend that runs tabletop RPG sessions using a Retrieval-Augmented Generation (RAG) pipeline. The agent draws from two separate knowledge bases — **World Lore** and **Campaign History** — to narrate vivid, consistent, and contextually grounded sessions powered by Google Gemini.

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
   Google Gemini LLM
        │
        ▼
  DM Response + Auto-save turn to Campaign KB
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn |
| LLM | Google Gemini (`google-generativeai`) |
| Embeddings | HuggingFace Sentence Transformers |
| Vector Store | ChromaDB (via `langchain-chroma`) |
| Reranker | CrossEncoder (`sentence-transformers`) |
| Document Loaders | LangChain (PDF + Text) |
| Config | Pydantic Settings + `.env` |

---

## Project Structure

```
backend/
├── app/
│   ├── api/
│   │   ├── chat.py          # POST /api/chat/ — player turn endpoint
│   │   └── documents.py     # Document ingestion endpoints
│   ├── core/
│   │   ├── config.py        # Environment & settings
│   │   └── vectorstore.py   # ChromaDB setup, chunking, retrievers
│   └── services/
│       ├── rag_chain.py     # Main RAG pipeline (retrieve → rerank → generate → save)
│       └── reranker.py      # CrossEncoder reranking logic
└── requirements.txt
```

---

## How It Works

1. **Document Ingestion** — Lore PDFs and text files are uploaded via `/api/documents`, chunked, embedded, and stored in the lore ChromaDB collection.

2. **Turn Execution** — When a player submits an action via `/api/chat/`, the system concurrently retrieves relevant chunks from both the lore and campaign history stores.

3. **Reranking** — Both retrieved sets are independently reranked using a CrossEncoder to surface the most contextually relevant chunks.

4. **Generation** — A structured prompt combining the system instructions, lore context, campaign context, and player action is sent to Gemini, which responds in character as the Dungeon Master.

5. **Auto-Persistence** — Each completed turn (player action + DM response) is automatically saved back into the campaign vectorstore, building a living, searchable campaign history.

---

## Setup

1. Clone the repo and navigate to `backend/`:
   ```bash
   cd backend
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Copy and fill in the environment file:
   ```bash
   cp .env.example .env
   ```

4. Run the server:
   ```bash
   uvicorn app.main:app --reload
   ```

---

## API Endpoints

### `POST /api/chat/`
Submit a player action and receive a DM response.

**Request:**
```json
{ "query": "I approach the hooded figure at the bar and ask what they know about the missing caravan." }
```

**Response:**
```json
{
  "turn": 4,
  "answer": "The figure slowly lifts their gaze...",
  "sources": {
    "lore": [{ "source": "world_lore.pdf", "snippet": "..." }],
    "campaign": [{ "source": "campaign_history", "snippet": "..." }]
  }
}
```

### `POST /api/documents/lore`
Upload a PDF or `.txt` file to the World Lore knowledge base.

### `POST /api/documents/campaign`
Upload a PDF or `.txt` file to seed the Campaign History knowledge base.

---

## Environment Variables

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Google Gemini API key |
| `LLM_MODEL` | Gemini model name (e.g. `gemini-1.5-flash`) |
| `EMBED_MODEL` | HuggingFace embedding model name |
| `RERANK_MODEL` | CrossEncoder model name |
| `LORE_DB_PERSIST_DIR` | ChromaDB persistence path for lore |
| `CAMPAIGN_DB_PERSIST_DIR` | ChromaDB persistence path for campaign |
| `TOP_K_RETRIEVE` | Number of chunks to retrieve per store |
| `TOP_K_RERANK` | Number of chunks to keep after reranking |
| `CHUNK_SIZE` | Document chunk size (characters) |
| `CHUNK_OVERLAP` | Chunk overlap (characters) |

---

## Current Status

The backend is fully implemented and functional:

- ✅ RAG pipeline with dual knowledge bases (lore + campaign)
- ✅ Concurrent retrieval with CrossEncoder reranking
- ✅ Gemini-powered DM narration
- ✅ Auto-saving turns to campaign history
- ✅ PDF and text document ingestion
- ✅ REST API via FastAPI

---

## Next Steps

### 1. 🖥️ Frontend
Build a player-facing web interface for the game. This includes a chat-style UI for submitting player actions and reading DM responses, a document upload panel for ingesting lore and campaign files, a turn history log with source citations displayed per turn, and theming that fits the fantasy/RPG aesthetic.

### 2. 📊 Evaluation
Establish a systematic way to measure the quality of DM responses. This means building a benchmark dataset of player actions paired with ideal DM responses, defining metrics such as lore consistency, narrative immersion, context relevance, and decision-point clarity, integrating an LLM-as-judge pipeline to score responses automatically, and running regression checks when the prompt, retrieval strategy, or model is changed.

### 3. 🌐 Multiplayer
Extend the system to support multiple players in a shared session. This requires a session management layer to scope each campaign to a group of players, real-time communication (e.g. WebSockets) so all players see DM responses simultaneously, per-player character state tracking, and a shared campaign history that reflects all players' actions rather than a single user's.
