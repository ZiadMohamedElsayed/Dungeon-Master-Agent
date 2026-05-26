# Dungeon Master Agent

An AI-powered Dungeon Master that runs tabletop RPG sessions using a Retrieval-Augmented Generation (RAG) pipeline. The agent draws from two separate knowledge bases — **World Lore** and **Campaign History** — to narrate vivid, consistent, and contextually grounded sessions powered by Google Gemini.

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
        │
        ▼
  DM Response + Auto-save turn to Campaign KB
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML5 / JavaScript |
| Styling | CSS3 |
| API Framework | FastAPI + Uvicorn |
| LLM | Google Gemini via `langchain-google-genai` |
| Embeddings | HuggingFace Sentence Transformers |
| Vector Store | ChromaDB (via `langchain-chroma`) |
| Reranker | CrossEncoder (`sentence-transformers`) |
| Orchestration | LangChain LCEL |
| Document Loaders | LangChain (PDF + Text) |
| Evaluation | RAGAS (RubricsScore + faithfulness) |
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
│       ├── reranker.py      # CrossEncoder reranking logic
│       └── evaluator.py     # RAGAS evaluation pipeline
└── requirements.txt

frontend/
├── css/
│   └── styles.css           # Layout & chat UI styling
├── js/
│   ├── api.js               # Fetch wrapper for backend communication
│   ├── chat.js              # Chat UI logic: message rendering & input
│   ├── main.js              # Main entry point; coordinates JS modules
│   └── sidebar.js           # Document list & sidebar toggle logic
└── index.html               # Main HTML entry point
```

---

## How It Works

1. **Document Ingestion** — Lore PDFs and text files are uploaded via `/api/documents`, chunked, embedded, and stored in the lore ChromaDB collection.

2. **Turn Execution** — When a player submits an action via `/api/chat/`, the system concurrently retrieves relevant chunks from both the lore and campaign history stores using a LangChain `RunnableParallel`.

3. **Reranking** — Both retrieved sets are independently reranked using a CrossEncoder to surface the most contextually relevant chunks.

4. **Generation** — A structured prompt combining the system instructions, lore context, campaign context, and player action is sent to Gemini via `ChatGoogleGenerativeAI`. The full response is awaited and returned as JSON.

5. **Auto-Persistence** — Each completed turn (player action + DM response) is automatically saved back into the campaign vectorstore, building a living, searchable campaign history.

6. **Evaluation** — Responses can be optionally evaluated by passing `"evaluate": true` in the request. Three custom DM-appropriate metrics scored 1–5 via an LLM judge are computed: `dm_relevance`, `lore_consistency`, and `narrative_quality`. When a `reference` answer is provided, `context_precision` and `context_recall` are also computed. Evaluation runs in a thread pool to avoid event loop conflicts with the RAGAS internals.

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

5. Run the frontend:
   ```bash
   cd ../frontend
   python -m http.server 5000
   ```

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

## Next Steps

💾 **Save / Load Campaign**
Allow players to export the current campaign history and reload it in a future session. This means serializing the ChromaDB campaign collection to a portable format (JSON or ZIP), exposing `/api/campaign/export` and `/api/campaign/import` endpoints, and wiring up save/load controls in the frontend sidebar.

🌊 **Streaming**
Stream the DM's response token by token instead of waiting for the full generation. This means switching `dm_chain.ainvoke()` to `dm_chain.astream()` in `rag_chain.py`, returning a `StreamingResponse` from the chat endpoint, and updating the frontend to consume the token stream and append to the DM bubble in real time.