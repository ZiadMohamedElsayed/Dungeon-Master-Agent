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
   Google Gemini LLM
        │
        ▼
  DM Response + Auto-save turn to Campaign KB
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | FastAPI + Uvicorn |
| LLM | HTML5 / JavaScript |
| Styling | CSS3 |
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

frontend/
├── css/              
│   └── styles.css           # Layout & chat UI styling.
├── js/      
│   ├── api.js               # Fetch wrapper for backend communication.
│   ├── chat.js              # Chat UI logic: message rendering & input.
│   ├── main.js              # Main entry point; coordinates JS modules.
│   └── sidebar.js           # Document list & sidebar toggle logic.
└── index.html               # Main HTML entry point.
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

📊 Evaluation
Establish a systematic way to measure the quality of DM responses. This means building a benchmark dataset of player actions paired with ideal DM responses, defining metrics such as lore consistency, narrative immersion, context relevance, and decision-point clarity, integrating an LLM-as-judge pipeline to score responses automatically, and running regression checks when the prompt, retrieval strategy, or model is changed.
