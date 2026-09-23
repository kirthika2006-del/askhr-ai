# AskHR AI — HR Policy Assistant

A production-ready RAG chatbot for company HR policies. Employees upload HR documents
(employee handbook, leave policy, onboarding guide, code of conduct — PDF, DOCX, or TXT)
and ask questions grounded strictly in their content, with sources and page numbers
shown for every answer.

## Architecture

```
Document ──▶ Validation ──▶ Text Extraction ──▶ Chunking ──▶ Embedding (Gemini) ──▶ Qdrant

Question ──▶ Validation ──▶ Embedding (Gemini) ──▶ Qdrant Similarity Search
         ──▶ Context Construction ──▶ Gemini generateContent ──▶ Grounded Answer + Sources
```

## Features

- Drag-and-drop document upload with live processing status (Uploading → Extracting →
  Chunking → Embedding → Indexing → Completed)
- PDF / DOCX / TXT text extraction with page-number tracking for PDFs
- Configurable chunking (size, overlap) with paragraph/sentence-aware splitting
- Gemini multimodal embeddings (`gemini-embedding-2`) stored in Qdrant
- Grounded question answering using `gemini-flash-latest`, with a similarity threshold
  so the model says "not enough information" instead of hallucinating
- Source cards with filename, page number (when available), and relevance %
- Document management: list, chunk counts, delete (removes vectors from Qdrant too)
- Live system status (Qdrant connection, Gemini configuration, doc/chunk counts)
- Markdown-lite chat rendering, copy/regenerate, auto-scroll, Enter-to-send / Shift+Enter
- Dark glassmorphism UI, fully responsive (desktop/tablet/mobile)

## Technologies

- **Backend:** Python 3, Flask
- **LLM:** Google Gemini API (`google-genai` SDK)
- **Vector DB:** Qdrant (cloud or self-hosted)
- **Frontend:** HTML5, CSS3, vanilla JavaScript (no framework)

## Requirements

- Python 3.10+
- A Gemini API key — https://aistudio.google.com/apikey
- A Qdrant cluster (Qdrant Cloud free tier or local) — https://cloud.qdrant.io

## Environment variables

Copy `.env.example` to `.env` and fill in:

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Your Gemini API key |
| `GEMINI_MODEL` | Generation model (default `gemini-flash-latest`) |
| `GEMINI_EMBEDDING_MODEL` | Embedding model (default `gemini-embedding-2`) |
| `EMBEDDING_DIMENSIONS` | Output vector size, must match the Qdrant collection (default `768`) |
| `QDRANT_URL` | Your Qdrant cluster URL |
| `QDRANT_API_KEY` | Your Qdrant API key |
| `QDRANT_COLLECTION_NAME` | Collection name (auto-created if missing) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Chunking configuration (characters) |
| `TOP_K` | Number of chunks retrieved per question |
| `SIMILARITY_THRESHOLD` | Minimum cosine score (0–1) to consider a chunk relevant |
| `MAX_UPLOAD_SIZE_MB` | Max upload size |
| `FLASK_SECRET_KEY` | Any random string |

**Never commit your `.env` file.** `.gitignore` already excludes it.

## Installation

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then edit .env with your keys
```

## Running locally

```bash
python app.py
```

Open http://localhost:5000 in your browser.

## Document ingestion workflow

1. Drag a PDF/DOCX/TXT file onto the upload zone (or click to browse).
2. The server validates the file (type, size, non-empty).
3. Text is extracted (page numbers are tracked for PDFs).
4. The text is split into overlapping chunks (`CHUNK_SIZE` / `CHUNK_OVERLAP`).
5. Each chunk is embedded with Gemini and upserted into Qdrant with metadata
   (document id, filename, chunk index, page number).
6. The document appears in the sidebar list with its chunk count.

## RAG query workflow

1. You ask a question in the chat box.
2. The question is embedded and Qdrant is searched for the `TOP_K` most similar chunks
   above `SIMILARITY_THRESHOLD`.
3. The retrieved chunks are assembled into a context block and sent to Gemini along with
   the system prompt (`config/rag_prompt.txt`) and recent conversation history.
4. Gemini answers strictly from the provided context, and the response includes source
   cards for every chunk that was actually used.
5. If nothing sufficiently relevant is found, the app tells you directly instead of
   guessing.

## API overview

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Qdrant/Gemini connectivity, document & chunk counts |
| GET | `/api/documents` | List uploaded documents |
| POST | `/api/documents` | Upload a document (`multipart/form-data`, field `file`) |
| DELETE | `/api/documents/<id>` | Delete a document and its vectors |
| POST | `/api/chat` | `{ "question": "...", "history": [...] }` → grounded answer + sources |

All responses follow `{ "success": true, "data": {...} }` or
`{ "success": false, "error": { "code": "...", "message": "..." } }`.

## Troubleshooting

- **Qdrant shows "Offline"** — check `QDRANT_URL` / `QDRANT_API_KEY` in `.env`, and that
  your Qdrant Cloud cluster is running (not paused).
- **Gemini shows "Not set"** — check `GEMINI_API_KEY` is present and valid.
- **"Vector dimension mismatch" on upload** — if you change `EMBEDDING_DIMENSIONS` after
  the Qdrant collection already exists, delete the collection (or use a new
  `QDRANT_COLLECTION_NAME`) so it's recreated with the new size.
- **PDF extraction fails** — the PDF may be scanned images with no embedded text (OCR is
  not included), or password-protected.
- **Gemini 503 / high demand errors** — the request will surface a clear error in the
  chat; simply retry using the Regenerate button.

## Deployment considerations

- Set `FLASK_DEBUG=False` in production.
- Put real secrets in your hosting platform's environment variable settings, never in
  the repo.
- The in-memory document registry (`RagService.documents`) resets on server restart —
  the chunks stay in Qdrant, but the "Documents" list in the UI will be empty until
  re-populated from a persistent store if you extend this for multi-instance/production
  use.
- Run with a production WSGI server, e.g. `gunicorn --workers 1 --timeout 120 app:app`.
- Configure CORS explicitly if the frontend will be served from a different origin.
