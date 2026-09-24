"""Orchestrates document ingestion and grounded question answering."""
import os
import json
import logging
from qdrant_client.http import models as qmodels
 
from services import document_service
from utils.chunking import chunk_text
from utils.helpers import new_id, now_iso
 
logger = logging.getLogger(__name__)
 
MAX_HISTORY_TURNS = 6
MAX_CONTEXT_CHARS = 12000
 
 
class RagService:
    def __init__(
        self,
        embedding_service,
        qdrant_service,
        gemini_service,
        rag_prompt_template: str,
        top_k: int,
        similarity_threshold: float,
        chunk_size: int,
        chunk_overlap: int,
        registry_path: str = None,
    ):
        self.embeddings = embedding_service
        self.qdrant = qdrant_service
        self.gemini = gemini_service
        self.prompt_template = rag_prompt_template
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
 
        # Document registry. Kept in memory for fast reads, and mirrored to a
        # small JSON file on disk so the list survives a server restart
        # (the underlying vectors always live in Qdrant regardless).
        self.registry_path = registry_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "documents.json"
        )
        self.documents = {}
        self._load_registry()
 
    # ---------------- Registry persistence ----------------
 
    def _load_registry(self):
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    self.documents = json.load(f)
                if self.documents:
                    logger.info(
                        "Loaded %d document(s) from registry at %s",
                        len(self.documents), self.registry_path,
                    )
                    return
            except Exception as exc:
                logger.warning("Could not load document registry (%s): %s", self.registry_path, exc)
                self.documents = {}
 
        # Local file is missing or empty -- this happens on hosts with an
        # ephemeral disk (e.g. Render's free tier) after every restart.
        # The vectors in Qdrant always survive, so rebuild the list from
        # there instead of showing an empty state.
        rebuilt = self.qdrant.list_document_summaries()
        if rebuilt:
            self.documents = rebuilt
            self._save_registry()
            logger.info("Rebuilt %d document(s) from Qdrant.", len(rebuilt))
 
    def _save_registry(self):
        try:
            os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
            with open(self.registry_path, "w", encoding="utf-8") as f:
                json.dump(self.documents, f, indent=2)
        except Exception as exc:
            # Non-fatal: the upload/delete itself already succeeded against
            # Qdrant, so we just log — the in-memory list is still correct
            # for the current process.
            logger.warning("Could not save document registry: %s", exc)
 
    # ---------------- Ingestion ----------------
 
    def ingest_document(self, file_path: str, original_filename: str, extension: str):
        document_id = new_id()
 
        full_text, pages = document_service.extract_text(file_path, extension)
 
        chunks = chunk_text(full_text, self.chunk_size, self.chunk_overlap)
        if not chunks:
            raise document_service.ExtractionError(
                "Document produced no usable chunks.", "NO_CHUNKS"
            )
 
        chunk_texts = [c["text"] for c in chunks]
        vectors = self.embeddings.embed_documents(chunk_texts)
 
        points = []
        for chunk, vector in zip(chunks, vectors):
            page_number = self._infer_page(chunk["text"], pages)
            payload = {
                "document_id": document_id,
                "filename": original_filename,
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
                "page_number": page_number,
            }
            points.append(
                qmodels.PointStruct(id=new_id(), vector=vector, payload=payload)
            )
 
        self.qdrant.upsert_chunks(points)
 
        file_size = os.path.getsize(file_path)
        self.documents[document_id] = {
            "id": document_id,
            "filename": original_filename,
            "extension": extension,
            "size_bytes": file_size,
            "uploaded_at": now_iso(),
            "chunk_count": len(points),
            "status": "completed",
        }
        self._save_registry()
        return self.documents[document_id]
 
    @staticmethod
    def _infer_page(chunk_text_value: str, pages):
        """Best-effort page number: find which page's text the chunk starts in."""
        if not pages:
            return None
        snippet = chunk_text_value[:80].strip()
        if not snippet:
            return None
        for page_number, page_text in pages:
            if snippet[:40] and snippet[:40] in page_text:
                return page_number
        return None
 
    def list_documents(self):
        return list(self.documents.values())
 
    def delete_document(self, document_id: str):
        if document_id not in self.documents:
            return False
        self.qdrant.delete_document(document_id)
        del self.documents[document_id]
        self._save_registry()
        return True
 
    # ---------------- Question answering ----------------
 
    def answer_question(self, question: str, history: list):
        query_vector = self.embeddings.embed_query(question)
        results = self.qdrant.search(
            query_vector, self.top_k, self.similarity_threshold
        )
 
        if not results:
            return {
                "answer": (
                    "The uploaded HR policies don't cover this clearly enough to "
                    "answer confidently. Try rephrasing, upload the relevant policy "
                    "document, or check with HR directly."
                ),
                "sources": [],
            }
 
        context_parts = []
        sources = []
        used_chars = 0
 
        for point in results:
            payload = point.payload or {}
            text = payload.get("text", "")
            if used_chars + len(text) > MAX_CONTEXT_CHARS:
                continue
            used_chars += len(text)
 
            context_parts.append(
                f"[Source: {payload.get('filename', 'unknown')}"
                + (f", page {payload.get('page_number')}" if payload.get("page_number") else "")
                + f"]\n{text}"
            )
            sources.append(
                {
                    "filename": payload.get("filename", "unknown"),
                    "page_number": payload.get("page_number"),
                    "relevance": round(point.score * 100, 1),
                    "chunk_index": payload.get("chunk_index"),
                }
            )
 
        context = "\n\n---\n\n".join(context_parts)
        history_text = self._format_history(history)
 
        prompt = self.prompt_template.format(
            context=context, history=history_text, question=question
        )
 
        answer = self.gemini.generate_answer(prompt)
        return {"answer": answer, "sources": sources}
 
    @staticmethod
    def _format_history(history: list) -> str:
        if not history:
            return "(no prior conversation)"
        recent = history[-MAX_HISTORY_TURNS:]
        lines = []
        for turn in recent:
            role = "User" if turn.get("role") == "user" else "Assistant"
            lines.append(f"{role}: {turn.get('content', '')}")
        return "\n".join(lines)
 