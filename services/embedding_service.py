"""Wraps Gemini embedding calls for documents and queries."""
import logging
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# Gemini's embed_content batch endpoint accepts at most 100 requests per call.
GEMINI_BATCH_LIMIT = 100


class EmbeddingService:
    def __init__(self, api_key: str, model: str, dimensions: int):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.dimensions = dimensions

    def _embed_batch(self, texts, task_type: str):
        try:
            result = self.client.models.embed_content(
                model=self.model,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=self.dimensions,
                ),
            )
            return [e.values for e in result.embeddings]
        except Exception as exc:
            logger.error("Embedding call failed: %s", exc)
            raise RuntimeError(f"Embedding generation failed: {exc}")

    def _embed(self, texts, task_type: str):
        """Splits large chunk lists into batches of <= GEMINI_BATCH_LIMIT."""
        if len(texts) <= GEMINI_BATCH_LIMIT:
            return self._embed_batch(texts, task_type)

        all_vectors = []
        for i in range(0, len(texts), GEMINI_BATCH_LIMIT):
            batch = texts[i : i + GEMINI_BATCH_LIMIT]
            logger.info(
                "Embedding batch %d-%d of %d chunks",
                i + 1, i + len(batch), len(texts),
            )
            all_vectors.extend(self._embed_batch(batch, task_type))
        return all_vectors

    def embed_documents(self, texts):
        """Embed a batch of document chunks (auto-batched under the API limit)."""
        if not texts:
            return []
        return self._embed(texts, task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str):
        """Embed a single user query."""
        vectors = self._embed([text], task_type="RETRIEVAL_QUERY")
        return vectors[0] if vectors else []
