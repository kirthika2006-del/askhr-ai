"""Qdrant collection management, upsert, search, and deletion."""
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse
 
logger = logging.getLogger(__name__)
 
 
class QdrantServiceError(Exception):
    def __init__(self, message: str, code: str = "QDRANT_ERROR"):
        super().__init__(message)
        self.message = message
        self.code = code
 
 
class QdrantService:
    def __init__(self, url: str, api_key: str, collection_name: str, vector_size: int):
        if not url:
            raise ValueError("QDRANT_URL is not configured.")
        self.collection_name = collection_name
        self.vector_size = vector_size
        try:
            self.client = QdrantClient(url=url, api_key=api_key or None, timeout=30)
        except Exception as exc:
            logger.error("Qdrant client init failed: %s", exc)
            raise QdrantServiceError(
                "Could not connect to Qdrant. Check QDRANT_URL and QDRANT_API_KEY.",
                "CONNECTION_FAILED",
            )
        self._ensure_collection()
 
    def _ensure_collection(self):
        try:
            existing = [c.name for c in self.client.get_collections().collections]
        except Exception as exc:
            logger.error("Qdrant health check failed: %s", exc)
            raise QdrantServiceError(
                "Could not reach the Qdrant cluster. Check network/credentials.",
                "CONNECTION_FAILED",
            )
 
        if self.collection_name not in existing:
            try:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=self.vector_size,
                        distance=qmodels.Distance.COSINE,
                    ),
                )
                logger.info("Created Qdrant collection '%s'", self.collection_name)
            except Exception as exc:
                logger.error("Collection creation failed: %s", exc)
                raise QdrantServiceError(
                    f"Failed to create collection: {exc}", "COLLECTION_CREATE_FAILED"
                )
 
    def health_check(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception as exc:
            logger.warning("Qdrant health check failed: %s", exc)
            return False
 
    def collection_stats(self):
        try:
            info = self.client.get_collection(self.collection_name)
            return {"points_count": info.points_count or 0}
        except Exception as exc:
            logger.warning("Could not fetch collection stats: %s", exc)
            return {"points_count": 0}
 
    def upsert_chunks(self, points):
        """points: list of qmodels.PointStruct"""
        try:
            self.client.upsert(collection_name=self.collection_name, points=points)
        except Exception as exc:
            logger.error("Qdrant upsert failed: %s", exc)
            raise QdrantServiceError(f"Failed to store vectors: {exc}", "UPSERT_FAILED")
 
    def search(self, query_vector, top_k: int, similarity_threshold: float):
        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                score_threshold=similarity_threshold,
                with_payload=True,
            ).points
            return results
        except Exception as exc:
            logger.error("Qdrant search failed: %s", exc)
            raise QdrantServiceError(f"Search failed: {exc}", "SEARCH_FAILED")
 
    def delete_document(self, document_id: str):
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="document_id",
                                match=qmodels.MatchValue(value=document_id),
                            )
                        ]
                    )
                ),
            )
        except Exception as exc:
            logger.error("Qdrant deletion failed for doc %s: %s", document_id, exc)
            raise QdrantServiceError(
                f"Failed to delete document vectors: {exc}", "DELETE_FAILED"
            )
 
    def list_document_summaries(self):
        """Reconstruct a document list by scanning stored chunk payloads.
 
        Used to recover the document list if the local registry file was
        lost (e.g. after a redeploy on a host with an ephemeral disk, like
        Render's free tier) -- the vectors in Qdrant are the source of
        truth and always survive a restart.
        """
        summaries = {}
        next_offset = None
        try:
            while True:
                points, next_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    with_payload=True,
                    with_vectors=False,
                    limit=256,
                    offset=next_offset,
                )
                for point in points:
                    payload = point.payload or {}
                    doc_id = payload.get("document_id")
                    if not doc_id:
                        continue
                    if doc_id not in summaries:
                        summaries[doc_id] = {
                            "id": doc_id,
                            "filename": payload.get("filename", "unknown"),
                            "extension": None,
                            "size_bytes": None,
                            "uploaded_at": None,
                            "chunk_count": 0,
                            "status": "completed",
                        }
                    summaries[doc_id]["chunk_count"] += 1
                if next_offset is None:
                    break
        except Exception as exc:
            logger.warning("Could not rebuild document list from Qdrant: %s", exc)
        return summaries
 
    def count_chunks(self, document_id: str) -> int:
        try:
            result = self.client.count(
                collection_name=self.collection_name,
                count_filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="document_id",
                            match=qmodels.MatchValue(value=document_id),
                        )
                    ]
                ),
            )
            return result.count
        except Exception as exc:
            logger.warning("Could not count chunks for doc %s: %s", document_id, exc)
            return 0
 