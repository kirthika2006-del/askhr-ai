"""Flask entrypoint for the RAG Knowledge Assistant."""
import os
import logging
import tempfile

from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

from utils.helpers import success_response, error_response
from utils.file_validation import validate_upload, get_extension, FileValidationError
from services.document_service import ExtractionError
from services.embedding_service import EmbeddingService
from services.qdrant_service import QdrantService, QdrantServiceError
from services.gemini_service import GeminiService
from services.rag_service import RagService

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rag_app")

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-key-change-me")
app.config["MAX_CONTENT_LENGTH"] = int(
    os.environ.get("MAX_UPLOAD_SIZE_MB", "20")
) * 1024 * 1024

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
GEMINI_EMBEDDING_MODEL = os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
EMBEDDING_DIMENSIONS = int(os.environ.get("EMBEDDING_DIMENSIONS", "768"))

QDRANT_URL = os.environ.get("QDRANT_URL", "")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
QDRANT_COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME", "rag_knowledge_base")

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "150"))
TOP_K = int(os.environ.get("TOP_K", "5"))
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.5"))
MAX_UPLOAD_SIZE_BYTES = int(os.environ.get("MAX_UPLOAD_SIZE_MB", "20")) * 1024 * 1024

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "config", "rag_prompt.txt")
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    RAG_PROMPT_TEMPLATE = f.read()

# ---------------------------------------------------------------------------
# Service initialization (fail gracefully; report status via /api/health)
# ---------------------------------------------------------------------------
init_errors = {}
embedding_service = None
qdrant_service = None
gemini_service = None
rag_service = None

try:
    embedding_service = EmbeddingService(
        GEMINI_API_KEY, GEMINI_EMBEDDING_MODEL, EMBEDDING_DIMENSIONS
    )
except Exception as exc:
    logger.error("Embedding service init failed: %s", exc)
    init_errors["embedding"] = str(exc)

try:
    qdrant_service = QdrantService(
        QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION_NAME, EMBEDDING_DIMENSIONS
    )
except Exception as exc:
    logger.error("Qdrant service init failed: %s", exc)
    init_errors["qdrant"] = str(exc)

try:
    gemini_service = GeminiService(GEMINI_API_KEY, GEMINI_MODEL)
except Exception as exc:
    logger.error("Gemini service init failed: %s", exc)
    init_errors["gemini"] = str(exc)

if embedding_service and qdrant_service and gemini_service:
    rag_service = RagService(
        embedding_service,
        qdrant_service,
        gemini_service,
        RAG_PROMPT_TEMPLATE,
        TOP_K,
        SIMILARITY_THRESHOLD,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
    )


def require_rag_service():
    if rag_service is None:
        return jsonify(
            error_response(
                "SERVICE_NOT_CONFIGURED",
                "The server is missing required configuration. Check server logs "
                "and your .env file (GEMINI_API_KEY / QDRANT_URL / QDRANT_API_KEY).",
            )
        ), 503
    return None


# ---------------------------------------------------------------------------
# Routes — pages
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Routes — API
# ---------------------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health():
    qdrant_ok = qdrant_service.health_check() if qdrant_service else False
    gemini_ok = gemini_service.is_configured() if gemini_service else False
    stats = qdrant_service.collection_stats() if qdrant_service else {"points_count": 0}
    doc_count = len(rag_service.documents) if rag_service else 0

    return jsonify(
        success_response(
            {
                "qdrant_connected": qdrant_ok,
                "gemini_configured": gemini_ok,
                "document_count": doc_count,
                "chunk_count": stats.get("points_count", 0),
                "errors": init_errors,
            }
        )
    )


@app.route("/api/documents", methods=["GET"])
def list_documents():
    err = require_rag_service()
    if err:
        return err
    return jsonify(success_response({"documents": rag_service.list_documents()}))


@app.route("/api/documents", methods=["POST"])
def upload_document():
    err = require_rag_service()
    if err:
        return err

    file_storage = request.files.get("file")
    try:
        safe_name = validate_upload(file_storage, MAX_UPLOAD_SIZE_BYTES)
    except FileValidationError as exc:
        return jsonify(error_response(exc.code, exc.message)), 400

    extension = get_extension(safe_name)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=extension, dir=tempfile.gettempdir()
        ) as tmp:
            file_storage.save(tmp.name)
            tmp_path = tmp.name

        document = rag_service.ingest_document(tmp_path, safe_name, extension)
        return jsonify(success_response({"document": document})), 201

    except ExtractionError as exc:
        return jsonify(error_response(exc.code, exc.message)), 422
    except QdrantServiceError as exc:
        return jsonify(error_response(exc.code, exc.message)), 502
    except Exception as exc:
        logger.exception("Unexpected upload failure")
        return (
            jsonify(
                error_response(
                    "INTERNAL_ERROR", "An unexpected error occurred while processing the file."
                )
            ),
            500,
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


@app.route("/api/documents/<document_id>", methods=["DELETE"])
def delete_document(document_id):
    err = require_rag_service()
    if err:
        return err
    try:
        deleted = rag_service.delete_document(document_id)
    except QdrantServiceError as exc:
        return jsonify(error_response(exc.code, exc.message)), 502

    if not deleted:
        return jsonify(error_response("NOT_FOUND", "Document not found.")), 404
    return jsonify(success_response({"deleted_id": document_id}))


@app.route("/api/chat", methods=["POST"])
def chat():
    err = require_rag_service()
    if err:
        return err

    body = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()
    history = body.get("history") or []

    if not question:
        return jsonify(error_response("EMPTY_QUESTION", "Question cannot be empty.")), 400
    if len(question) > 2000:
        return jsonify(error_response("QUESTION_TOO_LONG", "Question is too long.")), 400
    if not isinstance(history, list):
        history = []

    try:
        result = rag_service.answer_question(question, history)
        return jsonify(success_response(result))
    except QdrantServiceError as exc:
        return jsonify(error_response(exc.code, exc.message)), 502
    except RuntimeError as exc:
        return jsonify(error_response("GEMINI_ERROR", str(exc))), 502
    except Exception:
        logger.exception("Unexpected chat failure")
        return (
            jsonify(
                error_response("INTERNAL_ERROR", "An unexpected error occurred answering the question.")
            ),
            500,
        )


@app.errorhandler(413)
def too_large(_e):
    return jsonify(error_response("FILE_TOO_LARGE", "Uploaded file exceeds the size limit.")), 413


@app.errorhandler(404)
def not_found(_e):
    return jsonify(error_response("NOT_FOUND", "Resource not found.")), 404


@app.errorhandler(500)
def server_error(_e):
    return jsonify(error_response("INTERNAL_ERROR", "Internal server error.")), 500


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=debug)
