"""File validation helpers for document uploads."""
import os
import re

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}

ALLOWED_MIME_TYPES = {
    ".pdf": {"application/pdf"},
    ".docx": {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    },
    ".txt": {"text/plain"},
}


class FileValidationError(Exception):
    """Raised when an uploaded file fails validation."""

    def __init__(self, message: str, code: str = "INVALID_FILE"):
        super().__init__(message)
        self.message = message
        self.code = code


def get_extension(filename: str) -> str:
    return os.path.splitext(filename)[1].lower()


def sanitize_filename(filename: str) -> str:
    """Strip path components and unsafe characters to prevent path traversal."""
    filename = os.path.basename(filename)
    filename = filename.replace("\x00", "")
    filename = re.sub(r"[^A-Za-z0-9._\-]", "_", filename)
    filename = filename.lstrip(".")
    if not filename:
        filename = "unnamed_file"
    return filename[:200]


def validate_upload(file_storage, max_size_bytes: int) -> str:
    """
    Validate an uploaded Werkzeug FileStorage object.
    Returns the sanitized filename on success, raises FileValidationError otherwise.
    """
    if file_storage is None or file_storage.filename == "":
        raise FileValidationError("No file was selected for upload.", "NO_FILE")

    original_name = file_storage.filename
    ext = get_extension(original_name)

    if ext not in ALLOWED_EXTENSIONS:
        raise FileValidationError(
            f"Unsupported file type '{ext}'. Allowed types: PDF, DOCX, TXT.",
            "UNSUPPORTED_TYPE",
        )

    mimetype = (file_storage.mimetype or "").lower()
    allowed_mimes = ALLOWED_MIME_TYPES.get(ext, set())
    if mimetype and allowed_mimes and mimetype not in allowed_mimes:
        raise FileValidationError(
            f"File content type '{mimetype}' does not match extension '{ext}'.",
            "MIME_MISMATCH",
        )

    file_storage.stream.seek(0, os.SEEK_END)
    size = file_storage.stream.tell()
    file_storage.stream.seek(0)

    if size == 0:
        raise FileValidationError("The uploaded file is empty.", "EMPTY_FILE")

    if size > max_size_bytes:
        max_mb = max_size_bytes / (1024 * 1024)
        raise FileValidationError(
            f"File exceeds the maximum allowed size of {max_mb:.0f} MB.",
            "FILE_TOO_LARGE",
        )

    return sanitize_filename(original_name)
