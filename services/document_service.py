"""Text extraction for PDF, DOCX, and TXT documents."""
import logging
from pypdf import PdfReader
from docx import Document as DocxDocument

logger = logging.getLogger(__name__)

MIN_USABLE_CHARS = 20


class ExtractionError(Exception):
    def __init__(self, message: str, code: str = "EXTRACTION_FAILED"):
        super().__init__(message)
        self.message = message
        self.code = code


def extract_pdf(file_path: str):
    """Returns (full_text, list_of_(page_number, page_text))."""
    try:
        reader = PdfReader(file_path)
    except Exception as exc:
        logger.error("PDF read failure for %s: %s", file_path, exc)
        raise ExtractionError(
            "Could not open the PDF. It may be corrupted or encrypted.",
            "CORRUPTED_PDF",
        )

    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            raise ExtractionError(
                "This PDF is password-protected and cannot be read.",
                "ENCRYPTED_PDF",
            )

    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            logger.warning("Failed extracting page %d of %s: %s", i, file_path, exc)
            text = ""
        pages.append((i + 1, text.strip()))

    full_text = "\n\n".join(t for _, t in pages if t)
    return full_text, pages


def extract_docx(file_path: str):
    try:
        doc = DocxDocument(file_path)
    except Exception as exc:
        logger.error("DOCX read failure for %s: %s", file_path, exc)
        raise ExtractionError(
            "Could not open the DOCX file. It may be corrupted.",
            "CORRUPTED_DOCX",
        )

    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    full_text = "\n\n".join(paragraphs)
    return full_text, None  # DOCX has no reliable page numbers


def extract_txt(file_path: str):
    encodings = ["utf-8", "utf-8-sig", "latin-1"]
    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                text = f.read()
            return text.strip(), None
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as exc:
            logger.error("TXT read failure for %s: %s", file_path, exc)
            raise ExtractionError("Could not read the text file.", "CORRUPTED_TXT")
    raise ExtractionError(
        "Could not decode the text file with any supported encoding.",
        "ENCODING_ERROR",
    )


def extract_text(file_path: str, extension: str):
    """
    Dispatch extraction by extension.
    Returns (full_text, pages) where pages is a list of (page_number, text) or None.
    Raises ExtractionError if no usable text was found.
    """
    if extension == ".pdf":
        full_text, pages = extract_pdf(file_path)
    elif extension == ".docx":
        full_text, pages = extract_docx(file_path)
    elif extension == ".txt":
        full_text, pages = extract_txt(file_path)
    else:
        raise ExtractionError(f"Unsupported extension: {extension}", "UNSUPPORTED_TYPE")

    if not full_text or len(full_text.strip()) < MIN_USABLE_CHARS:
        raise ExtractionError(
            "No usable text could be extracted from this document.",
            "NO_TEXT_EXTRACTED",
        )

    return full_text, pages
