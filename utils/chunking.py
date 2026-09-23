"""Configurable text chunking with overlap, splitting on natural boundaries."""
import re
from typing import List, Dict


def _split_into_paragraphs(text: str) -> List[str]:
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Dict]:
    """
    Split text into overlapping chunks, preferring paragraph/sentence
    boundaries over arbitrary character cuts.

    Returns a list of {"text": str, "chunk_index": int} dicts.
    """
    if not text or not text.strip():
        return []

    chunk_size = max(chunk_size, 200)
    chunk_overlap = max(0, min(chunk_overlap, chunk_size // 2))

    paragraphs = _split_into_paragraphs(text)
    if not paragraphs:
        paragraphs = [text.strip()]

    chunks = []
    current = ""

    def flush(buf: str):
        buf = buf.strip()
        if buf:
            chunks.append(buf)

    for para in paragraphs:
        if len(para) > chunk_size:
            # paragraph itself is too long — split on sentences
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                if len(current) + len(sent) + 1 <= chunk_size:
                    current = f"{current} {sent}".strip()
                else:
                    flush(current)
                    overlap_tail = current[-chunk_overlap:] if chunk_overlap else ""
                    current = f"{overlap_tail} {sent}".strip()
        else:
            if len(current) + len(para) + 1 <= chunk_size:
                current = f"{current}\n\n{para}".strip()
            else:
                flush(current)
                overlap_tail = current[-chunk_overlap:] if chunk_overlap else ""
                current = f"{overlap_tail}\n\n{para}".strip()

    flush(current)

    return [
        {"text": chunk, "chunk_index": idx}
        for idx, chunk in enumerate(chunks)
        if chunk.strip()
    ]
