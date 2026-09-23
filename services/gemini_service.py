"""Wraps Gemini generateContent calls for grounded answers."""
import logging
import time
import random

from google import genai

logger = logging.getLogger(__name__)

# Retry on transient overload/rate-limit errors only (503, 429).
RETRYABLE_MARKERS = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED")
MAX_RETRIES = 3
BASE_DELAY_SECONDS = 1.5


class GeminiService:
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def is_configured(self) -> bool:
        return bool(self.client)

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        message = str(exc)
        return any(marker in message for marker in RETRYABLE_MARKERS)

    def generate_answer(self, prompt: str) -> str:
        last_exc = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                )
                text = getattr(response, "text", None)
                if not text:
                    raise RuntimeError("Empty response from Gemini.")
                return text.strip()
            except Exception as exc:
                last_exc = exc
                if attempt < MAX_RETRIES and self._is_retryable(exc):
                    delay = BASE_DELAY_SECONDS * (2 ** attempt) + random.uniform(0, 0.5)
                    logger.warning(
                        "Gemini call attempt %d failed (retryable), retrying in %.1fs: %s",
                        attempt + 1, delay, exc,
                    )
                    time.sleep(delay)
                    continue
                logger.error("Gemini generation failed: %s", exc)
                raise RuntimeError(f"Gemini API call failed: {exc}")
        raise RuntimeError(f"Gemini API call failed after retries: {last_exc}")
