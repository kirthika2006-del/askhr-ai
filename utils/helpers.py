"""Shared small helpers: response envelopes and id generation."""
import uuid
from datetime import datetime, timezone


def success_response(data=None):
    return {"success": True, "data": data if data is not None else {}}


def error_response(code: str, message: str):
    return {"success": False, "error": {"code": code, "message": message}}


def new_id() -> str:
    return str(uuid.uuid4())


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
