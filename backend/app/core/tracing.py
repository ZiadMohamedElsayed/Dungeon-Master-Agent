"""LangSmith tracing setup (env-gated, off by default).

LangChain auto-traces every runnable/LLM call once the standard
LANGSMITH_* env vars are present — no call-site changes needed.
This module copies the pydantic settings into os.environ at startup
so a single backend/.env controls everything.
"""

import os

from app.core.config import settings


def configure_tracing() -> bool:
    """Export LANGSMITH_* into the environment if tracing is enabled.

    Returns True when tracing was enabled, False when left disabled
    (no key or LANGSMITH_TRACING=false). Never raises: tracing must
    not be able to crash server boot.
    """
    try:
        if not settings.langsmith_tracing or not settings.langsmith_api_key:
            return False
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
        return True
    except Exception:
        return False
