"""Pytest bootstrap: isolate state BEFORE any app import.

- Point both Chroma stores at a temp dir (never touch ./dp).
- Strip LangSmith env so tests never emit traces.
- Force tracing off on the settings object per test.
"""

import os
import tempfile

_tmp = tempfile.mkdtemp(prefix="dm-test-")
os.environ["LORE_DB_PERSIST_DIR"] = os.path.join(_tmp, "lore")
os.environ["CAMPAIGN_DB_PERSIST_DIR"] = os.path.join(_tmp, "campaign")
# The LLM client is constructed at import time and newer langchain-google-genai
# requires a key present (never called in tests — all LLM uses are mocked).
# Local dev supplies the real key via backend/.env; CI has neither.
os.environ.setdefault("GEMINI_API_KEY", "test-dummy-key-never-used")
for _v in ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT", "LANGSMITH_ENDPOINT"):
    os.environ.pop(_v, None)

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _no_tracking(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "langsmith_tracing", False)
    monkeypatch.setattr(settings, "langsmith_api_key", "")
