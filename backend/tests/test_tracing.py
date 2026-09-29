"""Tracing helpers: safe defaults, no-op paths, background feedback."""

import os
from unittest.mock import MagicMock, patch

from app.core import tracing as tr


def test_configure_disabled_without_key(monkeypatch):
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    from app.core.config import settings

    monkeypatch.setattr(settings, "langsmith_tracing", False)
    monkeypatch.setattr(settings, "langsmith_api_key", "")
    assert tr.configure_tracing() is False
    assert "LANGSMITH_API_KEY" not in os.environ


def test_configure_enabled_sets_env(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "langsmith_tracing", True)
    monkeypatch.setattr(settings, "langsmith_api_key", "fake-key")
    monkeypatch.setattr(settings, "langsmith_project", "proj")
    assert tr.configure_tracing() is True
    assert os.environ["LANGSMITH_API_KEY"] == "fake-key"
    assert os.environ["LANGSMITH_PROJECT"] == "proj"


def test_current_run_id_none_when_disabled(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "langsmith_tracing", False)
    assert tr.current_run_id() is None


def test_post_feedback_noops_safely(monkeypatch):
    """Disabled, missing run, or error scores must never raise or post."""
    from app.core.config import settings

    monkeypatch.setattr(settings, "langsmith_tracing", False)
    mock_client = MagicMock()
    with patch.object(tr, "_LSClient", return_value=mock_client):
        tr.post_feedback_background(None, {"a": 1})
        tr.post_feedback_background("run-1", None)
        tr.post_feedback_background("run-1", {"error": "boom"})
        tr.post_feedback_background("run-1", {"dm_relevance": 5})
    mock_client.create_feedback.assert_not_called()
