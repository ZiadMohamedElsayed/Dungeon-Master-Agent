"""LangSmith tracing setup (env-gated, off by default).

LangChain auto-traces every runnable/LLM call once the standard
LANGSMITH_* env vars are present — no call-site changes needed.
This module copies the pydantic settings into os.environ at startup
so a single backend/.env controls everything.

It also hosts the turn-level helpers: `traceable` (real one when the
SDK is installed, identity fallback otherwise), `current_run_id()`,
and `post_feedback_background()` which attaches judge scores to a
turn's root run without blocking the response.
"""

import os
import threading
import time

from app.core.config import settings

try:
    from langsmith import Client as _LSClient
    from langsmith import traceable as _traceable
    from langsmith.run_helpers import get_current_run_tree as _get_run_tree

    _LANGSMITH_AVAILABLE = True
except ImportError:  # pragma: no cover - old image without the SDK
    _LANGSMITH_AVAILABLE = False


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


def _enabled() -> bool:
    return (
        _LANGSMITH_AVAILABLE
        and settings.langsmith_tracing
        and bool(settings.langsmith_api_key)
    )


def traceable(*dargs, **dkwargs):
    """LangSmith @traceable, or an identity decorator when unavailable."""
    if _LANGSMITH_AVAILABLE:
        return _traceable(*dargs, **dkwargs)
    if dargs and callable(dargs[0]) and len(dargs) == 1 and not dkwargs:
        return dargs[0]

    def wrap(fn):
        return fn

    return wrap


def current_run_id() -> str | None:
    """ID of the enclosing traced run, or None outside one / when disabled."""
    if not _enabled():
        return None
    try:
        tree = _get_run_tree()
        return str(tree.id) if tree is not None else None
    except Exception:
        return None


def post_feedback_background(run_id: str | None, scores: dict | None) -> None:
    """Attach judge scores as LangSmith feedback without blocking the turn.

    Runs in a daemon thread: waits for the run batch to flush, then posts
    one feedback per numeric score. Never raises.
    """
    if not run_id or not scores or not _enabled():
        return
    clean = {
        k: v for k, v in scores.items() if isinstance(v, (int, float)) and k != "error"
    }
    if not clean:
        return

    def _post() -> None:
        try:
            client = _LSClient(
                api_key=settings.langsmith_api_key,
                api_url=settings.langsmith_endpoint,
            )
            time.sleep(5)  # let the run batch flush so feedback has a parent
            for key, score in clean.items():
                try:
                    client.create_feedback(
                        run_id=run_id, key=key, score=score,
                        feedback_source_type="api",
                    )
                except Exception:
                    pass  # create_feedback already retries internally
        except Exception:
            pass

    threading.Thread(target=_post, daemon=True).start()
