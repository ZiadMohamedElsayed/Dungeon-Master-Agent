"""Short-term memory: windowing, order, limits, truncation."""

import pytest

from langchain_core.documents import Document

from app.core.vectorstore import campaign_vectorstore
from app.services import rag_chain as rc


def _seed(turns):
    campaign_vectorstore.add_documents([
        Document(
            page_content=f"[Turn {t}] Player: action{t} DM: outcome{t} CODE{t}",
            metadata={
                "source": "campaign_history",
                "turn": t,
                "timestamp": "t",
                "type": "turn_log",
            },
        )
        for t in turns
    ])


@pytest.fixture(autouse=True)
def _clean_campaign():
    campaign_vectorstore._collection.delete(where={"source": {"$ne": ""}})
    yield
    campaign_vectorstore._collection.delete(where={"source": {"$ne": ""}})


def test_window_returns_last_n_oldest_first():
    _seed(range(1, 9))
    recent = rc.get_recent_turns()
    assert [t["turn"] for t in recent] == [4, 5, 6, 7, 8]
    ctx = rc._format_recent_turns(recent)
    assert ctx.index("CODE4") < ctx.index("CODE8")


def test_window_smaller_than_history_returns_all():
    _seed([1, 2])
    assert [t["turn"] for t in rc.get_recent_turns()] == [1, 2]


def test_limit_and_empty():
    _seed(range(1, 6))
    assert [t["turn"] for t in rc.get_recent_turns(limit=1)] == [5]
    assert rc.get_recent_turns(limit=0) == []


def test_empty_db_message():
    assert "none yet" in rc._format_recent_turns([])


def test_long_turn_truncated(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "short_term_max_chars", 50)
    out = rc._format_recent_turns([{"turn": 1, "content": "x" * 200}])
    assert "[truncated]" in out
