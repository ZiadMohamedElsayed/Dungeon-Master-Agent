"""Turn execution with a mocked LLM: shape, streaming, prompt wiring."""

from unittest.mock import AsyncMock, patch

import pytest

from app.services import rag_chain as rc


@pytest.mark.asyncio
async def test_play_turn_shape():
    with (
        patch.object(rc, "_generate_with_tools", new=AsyncMock(return_value="Mocked narration.")),
        patch.object(rc, "_get_turn_count", return_value=7),
        patch.object(rc, "_save_turn_to_campaign") as save,
        patch.object(rc, "get_recent_turns", return_value=[{"turn": 6, "content": "prev"}]),
    ):
        res = await rc.play_turn("i attack", evaluate=False)
    assert res["turn"] == 7
    assert res["answer"] == "Mocked narration."
    assert res["recent_turns"] == [6]
    assert set(res["sources"]) == {"lore", "campaign"}
    save.assert_called_once_with("i attack", "Mocked narration.", 7)


@pytest.mark.asyncio
async def test_stream_turn_event_sequence(monkeypatch):
    class FakeResp:
        content = "hello world"
        tool_calls = []

    class FakeLLM:
        async def ainvoke(self, messages):
            return FakeResp()

    monkeypatch.setattr(rc, "llm_with_tools", FakeLLM())
    monkeypatch.setattr(rc, "_get_turn_count", lambda: 8)
    monkeypatch.setattr(rc, "_save_turn_to_campaign", lambda *a: None)

    events = [e async for e in rc.stream_turn("look around")]
    kinds = [e["type"] for e in events]
    assert kinds[0] == "sources"
    assert kinds[-1] == "done"
    assert "token" in kinds
    assert events[-1]["turn"] == 8
    assert "".join(e["token"] for e in events if e["type"] == "token") == "hello world"


@pytest.mark.asyncio
async def test_evaluation_error_surfaced_not_raised():
    async def boom(*a, **k):
        raise RuntimeError("judge down")

    fake_eval = type("M", (), {"evaluate_without_reference": staticmethod(boom)})()
    with (
        patch.object(rc, "_generate_with_tools", new=AsyncMock(return_value="ans")),
        patch.object(rc, "_get_turn_count", return_value=1),
        patch.object(rc, "_save_turn_to_campaign"),
        patch.object(rc, "get_recent_turns", return_value=[]),
        patch.dict("sys.modules", {"app.services.evaluator": fake_eval}),
    ):
        res = await rc.play_turn("q", evaluate=True)
    assert "error" in res["evaluation"]
