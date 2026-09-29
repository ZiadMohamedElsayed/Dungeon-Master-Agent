"""Evaluator: parsing, judge caching, total-failure signal."""

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.documents import Document

from app.services import evaluator as ev


class FakeMsg:
    def __init__(self, text):
        self.content = text


class FakeJudgeLLM:
    def __init__(self, reply="4"):
        self.reply = reply
        self.calls = 0

    async def ainvoke(self, prompt):
        self.calls += 1
        assert "Reply with ONLY the integer score" in prompt
        return FakeMsg(self.reply)


@pytest.fixture
def docs():
    return [Document(page_content="Mira tends the bar.", metadata={})]


def test_parse_score():
    assert ev._parse_score("Score: 5") == 5
    assert ev._parse_score("4") == 4
    assert ev._parse_score("no digits here") is None
    assert ev._parse_score("") is None


def test_cache_key_stable_and_sensitive(docs):
    ctx = ev._build_contexts(docs, [])
    k1 = ev._judge_cache_key("q", "a", ctx, None, ["dm_relevance"])
    assert k1 == ev._judge_cache_key("q", "a", ctx, None, ["dm_relevance"])
    assert k1 != ev._judge_cache_key("q!", "a", ctx, None, ["dm_relevance"])


@pytest.mark.asyncio
async def test_second_identical_eval_makes_zero_calls(docs):
    llm = FakeJudgeLLM()
    with patch.object(ev, "_judge_llm", return_value=llm):
        ev._JUDGE_CACHE.clear()
        r1 = await ev.evaluate_without_reference("q?", "a!", docs, [])
        assert llm.calls == 3
        r2 = await ev.evaluate_without_reference("q?", "a!", docs, [])
        assert llm.calls == 3
    assert r1 == r2 == {
        "dm_relevance": 4,
        "lore_consistency": 4,
        "narrative_quality": 4,
    }


@pytest.mark.asyncio
async def test_reference_adds_two_metrics(docs):
    with patch.object(ev, "_judge_llm", return_value=FakeJudgeLLM("5")):
        ev._JUDGE_CACHE.clear()
        res = await ev.evaluate_with_reference("q", "a", docs, [], "ref")
    assert set(res) == {
        "dm_relevance",
        "lore_consistency",
        "narrative_quality",
        "context_precision",
        "context_recall",
    }
    assert all(v == 5 for v in res.values())


@pytest.mark.asyncio
async def test_total_failure_raises(docs):
    with patch.object(ev, "_judge_llm", return_value=FakeJudgeLLM("garbage no score")):
        ev._JUDGE_CACHE.clear()
        with pytest.raises(RuntimeError):
            await ev.evaluate_without_reference("q", "a", docs, [])
