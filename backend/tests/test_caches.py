"""Caches: embedding hits, retrieval hits + epoch invalidation, save bump."""

from unittest.mock import patch

import pytest

from app.core import vectorstore as vs
from app.services import rag_chain as rc


def test_embed_query_hit_after_miss():
    vs.embeddings.cache_clear()
    a = vs.embeddings.embed_query("look around the tavern")
    b = vs.embeddings.embed_query("look around the tavern")
    info = vs.embeddings.cache_info()["embed_query"]
    assert a == b and len(a) > 100
    assert info["misses"] == 1 and info["hits"] == 1


def test_embed_documents_hit_after_miss():
    vs.embeddings.cache_clear()
    texts = ["alpha beta gamma", "delta epsilon zeta"]
    assert vs.embeddings.embed_documents(texts) == vs.embeddings.embed_documents(texts)
    info = vs.embeddings.cache_info()["embed_documents"]
    assert info["misses"] == 1 and info["hits"] == 1


@pytest.mark.asyncio
async def test_retrieval_hit_and_per_store_invalidation():
    rc._RETRIEVAL_CACHE.clear()
    lore0, camp0 = vs.collection_epoch("lore"), vs.collection_epoch("campaign")

    first = await rc._cached_retrieval("tell me about Mira")
    assert len(rc._RETRIEVAL_CACHE) == 2

    # case/whitespace-insensitive hit, same objects
    second = await rc._cached_retrieval("  TELL me about   mira ")
    assert second[0] is first[0] and second[1] is first[1]

    # lore write busts lore only; campaign entry reused
    vs.bump_collection_epoch("lore")
    third = await rc._cached_retrieval("tell me about Mira")
    assert third[0] is not first[0]
    assert third[1] is first[1]
    assert vs.collection_epoch("lore") == lore0 + 1
    assert vs.collection_epoch("campaign") == camp0


def test_save_turn_bumps_campaign_epoch():
    before = vs.collection_epoch("campaign")
    with patch.object(vs.campaign_vectorstore, "add_documents", return_value=None):
        rc._save_turn_to_campaign("q", "a", 999)
    assert vs.collection_epoch("campaign") == before + 1


def test_cache_stats_shape():
    stats = rc.cache_stats()
    assert set(stats) == {"embeddings", "retrieval_entries", "judge_entries", "epochs"}
    assert set(stats["epochs"]) == {"lore", "campaign"}
