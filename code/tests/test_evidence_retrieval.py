"""Tests for TF-IDF evidence retrieval over the real dataset."""

import pytest

from src.context_builder import ContextBuilder


@pytest.fixture(scope="module")
def builder():
    return ContextBuilder()


def test_similar_content_is_retrieved_across_senders(builder):
    # u_008 historically received an OTP/fee delivery scam (message_0107).
    # A scam-like query should surface it by content similarity.
    evidence = builder.get_historical_evidence(
        user_id="u_008",
        query_text="Delivery failed. Pay a small reattempt fee and enter the OTP to release your package.",
    )
    assert evidence, "expected content-similar evidence"
    assert any("fee" in (e["message_text"] or "").lower() or "otp" in (e["message_text"] or "").lower() for e in evidence)


def test_no_query_text_falls_back_to_relationship_recency(builder):
    with_query = builder.get_historical_evidence(user_id="u_002", business_id="business_002", query_text="")
    assert isinstance(with_query, list)  # empty query must not raise
    for item in with_query:
        assert item["message_id"].startswith("message_")


def test_unknown_user_returns_empty(builder):
    assert builder.get_historical_evidence(user_id="u_does_not_exist", query_text="hello") == []


def test_evidence_ids_always_come_from_users_own_history(builder):
    evidence = builder.get_historical_evidence(
        user_id="u_008", query_text="package delivery fee otp"
    )
    own_history = set(builder.message_history[builder.message_history.user_id == "u_008"]["message_id"])
    assert all(e["message_id"] in own_history for e in evidence)


def test_top_n_is_respected(builder):
    evidence = builder.get_historical_evidence(user_id="u_008", query_text="delivery fee otp package", top_n=2)
    assert len(evidence) <= 2
