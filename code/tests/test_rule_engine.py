"""Unit tests for the deterministic fast-path rules."""

import pytest

from src.rule_engine import RuleEngine


@pytest.fixture
def engine():
    return RuleEngine()


def _business_context(**overrides):
    business = {
        "known": True,
        "verified": True,
        "sender_domain_matches_official": True,
        "business_messages_sent_30d": 1000,
        "business_user_reports_30d": 2,
        "domain_used_by_sender_age_days": 900,
        "domain_used_by_sender": "brand.com",
        "official_domain": "brand.com",
    }
    business.update(overrides)
    return {"business": business, "evidence": []}


# ---- OTP handling ------------------------------------------------------------

def test_genuine_otp_from_verified_business_notifies(engine):
    message = {"message_text": "Your OTP is 482910. Valid for 10 minutes.", "user_id": "u_001"}
    decision = engine.decide(message, _business_context())
    assert decision is not None
    assert decision.rule_name == "otp_delivery"
    assert decision.action == "notify"
    assert decision.message_type == "urgent"


def test_otp_from_unverified_sender_is_not_fast_pathed(engine):
    message = {"message_text": "Your OTP is 482910.", "user_id": "u_001"}
    context = _business_context(verified=False)
    decision = engine.decide(message, context)
    # falls through to the LLM rather than blindly notifying
    assert decision is None or decision.rule_name != "otp_delivery"


def test_otp_solicitation_is_muted_as_scam(engine):
    message = {
        "message_text": "Please reply with the 6 digit login code you just received to keep your account active.",
        "user_id": "u_001",
    }
    decision = engine.decide(message, {"evidence": []})
    assert decision is not None
    assert decision.rule_name == "otp_solicitation"
    assert decision.action == "mute"
    assert decision.message_type == "scam"


# ---- business scam filter ----------------------------------------------------

def test_domain_mismatch_plus_reports_is_scam(engine):
    context = _business_context(
        sender_domain_matches_official=False,
        domain_used_by_sender="brand-refunds.help",
        business_user_reports_30d=200,
    )
    decision = engine.decide({"message_text": "Refund pending, pay fee.", "user_id": "u_001"}, context)
    assert decision is not None
    assert decision.rule_name == "scam_business_sender"
    assert decision.action == "mute"
    assert decision.message_type == "scam"


def test_single_bad_signal_is_not_enough(engine):
    # unverified alone (one signal) must NOT auto-brand a business as scam
    context = _business_context(verified=False)
    decision = engine.decide({"message_text": "Hello from our shop!", "user_id": "u_001"}, context)
    assert decision is None


def test_healthy_verified_business_passes_through(engine):
    decision = engine.decide({"message_text": "Your order shipped.", "user_id": "u_001"}, _business_context())
    assert decision is None


# ---- muted groups ------------------------------------------------------------

def _muted_group_context(group_type="friends"):
    return {"group": {"known": True, "user_muted_group": True, "group_type": group_type}, "evidence": []}


def test_muted_group_without_mention_is_muted(engine):
    message = {"message_text": "Anyone up for cricket tonight?", "user_id": "u_007", "forwarded_count": 0}
    decision = engine.decide(message, _muted_group_context())
    assert decision is not None
    assert decision.rule_name == "muted_group_no_mention"
    assert decision.action == "mute"


def test_muted_group_with_direct_mention_falls_through(engine):
    message = {"message_text": "@u_007 your car is blocking the gate", "user_id": "u_007", "forwarded_count": 0}
    assert engine.decide(message, _muted_group_context()) is None


def test_muted_group_greeting_gets_content_type(engine):
    message = {"message_text": "Good morning, stay blessed everyone!", "user_id": "u_007", "forwarded_count": 0}
    decision = engine.decide(message, _muted_group_context())
    assert decision is not None
    assert decision.message_type == "greeting"


def test_muted_marketplace_group_defaults_to_promotion(engine):
    # a listing with no promo keywords still classifies by the group's purpose
    message = {"message_text": "Photos are attached. Pickup near Gate 2 this weekend.", "user_id": "u_007", "forwarded_count": 0}
    decision = engine.decide(message, _muted_group_context(group_type="marketplace"))
    assert decision is not None
    assert decision.rule_name == "muted_group_no_mention"
    assert decision.message_type == "promotion"


# ---- opted-out promotions ----------------------------------------------------

def test_promo_to_opted_out_user_is_muted(engine):
    context = _business_context(user_has_relationship=True, allows_promotions=False)
    message = {"message_text": "Flash sale! 50% off everything, limited time offer.", "user_id": "u_001"}
    decision = engine.decide(message, context)
    assert decision is not None
    assert decision.rule_name == "optout_promotion"
    assert decision.action == "mute"
    assert decision.message_type == "promotion"


def test_promo_to_opted_in_user_falls_through(engine):
    context = _business_context(user_has_relationship=True, allows_promotions=True)
    message = {"message_text": "Flash sale! 50% off everything.", "user_id": "u_001"}
    assert engine.decide(message, context) is None


def test_promo_without_relationship_falls_through(engine):
    message = {"message_text": "Huge discount offer today!", "user_id": "u_001"}
    assert engine.decide(message, _business_context()) is None


# ---- mass-forwarded chains ---------------------------------------------------

def test_high_forward_count_is_muted_as_chain(engine):
    message = {"message_text": "Doctors don't tell you this health secret...", "user_id": "u_001", "forwarded_count": 9}
    decision = engine.decide(message, {"evidence": []})
    assert decision is not None
    assert decision.rule_name == "high_forward_chain"
    assert decision.action == "mute"
    assert decision.message_type == "forward"


def test_high_forward_greeting_keeps_greeting_type(engine):
    message = {"message_text": "Good morning! Stay blessed and share this.", "user_id": "u_001", "forwarded_count": 6}
    decision = engine.decide(message, {"evidence": []})
    assert decision is not None
    assert decision.rule_name == "high_forward_chain"
    assert decision.message_type == "greeting"


def test_low_forward_count_falls_through(engine):
    message = {"message_text": "Sharing this article, quite interesting.", "user_id": "u_001", "forwarded_count": 2}
    assert engine.decide(message, {"evidence": []}) is None
