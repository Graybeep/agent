"""Unit tests for the output-contract validator: malformed LLM outputs must be
sanitized into contract-safe rows without ever raising."""

from src.config import (
    FALLBACK_ACTION,
    FALLBACK_MESSAGE_TYPE,
    LLM_CONFIDENCE_CAP,
    RULE_CONFIDENCE,
)
from src.validator import apply_quiet_hours_policy, validate_prediction


def test_completely_empty_prediction_gets_fallbacks():
    row = validate_prediction("msg_x", {})
    assert row["action"] == FALLBACK_ACTION
    assert row["message_type"] == FALLBACK_MESSAGE_TYPE
    assert row["evidence_message_ids"] == "none"
    assert 0.0 <= row["confidence"] <= 1.0
    assert row["reason"]  # never empty


def test_invalid_enum_values_are_repaired():
    row = validate_prediction("msg_x", {"action": "URGENT!!", "message_type": "phishing"})
    assert row["action"] == FALLBACK_ACTION
    assert row["message_type"] == FALLBACK_MESSAGE_TYPE


def test_valid_values_pass_through_with_case_normalization():
    row = validate_prediction("msg_x", {"action": " Notify ", "message_type": "SCAM", "reason": "r", "confidence": 0.5})
    assert row["action"] == "notify"
    assert row["message_type"] == "scam"


def test_garbage_confidence_does_not_raise():
    for bad in ("high", None, [], {}, float("nan"), float("inf")):
        row = validate_prediction("msg_x", {"confidence": bad})
        assert 0.0 <= row["confidence"] <= 1.0  # also fails on NaN leak
    assert validate_prediction("msg_x", {"confidence": "0.7"})["confidence"] == 0.7


def test_out_of_range_confidence_is_clamped():
    assert validate_prediction("msg_x", {"confidence": 7})["confidence"] <= 1.0
    assert validate_prediction("msg_x", {"confidence": -3})["confidence"] == 0.0


def test_llm_confidence_is_capped_but_rule_confidence_is_not():
    llm = validate_prediction("msg_x", {"action": "mute", "confidence": 1.0}, source="llm")
    assert llm["confidence"] == LLM_CONFIDENCE_CAP
    rule = validate_prediction("msg_x", {"action": "mute", "confidence": 1.0}, source="rule")
    assert rule["confidence"] == 1.0


def test_rule_confidences_systematically_exceed_llm_cap():
    # every deterministic rule must score at least as certain as any LLM decision
    assert all(conf >= LLM_CONFIDENCE_CAP for conf in RULE_CONFIDENCE.values())


def test_fabricated_evidence_ids_are_dropped():
    row = validate_prediction(
        "msg_x",
        {"evidence_message_ids": ["message_0001", "message_9999", "totally_fake"]},
        valid_evidence_ids={"message_0001", "message_9999"},
        known_history_ids={"message_0001"},
    )
    # context filter drops totally_fake; history filter drops message_9999
    assert row["evidence_message_ids"] == "message_0001"


def test_evidence_as_semicolon_string_is_parsed_and_filtered():
    row = validate_prediction(
        "msg_x",
        {"evidence_message_ids": "message_0001;none;message_0002;message_0001"},
        valid_evidence_ids={"message_0001", "message_0002"},
        known_history_ids={"message_0001", "message_0002"},
    )
    assert row["evidence_message_ids"] == "message_0001;message_0002"  # deduped, 'none' token ignored


def test_reason_newlines_are_flattened_for_csv_safety():
    row = validate_prediction("msg_x", {"reason": "line one\nline two"})
    assert "\n" not in row["reason"]


def test_quiet_hours_downgrades_notify_to_digest():
    row = {"action": "notify", "message_type": "personal", "reason": "Friend asked something."}
    out = apply_quiet_hours_policy(row, in_quiet_hours=True)
    assert out["action"] == "digest"
    assert "quiet hours" in out["reason"]


def test_quiet_hours_urgent_override_still_notifies():
    row = {"action": "notify", "message_type": "urgent", "reason": "Emergency."}
    assert apply_quiet_hours_policy(row, in_quiet_hours=True)["action"] == "notify"


def test_otp_inside_quiet_hours_still_notifies_end_to_end():
    # OTP codes expire in minutes; the otp_delivery rule assigns type 'urgent',
    # which must be the quiet-hours exemption — batching an OTP defeats the rule.
    from src.rule_engine import RuleEngine

    message = {"message_text": "Your OTP is 482910. Valid for 10 minutes.", "user_id": "u_001"}
    context = {"business": {"known": True, "verified": True, "sender_domain_matches_official": True}, "evidence": []}
    decision = RuleEngine().decide(message, context)
    assert decision.rule_name == "otp_delivery"
    row = validate_prediction("m", {"action": decision.action, "message_type": decision.message_type, "reason": "r", "confidence": decision.confidence}, source="rule")
    assert apply_quiet_hours_policy(row, in_quiet_hours=True)["action"] == "notify"


def test_quiet_hours_leaves_other_actions_and_daytime_untouched():
    mute = {"action": "mute", "message_type": "spam", "reason": "r"}
    assert apply_quiet_hours_policy(mute, in_quiet_hours=True)["action"] == "mute"
    day = {"action": "notify", "message_type": "personal", "reason": "r"}
    assert apply_quiet_hours_policy(day, in_quiet_hours=False)["action"] == "notify"


def test_columns_match_output_contract():
    row = validate_prediction("msg_x", {})
    assert list(row.keys()) == [
        "message_id",
        "action",
        "message_type",
        "reason",
        "confidence",
        "evidence_message_ids",
    ]
