"""Stage 4 guardrails: validate and repair every prediction before it is
written to output.csv, so the submission can never contain out-of-contract
values regardless of which stage produced the row.
"""

from __future__ import annotations

import math
from typing import Any

from .config import (
    ALLOWED_ACTIONS,
    ALLOWED_MESSAGE_TYPES,
    FALLBACK_ACTION,
    FALLBACK_CONFIDENCE,
    FALLBACK_MESSAGE_TYPE,
    FALLBACK_REASON,
    LLM_CONFIDENCE_CAP,
    OUTPUT_COLUMNS,
    QUIET_HOURS_OVERRIDE_TYPES,
)

__all__ = ["validate_prediction", "apply_quiet_hours_policy", "OUTPUT_COLUMNS"]


def apply_quiet_hours_policy(row: dict[str, Any], in_quiet_hours: bool) -> dict[str, Any]:
    """Downgrade notify -> digest inside the user's do-not-disturb window,
    unless the message type is an override (e.g. urgent). Call on a validated
    output row."""
    if (
        in_quiet_hours
        and row.get("action") == "notify"
        and row.get("message_type") not in QUIET_HOURS_OVERRIDE_TYPES
    ):
        row = dict(row)
        row["action"] = "digest"
        row["reason"] = (
            row.get("reason", "").rstrip(".")
            + "; delivered as digest because it arrived during the user's quiet hours."
        )
    return row


def validate_prediction(
    message_id: str,
    prediction: dict[str, Any],
    valid_evidence_ids: set[str] | None = None,
    known_history_ids: set[str] | None = None,
    source: str = "llm",
) -> dict[str, Any]:
    """Return a contract-safe output row, repairing any invalid field.

    source: 'rule' (deterministic, confidence trusted as calibrated) or 'llm'
    (confidence capped). Evidence IDs are filtered against the per-message
    context (valid_evidence_ids) and, independently, against the full set of
    real historical message IDs (known_history_ids) so a fabricated ID can
    never reach output.csv even if a caller forgets the context filter.
    """
    action = str(prediction.get("action", "")).strip().lower()
    if action not in ALLOWED_ACTIONS:
        action = FALLBACK_ACTION

    message_type = str(prediction.get("message_type", "")).strip().lower()
    if message_type not in ALLOWED_MESSAGE_TYPES:
        message_type = FALLBACK_MESSAGE_TYPE

    reason = str(prediction.get("reason", "")).strip().replace("\n", " ")
    if not reason:
        reason = FALLBACK_REASON

    try:
        confidence = float(prediction.get("confidence"))
        if math.isnan(confidence) or math.isinf(confidence):
            confidence = FALLBACK_CONFIDENCE
    except (TypeError, ValueError):
        confidence = FALLBACK_CONFIDENCE
    confidence = min(max(confidence, 0.0), 1.0)
    if source == "llm":
        confidence = min(confidence, LLM_CONFIDENCE_CAP)
    confidence = round(confidence, 2)

    evidence = prediction.get("evidence_message_ids") or []
    if isinstance(evidence, str):
        evidence = [e.strip() for e in evidence.split(";") if e.strip() and e.strip() != "none"]
    if valid_evidence_ids is not None:
        evidence = [mid for mid in evidence if mid in valid_evidence_ids]
    if known_history_ids is not None:
        evidence = [mid for mid in evidence if mid in known_history_ids]
    evidence_str = ";".join(dict.fromkeys(evidence)) if evidence else "none"

    return {
        "message_id": message_id,
        "action": action,
        "message_type": message_type,
        "reason": reason,
        "confidence": confidence,
        "evidence_message_ids": evidence_str,
    }
