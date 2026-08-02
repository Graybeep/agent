"""Stage 3A: deterministic fast-path rules that decide clear-cut messages
before any LLM call.

Rules are intentionally conservative: they only fire on high-confidence
patterns, and everything ambiguous falls through (returns None) to the LLM
reasoning engine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .config import (
    FORWARD_CHAIN_MIN_COUNT,
    MUTED_FORWARD_MIN_COUNT,
    RULE_CONFIDENCE,
    SCAM_MIN_SIGNALS,
    SCAM_REPORT_RATE_THRESHOLD,
    YOUNG_DOMAIN_AGE_DAYS,
)


@dataclass
class RuleDecision:
    action: str
    message_type: str
    reason: str
    confidence: float
    evidence_message_ids: list[str] = field(default_factory=list)
    rule_name: str = ""


# Genuine code DELIVERY: "Your OTP is 482910", "use code 123456 to log in".
OTP_DELIVERY_PATTERN = re.compile(
    r"\b(otp|one[- ]time (password|code)|verification code|security code|login code)\b.{0,40}?\b\d{4,8}\b"
    r"|\b\d{4,8}\b.{0,40}?\b(is your|as your) (otp|code|verification code)\b",
    re.IGNORECASE | re.DOTALL,
)

# Code SOLICITATION: someone asking the user to send a code back — an
# account-takeover pattern, never a legitimate OTP notification.
OTP_SOLICITATION_PATTERN = re.compile(
    r"\b(reply|send|share|forward|tell)\b.{0,60}?\b(otp|(\d[- ]?digit|login|verification|security) code)\b"
    r"|\b(otp|code)\b.{0,40}?\byou (just )?received\b",
    re.IGNORECASE | re.DOTALL,
)

MENTION_PATTERN = "@{user_id}"

# Light content classification for messages muted by group preference, so the
# message_type still reflects what the message IS rather than 'unknown'.
GREETING_PATTERN = re.compile(
    r"\b(good morning|good night|stay blessed|blessing|stay positive|smile today|"
    r"bhagwan|god bless|have a (good|great|nice) day)\b",
    re.IGNORECASE,
)
PROMOTION_PATTERN = re.compile(
    r"\b(selling|sale|offer|discount|price|buy now|token|booking open|plots?|"
    r"size [SML]|barely used|final few)\b|₹|\bRs\.?\s?\d",
    re.IGNORECASE,
)

# Marketing language from a business the user opted out of promotions from.
PROMO_TERMS_PATTERN = re.compile(
    r"\b(discount|offer|sale|deal|coupon|% ?off|cashback|flash sale|"
    r"limited time|exclusive)\b",
    re.IGNORECASE,
)


def _classify_muted_content(text: str, forwarded_count: int, group_type: str | None = None) -> str:
    if text and GREETING_PATTERN.search(text):
        return "greeting"
    if text and PROMOTION_PATTERN.search(text):
        return "promotion"
    if forwarded_count >= MUTED_FORWARD_MIN_COUNT:
        return "forward"
    # Marketplace groups exist to sell things: listings without explicit promo
    # keywords ("photos attached, pickup this weekend") are still promotions.
    if group_type == "marketplace":
        return "promotion"
    return "unknown"


class RuleEngine:
    """Applies deterministic routing rules; returns None when no rule fires."""

    def decide(
        self,
        message: dict[str, Any],
        context: dict[str, Any],
        media_text: str | None = None,
    ) -> RuleDecision | None:
        text = _combined_text(message, media_text)

        for rule in (
            self._rule_scam_business_sender,
            self._rule_otp_solicitation,
            self._rule_otp_delivery,
            self._rule_optout_promotion,
            self._rule_high_forward_chain,
            self._rule_muted_group_no_mention,
        ):
            decision = rule(message, context, text)
            if decision is not None:
                return decision
        return None

    # ---- rule 1: untrustworthy business sender -> mute as scam ---------------

    def _rule_scam_business_sender(
        self, message: dict[str, Any], context: dict[str, Any], text: str
    ) -> RuleDecision | None:
        business = context.get("business")
        if not business or not business.get("known"):
            return None

        domain_mismatch = not business.get("sender_domain_matches_official", True)
        unverified = not business.get("verified", False)
        sent = business.get("business_messages_sent_30d") or 0
        reports = business.get("business_user_reports_30d") or 0
        high_report_rate = sent > 0 and (reports / sent) > SCAM_REPORT_RATE_THRESHOLD
        young_domain = (business.get("domain_used_by_sender_age_days") or 9999) < YOUNG_DOMAIN_AGE_DAYS

        # Require two independent bad signals so a merely-unverified small
        # business is not auto-branded a scam.
        signals = [domain_mismatch, unverified, high_report_rate, young_domain]
        if sum(signals) < SCAM_MIN_SIGNALS:
            return None

        details = []
        if domain_mismatch:
            details.append(
                f"sender domain {business.get('domain_used_by_sender')!r} does not match "
                f"official domain {business.get('official_domain')!r}"
            )
        if unverified:
            details.append("account is unverified")
        if high_report_rate:
            details.append("high user report rate")
        if young_domain:
            details.append("sender domain is very new")

        return RuleDecision(
            action="mute",
            message_type="scam",
            reason=f"Business sender shows scam signals: {'; '.join(details)}.",
            confidence=RULE_CONFIDENCE["scam_business_sender"],
            evidence_message_ids=_reported_or_dismissed_evidence(context),
            rule_name="scam_business_sender",
        )

    # ---- rule 2a: someone asking the user to share a code -> scam ------------

    def _rule_otp_solicitation(
        self, message: dict[str, Any], context: dict[str, Any], text: str
    ) -> RuleDecision | None:
        if not text or not OTP_SOLICITATION_PATTERN.search(text):
            return None
        return RuleDecision(
            action="mute",
            message_type="scam",
            reason="Message asks the user to share a security code, a known account-takeover pattern.",
            confidence=RULE_CONFIDENCE["otp_solicitation"],
            evidence_message_ids=_reported_or_dismissed_evidence(context),
            rule_name="otp_solicitation",
        )

    # ---- rule 2b: genuine time-sensitive code delivery -> notify -------------

    def _rule_otp_delivery(
        self, message: dict[str, Any], context: dict[str, Any], text: str
    ) -> RuleDecision | None:
        if not text or not OTP_DELIVERY_PATTERN.search(text):
            return None

        # Only trust code delivery from a business sender that is verified and
        # using its official domain; anything else goes to the LLM for judgment.
        business = context.get("business")
        if business is None or not (
            business.get("verified") and business.get("sender_domain_matches_official")
        ):
            return None

        return RuleDecision(
            action="notify",
            message_type="urgent",
            reason="A verified sender delivered a time-sensitive security code.",
            confidence=RULE_CONFIDENCE["otp_delivery"],
            evidence_message_ids=[],
            rule_name="otp_delivery",
        )

    # ---- rule 2c: promotion to a user who opted out -> mute ------------------

    def _rule_optout_promotion(
        self, message: dict[str, Any], context: dict[str, Any], text: str
    ) -> RuleDecision | None:
        business = context.get("business")
        if not business or not business.get("known") or not business.get("user_has_relationship"):
            return None
        # Only fire on an explicit opt-out; opted-in users' promotions are
        # digest-worthy and belong to the LLM.
        if business.get("allows_promotions") is not False:
            return None
        if not text or not PROMO_TERMS_PATTERN.search(text):
            return None

        return RuleDecision(
            action="mute",
            message_type="promotion",
            reason="The user opted out of promotions from this business, so marketing messages are muted.",
            confidence=RULE_CONFIDENCE["optout_promotion"],
            evidence_message_ids=_reported_or_dismissed_evidence(context),
            rule_name="optout_promotion",
        )

    # ---- rule 2d: mass-forwarded chain content -> mute -----------------------

    def _rule_high_forward_chain(
        self, message: dict[str, Any], context: dict[str, Any], text: str
    ) -> RuleDecision | None:
        forwarded = int(message.get("forwarded_count") or 0)
        if forwarded < FORWARD_CHAIN_MIN_COUNT:
            return None

        # Type still reflects the content (a forwarded good-morning chain is a
        # greeting); default to 'forward' for generic chain content.
        group_type = (context.get("group") or {}).get("group_type")
        message_type = _classify_muted_content(text, forwarded, group_type)
        if message_type == "unknown":
            message_type = "forward"

        return RuleDecision(
            action="mute",
            message_type=message_type,
            reason=f"The message was forwarded {forwarded} times, indicating mass-forwarded chain content rather than a personal message.",
            confidence=RULE_CONFIDENCE["high_forward_chain"],
            evidence_message_ids=_reported_or_dismissed_evidence(context),
            rule_name="high_forward_chain",
        )

    # ---- rule 3: muted group without a direct mention -> mute ----------------

    def _rule_muted_group_no_mention(
        self, message: dict[str, Any], context: dict[str, Any], text: str
    ) -> RuleDecision | None:
        group = context.get("group")
        if not group or not group.get("known") or not group.get("user_muted_group"):
            return None

        mention = MENTION_PATTERN.format(user_id=message.get("user_id", ""))
        if mention.lower() in (text or "").lower():
            return None  # direct mention overrides the mute; let the LLM weigh urgency

        forwarded = int(message.get("forwarded_count") or 0)
        return RuleDecision(
            action="mute",
            message_type=_classify_muted_content(text, forwarded, group.get("group_type")),
            reason="The user muted this group and the message does not mention them directly.",
            confidence=RULE_CONFIDENCE["muted_group_no_mention"],
            evidence_message_ids=_reported_or_dismissed_evidence(context),
            rule_name="muted_group_no_mention",
        )


def _combined_text(message: dict[str, Any], media_text: str | None) -> str:
    parts = []
    text = message.get("message_text")
    if isinstance(text, str):
        parts.append(text)
    if media_text:
        parts.append(media_text)
    return "\n".join(parts)


def _reported_or_dismissed_evidence(context: dict[str, Any]) -> list[str]:
    """Historical messages the user reported/dismissed/muted support a mute call."""
    return [
        item["message_id"]
        for item in context.get("evidence", [])
        if item.get("was_reported") or item.get("muted_after") or item.get("was_dismissed")
    ]
