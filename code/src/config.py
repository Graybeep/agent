"""Central configuration for the message notification router.

Every path, model identifier, retry parameter, and threshold used by the
pipeline lives here so behavior can be tuned in one place.
"""

from __future__ import annotations

from pathlib import Path

# ---- paths -------------------------------------------------------------------

CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
DATASET_DIR = REPO_ROOT / "dataset"
OUTPUT_FILE = DATASET_DIR / "output.csv"
SUMMARY_REPORT_FILE = DATASET_DIR / "summary_report.json"
CACHE_DIR = CODE_DIR / "cache"

# ---- Gemini models -----------------------------------------------------------

# Fallback chain, tried in order. The 2.5-generation names are retired for new
# accounts and gemini-2.0-flash has zero free-tier quota; -latest tracks
# whatever Google currently serves.
GEMINI_MODELS = ["gemini-flash-latest", "gemini-3-flash-preview", "gemini-3.1-flash-lite"]

# A 429/503 is retried per model with these waits (5s, 15s, 45s) before the
# model is considered exhausted and the chain moves on — a single transient 429
# must never write a model off, since most of the workload can depend on one model.
RATE_LIMIT_BACKOFF_S = [5, 15, 45]
ALL_MODELS_LIMITED_WAIT_S = 45  # extra wait before one full second pass over the chain

# ---- output contract ---------------------------------------------------------

OUTPUT_COLUMNS = [
    "message_id",
    "action",
    "message_type",
    "reason",
    "confidence",
    "evidence_message_ids",
]

ALLOWED_ACTIONS = {"notify", "digest", "mute"}
ALLOWED_MESSAGE_TYPES = {
    "personal",
    "urgent",
    "event",
    "payment",
    "business_update",
    "promotion",
    "greeting",
    "forward",
    "spam",
    "scam",
    "unknown",
}

# ---- confidence policy -------------------------------------------------------

# LLMs are habitually overconfident (0.98-1.0 on guesses); ground-truth sample
# labels top out at 0.91. Confidence encodes the decision source: deterministic
# rule hits occupy the 0.90-0.95 band, ambiguous LLM judgments sit BELOW it, so
# a rule decision always scores at least as certain as any LLM one.
#
# Hard-clamping every LLM score to the cap destroyed the model's own ranking
# (69 of 70 rows landed on exactly 0.90). Instead the raw self-reported score
# is linearly rescaled from its practical range into the LLM band, preserving
# relative ordering while staying under the rule tier. LLM_CONFIDENCE_CAP
# remains as a backstop so a rescaling bug can never cross into rule territory.
LLM_CONFIDENCE_CAP = 0.90
LLM_RAW_MIN = 0.70  # raw scores at/below this map to the bottom of the band
LLM_RAW_MAX = 1.00
LLM_BAND_MIN = 0.75
LLM_BAND_MAX = 0.90

RULE_CONFIDENCE = {
    "otp_solicitation": 0.95,
    "scam_business_sender": 0.94,
    "otp_delivery": 0.93,
    "optout_promotion": 0.92,
    "high_forward_chain": 0.91,
    "muted_group_no_mention": 0.90,
}

FALLBACK_ACTION = "digest"  # safest middle ground when a prediction is broken
FALLBACK_MESSAGE_TYPE = "unknown"
FALLBACK_CONFIDENCE = 0.3
FALLBACK_REASON = "Automatic fallback: the original prediction was invalid."

# Written for every remaining LLM-bound row once the whole model chain is
# exhausted mid-run, so the run completes deterministically instead of dying.
QUOTA_EXHAUSTED_PREDICTION = {
    "action": "digest",
    "message_type": "unknown",
    "confidence": 0.3,
    "reason": "LLM quota exhausted mid-run; deferred to digest for safe later review.",
    "evidence_message_ids": [],
}

# ---- rule thresholds ---------------------------------------------------------

SCAM_MIN_SIGNALS = 2  # independent bad signals before branding a business scam
SCAM_REPORT_RATE_THRESHOLD = 0.05  # user reports / messages sent (30d)
YOUNG_DOMAIN_AGE_DAYS = 60
MUTED_FORWARD_MIN_COUNT = 3  # forwarded_count for 'forward' type in muted groups
FORWARD_CHAIN_MIN_COUNT = 5  # forwarded_count that marks a mass-forwarded chain

# ---- context enrichment ------------------------------------------------------

EVIDENCE_TOP_N = 3  # historical messages surfaced per message
NOTIFICATION_LOAD_DAYS = 7  # recent days for notification-load summary

# TF-IDF evidence retrieval: cosine similarity over the user's history, with a
# score boost for messages from the same sender/business/group, falling back to
# relationship-recency when the query has no usable text.
EVIDENCE_MIN_SIMILARITY = 0.08
EVIDENCE_RELATIONSHIP_BOOST = 0.2

# ---- quiet-hours policy ------------------------------------------------------

# notify decisions inside the user's do-not-disturb window are downgraded to
# digest unless the message_type is in the override set.
QUIET_HOURS_OVERRIDE_TYPES = {"urgent"}
