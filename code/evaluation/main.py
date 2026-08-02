"""Evaluate the routing pipeline against the solved examples in
dataset/sample_messages.csv: run every sample through the same
rules -> LLM path used for real predictions and score agreement on
action and message_type.

Run from code/:  python evaluation/main.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.context_builder import ContextBuilder, is_within_quiet_hours
from src.llm_router import LLMRouter
from src.media_extractor import MediaExtractor
from src.rule_engine import RuleEngine
from src.validator import apply_quiet_hours_policy, validate_prediction

DATASET_DIR = Path(__file__).resolve().parents[2] / "dataset"


def main() -> int:
    samples = pd.read_csv(DATASET_DIR / "sample_messages.csv")
    builder = ContextBuilder(DATASET_DIR)
    extractor = MediaExtractor(DATASET_DIR)
    rules = RuleEngine()
    router = LLMRouter()

    known_history_ids = set(builder.message_history["message_id"])
    results = []
    for _, row in samples.iterrows():
        message = row.to_dict()
        media_text = extractor.get_media_text(message.get("media_type"), message.get("media_id"))
        context = builder.build_context(row, media_text)
        valid_evidence = {item["message_id"] for item in context.get("evidence", [])}

        decision = rules.decide(message, context, media_text)
        if decision is not None:
            prediction = {
                "action": decision.action,
                "message_type": decision.message_type,
                "reason": decision.reason,
                "confidence": decision.confidence,
                "evidence_message_ids": decision.evidence_message_ids,
            }
            source = f"rule:{decision.rule_name}"
        else:
            prediction = router.decide(message, context, media_text)
            source = "llm"

        pred = validate_prediction(
            message["message_id"],
            prediction,
            valid_evidence,
            known_history_ids,
            "rule" if source.startswith("rule") else "llm",
        )
        dnd = context.get("user", {}).get("do_not_disturb_window")
        in_quiet = False
        if dnd and message.get("created_at"):
            try:
                in_quiet = is_within_quiet_hours(
                    dnd, datetime.strptime(str(message["created_at"]), "%Y-%m-%d %H:%M")
                )
            except ValueError:
                pass
        pred = apply_quiet_hours_policy(pred, in_quiet)
        results.append(
            {
                "message_id": message["message_id"],
                "source": source,
                "expected_action": row["action"],
                "predicted_action": pred["action"],
                "expected_type": row["message_type"],
                "predicted_type": pred["message_type"],
                "action_match": row["action"] == pred["action"],
                "type_match": row["message_type"] == pred["message_type"],
            }
        )

    df = pd.DataFrame(results)
    n = len(df)
    print(f"\nsamples: {n}")
    print(f"action accuracy:       {df.action_match.mean():.0%} ({df.action_match.sum()}/{n})")
    print(f"message_type accuracy: {df.type_match.mean():.0%} ({df.type_match.sum()}/{n})")
    print(f"both correct:          {(df.action_match & df.type_match).mean():.0%}")

    misses = df[~df.action_match | ~df.type_match]
    if not misses.empty:
        print("\nmismatches:")
        print(
            misses[
                ["message_id", "source", "expected_action", "predicted_action", "expected_type", "predicted_type"]
            ].to_string(index=False)
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
