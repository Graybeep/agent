"""Route an arbitrary messages CSV through the pipeline without touching the
submission artifacts.

Uses the same stages as main.py (context -> media -> rules -> LLM -> validate
-> quiet hours) but reads and writes caller-supplied paths, so dataset/output.csv
is never opened.

    python run_sample.py ../sample_test_messages.csv ../sample_output.csv
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import DATASET_DIR, OUTPUT_COLUMNS
from src.context_builder import ContextBuilder, is_within_quiet_hours
from src.llm_router import LLMRouter
from src.media_extractor import MediaExtractor
from src.rule_engine import RuleEngine
from src.validator import apply_quiet_hours_policy, validate_prediction


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    source, target = Path(sys.argv[1]), Path(sys.argv[2])

    messages = pd.read_csv(source)
    builder = ContextBuilder(DATASET_DIR)
    extractor = MediaExtractor(DATASET_DIR)
    rules = RuleEngine()
    router = LLMRouter()
    known_history_ids = set(builder.message_history["message_id"])

    rows, sources = [], {"rule": 0, "llm": 0, "fallback": 0}
    for _, row in messages.iterrows():
        message = row.to_dict()
        message_id = message["message_id"]

        try:
            media_text = extractor.get_media_text(
                message.get("media_type"), message.get("media_id")
            )
        except Exception as exc:
            print(f"  {message_id}: media extraction failed ({exc}); text only")
            media_text = None

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
            source_kind, label = "rule", decision.rule_name
        else:
            source_kind, label = "llm", "llm"
            try:
                prediction = router.decide(message, context, media_text)
            except Exception as exc:
                print(f"  {message_id}: LLM failed ({exc})")
                prediction, source_kind, label = {}, "fallback", "fallback"

        validated = validate_prediction(
            message_id, prediction, valid_evidence, known_history_ids,
            "rule" if source_kind == "rule" else "llm",
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
        validated = apply_quiet_hours_policy(validated, in_quiet)

        sources[source_kind] += 1
        rows.append(validated)
        print(f"  {message_id}: {validated['action']}/{validated['message_type']} "
              f"({validated['confidence']}) via {label}")

    with target.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nwrote {target} ({len(rows)} rows)")
    print(f"sources: {sources}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
