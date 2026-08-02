"""Pre-package checks. Run before every code.zip rebuild:

    python preflight.py

Exits non-zero if anything fails, so a broken artifact can never be zipped.
Checks output.csv against the submission contract and README.md for the
non-ASCII characters that a cp437/cp1252 round-trip bakes in.
"""

from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

import pandas as pd

from src.config import (
    ALLOWED_ACTIONS,
    ALLOWED_MESSAGE_TYPES,
    CODE_DIR,
    DATASET_DIR,
    OUTPUT_COLUMNS,
    OUTPUT_FILE,
)

README_FILE = CODE_DIR / "README.md"


def check_output_csv() -> list[str]:
    """Row count, exact schema, ordering, enums, confidence range."""
    failures = []
    if not OUTPUT_FILE.exists():
        return [f"output.csv missing at {OUTPUT_FILE}"]

    messages = pd.read_csv(DATASET_DIR / "messages.csv")
    out = pd.read_csv(OUTPUT_FILE)

    if list(out.columns) != OUTPUT_COLUMNS:
        failures.append(f"columns are {list(out.columns)}, expected {OUTPUT_COLUMNS}")
    if len(out) != len(messages):
        failures.append(f"row count {len(out)}, expected {len(messages)}")
    if out["message_id"].duplicated().any():
        dupes = out.loc[out["message_id"].duplicated(), "message_id"].tolist()
        failures.append(f"duplicate message_ids: {dupes}")

    missing = set(messages["message_id"]) - set(out["message_id"])
    if missing:
        failures.append(f"{len(missing)} message_ids missing from output: {sorted(missing)[:5]}")
    if len(out) == len(messages) and list(out["message_id"]) != list(messages["message_id"]):
        failures.append("row order does not match messages.csv")

    bad_actions = sorted(set(out["action"].dropna()) - ALLOWED_ACTIONS)
    if bad_actions:
        failures.append(f"invalid action values: {bad_actions}")
    bad_types = sorted(set(out["message_type"].dropna()) - ALLOWED_MESSAGE_TYPES)
    if bad_types:
        failures.append(f"invalid message_type values: {bad_types}")

    if out[["action", "message_type", "reason", "confidence"]].isna().any().any():
        failures.append("empty values in action/message_type/reason/confidence")
    if not out["confidence"].between(0.0, 1.0).all():
        failures.append("confidence values outside [0, 1]")

    return failures


def check_readme_ascii() -> list[str]:
    """Non-ASCII in the README usually means encoding corruption, not intent."""
    if not README_FILE.exists():
        return [f"README.md missing at {README_FILE}"]

    text = README_FILE.read_text(encoding="utf-8")
    offenders = sorted(
        {(hex(ord(c)), unicodedata.name(c, "UNNAMED")) for c in text if ord(c) > 127}
    )
    if offenders:
        return [f"README.md has {len(offenders)} non-ASCII char types: {offenders}"]
    return []


def main() -> int:
    all_failures = []
    for label, failures in (
        ("output.csv contract", check_output_csv()),
        ("README.md encoding", check_readme_ascii()),
    ):
        if failures:
            all_failures.extend(failures)
            print(f"FAIL  {label}")
            for f in failures:
                print(f"        - {f}")
        else:
            print(f"ok    {label}")

    if all_failures:
        print(f"\nPREFLIGHT FAILED ({len(all_failures)} issue(s)). Do not rebuild code.zip.")
        return 1
    print("\nPREFLIGHT PASSED. Safe to rebuild code.zip.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
