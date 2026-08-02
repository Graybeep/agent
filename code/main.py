"""Checkpointed execution loop for the WhatsApp message notification router.

Pipeline per message: media extraction (cached) -> context enrichment ->
deterministic rule engine -> LLM router (only when no rule fired) ->
validation + quiet-hours policy -> append to dataset/output.csv.

The output file is the checkpoint: rows already predicted are skipped on
restart, so an interrupted run (rate limits, network, Ctrl+C) resumes safely.

Usage:
    python main.py                 # process all pending messages
    python main.py --limit 10      # only the first 10 pending messages
    python main.py --fresh         # discard previous predictions, start over
    python main.py --dry-run       # print decisions, write nothing
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from src.config import (
    DATASET_DIR,
    OUTPUT_COLUMNS,
    OUTPUT_FILE,
    QUOTA_EXHAUSTED_PREDICTION,
    SUMMARY_REPORT_FILE,
)
from src.context_builder import ContextBuilder, is_within_quiet_hours
from src.llm_router import LLMRouter
from src.media_extractor import MediaExtractor
from src.rule_engine import RuleEngine
from src.validator import apply_quiet_hours_policy, validate_prediction

console = Console()


def _in_quiet_hours(context: dict, created_at) -> bool:
    dnd = context.get("user", {}).get("do_not_disturb_window")
    if not dnd or not created_at:
        return False
    try:
        when = datetime.strptime(str(created_at), "%Y-%m-%d %H:%M")
    except ValueError:
        return False
    return is_within_quiet_hours(dnd, when)


def load_completed_ids(output_file: Path) -> set[str]:
    """Message IDs already predicted (non-empty action) in a previous run."""
    if not output_file.exists():
        return set()
    df = pd.read_csv(output_file, dtype=str)
    if "action" not in df.columns:
        return set()
    done = df[df["action"].notna() & (df["action"].str.strip() != "")]
    return set(done["message_id"])


def reset_output(output_file: Path, completed_ids: set[str]) -> None:
    """Rewrite output.csv keeping only genuinely completed rows.

    The provided template contains every message_id with empty prediction
    columns; those placeholder rows must not survive into the final file.
    """
    if output_file.exists() and completed_ids:
        df = pd.read_csv(output_file, dtype=str)
        df = df[df["message_id"].isin(completed_ids)]
        df.to_csv(output_file, index=False)
    else:
        with output_file.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(OUTPUT_COLUMNS)


def append_row(output_file: Path, row: dict) -> None:
    with output_file.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS).writerow(row)


def write_summary_report(stats: dict, report_file: Path) -> None:
    report_file.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def print_summary(stats: dict) -> None:
    table = Table(title="Routing Summary", show_header=True, header_style="bold")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Messages processed", str(stats["messages_processed"]))
    table.add_row("Rule-engine decisions", str(stats["sources"]["rule"]))
    table.add_row("LLM decisions", str(stats["sources"]["llm"]))
    table.add_row("Fallbacks", str(stats["sources"]["fallback"]))
    for action, count in stats["actions"].items():
        table.add_row(f"action: {action}", str(count))
    table.add_row("Elapsed", f"{stats['elapsed_seconds']}s")
    table.add_row("Avg / message", f"{stats['avg_seconds_per_message']}s")
    console.print(table)

    types_line = ", ".join(f"{t}: {c}" for t, c in stats["message_types"].items())
    console.print(f"[dim]message types — {types_line}[/dim]")


def main() -> int:
    parser = argparse.ArgumentParser(description="Route WhatsApp messages to notify/digest/mute")
    parser.add_argument("--fresh", action="store_true", help="discard previous predictions and start over")
    parser.add_argument("--limit", type=int, default=None, metavar="N", help="only process the first N pending messages")
    parser.add_argument("--dry-run", action="store_true", help="print decisions to the console without writing output.csv or the summary report")
    args = parser.parse_args()

    messages = pd.read_csv(DATASET_DIR / "messages.csv")
    # A dry run writes nothing, so it ignores the checkpoint and can re-inspect
    # any slice even when every message already has a prediction.
    completed = set() if (args.fresh or args.dry_run) else load_completed_ids(OUTPUT_FILE)
    if not args.dry_run:
        reset_output(OUTPUT_FILE, completed)

    pending = messages[~messages["message_id"].isin(completed)]
    if args.limit:
        pending = pending.head(args.limit)

    mode = " [yellow](dry run — nothing will be written)[/yellow]" if args.dry_run else ""
    console.print(
        f"[bold]{len(messages)}[/bold] messages total, "
        f"[green]{len(completed)}[/green] already done, "
        f"[cyan]{len(pending)}[/cyan] to process{mode}"
    )
    if pending.empty:
        console.print("Nothing to do.")
        return 0

    builder = ContextBuilder(DATASET_DIR)
    extractor = MediaExtractor(DATASET_DIR)
    rules = RuleEngine()
    router = LLMRouter()
    known_history_ids = set(builder.message_history["message_id"])

    sources = Counter(rule=0, llm=0, fallback=0)
    actions: Counter = Counter()
    types: Counter = Counter()
    dry_rows: list[dict] = []
    llm_exhausted = False  # set once the whole model chain fails; skips further calls
    started = time.time()

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )
    with progress:
        task = progress.add_task("starting…", total=len(pending))
        for _, row in pending.iterrows():
            message = row.to_dict()
            message_id = message["message_id"]

            def stage(name: str) -> None:
                progress.update(
                    task,
                    description=(
                        f"[bold]{message_id}[/bold] [cyan]{name:<10}[/cyan] "
                        f"rules {sources['rule']} · llm {sources['llm']} · fb {sources['fallback']}"
                    ),
                )

            stage("media")
            try:
                media_text = extractor.get_media_text(
                    message.get("media_type"), message.get("media_id")
                )
            except Exception as exc:
                # Corrupt image, unreadable audio, unsupported format, or an
                # empty model response must not end the run: route on the
                # message text alone and carry on.
                progress.console.print(
                    f"[yellow]{message_id}: media extraction failed ({exc}); "
                    f"routing on text only[/yellow]"
                )
                media_text = None

            stage("context")
            context = builder.build_context(row, media_text)
            valid_evidence = {item["message_id"] for item in context.get("evidence", [])}

            stage("rules")
            decision = rules.decide(message, context, media_text)
            if decision is not None:
                prediction = {
                    "action": decision.action,
                    "message_type": decision.message_type,
                    "reason": decision.reason,
                    "confidence": decision.confidence,
                    "evidence_message_ids": decision.evidence_message_ids,
                }
                source = "rule"
            elif llm_exhausted:
                # Whole chain already failed: deterministic low-confidence digest
                # rows for the rest, no pointless retry loops per message.
                prediction = dict(QUOTA_EXHAUSTED_PREDICTION)
                source = "fallback"
            else:
                stage("llm")
                source = "llm"
                try:
                    prediction = router.decide(message, context, media_text)
                except Exception as exc:  # keep the run alive with an explicit marker
                    llm_exhausted = True
                    progress.console.print(
                        f"[red]{message_id}: model chain exhausted ({exc}); "
                        f"remaining LLM-bound rows get quota-fallback digests[/red]"
                    )
                    prediction = dict(QUOTA_EXHAUSTED_PREDICTION)
                    source = "fallback"

            validated = validate_prediction(
                message_id, prediction, valid_evidence, known_history_ids,
                "rule" if source == "rule" else "llm",
            )
            validated = apply_quiet_hours_policy(
                validated, _in_quiet_hours(context, message.get("created_at"))
            )
            sources[source] += 1
            actions[validated["action"]] += 1
            types[validated["message_type"]] += 1

            if args.dry_run:
                dry_rows.append(validated)
            else:
                append_row(OUTPUT_FILE, validated)
            progress.advance(task)

        stage("done")

    elapsed = time.time() - started
    stats = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dry_run": args.dry_run,
        "messages_processed": len(pending),
        "sources": dict(sources),
        "actions": dict(actions),
        "message_types": dict(types),
        "elapsed_seconds": round(elapsed, 1),
        "avg_seconds_per_message": round(elapsed / len(pending), 2),
        "output_file": None if args.dry_run else str(OUTPUT_FILE),
    }

    if args.dry_run:
        table = Table(title="Dry-run decisions", show_header=True, header_style="bold")
        for col in ("message_id", "action", "message_type", "confidence", "reason"):
            table.add_column(col, overflow="fold", max_width=60 if col == "reason" else None)
        for r in dry_rows:
            table.add_row(r["message_id"], r["action"], r["message_type"], str(r["confidence"]), r["reason"])
        console.print(table)
    else:
        write_summary_report(stats, SUMMARY_REPORT_FILE)
        console.print(f"[dim]summary report: {SUMMARY_REPORT_FILE}[/dim]")

    print_summary(stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
