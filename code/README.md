# Message Notification Router

An AI-powered routing system for WhatsApp messages, built for the HackerRank
Orchestrate hackathon. For every message in `dataset/messages.csv` it decides
`notify` (interrupt now), `digest` (show later), or `mute` (suppress), with a
message-type classification, a one-sentence reason, a calibrated confidence,
and supporting historical evidence.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # then put your Gemini API key in .env
```

Requires Python 3.11+. The only secret is `GEMINI_API_KEY` (free tier works;
get one at https://aistudio.google.com/apikey). It is read from the
environment only ΓÇö never hardcoded.

## Run

```bash
python main.py               # route all messages -> ../dataset/output.csv
python main.py --limit 10    # only the first 10 pending messages
python main.py --dry-run     # print decisions without writing anything
python main.py --fresh       # discard previous predictions and start over
```

The run is **checkpointed**: every prediction is appended to `output.csv`
immediately, so an interrupted run (rate limit, network, Ctrl+C) resumes where
it left off. A summary table is printed at the end and written to
`../dataset/summary_report.json`.

Evaluate against the solved examples, and run the unit tests:

```bash
python evaluation/main.py    # accuracy vs dataset/sample_messages.csv
python -m pytest tests/      # 35 unit tests (rules, validator, retrieval)
```

## Architecture

```
messages.csv row
   Γöé
   Γö£ΓöÇ 1. media_extractor.py   Gemini vision/audio -> text, cached to JSON
   Γö£ΓöÇ 2. context_builder.py   pandas joins: user habits, group role, business
   Γöé                          trust, notification load + TF-IDF evidence
   Γö£ΓöÇ 3a. rule_engine.py      6 deterministic rules decide clear cases
   Γö£ΓöÇ 3b. llm_router.py       Gemini structured output for everything else
   ΓööΓöÇ 4. validator.py         enum repair, confidence policy, evidence
                              filtering, quiet-hours downgrade
```

- **Stage 1 ΓÇö media pre-extraction**: voice notes are transcribed and images
  get verbatim text + description via Gemini; results cache to `cache/*.json`
  so repeated runs make zero media calls.
- **Stage 2 ΓÇö context enrichment**: per-message joins across all eight
  background CSVs (quiet hours, open/dismiss/report rates, group membership
  and mute state, business verification and domain match, opt-in state, daily
  notification load). Historical evidence is retrieved with **TF-IDF + cosine
  similarity** over the user's whole message history (scikit-learn), boosted
  for same-sender/business/group, with recency fallback ΓÇö so the same scam
  text arriving from a new sender is still found.
- **Stage 3a ΓÇö rule engine**: six conservative deterministic rules resolve 40
  of the 110 messages (36%) with zero LLM calls: business-scam signals
  (requires 2+ independent signals), OTP solicitation (mute) vs verified OTP
  delivery (notify), opted-out promotions, mass-forwarded chains (>= 5), and
  muted groups without a direct @mention.
- **Stage 3b ΓÇö LLM router**: Gemini with a pydantic `response_schema` so
  invalid enums are impossible; the prompt separates trusted context from
  untrusted sender text (prompt-injection defense), and encodes the decision
  principles (urgency, fatigue, opt-in, response-blocking personal messages).
  Model fallback chain with 5s/15s/45s backoff per model; if the whole chain
  is exhausted mid-run, remaining rows get explicit low-confidence digest
  rows instead of the run dying.
- **Stage 4 ΓÇö guardrails**: every row is validated (enum repair, confidence
  clamping incl. NaN/inf, evidence IDs cross-checked against
  `message_history.csv` so fabricated IDs are dropped) and a deterministic
  quiet-hours policy downgrades notify -> digest inside the user's DND window
  unless the type is `urgent` (OTPs still break through).

All paths, thresholds, model names, and policies live in `src/config.py`.

## Accuracy (vs the 30 solved sample messages)

- Action: **97%** (29/30) ┬╖ message type: **83%** (25/30)
- Confidence is banded by source and empirically calibrated: deterministic
  rules occupy 0.90ΓÇô0.95 and scored **100% action accuracy** in every band on
  the samples; LLM decisions are capped at 0.90 and scored 96%. No bucket
  with lower confidence outperforms a higher one on action accuracy.

**Calibration caveat (small n):** the two newest rules fire on very few
solved samples ΓÇö `high_forward_chain` on 2 (both exactly correct, action and
type) and `optout_promotion` on 1 (exactly correct). Their 100% figures are
consistent with, but not strong evidence of, their assigned confidence bands
(0.91/0.92). The four original rules and the LLM path have more support
(7 and 23 sample decisions respectively).

## Output

`../dataset/output.csv`, one row per message in `messages.csv` order:

```
message_id,action,message_type,reason,confidence,evidence_message_ids
```

`evidence_message_ids` is a semicolon-separated list of real
`message_history.csv` IDs (or `none`), verified against the dataset.

**Provenance note:** the shipped `output.csv` was finalized (2026-08-02
01:54 IST) shortly before a retry-logic hardening landed in the code
(02:42 IST). It was deliberately not regenerated afterwards: the hardening
only changes behavior on API *failures*, and the finalized file was audited
to contain zero failure-mode rows (no fallback reasons, no quota reasons, no
digest/unknown/0.3 signatures; minimum confidence 0.85). The failure-mode
rows the old retry logic did produce ΓÇö written by a stale process running
against an exhausted API key ΓÇö were identified, dropped, and regenerated
with a working key during that same finalization, and the run that produced
the surviving rows reported 0 fallbacks in its own summary. Regenerating
110 verified-good rows would only have introduced model-serving variance.
Root cause of the corruption: an imperfect process kill ΓÇö `taskkill
python.exe` missed the stale run because Microsoft Store Python executes as
`python3.11.exe`, so the process survived and kept appending; it was later
found via `Get-CimInstance` and stopped by PID.
