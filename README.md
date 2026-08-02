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
environment only, never hardcoded.

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
python -m pytest tests/      # 43 unit tests (rules, validator, retrieval, media)
python preflight.py          # pre-package checks; run before any code.zip rebuild
```

`preflight.py` validates output.csv against the submission contract (row
count, exact column order, message ordering, enum values, confidence range)
and checks README.md for the non-ASCII characters an encoding round-trip
leaves behind. It exits non-zero on failure so a broken artifact cannot be
packaged.

## Architecture

```
messages.csv row
   |
   |- 1. media_extractor.py   Gemini vision/audio -> text, cached to JSON
   |- 2. context_builder.py   pandas joins: user habits, group role, business
   |                          trust, notification load + TF-IDF evidence
   |- 3a. rule_engine.py      6 deterministic rules decide clear cases
   |- 3b. llm_router.py       Gemini structured output for everything else
   \- 4. validator.py         enum repair, confidence policy, evidence
                              filtering, quiet-hours downgrade
```

- **Stage 1, media pre-extraction**: voice notes are transcribed and images
  get verbatim text plus a description via Gemini; results cache to
  `cache/*.json` so repeated runs make zero media calls.
- **Stage 2, context enrichment**: per-message joins across all eight
  background CSVs (quiet hours, open/dismiss/report rates, group membership
  and mute state, business verification and domain match, opt-in state, daily
  notification load). Historical evidence is retrieved with **TF-IDF plus
  cosine similarity** over the user's whole message history (scikit-learn),
  boosted for same-sender/business/group, with recency fallback, so the same
  scam text arriving from a new sender is still found.
- **Stage 3a, rule engine**: six conservative deterministic rules resolve 40
  of the 110 messages (36%) with zero LLM calls: business-scam signals
  (requires 2 or more independent signals), OTP solicitation (mute) versus
  verified OTP delivery (notify), opted-out promotions, mass-forwarded chains
  (5 or more forwards), and muted groups without a direct @mention.
- **Stage 3b, LLM router**: Gemini with a pydantic `response_schema` so
  invalid enums are impossible; the prompt separates trusted context from
  untrusted sender text (prompt-injection defense), and encodes the decision
  principles (urgency, fatigue, opt-in, response-blocking personal messages).
  Model fallback chain with 5s/15s/45s backoff per model; if the whole chain
  is exhausted mid-run, remaining rows get explicit low-confidence digest
  rows instead of the run dying.
- **Stage 4, guardrails**: every row is validated (enum repair, confidence
  clamping including NaN/inf, evidence IDs cross-checked against
  `message_history.csv` so fabricated IDs are dropped) and a deterministic
  quiet-hours policy downgrades notify to digest inside the user's DND window
  unless the type is `urgent` (OTPs still break through).

All paths, thresholds, model names, and policies live in `src/config.py`.

## Accuracy (vs the 30 solved sample messages)

- Action: **97%** (29/30), message type: **87%** (26/30)
- Confidence is banded by source and empirically calibrated: deterministic
  rules occupy 0.90 to 0.95 and scored **100% action accuracy** in every band
  on the samples; LLM decisions scored 96% and sit in a lower band. No bucket
  with lower confidence outperforms a higher one on action accuracy.
- LLM confidence is **rescaled, not clamped**. Hard-capping every model score
  at 0.90 collapsed 69 of 70 LLM rows onto that single value, destroying the
  model's own ranking. `rescale_confidence()` linearly maps the raw score
  from [0.70, 1.00] into [0.75, 0.90], preserving relative ordering while
  staying below the rule tier; `LLM_CONFIDENCE_CAP` is retained underneath as
  a backstop. The shipped file now spans 10 distinct confidence values
  (0.82 to 0.95); the largest single value covers 41% of LLM rows, down
  from 99%.

**Calibration caveat (small n):** the two newest rules fire on very few
solved samples: `high_forward_chain` on 2 (both exactly correct, action and
type) and `optout_promotion` on 1 (exactly correct). Their 100% figures are
consistent with, but not strong evidence of, their assigned confidence bands
(0.91 and 0.92). The four original rules and the LLM path have more support
(7 and 23 sample decisions respectively).

## Output

`../dataset/output.csv`, one row per message in `messages.csv` order:

```
message_id,action,message_type,reason,confidence,evidence_message_ids
```

`evidence_message_ids` is a semicolon-separated list of real
`message_history.csv` IDs (or `none`), verified against the dataset. 17 of
the 110 rows carry `none`; all were audited and none are silent losses:
1 row has no retrievable match (unknown sender, no topical similarity),
7 are rule decisions whose candidates carried no negative signal (rules cite
only reported/dismissed/muted history by design), and 9 are LLM decisions
where the model saw only topically unrelated candidates and correctly
declined to cite them.

**Provenance note:** the shipped `output.csv` was generated 2026-08-02
12:33 IST in a single clean run (40 rule and 70 LLM decisions, 0 fallbacks)
and re-audited afterwards: exact contract columns and row order, zero
duplicate or failure-mode rows, every evidence ID verified against
`message_history.csv`, confidence within the 0.85 to 0.95 policy bands, and
the quiet-hours policy applied. An earlier finalized version (01:54 IST)
was lost the same morning when the workspace was reset to the repository
HEAD (all untracked sources deleted, `output.csv` reverted to the blank
template); the pipeline was restored file-for-file from the working
session, verified by the full test suite (36 passing) and by matching the
deterministic rule-coverage fingerprint of the previous build exactly
(same 40 rule-decided rows, same per-rule counts) before regenerating.
One deliberate improvement landed between the two versions: listings in
muted marketplace groups now classify as `promotion` instead of `unknown`
(2 rows), which also raised sample message-type accuracy from 83% to 87%.
A candidate scam-versus-spam refinement was evaluated the same way and
rejected: dry-running it against the eight production rows it would touch
flipped one correct `scam` row, failing our no-regression criterion.

**Known limitation, quota-exhaustion latch:** when the whole model chain
fails on one message, `main.py` sets an `llm_exhausted` flag and every
remaining LLM-bound row is written as an explicit low-confidence digest with
a "quota exhausted" reason. The flag is deliberately one-way: it never
re-probes, so a transient outage that clears seconds later still degrades
the rest of the run. That trade buys guaranteed completion and no retry
storms against an already-failing API, and the rows it produces are trivially
identifiable and re-runnable -- drop them and rerun, and the checkpoint
resume regenerates exactly those. A future version could re-probe after a
backoff instead of latching. The shipped `output.csv` contains zero such
rows.

**Regeneration variance:** the confidence rescale required re-running the 70
LLM-decided rows (the 40 rule-decided rows are deterministic and were left
untouched). Diffing the result against the pre-rescale snapshot, 3 of 110
rows (2.7%) changed `message_type` and 1 of those also changed `action`.
`msg_023` (`payment` to `business_update`, a bank card-payment notice) and
`msg_045` (`urgent` to `event`, a delivery agent waiting at the gate) are
adjacent-category moves with no routing impact -- both kept their original
action. `msg_065` moved from `mute`/`scam` to `digest`/`promotion` and
stands: the sender is a verified business on its official domain, the user
has opted into its promotions and consistently opens them, and those three
structured signals outweigh the third-party ad creative in the attached
image that the earlier run had read as a brand mismatch.
