# Interview Notes - Message Notification Router

One page of honest answers to the questions most likely to be asked. Numbers
are measured, not estimated.

## The system in one breath

110 incoming messages, each routed to notify / digest / mute with a type, a
reason, a confidence and historical evidence. Six deterministic rules resolve
40 rows (36%) with zero API calls; the remaining 70 go to Gemini with
structured output. Measured on the 30 solved samples: **97% action accuracy
(29/30), 87% message-type (26/30)**.

---

## 1. Why 64% of the predictions depend on one model

70 of 110 rows (63.6%) are decided by the LLM, and in practice all of them
are served by a single model: `gemini-3.1-flash-lite`.

The fallback chain has three models, but the free tier allocates them very
differently: `gemini-flash-latest` and `gemini-3-flash-preview` are capped at
**20 requests/day each**, while flash-lite gets **500**. Twenty requests do
not survive a single 70-row run, so after the first few calls the chain
collapses to its third entry for the rest of the day. The redundancy is real
in code and nearly absent in practice.

What that concentration actually risks:
- **Availability.** One model's outage or quota exhaustion stalls 64% of the
  output. Mitigated, not solved, by 5s/15s/45s per-model backoff and a
  quota-exhaustion path that completes the run with flagged low-confidence
  digests rather than dying.
- **Correlated error.** Every LLM judgment shares one model's biases. The
  four remaining sample misses are all boundary calls (urgent vs event,
  scam vs spam), which is what single-model correlated error looks like.

Why it was still the right call: the deterministic layer covers exactly the
cases where being wrong is expensive - scam solicitation, opted-out
promotions, mass-forwarded chains - so the single-model dependency sits on
the *judgment* tier, not the *safety* tier.

## 2. Why two confidence bands rest on n=1 and n=2

Confidence is banded by decision source: rules occupy 0.90-0.95, LLM
decisions are rescaled into 0.75-0.90, so a rule always outranks a judgment
call. Each band was checked against real sample accuracy, and rules scored
100% action accuracy in every band versus 96% for the LLM tier - no inversion.

But the support behind each band is very uneven:

| Rule | Sample decisions | Band |
|---|---|---|
| Four original rules (combined) | 7 | 0.90-0.95 |
| LLM tier | 23 | 0.75-0.90 |
| `high_forward_chain` | **2** | 0.91 |
| `optout_promotion` | **1** | 0.92 |

Both newer rules were exactly correct on every sample they touched, action
and type. But 2-for-2 and 1-for-1 are consistent with their assigned bands
rather than evidence for them; the honest reading is that those two numbers
are asserted, not demonstrated. The ground truth simply does not contain
enough mass-forwarded or opted-out-promotion examples to say more, and
inventing a band from a single row is the same mistake as inventing a rule
from one - which we explicitly refused to do elsewhere (see below).

## 3. What I would do with more time or quota

- **Break the single-model dependency.** A second provider key, or billing on
  the current project, would make the fallback chain real instead of nominal.
- **Earn the thin calibration bands.** With quota to spare, run the two
  newer rules against held-out synthetic cases built from the same generator
  patterns, and move their bands to whatever the measurement supports.
- **Resolve the urgent/event boundary properly.** It accounts for 2 of the 4
  remaining misses, and both directions have appeared. That needs more
  labelled examples, not more prompt text - we tried prompt fixes twice and
  each fixed one sample while breaking another.
- **Re-probe instead of latching.** The quota-exhaustion flag is one-way by
  design; a backoff-and-retry would degrade fewer rows during a transient
  outage.
- **Widen preflight.** It catches encoding damage and contract violations but
  not prose damage - a shell quoting bug once silently deleted identifiers
  from the README and preflight passed it.

## 4. The two workspace wipes, and why recovery held

Twice today the working tree was reset to repository HEAD, deleting untracked
work.

**First wipe (~12:02).** Every file in `src/` and `tests/`, plus
`requirements.txt`, `.gitignore` and `code.zip`, was deleted; `main.py` and
`evaluation/main.py` reverted to empty stubs; `output.csv` reverted to the
blank template, losing all 110 predictions. Recovery: rewrote 13 files from
the working session, then proved fidelity two ways before trusting it - the
full test suite passed (36 at the time), and the deterministic rule-coverage
fingerprint matched the pre-wipe build **exactly**: same 40 rule-decided
rows, same per-rule counts. Only then was the LLM tier regenerated.

**Second wipe (~15:30).** Narrower - `main.py` and `README.md` deleted,
everything else intact. Git HEAD turned out to be **stale** (it predated the
media guard and several README sections), so the correct source was the
verified `code.zip`, whose contents had been checked from inside the archive
after the last build. Both files restored from it, verified, 43 tests green.

The transferable point: the artifact that saved the work both times was the
one that had been **independently verified**, not the one that was merely
recent. Git was clean but stale; the zip was untracked but checked. That is
also the strongest argument for the habits used throughout - confirming exact
costs before spending them, checking claims against timestamped records
rather than memory, and dry-running a candidate change against the rows it
would touch before applying it. One such dry-run rejected a scam/spam
refinement because it flipped a currently-correct row: the change looked good
in isolation and was wrong in aggregate.
