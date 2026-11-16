# Error analysis

All scores use the assignment's scoring contract, scored by `aster-vale evaluate`:
- **Exact string match** against the answer key in "50 Labeled Calls".
- **Sub-disposition** counts only when the disposition is also right.
- **`next_call_at`** must match exactly, including seconds and `+05:30`, or be JSON `null`.
- **Weights:** disposition 15, sub-disposition 20, lead quality 15, next call time 20 (70 points in total).

Model for every run: Groq `openai/gpt-oss-120b`, temperature 0, seed 7, reasoning effort medium, strict
JSON-schema output.

## 1. Final run (the submitted predictions)

`predictions_50.jsonl` comes from **one run of the frozen prompt v3 over all 50 calls**, using the README
command:
- 50 lines, in input order
- 0 fallbacks, 0 contract violations: exactly 5 keys, valid pairs, `+05:30` with seconds or JSON `null`

The run was throttled by the Groq free-tier daily token limit, so it completed over several passes. Finished
calls were cached and never re-sent. The 25 calls run fresh in the final passes used 135,021 input and
32,543 output tokens (~$0.04 at list price).

| Field | Correct | Accuracy | Points |
|---|---|---|---|
| Disposition | 49 / 50 | 98% | 14.7 / 15 |
| Sub-disposition | 37 / 50 | 74% | 14.8 / 20 |
| Lead quality | 49 / 50 | 98% | 14.7 / 15 |
| Next call time | 47 / 50 | 94% | 18.8 / 20 |
| **Total** | **36 / 50 fully correct (72%)** | | **63.0 / 70** |

## 2. Results

| Run | Calls | Disposition | Sub-disp. | Lead quality | Next call | All 4 right | Points /70 |
|---|---|---|---|---|---|---|---|
| **Prompt v3 final run (submitted)** | **AVT-001–050 (50)** | **98%** | **74%** | **98%** | **94%** | **72%** | **63.0** |
| Prompt v3, first 23 calls of the final run | AVT-001–023 (23) | **100%** | **91.3%** | **100%** | **100%** | **91.3%** | **68.3** |
| Prompt v1 + final rules | AVT-001–032 (32) | 100% | 84.4% | 93.8% | 100% | 78.1% | 65.9 |
| **Prompt v2, held-out** | AVT-033–050 (18, not used while writing v2) | 94.4% | 66.7% | 88.9% | 88.9% | 55.6% | **58.6** |

**How to read this honestly:**
- The v3 numbers are measured on calls whose errors helped shape v3, so they are optimistic.
- The cleanest unseen-data estimate we have is the **v2 hold-out: 58.6 / 70**.
- v3 was then improved using errors from *all* 50 calls, so no fully unseen calls remain to test v3 on.
- A fair expectation for the private set is **around 58–63 / 70**: between the v2 hold-out and the final
  v3 run.

## 3. Failure patterns

Most lost points come from **sub-disposition**. In the v1+v2 evidence it cost 3.6 of the 5.6 missed points.

| Pattern | Example | What went wrong | Fix |
|---|---|---|---|
| **Earlier objection chosen over the final blocker** | AVT-034 (key Family, ours Travel) | Distance came up early; the call ends with "I will talk at home". D4 says the *final* main blocker wins | v3 prompt: "the customer's final stated next step outranks earlier complaints" |
| **Travel Constraint over-predicted** | AVT-030 (cataract follow-up → Current Treatment), AVT-047 (reports review → Treatment Concern) | Any mention of travel or timing pulled the label to Travel | v3 prompt: a medical follow-up counts as Current Treatment; clinic review of reports counts as Treatment Concern |
| **Family needed as companion vs family decision** | AVT-001, AVT-020 | "Husband is away", "wife must manage timing" are logistics (Travel), not a family decision | v2/v3 prompt clarification; **AVT-020 is still wrong in v3** |
| **Insurance check read as family** | AVT-049 (key Cost) | "Speak to my office … insurance person" plus "discuss at home as well" | v3: insurance or office reimbursement checks count as Cost |
| **Customer preference treated as a booking** | AVT-041 (all 4 fields wrong) | The customer named "four to five", but the agent said a Wednesday call would confirm the time. D3 says a callback to confirm is not a booking | v3: a booking needs the agent to confirm the slot |
| **Low intent missed** | AVT-022 | "I do not want that treatment… I'll call if I want" means Low | v3 low-intent cues |
| **Need unclear** | AVT-042 | "Not exactly pain, it's stiffness" means Not Assessable (it comes before Low in the LQ order) | v3 wording |
| **Accepted callback not extracted** | AVT-040 | "Should I call next week?" "Yes" means next Monday (F4) | v3: an agent proposal answered "yes" is acknowledged |
| **Vague callback not treated as vague** | AVT-024 | "Call back next week after Dussehra" is event-qualified, so F12 says vague → F7 (+3 days) | Rule fix in code (no LLM change) |
| **Vague deferral missed while a blocker is present** | AVT-001, AVT-003 under v3 | The model set `vague_postponement=false` whenever it found a blocker | Deterministic backstop: customer deferral phrases in the last turns → F7. Checked against all 50 gold labels: it never overrides a correct answer |

**Final-run (v3) failure patterns** (14 calls with a wrong label; details per call in the dashboard's Model
quality tab):

| Pattern | Calls | What happened |
|---|---|---|
| **Blocker under-extraction** (main cause) | AVT-018, 028, 031, 034, 041, 047, 048, 050 | v3 returned no blocker although the customer gave one (travel, cost, treatment doubts, family), so the label fell through to Vague Deferral, Information Requested, Specific Callback or Further Discussion. v3's stricter wording ("the blocker quote must be the customer's own words") likely made the model too cautious. **8 of the 13 sub-disposition misses.** |
| Wrong blocker chosen | AVT-020, 038, 049 | Logistics read as family (020), an earlier medicine mention lost to a later price question (038), insurance outranked by "discuss at home as well" (049) |
| "Forget it" read as a final refusal | AVT-042 | The key treats the call as still open (Follow-up / Treatment Concern) |
| Postponement or acknowledgement missed | AVT-043, AVT-048 | "Consult my family and then decide" not flagged vague (+3 days); "I'll call you back tomorrow" → "Alright" not marked acknowledged |
| Routine location request read as an information request | AVT-032 | |

**What v3 fixed compared with earlier runs:** AVT-041 is no longer a false booking. AVT-022, AVT-030 and
AVT-040 are now correct.

**Lesson:** each prompt change fixed some calls and broke others. The single frozen run (63.0) scores below
the best-per-call development mix (64.4), which is why the submission uses one run rather than cherry-picked
results. The most promising next change would be to relax the blocker-quote requirement and add an explicit
"if the customer names any reason for not booking, list it" instruction, validated on fresh, unseen calls.

**Earlier observation from the first 23 calls of this run:**
- **AVT-018** (key Treatment Concern → ours Vague Deferral). v1 had this right, which shows a prompt change can fix some calls and break others. This is why the submission uses one frozen run, not the best result per call.
- **AVT-020** (key Travel → ours Family).

## 4. Changes made (iteration log)

| Step | Type | Measured on | Score |
|---|---|---|---|
| Prompt v1 baseline | prompt | 32 calls | 65.3 / 70 |
| Event-qualified callback is vague (F12 → F7) | rule only | same 32, re-scored | 65.9 (next call 97% → 100%) |
| Prompt v2 (companion = Travel, cataract = Current Treatment, customer-only blocker quotes, routine location ≠ info request, bare weekday ≠ "next", low-intent cues) | prompt | 18 unseen calls | 58.6 (honest hold-out) |
| Prompt v3 (agent must confirm the slot, insurance = Cost, report review = Treatment Concern, final step wins, accepted callback proposals, unclear need) | prompt | 14 calls | 67.1 |
| Deferral-phrase backstop for F7 | rule only | same 14, re-scored | 70.0 |
| Policy audit hardening: F9 final vague cancels an earlier time, F6 year roll-forward, D3 slot must have a time, LQ High needs a proven 0–6 day offset | rule only | all saved evidence | unchanged (rare cases); see `docs/RULES_AUDIT.md` |

Rule changes are re-scored on saved LLM evidence with `aster-vale rescore`, at no token cost. 47 unit tests
pin every policy rule and the EX1–EX6 boundary examples.

## 5. Runtime, tokens and cost

Measured on 62 Groq requests during development (cached repeats excluded):

| | Per call | 50 calls |
|---|---|---|
| Tokens | 6.4K (5.2K in, 1.2K out) | ~320K |
| Cost at Groq list price ($0.15/M in, $0.60/M out) | $0.0015 | **~$0.08** |
| Time (free tier, sequential, 8K tokens/min limit) | ~41 s | ~35 min |

On a paid tier the per-call time is a few seconds. The completed development runs produced no fallback
records. An attempt at the final run that hit the daily limit was stopped, and its partial output discarded;
fallback records are never submitted.

## 6. Task 2: gender detection (no transformer)

- **Method:** rule-based scoring for both caller and patient:
  - the agent's form of address (sir/ma'am, recency-weighted, agent turns only)
  - self-kinship ("my husband", "I am Renu's husband")
  - the stated first name against a name lexicon
  - the patient relation ("my mother")
  - it abstains when the evidence is weak
- **Result:** 50 of 50 callers match hand labels. Caveat: the hand labels use the same kinds of cues, so this
  measures consistency rather than an independent benchmark.
- **Fix made:** AVT-042 contains a single "Sir" in an early agent turn. It was an abstention until the
  threshold was set so that one explicit agent address term counts. This introduced no new errors.

## 7. Limitations

1. "Final main blocker" and "exploratory vs genuine intent" are judgement calls left to the LLM, guided by
   explicit definitions. Most residual error comes from them.
2. Categories absent from the 50 calls (Invalid Lead, Do Not Contact, Online Consultation, Rescheduled, Unable
   to Assess) are fully unit-tested in the rules, but how well the LLM detects them is unmeasured.
3. Temperature 0 and a fixed seed reduce variation, but Groq does not guarantee identical outputs. Results are
   cached so they can be re-scored reproducibly.
