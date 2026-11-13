# Policy audit: Label Rules (assignment-v2) → implementation → proof

Each rule in the "Label Rules" tab is listed below with **where it is enforced**, **who enforces it**, and
**the proof**:

- **Python**: deterministic code. The same evidence always gives the same answer.
- **LLM**: the model extracts the evidence the rule needs, guided by an instruction in `core/prompts.py`.
- **Test**: a unit test in `tests/` that fails if the rule is broken. Run with `pytest -q` (47 tests).
- **Gold**: confirmed on the 50 labelled calls.

## Contracts

| Rule | Requirement | Enforced in | Proof |
|---|---|---|---|
| Input contract | Only `transcript_id, call_started_at, timezone, transcript` reach the model; gold columns never do | `schemas.InputRecord` (`extra="ignore"`), `pipeline.build_user_message` sends only call time + transcript | Code; gold columns are dropped at parse time |
| Output contract / P2 | Exactly 5 keys; sub must be a valid child; ISO with seconds and `+05:30` or JSON `null` | `schemas.Prediction` (`extra="forbid"`), `rules.decide` asserts `sub ∈ ALLOWED_PAIRS[disp]`, `callback.fmt` uses `isoformat(timespec="seconds")` | Every test asserts exact strings like `2026-11-10T16:00:00+05:30`; `predictions` written via `model_dump_json` (`None` → `null`) |
| Scoring contract | No missing or duplicate IDs; no unparseable records | `pipeline.process_record` never raises (valid fallback record); `read_jsonl` drops duplicate IDs | Code; every Groq run so far (64 calls) had 0 fallbacks |
| PAIR | Allowed parent/child pairs | `schemas.ALLOWED_PAIRS` copied verbatim from the sheet | Assertion in `decide`; dashboard validates the same table |

## Disposition (D1–D6)

| Rule | Implementation | Who | Proof |
|---|---|---|---|
| D6 precedence Invalid → Not Interested → Appointment → Follow-up → Unable | `rules.disposition` checks in exactly this order | Python | `test_precedence_invalid_beats_refusal_and_booking`, `test_ex5_final_do_not_contact_after_callback` |
| D1 Invalid: wrong contact / no pain need / test / policy exclusion / out-of-area with travel AND remote refused; distance alone never invalid | LLM `invalid_reasons` plus prompt wording "distance alone is NOT enough"; Python **removes** any policy exclusion unless the known age is < 15 and **adds** it when age < 15 | LLM + Python | `test_invalid_tie_break_order_and_no_inferred_exclusion` (age unknown, so not excluded) |
| D1 missing age/duration is not exclusion | Same as above | Python | Same test |
| D2 Not Interested: unconditional final refusal; price objection / think / treatment elsewhere are not refusals; reversed refusals superseded | Prompt REFUSAL section lists the non-refusals explicitly | LLM | Gold: AVT-010 and AVT-033 refusals correct; no follow-up call mislabelled as refusal on the 50 |
| D3 Appointment: day **and** clock time/bounded window mutually confirmed; offers / visit intentions / callbacks-to-confirm are not bookings | Prompt APPOINTMENT section, plus a **Python guard**: a "confirmed" slot with no time or window is downgraded | LLM + Python | `test_d3_guard_confirmed_without_time_is_not_booking`; gold: all 12 real bookings detected; 1 false booking (AVT-041, prompt v2) is addressed by prompt v3's "agent must confirm the slot" rule, awaiting re-run |
| D3 Rescheduled beats modality; remote → Online Consultation | `rules.disposition` | Python | `test_reschedule_beats_online_and_online_beats_clinic` |
| D4 Follow-up: final main blocker, not historical objections; callback extracted independently | Prompt "final, unresolved… most primary first"; `next_call_at` is computed separately from the sub-disposition | LLM + Python | `test_blocker_beats_specific_callback_for_sub_but_callback_time_kept` |
| D5 Unable to Assess only when there is no meaningful evidence; a long unresolved pain discussion is Follow-up | `conversation_meaningful=false` **and** pain need not established | LLM + Python | `test_unable_to_assess_and_suppression` |

## Sub-disposition (SUB)

| Rule | Implementation | Proof |
|---|---|---|
| Follow-up tie-break Family → Cost → Travel → Current Treatment → Treatment Concern → Specific Callback → Information Requested → Vague Deferral → Further Discussion | LLM ranks blockers by the customer's final primacy; Python takes the first, then falls back in exactly the sheet's order | `test_follow_up_tie_break_family_first`, `test_deferral_backstop_sets_plus_three` |
| Invalid tie-break Test → Wrong Contact → No Pain Need → Policy Exclusion → Outside Area | `rules.INVALID_ORDER` | `test_invalid_tie_break_order_and_no_inferred_exclusion` |
| Do Not Contact beats Explicit Refusal | `rules.disposition` | `test_do_not_contact_beats_explicit_refusal` |
| Specific Callback needs a mutually acknowledged, resolvable day or time | `callback.final_specific_callback` | `test_unacknowledged_callback_ignored` |
| Information Requested excludes routine location requests | Prompt wording (location/address/map/number excluded) | Gold: AVT-013 correct on the final prompt |

## Lead quality (LQ)

| Rule | Implementation | Proof |
|---|---|---|
| Order Not Qualified → Not Assessable → Low → High → Medium | `rules.lead_quality` returns in exactly this order | `test_lq_order_low_before_high_and_defer_beyond_30` |
| Not Qualified ⇔ Invalid Lead | First branch | `test_precedence_invalid_beats_refusal_and_booking` |
| Not Assessable: Unable, or need/willingness cannot be established | `pain_need == "unclear"` or `intent == "unclear"` | `test_unable_to_assess_and_suppression` |
| Low: refusal, mere curiosity, or defers > 30 days without a confirmed appointment; severity never upgrades | `intent` in (refused, exploratory_only), or `defers_beyond_30_days` | Same test; gold AVT-006 Low correct |
| High: consultation on **date offsets 0–6** with no unresolved blocker; a callback alone is not commitment | Python computes the offset from `call_started_at` and requires 0 ≤ offset ≤ 6 **and** no blocker; an unresolvable date is never High | `test_appointment_offsets_drive_lead_quality` (offset 6 → High, 7 → Medium = EX6), `test_lq_high_requires_resolvable_offset` |
| Medium default | Final return | Gold: 32 of 33 Medium leads correct |

## Next call time (F1–F12): all date arithmetic is Python

The LLM reports **what was said** (`TimeSpec`: kind, weekday plus this/next, day/month/year, hour as spoken,
am/pm, part of day). `core/callback.py` turns that into a timestamp.

| Rule | Implementation | Proof (tests in `tests/test_callback.py`) |
|---|---|---|
| F1 reference = `call_started_at` in Asia/Kolkata; never today's run date | `parse_call_time` + `ZoneInfo`; `datetime.now()` is never used | Every test passes its own reference time |
| F2 suppression for Invalid / Not Interested / Unable | First line of `next_call_at` | `test_ex5_suppression`, `test_unable_to_assess_and_suppression` |
| F3 final callback acknowledged by the other party; customer-only times ignored; amendments replace; appointment is not a callback; distinct callback can coexist with a booking | `final_specific_callback` (acknowledgement required, last one wins) | `test_ex1…`, `test_unacknowledged_callback_ignored`, `test_amendment_replaces_earlier`, `test_ex3…`, `test_appointment_can_coexist_with_distinct_callback` |
| F4 tomorrow +1, parso +2, in N days, bare weekday = next strictly future, "this" = today if time still future, "next" = following Mon–Sun week, "next week" = Monday, "this week" = vague | `resolve` | `test_relative_days`, `test_weekday_nearest_strictly_future`, `test_this_weekday_today_if_time_future`, `test_next_weekday_is_following_week`, `test_next_week_alone_is_monday`, `test_f12_this_week_vague…` |
| F5 day only → 11:00; time only → today if future, else tomorrow; 1–7 → PM, 8–11 → AM, 12 → noon; explicit am/pm/morning/evening wins; morning 10, afternoon 15, evening 18, lunch 13; window → its start | `_clock`, `resolve` | `test_clock_defaults`, `test_time_only_today_or_tomorrow`, `test_f12…window_uses_start`, `test_ex1_tomorrow_at_4_is_pm` |
| F6 month/day without year → next future occurrence; explicit past full dates and past same-day times rejected, never repaired | `_resolve_date` + roll-forward; final `result > ref` check | `test_absolute_dates`, `test_f6_month_day_without_year_today_but_past_rolls_forward`, `test_f6_explicit_past_full_date_rejected_not_repaired`, `test_past_same_day_time_rejected` |
| F7 any final vague postponement → +3 days 11:00, regardless of blocker; a routine agent promise is not one | LLM flag **or** a deterministic backstop on the customer's last 8 turns ("let you know", "I'll think"…), customer turns only | `test_ex2_vague_deferral_plus_3`, `test_deferral_backstop_sets_plus_three`, `test_backstop_never_overrides_specific_callback`; precision check on gold: the backstop never overrode a correct answer |
| F8 Follow-up without callback or vague → +1 day 11:00; Appointment without distinct callback → null | End of `next_call_at` | `test_ex4_unspecified_agent_call_plus_1`, `test_ex3_appointment_without_callback_is_null` |
| F9 only the final agreement counts; a final vague answer cancels an earlier specific time; the result must be in the future | Last acknowledged callback wins even when vague; past results → None | `test_f9_final_vague_agreement_cancels_earlier_specific_time`, `test_cancelled_callback_falls_back_to_vague` |
| F10 "in N minutes/hours" keeps seconds; alternatives → earliest; "after X" → X; "around X" → X; open-ended "after date/event" → vague | `resolve` (minute/hour intervals use exact `ref + delta`); prompt for the rest | `test_intervals_keep_seconds` (AVT-047 → `14:26:01`), `test_gold_avt_029_window_uses_earliest` |
| F11 in 1 month → same day, clamped; bare next month → 1st at 11:00 | `_add_months`, `next_month` | `test_months` (31 Jan → 28 Feb) |
| F12 "may call" / opening hours are not commitments; "next week after event" → vague; definite "will call tomorrow" + ack is a commitment | Prompt; vague callbacks route to F7 | `test_acknowledged_event_qualified_callback_is_vague` (AVT-024 → +3) |

## Measured results on the supplied data

| Configuration | Calls | Points /70 | Exact match |
|---|---|---|---|
| Final prompt v3 + final rules | 14 | 70.0 | 100% |
| Prompt v1 evidence + final rules | 32 | 65.9 | 78% |
| Prompt v2 evidence + final rules (calls unseen during v1 tuning) | 18 | 58.6 | 56% |

Re-scoring all saved evidence after this audit's fixes produced identical scores (no regressions).

## What the code does NOT guarantee (honest limits)

1. **Judgement calls stay with the LLM:** which blocker is "final and primary", whether intent is exploratory,
   whether a refusal is unconditional. These are guided by explicit prompt definitions but are not provable
   in code. They cause most of the remaining sub-disposition errors.
2. **"Equally primary" blockers:** the LLM orders blockers by primacy. A true tie is resolved by the LLM's
   ordering rather than the sheet's Family → Cost → … order.
3. **Rare categories absent from the 50 calls** (Invalid Lead, Do Not Contact, Online Consultation,
   Rescheduled, Unable to Assess): the rule logic is unit-tested, but the LLM's detection of them is only
   prompt-guided. No real example exists to measure it.
4. **Run-to-run variation:** temperature 0 and a fixed seed, but Groq does not guarantee bit-identical
   outputs. Results are cached for reproducible re-scoring.
