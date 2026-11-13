"""Deterministic policy: Evidence -> disposition, sub_disposition, lead_quality, next_call_at.

Every branch cites the rule ID from the "Label Rules" tab (policy assignment-v2).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from .callback import day_offset, final_specific_callback, fmt, has_vague_postponement, next_call_at
from .schemas import ALLOWED_PAIRS, Evidence, Prediction

INVALID_ORDER = [  # SUB tie-break for multiple invalid reasons
    ("test_enquiry", "Test Enquiry"),
    ("wrong_contact", "Wrong Contact"),
    ("no_pain_need", "No Pain Need"),
    ("policy_exclusion_under_15", "Policy Exclusion"),
    ("outside_service_area", "Outside Service Area"),
]
BLOCKER_ORDER = [
    ("family_discussion", "Family Discussion"),
    ("cost_concern", "Cost Concern"),
    ("travel_constraint", "Travel Constraint"),
    ("current_treatment", "Current Treatment"),
    ("treatment_concern", "Treatment Concern"),
]
BLOCKER_LABEL = dict(BLOCKER_ORDER)


# Deterministic backstop for F7: customer deferral phrases in the final customer turns. Only matters when no
# specific acknowledged callback exists (F3 takes precedence), so it cannot override an agreed time.
DEFERRAL_RE = re.compile(
    r"\b(let you know|i'?ll (?:tell|inform|call|contact|get back to|update) you|"
    r"i will (?:tell|inform|call|contact|get back to|update) you|"
    r"(?:i'?ll|i will|we'?ll|we will|let me) (?:see|think|check|discuss|consult|talk)|get in touch|"
    r"need (?:some )?time|think about it|will decide|i'?ll come when|whenever i|call you myself)\b",
    re.I,
)


def deferral_cue(transcript: str, last_n: int = 8) -> str | None:
    turns = [ln for ln in transcript.splitlines() if ln.lower().startswith("customer:")][-last_n:]
    for ln in reversed(turns):
        m = DEFERRAL_RE.search(ln)
        if m:
            return m.group(0)
    return None


@dataclass
class Decision:
    prediction: Prediction
    trace: list[str] = field(default_factory=list)  # human-readable rule provenance


def _invalid_reasons(ev: Evidence) -> list[str]:
    reasons = set(ev.invalid_reasons)
    if ev.patient_age is not None and ev.patient_age < 15:
        reasons.add("policy_exclusion_under_15")
    else:
        reasons.discard("policy_exclusion_under_15")  # D1: no other exclusion may be inferred
    if ev.pain_need == "no":
        reasons.add("no_pain_need")
    return [label for key, label in INVALID_ORDER if key in reasons]


def disposition(ev: Evidence, trace: list[str]) -> tuple[str, str]:
    invalid = _invalid_reasons(ev)
    if invalid:
        trace.append(f"D1 Invalid Lead: {invalid}")
        return "Invalid Lead", invalid[0]
    if ev.do_not_contact_request:
        trace.append("D2 Not Interested: do-not-contact request")
        return "Not Interested", "Do Not Contact"
    if ev.final_unconditional_refusal:
        trace.append("D2 Not Interested: unconditional final refusal")
        return "Not Interested", "Explicit Refusal"
    w = ev.appointment_when
    slot_ok = w is not None and w.kind != "vague" and (w.hour is not None or w.part_of_day != "none")
    if ev.appointment_confirmed and not slot_ok:
        trace.append("D3 guard: 'confirmed' without a clock time/window is not a booking")
    if ev.appointment_confirmed and slot_ok:
        if ev.appointment_is_reschedule:
            sub = "Rescheduled"
        elif ev.appointment_mode == "online":
            sub = "Online Consultation"
        else:
            sub = "Clinic Visit"
        trace.append(f"D3 Appointment Scheduled: {sub}")
        return "Appointment Scheduled", sub
    if not ev.conversation_meaningful and ev.pain_need != "yes":
        trace.append("D5 Unable to Assess: no meaningful evidence")
        return "Unable to Assess", "Incomplete Conversation"
    trace.append("D4 Follow-up Required")
    return "Follow-up Required", ""


def follow_up_sub(ev: Evidence, ref: datetime, trace: list[str]) -> str:
    # SUB tie-break: customer's primary blocker first; the LLM lists them most-primary first.
    for b in ev.final_blockers:
        if b in BLOCKER_LABEL:
            trace.append(f"SUB blocker: {BLOCKER_LABEL[b]}")
            return BLOCKER_LABEL[b]
    if final_specific_callback(ev, ref) is not None:
        trace.append("SUB Specific Callback: acknowledged resolvable callback")
        return "Specific Callback"
    if ev.reviewing_requested_information:
        trace.append("SUB Information Requested")
        return "Information Requested"
    if has_vague_postponement(ev, ref):
        trace.append("SUB Vague Deferral")
        return "Vague Deferral"
    trace.append("SUB Further Discussion (default)")
    return "Further Discussion"


def lead_quality(disp: str, ev: Evidence, ref: datetime, trace: list[str]) -> str:
    # LQ order: Not Qualified, Not Assessable, Low, High, then Medium.
    if disp == "Invalid Lead":
        return "Not Qualified"
    if disp == "Unable to Assess" or ev.pain_need == "unclear" or ev.intent == "unclear":
        trace.append("LQ Not Assessable: need/willingness not established")
        return "Not Assessable"
    if disp == "Not Interested" or ev.intent in ("refused", "exploratory_only"):
        trace.append("LQ Low: refusal or exploratory curiosity")
        return "Low"
    if ev.defers_beyond_30_days and not ev.appointment_confirmed:
        trace.append("LQ Low: defers > 30 days")
        return "Low"
    if disp == "Appointment Scheduled":
        off = day_offset(ev.appointment_when, ref)
        trace.append(f"LQ appointment day offset = {off}")
        if off is not None and 0 <= off <= 6 and not ev.final_blockers:
            return "High"  # EX6: offsets 0-6, no unresolved blocker
        return "Medium"  # offset >= 7, unresolvable date, or a blocker remains
    if ev.commitment_within_7_days and not ev.final_blockers and ev.intent == "committed":
        trace.append("LQ High: explicit feasible commitment within 7 days, no blocker")
        return "High"
    return "Medium"


def decide(transcript_id: str, ev: Evidence, ref: datetime, transcript: str | None = None) -> Decision:
    trace: list[str] = []
    if transcript and not ev.vague_postponement:
        cue = deferral_cue(transcript)
        if cue:
            ev = ev.model_copy(update={"vague_postponement": True, "vague_postponement_quote": cue})
            trace.append(f"F7 backstop: customer deferral phrase '{cue}'")
    disp, sub = disposition(ev, trace)
    if disp == "Follow-up Required":
        sub = follow_up_sub(ev, ref, trace)
    assert sub in ALLOWED_PAIRS[disp], (disp, sub)
    lq = lead_quality(disp, ev, ref, trace)
    nca = next_call_at(disp, ev, ref)
    trace.append(f"F next_call_at = {fmt(nca)}")
    pred = Prediction(
        transcript_id=transcript_id, disposition=disp, sub_disposition=sub, lead_quality=lq, next_call_at=fmt(nca)
    )
    return Decision(pred, trace)


def fallback_prediction(transcript_id: str, ref: datetime) -> Prediction:
    """Conservative valid record used only when extraction fails after all retries."""
    from datetime import timedelta

    from .callback import DEFAULT_TIME

    nca = datetime.combine(ref.date() + timedelta(days=1), DEFAULT_TIME, tzinfo=ref.tzinfo)
    return Prediction(
        transcript_id=transcript_id,
        disposition="Follow-up Required",
        sub_disposition="Further Discussion",
        lead_quality="Medium",
        next_call_at=fmt(nca),
    )
