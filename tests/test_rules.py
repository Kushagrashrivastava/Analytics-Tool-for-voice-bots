from aster_vale.core.callback import parse_call_time
from aster_vale.core.rules import decide, deferral_cue
from test_callback import cb, ev, ts

REF = parse_call_time("2026-10-17T20:42:51+05:30")


def test_precedence_invalid_beats_refusal_and_booking():
    e = ev(patient_age=12, final_unconditional_refusal=True, appointment_confirmed=True)
    p = decide("X", e, REF).prediction
    assert (p.disposition, p.sub_disposition, p.lead_quality, p.next_call_at) == (
        "Invalid Lead", "Policy Exclusion", "Not Qualified", None)


def test_do_not_contact_beats_explicit_refusal():
    p = decide("X", ev(final_unconditional_refusal=True, do_not_contact_request=True), REF).prediction
    assert (p.disposition, p.sub_disposition, p.lead_quality) == ("Not Interested", "Do Not Contact", "Low")


def test_appointment_offsets_drive_lead_quality():
    near = ev(appointment_confirmed=True, appointment_mode="clinic", appointment_when=ts("in_days", n=6, hour=2))
    far = ev(appointment_confirmed=True, appointment_mode="clinic", appointment_when=ts("in_days", n=7, hour=2))
    assert decide("X", near, REF).prediction.lead_quality == "High"
    assert decide("X", far, REF).prediction.lead_quality == "Medium"


def test_blocker_beats_specific_callback_for_sub_but_callback_time_kept():
    e = ev(final_blockers=["cost_concern"], callbacks=[cb(ts("tomorrow", part_of_day="evening"))])
    p = decide("X", e, REF).prediction
    assert p.sub_disposition == "Cost Concern" and p.next_call_at == "2026-10-18T18:00:00+05:30"


def test_deferral_backstop_sets_plus_three():
    t = "Agent: Can you come Monday?\nCustomer: No, I will let you know once my husband is back.\nAgent: Okay."
    assert deferral_cue(t) == "let you know"
    p = decide("X", ev(final_blockers=["travel_constraint"]), REF, t).prediction
    assert p.next_call_at == "2026-10-20T11:00:00+05:30"


def test_backstop_never_overrides_specific_callback():
    t = "Customer: I'll see. Call me on the 26th.\nAgent: Okay, I will call you on the 26th."
    p = decide("X", ev(callbacks=[cb(ts("date", day=26))]), REF, t).prediction
    assert p.next_call_at == "2026-10-26T11:00:00+05:30"


def test_d3_guard_confirmed_without_time_is_not_booking():
    e = ev(appointment_confirmed=True, appointment_mode="clinic", appointment_when=ts("tomorrow"))
    assert decide("X", e, REF).prediction.disposition == "Follow-up Required"


def test_lq_high_requires_resolvable_offset():
    e = ev(appointment_confirmed=True, appointment_mode="clinic",
           appointment_when=ts("date", day=1, month=10, year=2026, hour=2))  # explicit past date
    assert decide("X", e, REF).prediction.lead_quality == "Medium"


def test_reschedule_beats_online_and_online_beats_clinic():
    w = ts("tomorrow", hour=11, meridiem="am")
    assert decide("X", ev(appointment_confirmed=True, appointment_mode="online", appointment_is_reschedule=True,
                          appointment_when=w), REF).prediction.sub_disposition == "Rescheduled"
    assert decide("X", ev(appointment_confirmed=True, appointment_mode="online", appointment_when=w),
                  REF).prediction.sub_disposition == "Online Consultation"


def test_invalid_tie_break_order_and_no_inferred_exclusion():
    e = ev(invalid_reasons=["outside_service_area", "wrong_contact", "test_enquiry"])
    assert decide("X", e, REF).prediction.sub_disposition == "Test Enquiry"
    e2 = ev(invalid_reasons=["policy_exclusion_under_15"], patient_age=None)  # age unknown -> no exclusion
    assert decide("X", e2, REF).prediction.disposition == "Follow-up Required"


def test_follow_up_tie_break_family_first():
    e = ev(final_blockers=["family_discussion", "cost_concern"])
    assert decide("X", e, REF).prediction.sub_disposition == "Family Discussion"


def test_lq_order_low_before_high_and_defer_beyond_30():
    assert decide("X", ev(intent="exploratory_only", commitment_within_7_days=True), REF).prediction.lead_quality == "Low"
    assert decide("X", ev(defers_beyond_30_days=True), REF).prediction.lead_quality == "Low"


def test_unable_to_assess_and_suppression():
    p = decide("X", ev(conversation_meaningful=False, pain_need="unclear"), REF).prediction
    assert (p.disposition, p.sub_disposition, p.lead_quality, p.next_call_at) == (
        "Unable to Assess", "Incomplete Conversation", "Not Assessable", None)


def test_ex5_final_do_not_contact_after_callback():
    e = ev(callbacks=[cb(ts("tomorrow", hour=4))], do_not_contact_request=True)
    p = decide("X", e, REF).prediction
    assert (p.disposition, p.sub_disposition, p.lead_quality, p.next_call_at) == (
        "Not Interested", "Do Not Contact", "Low", None)
