from aster_vale.core.callback import fmt, next_call_at, parse_call_time, resolve
from aster_vale.core.schemas import Callback, Evidence, TimeSpec

MON = parse_call_time("2026-11-09T14:00:00+05:30")  # EX1-EX4 reference: Monday 14:00


def ts(kind="none", **kw) -> TimeSpec:
    base = dict(
        phrase="", kind=kind, n=None, weekday=None, weekday_qualifier="none", day=None, month=None,
        year=None, hour=None, minute=None, meridiem="none", part_of_day="none",
    )
    base.update(kw)
    return TimeSpec(**base)


def ev(**kw) -> Evidence:
    base = dict(
        conversation_meaningful=True, intended_patient="caller", patient_age=None, pain_need="yes",
        invalid_reasons=[], validity_quote=None, final_unconditional_refusal=False, do_not_contact_request=False,
        refusal_quote=None, appointment_confirmed=False, appointment_mode="none", appointment_is_reschedule=False,
        appointment_when=None, appointment_quote=None, final_blockers=[], blocker_quote=None,
        reviewing_requested_information=False, vague_postponement=False, vague_postponement_quote=None,
        callbacks=[], intent="considering", commitment_within_7_days=False, defers_beyond_30_days=False,
    )
    base.update(kw)
    return Evidence(**base)


def cb(when, ack=True, status="active", proposer="customer") -> Callback:
    return Callback(proposer=proposer, quote="", when=when, acknowledged_by_other_party=ack, status=status)


def r(spec, ref=MON):
    return fmt(resolve(spec, ref))


# --- Boundary examples EX1-EX5 ------------------------------------------------
def test_ex1_tomorrow_at_4_is_pm():
    e = ev(callbacks=[cb(ts("tomorrow", hour=4))])
    assert fmt(next_call_at("Follow-up Required", e, MON)) == "2026-11-10T16:00:00+05:30"


def test_ex2_vague_deferral_plus_3():
    e = ev(vague_postponement=True)
    assert fmt(next_call_at("Follow-up Required", e, MON)) == "2026-11-12T11:00:00+05:30"


def test_ex3_appointment_without_callback_is_null():
    assert next_call_at("Appointment Scheduled", ev(appointment_confirmed=True), MON) is None


def test_ex4_unspecified_agent_call_plus_1():
    assert fmt(next_call_at("Follow-up Required", ev(), MON)) == "2026-11-10T11:00:00+05:30"


def test_ex5_suppression():
    e = ev(callbacks=[cb(ts("tomorrow", hour=4))], do_not_contact_request=True)
    assert next_call_at("Not Interested", e, MON) is None


# --- F4 relative days / weekdays ---------------------------------------------
def test_relative_days():
    assert r(ts("day_after_tomorrow")) == "2026-11-11T11:00:00+05:30"
    assert r(ts("in_days", n=5)) == "2026-11-14T11:00:00+05:30"


def test_weekday_nearest_strictly_future():
    assert r(ts("weekday", weekday="monday")) == "2026-11-16T11:00:00+05:30"
    assert r(ts("weekday", weekday="wednesday")) == "2026-11-11T11:00:00+05:30"


def test_this_weekday_today_if_time_future():
    assert r(ts("weekday", weekday="monday", weekday_qualifier="this", hour=6)) == "2026-11-09T18:00:00+05:30"
    assert r(ts("weekday", weekday="monday", weekday_qualifier="this", hour=11)) == "2026-11-16T11:00:00+05:30"


def test_next_weekday_is_following_week():
    assert r(ts("weekday", weekday="wednesday", weekday_qualifier="next")) == "2026-11-18T11:00:00+05:30"


def test_next_week_alone_is_monday():
    assert r(ts("next_week")) == "2026-11-16T11:00:00+05:30"


# --- F5 clock defaults ------------------------------------------------------------
def test_clock_defaults():
    assert r(ts("tomorrow", hour=8)) == "2026-11-10T08:00:00+05:30"
    assert r(ts("tomorrow", hour=12)) == "2026-11-10T12:00:00+05:30"
    assert r(ts("tomorrow", hour=7, part_of_day="morning")) == "2026-11-10T07:00:00+05:30"
    assert r(ts("tomorrow", part_of_day="evening")) == "2026-11-10T18:00:00+05:30"
    assert r(ts("tomorrow", part_of_day="lunch")) == "2026-11-10T13:00:00+05:30"


def test_time_only_today_or_tomorrow():
    assert r(ts("none", hour=5)) == "2026-11-09T17:00:00+05:30"
    assert r(ts("none", hour=10)) == "2026-11-10T10:00:00+05:30"


def test_past_same_day_time_rejected():
    assert resolve(ts("today", hour=10, meridiem="am"), MON) is None


# --- F6 / F10 / F11 -----------------------------------------------------------------
def test_absolute_dates():
    assert r(ts("date", day=26)) == "2026-11-26T11:00:00+05:30"
    assert r(ts("date", day=4, month=11)) == "2027-11-04T11:00:00+05:30"  # month/day already past -> next year
    assert r(ts("date", day=1)) == "2026-12-01T11:00:00+05:30"
    assert resolve(ts("date", day=1, month=11, year=2026), MON) is None


def test_intervals_keep_seconds():
    ref = parse_call_time("2026-10-15T14:11:01+05:30")  # AVT-047
    assert r(ts("in_minutes", n=15), ref) == "2026-10-15T14:26:01+05:30"


def test_months():
    ref = parse_call_time("2026-01-31T10:00:00+05:30")
    assert r(ts("in_months", n=1), ref) == "2026-02-28T11:00:00+05:30"
    assert r(ts("next_month"), ref) == "2026-02-01T11:00:00+05:30"


# --- F3 / F9 acknowledgement and amendment ---------------------------------------------
def test_unacknowledged_callback_ignored():
    e = ev(callbacks=[cb(ts("weekday", weekday="wednesday"), ack=False, proposer="agent")])
    assert fmt(next_call_at("Follow-up Required", e, MON)) == "2026-11-10T11:00:00+05:30"


def test_amendment_replaces_earlier():
    e = ev(callbacks=[cb(ts("tomorrow"), status="amended"), cb(ts("day_after_tomorrow", hour=5))])
    assert fmt(next_call_at("Follow-up Required", e, MON)) == "2026-11-11T17:00:00+05:30"


def test_cancelled_callback_falls_back_to_vague():
    e = ev(callbacks=[cb(ts("tomorrow")), cb(ts("vague"), status="cancelled")], vague_postponement=True)
    assert fmt(next_call_at("Follow-up Required", e, MON)) == "2026-11-12T11:00:00+05:30"


def test_appointment_can_coexist_with_distinct_callback():
    e = ev(appointment_confirmed=True, callbacks=[cb(ts("tomorrow", part_of_day="morning"))])
    assert fmt(next_call_at("Appointment Scheduled", e, MON)) == "2026-11-10T10:00:00+05:30"


# --- Real gold cases ---------------------------------------------------------------
def test_gold_avt_014_friday():
    ref = parse_call_time("2026-10-14T15:13:56+05:30")
    e = ev(callbacks=[cb(ts("weekday", weekday="friday"), proposer="agent")])
    assert fmt(next_call_at("Follow-up Required", e, ref)) == "2026-10-16T11:00:00+05:30"


def test_gold_avt_004_tomorrow_evening():
    ref = parse_call_time("2026-10-14T13:19:21+05:30")
    e = ev(callbacks=[cb(ts("tomorrow", part_of_day="evening"), proposer="agent")])
    assert fmt(next_call_at("Follow-up Required", e, ref)) == "2026-10-15T18:00:00+05:30"


def test_gold_avt_029_window_uses_earliest():
    ref = parse_call_time("2026-10-17T18:05:01+05:30")
    e = ev(callbacks=[cb(ts("date", day=4, month=11))])
    assert fmt(next_call_at("Follow-up Required", e, ref)) == "2026-11-04T11:00:00+05:30"


def test_acknowledged_event_qualified_callback_is_vague():
    # AVT-024: "I will call you back once next week after Dussehra." / "Okay ma'am." -> F12 vague -> F7 (+3 days)
    ref = parse_call_time("2026-10-14T09:16:25+05:30")
    e = ev(callbacks=[cb(ts("vague", phrase="next week after Dussehra"), proposer="agent")])
    assert fmt(next_call_at("Follow-up Required", e, ref)) == "2026-10-17T11:00:00+05:30"


def test_f9_final_vague_agreement_cancels_earlier_specific_time():
    e = ev(callbacks=[cb(ts("tomorrow", hour=4)), cb(ts("vague", phrase="after Diwali"))], vague_postponement=True)
    assert fmt(next_call_at("Follow-up Required", e, MON)) == "2026-11-12T11:00:00+05:30"


def test_f6_month_day_without_year_today_but_past_rolls_forward():
    assert r(ts("date", day=9, month=11, hour=10, meridiem="am")) == "2027-11-09T10:00:00+05:30"


def test_f6_explicit_past_full_date_rejected_not_repaired():
    assert resolve(ts("date", day=9, month=11, year=2026, hour=10, meridiem="am"), MON) is None


def test_f12_this_week_vague_and_window_uses_start():
    assert resolve(ts("vague", phrase="this week"), MON) is None
    assert r(ts("tomorrow", hour=3, part_of_day="afternoon")) == "2026-11-10T15:00:00+05:30"  # "between 3 and 4"
