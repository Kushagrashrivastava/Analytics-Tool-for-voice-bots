"""Deterministic time resolution and next_call_at policy (rules F1-F12).

The LLM never computes timestamps. It returns a TimeSpec (what was said); this module turns it
into a concrete datetime relative to call_started_at.
"""

from __future__ import annotations

import calendar
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from .schemas import Evidence, TimeSpec

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
PART_OF_DAY_HOUR = {"morning": 10, "afternoon": 15, "lunch": 13, "evening": 18, "night": 20}
DEFAULT_TIME = time(11, 0)


def parse_call_time(call_started_at: str, tz: str = "Asia/Kolkata") -> datetime:
    dt = datetime.fromisoformat(call_started_at)
    zone = ZoneInfo(tz)
    return dt.astimezone(zone) if dt.tzinfo else dt.replace(tzinfo=zone)


def fmt(dt: datetime | None) -> str | None:
    return None if dt is None else dt.isoformat(timespec="seconds")


def _add_months(d: date, n: int) -> date:
    y, m = divmod(d.month - 1 + n, 12)
    y, m = d.year + y, m + 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def _clock(spec: TimeSpec) -> time | None:
    """F5: clock-time defaults. Returns None when no time of day was given."""
    h = spec.hour
    if h is not None:
        minute = spec.minute or 0
        if spec.meridiem == "pm" and h < 12:
            h += 12
        elif spec.meridiem == "am" and h == 12:
            h = 0
        elif spec.meridiem == "none" and h <= 12:
            if spec.part_of_day in ("afternoon", "evening", "night") and h < 12:
                h += 12
            elif spec.part_of_day == "morning":
                pass
            elif 1 <= h <= 7:
                h += 12
        if not 0 <= h <= 23 or not 0 <= minute <= 59:
            return None
        return time(h, minute)
    if spec.part_of_day in PART_OF_DAY_HOUR:
        return time(PART_OF_DAY_HOUR[spec.part_of_day], 0)
    return None


def resolve(spec: TimeSpec | None, ref: datetime) -> datetime | None:
    """Resolve a spoken time to an absolute datetime in ref's timezone; None if vague/invalid."""
    if spec is None or spec.kind == "vague":
        return None
    tz = ref.tzinfo
    today = ref.date()

    # F10: minute/hour intervals keep seconds.
    if spec.kind in ("in_minutes", "in_hours"):
        if not spec.n or spec.n <= 0:
            return None
        return ref + (timedelta(minutes=spec.n) if spec.kind == "in_minutes" else timedelta(hours=spec.n))

    clock = _clock(spec)
    t = clock or DEFAULT_TIME
    at = lambda d: datetime.combine(d, t, tzinfo=tz)  # noqa: E731

    k = spec.kind
    if k == "none":
        if clock is None:
            return None
        # F5: time only -> today if still future, otherwise tomorrow.
        return at(today) if at(today) > ref else at(today + timedelta(days=1))
    if k == "today":
        d = today
    elif k == "tomorrow":
        d = today + timedelta(days=1)
    elif k == "day_after_tomorrow":
        d = today + timedelta(days=2)
    elif k == "in_days":
        if spec.n is None or spec.n < 0:
            return None
        d = today + timedelta(days=spec.n)
    elif k == "weekday":
        if spec.weekday is None:
            return None
        wd = WEEKDAYS.index(spec.weekday)
        delta = (wd - today.weekday()) % 7
        if spec.weekday_qualifier == "next":
            # F4: occurrence in the following Monday-Sunday week.
            d = today - timedelta(days=today.weekday()) + timedelta(days=7 + wd)
        elif spec.weekday_qualifier == "this":
            d = today + timedelta(days=delta)
            if at(d) <= ref:
                d += timedelta(days=7)
        else:
            d = today + timedelta(days=delta or 7)  # nearest strictly future occurrence
    elif k == "next_week":
        d = today - timedelta(days=today.weekday()) + timedelta(days=7)
    elif k == "next_month":
        d = date(today.year + (today.month == 12), today.month % 12 + 1, 1)
    elif k == "in_months":
        d = _add_months(today, spec.n or 1)
    elif k == "date":
        d = _resolve_date(spec, today)
        if d is None:
            return None
    else:
        return None

    result = at(d)
    if result <= ref and k == "date" and spec.year is None and d == today:
        # F6: "next future occurrence" of a month/day lacking a year (today's date with a passed time)
        d = _add_months(d, 1) if spec.month is None else date(d.year + 1, d.month, d.day)
        result = at(d)
    # F6/F9: never return a past time; explicit past commitments are rejected, not repaired.
    return result if result > ref else None


def _resolve_date(spec: TimeSpec, today: date) -> date | None:
    if spec.day is None:
        return None
    try:
        if spec.month is None:
            # Bare day number: this month if not past, else next month.
            cand = date(today.year, today.month, spec.day)
            if cand < today:
                nxt = _add_months(date(today.year, today.month, 1), 1)
                cand = date(nxt.year, nxt.month, spec.day)
            return cand
        if spec.year is not None:
            return date(spec.year, spec.month, spec.day)
        cand = date(today.year, spec.month, spec.day)
        return cand if cand >= today else date(today.year + 1, spec.month, spec.day)
    except ValueError:
        return None


def day_offset(spec: TimeSpec | None, ref: datetime) -> int | None:
    """Calendar-day offset of a (consultation) time from the call date; None if unresolvable."""
    if spec is None:
        return None
    if spec.kind == "none":
        clock = _clock(spec)
        if clock is None:
            return 0  # slot agreed with no day -> same-day booking
    dt = resolve(spec, ref)
    if dt is None and spec.kind in ("today", "none"):
        return 0
    return None if dt is None else (dt.date() - ref.date()).days


def final_specific_callback(ev: Evidence, ref: datetime) -> datetime | None:
    """F3/F9: the FINAL acknowledged, still-active callback. If that final agreement is vague or resolves to a
    past time, there is no specific callback (an earlier specific time is not resurrected)."""
    chosen: datetime | None = None
    for cb in ev.callbacks:
        if cb.status == "cancelled":
            chosen = None
            continue
        if cb.status in ("amended", "rejected") or not cb.acknowledged_by_other_party:
            continue
        chosen = resolve(cb.when, ref)
    return chosen


def has_vague_postponement(ev: Evidence, ref: datetime) -> bool:
    """F7/F12: explicit vague postponement, or an agreed callback whose timing is event-based/unresolvable."""
    if ev.vague_postponement:
        return True
    return any(cb.status == "active" and cb.acknowledged_by_other_party and resolve(cb.when, ref) is None
               and cb.when.kind == "vague" for cb in ev.callbacks)


def next_call_at(disposition: str, ev: Evidence, ref: datetime) -> datetime | None:
    if disposition in ("Invalid Lead", "Not Interested", "Unable to Assess"):
        return None  # F2
    specific = final_specific_callback(ev, ref)
    if specific is not None:
        return specific  # F3, also allowed alongside Appointment Scheduled
    if disposition == "Appointment Scheduled":
        return None  # F8
    base = ref.date()
    if has_vague_postponement(ev, ref):
        return datetime.combine(base + timedelta(days=3), DEFAULT_TIME, tzinfo=ref.tzinfo)  # F7
    return datetime.combine(base + timedelta(days=1), DEFAULT_TIME, tzinfo=ref.tzinfo)  # F8
