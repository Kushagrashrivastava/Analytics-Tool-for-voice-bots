"""Data contracts: input records, LLM evidence (intermediate schema) and final predictions."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Label vocabulary (policy assignment-v2, "Label Rules" tab)
# ---------------------------------------------------------------------------
DISPOSITIONS = (
    "Appointment Scheduled",
    "Follow-up Required",
    "Not Interested",
    "Invalid Lead",
    "Unable to Assess",
)
ALLOWED_PAIRS: dict[str, tuple[str, ...]] = {
    "Appointment Scheduled": ("Clinic Visit", "Online Consultation", "Rescheduled"),
    "Follow-up Required": (
        "Specific Callback",
        "Vague Deferral",
        "Family Discussion",
        "Cost Concern",
        "Travel Constraint",
        "Current Treatment",
        "Treatment Concern",
        "Information Requested",
        "Further Discussion",
    ),
    "Not Interested": ("Explicit Refusal", "Do Not Contact"),
    "Invalid Lead": ("Wrong Contact", "No Pain Need", "Outside Service Area", "Policy Exclusion", "Test Enquiry"),
    "Unable to Assess": ("Incomplete Conversation",),
}
LEAD_QUALITIES = ("High", "Medium", "Low", "Not Qualified", "Not Assessable")


# ---------------------------------------------------------------------------
# I/O records
# ---------------------------------------------------------------------------
class InputRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")  # gold columns, if present, are dropped here

    transcript_id: str
    call_started_at: str
    timezone: str = "Asia/Kolkata"
    transcript: str


class Prediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transcript_id: str
    disposition: str
    sub_disposition: str
    lead_quality: str
    next_call_at: str | None


# ---------------------------------------------------------------------------
# Intermediate evidence returned by the LLM. Strict-JSON-schema friendly:
# every field required, no defaults, nullable via `X | None`.
# ---------------------------------------------------------------------------
Kind = Literal[
    "none",  # no day reference (time-only or nothing)
    "today",
    "tomorrow",
    "day_after_tomorrow",
    "in_days",
    "weekday",
    "date",
    "next_week",
    "next_month",
    "in_months",
    "in_minutes",
    "in_hours",
    "vague",  # event-only / unresolvable ("after Diwali", "next week after the function")
]


class TimeSpec(BaseModel):
    """Structured reading of a spoken time expression. Python resolves it to a timestamp."""

    model_config = ConfigDict(extra="forbid")

    phrase: str = Field(description="Exact words used, e.g. 'tomorrow evening', 'on the 26th', 'in 15 minutes'")
    kind: Kind
    n: int | None = Field(description="Count for in_days / in_months / in_minutes / in_hours")
    weekday: Literal["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"] | None
    weekday_qualifier: Literal["none", "this", "next"]
    day: int | None = Field(description="Day of month for kind=date")
    month: int | None = Field(description="Month number for kind=date, null if not said")
    year: int | None
    hour: int | None = Field(description="Clock hour exactly as spoken (1-12 or 0-23), null if none")
    minute: int | None
    meridiem: Literal["am", "pm", "none"]
    part_of_day: Literal["none", "morning", "afternoon", "lunch", "evening", "night"]


class Callback(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposer: Literal["agent", "customer"]
    quote: str
    when: TimeSpec
    acknowledged_by_other_party: bool
    status: Literal["active", "amended", "cancelled", "rejected"]


Blocker = Literal[
    "family_discussion", "cost_concern", "travel_constraint", "current_treatment", "treatment_concern"
]


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # --- validity -----------------------------------------------------------
    conversation_meaningful: bool
    intended_patient: Literal["caller", "family_member", "unknown"]
    patient_age: int | None
    pain_need: Literal["yes", "no", "unclear"]
    invalid_reasons: list[
        Literal["test_enquiry", "wrong_contact", "no_pain_need", "policy_exclusion_under_15", "outside_service_area"]
    ]
    validity_quote: str | None
    # --- refusal -------------------------------------------------------------
    final_unconditional_refusal: bool
    do_not_contact_request: bool
    refusal_quote: str | None
    # --- appointment -----------------------------------------------------------
    appointment_confirmed: bool
    appointment_mode: Literal["clinic", "online", "none"]
    appointment_is_reschedule: bool
    appointment_when: TimeSpec | None
    appointment_quote: str | None
    # --- open journey ---------------------------------------------------------
    final_blockers: list[Blocker] = Field(description="Final unresolved blockers, most primary first")
    blocker_quote: str | None
    reviewing_requested_information: bool
    vague_postponement: bool
    vague_postponement_quote: str | None
    callbacks: list[Callback] = Field(description="Every callback discussed, in chronological order")
    # --- readiness --------------------------------------------------------------
    intent: Literal["committed", "considering", "exploratory_only", "refused", "unclear"]
    commitment_within_7_days: bool
    defers_beyond_30_days: bool
