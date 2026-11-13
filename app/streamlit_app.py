"""Aster Vale Pain Clinic - enrollment call outcomes dashboard.

Run: streamlit run app/streamlit_app.py
Three audience views over the same labelled calls: Manager view (what to do), Model quality (how far to trust
the labels), Call explorer (evidence per call). API keys come from the environment / .env / Streamlit secrets.
"""

from __future__ import annotations

import io
import json
import os
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from aster_vale.core.callback import final_specific_callback, has_vague_postponement, parse_call_time, resolve
from aster_vale.core.gender import predict_gender
from aster_vale.core.llm import EvidenceExtractor, LLMConfig
from aster_vale.core.pipeline import process_batch, read_jsonl
from aster_vale.core.prompts import PROMPT_VERSION
from aster_vale.core.schemas import Evidence

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
MODEL_ID = "openai/gpt-oss-120b"
FIELDS = ["disposition", "sub_disposition", "lead_quality", "next_call_at"]
WEIGHTS = {"disposition": 15, "sub_disposition": 20, "lead_quality": 15, "next_call_at": 20}
FIELD_NAMES = {"disposition": "Outcome (disposition)", "sub_disposition": "Reason (sub-disposition)",
               "lead_quality": "Lead quality", "next_call_at": "Next call time"}

# ---- One colour = one meaning, everywhere (Datawrapper, EU Data Visualisation Guide) ----------------
OUTCOME = {"Appointment Scheduled": "#15803D", "Follow-up Required": "#D97706", "Not Interested": "#B91C1C",
           "Invalid Lead": "#6B7280", "Unable to Assess": "#9CA3AF"}
OUTCOME_SHORT = {"Appointment Scheduled": "Booked", "Follow-up Required": "Follow-up needed",
                 "Not Interested": "Not interested", "Invalid Lead": "Invalid lead",
                 "Unable to Assess": "Unable to assess"}
LEAD = {"High": "#4C1D95", "Medium": "#7C3AED", "Low": "#C4B5FD",  # ordinal -> one sequential hue
        "Not Assessable": "#D1D5DB", "Not Qualified": "#9CA3AF"}
QUEUE = {"Overdue": "#92400E", "Due today": "#D97706", "Next 7 days": "#F59E0B", "Later": "#FCD34D"}
NEUTRAL = "#475569"
# Reasons grouped by the kind of problem, so a manager sees 4 themes instead of 9 labels
THEMES = {"Practical": ["Travel Constraint", "Cost Concern"],
          "Doubts about treatment": ["Treatment Concern", "Current Treatment"],
          "Waiting on family": ["Family Discussion"],
          "No clear blocker": ["Information Requested", "Further Discussion", "Specific Callback", "Vague Deferral"]}
THEME_ICON = {"Practical": "🧳", "Doubts about treatment": "❓", "Waiting on family": "👨‍👩‍👧", "No clear blocker": "⏳"}
# Suggested next action per reason, based on what agents offered in the transcripts (ideas, not measured results)
PLAYBOOK = {
    "Travel Constraint": "offer the nearest centre or a slot on their free day, and book it during the call.",
    "Cost Concern": "mention the free 15-day follow-up and the 10% prepay discount; check if their insurance is accepted.",
    "Family Discussion": "offer a short call with the family member present; send the doctor's profile to share at home.",
    "Treatment Concern": "share the doctor's profile and how the procedure works; offer a senior-doctor slot.",
    "Current Treatment": "ask when the current treatment or review ends and set the callback for that date.",
    "Information Requested": "send what they asked for and agree a time to talk it through.",
    "Further Discussion": "call back with one clear question: which day suits you for the consultation?",
    "Specific Callback": "call exactly at the agreed time and offer two concrete slots.",
    "Vague Deferral": "ask for a specific day to call back instead of “later”.",
}
GENDER = {"Female": "#0E7490", "Male": "#B45309", "Multiple": "#64748B", "Unknown": "#CBD5E1"}  # no pink/blue

st.set_page_config(page_title="Aster Vale · Call Outcomes", page_icon="📞", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown("""
<style>
  [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="stToolbar"] {display:none}
  .block-container {padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1320px;}
  h1 {font-size: 1.9rem !important; margin-bottom: 0 !important;}
  .scope {color:#4B5563; font-size: .95rem; margin: 2px 0 10px 0;}
  .headline {font-size:1.15rem; color:#111827; background:#fff; border-left:5px solid #15803D;
             padding:10px 14px; border-radius:8px; margin:6px 0 12px 0}
  .kpi {background:#fff; border:1px solid #E5E7EB; border-radius:14px; padding:14px 16px; height:100%;}
  .kpi .label {color:#6B7280; font-size:.78rem; font-weight:650; text-transform:uppercase; letter-spacing:.03em}
  .kpi .value {font-size:1.95rem; font-weight:750; color:#111827; line-height:1.15; margin-top:4px}
  .kpi .sub {color:#4B5563; font-size:.88rem; margin-top:2px}
  .kpi {position:relative; cursor:help}
  .kpi .i {color:#9CA3AF; font-weight:700; margin-left:4px}
  .kpi .tip {display:none; position:absolute; z-index:50; left:0; right:0; top:calc(100% + 6px);
             background:#111827; color:#F9FAFB; font-size:.86rem; line-height:1.45; padding:12px 14px;
             border-radius:10px; box-shadow:0 8px 24px rgba(0,0,0,.25); text-transform:none; letter-spacing:0}
  .kpi .tip b {color:#fff}
  .kpi:hover .tip {display:block}
  .sec-title {font-size:1.15rem; font-weight:700; color:#111827; margin:0}
  .takeaway {color:#374151; font-size:.95rem; margin:2px 0 6px 0}
  .note {color:#6B7280; font-size:.84rem}
  .chip {display:inline-block; padding:3px 10px; border-radius:999px; font-size:.8rem; font-weight:600;
         color:#fff; margin:0 6px 6px 0}
  .quote {border-left:3px solid #94A3B8; background:#F8FAFC; padding:8px 12px; margin:6px 0; border-radius:6px;
          font-size:.9rem; color:#1F2937}
  .qcard {background:#fff; border:1px solid #E5E7EB; border-radius:12px; padding:12px 14px; height:100%}
  .qcard .t {font-weight:700; font-size:.95rem}
  .try {margin-top:8px; padding:8px 10px; background:#FFFBEB; border-radius:6px; font-size:.88rem; color:#78350F}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------------------
def pct(n: int, d: int) -> str:
    return f"{(100 * n / d):.0f}%" if d else "0%"


def calls(n: int) -> str:
    return f"{n} call" if n == 1 else f"{n} calls"


def nice_dt(iso: str | None, year: bool = False) -> str:
    """One human format everywhere (GOV.UK / Google style): 'Thu 15 Oct, 14:26' (+ seconds only if non-zero)."""
    if not iso:
        return "—"
    d = datetime.fromisoformat(iso)
    t = d.strftime("%H:%M:%S") if d.second else d.strftime("%H:%M")
    return d.strftime(f"%a %d %b{' %Y' if year else ''}, ") + t


def evidence(r: dict) -> Evidence | None:
    return Evidence.model_validate(r["evidence"]) if r.get("evidence") else None


def basis(r: dict) -> str:
    """Why this callback time (rules F3/F5/F7/F8)."""
    ev = evidence(r)
    if ev is None:
        return "fallback default"
    ref = parse_call_time(r["call_started_at"])
    if final_specific_callback(ev, ref) is not None:
        cb = [c for c in ev.callbacks if c.acknowledged_by_other_party and c.status == "active"][-1]
        no_time = cb.when.hour is None and cb.when.part_of_day == "none" and cb.when.kind not in (
            "in_minutes", "in_hours")
        return "agreed in the call" + (" (day agreed, no time → 11:00)" if no_time else "")
    if any(t.startswith("F7") for t in r.get("trace", [])) or has_vague_postponement(ev, ref):
        return "customer postponed without a date → call date + 3 days, 11:00"
    return "no next step agreed → next day, 11:00"


def basis_short(r: dict) -> str:
    """Short tag for the callback table (long form in basis())."""
    b = basis(r)
    if b.startswith("agreed in the call (day agreed"):
        return "Agreed day"
    if b.startswith("agreed in the call"):
        return "Agreed time"
    if b.startswith("customer postponed"):
        return "Vague → +3 days"
    if b.startswith("no next step"):
        return "No next step → next day"
    return "Fallback"


@st.cache_data
def load_sample() -> list[dict]:
    p = ROOT / "data/sample_results.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def get_api_key() -> str | None:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        try:
            key = st.secrets.get("GROQ_API_KEY")
        except Exception:  # noqa: BLE001 - no secrets file locally
            key = None
    return key


def parse_upload(file) -> list[dict]:
    raw = file.getvalue().decode("utf-8-sig")
    if file.name.lower().endswith(".csv"):
        df = pd.read_csv(io.StringIO(raw), dtype=str, keep_default_na=False)
        return [{k: (None if v in ("null", "") else v) for k, v in r.items()} for r in df.to_dict("records")]
    return [json.loads(x) for x in raw.splitlines() if x.strip()]


def analyse_upload(raw: list[dict], progress) -> list[dict]:
    records = read_jsonl(json.dumps(r) for r in raw)
    gold = {r["transcript_id"]: r for r in raw if r.get("transcript_id") and r.get("disposition")}
    extractor = EvidenceExtractor(LLMConfig(provider="groq", cache_path=str(ROOT / ".cache/llm_cache.sqlite")),
                                  api_key=get_api_key())
    done = {"n": 0}

    def tick(_, r):
        done["n"] += 1
        progress.progress(done["n"] / len(records), text=f"Analysed {done['n']} of {len(records)} calls")

    results = process_batch(records, extractor, concurrency=1, on_done=tick)
    return [{**res.prediction.model_dump(), "call_started_at": rec.call_started_at, "transcript": rec.transcript,
             "evidence": res.evidence.model_dump() if res.evidence else None, "trace": res.trace,
             "run": f"prompt {PROMPT_VERSION}" if res.status == "ok" else "fallback",
             "gold": {k: gold[rec.transcript_id].get(k) for k in FIELDS} if rec.transcript_id in gold else None,
             "gender": predict_gender(rec.transcript)} for rec, res in zip(records, results)]


def hbar(items: list[tuple[str, int]], colors, total: int, names: dict | None = None, per: int = 44):
    labels = [(names or {}).get(k, k) for k, _ in items][::-1]
    vals = [v for _, v in items][::-1]
    cols = [colors.get(k, "#9CA3AF") for k, _ in items][::-1] if isinstance(colors, dict) else list(colors)[::-1]
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h", marker_color=cols,
                           text=[f"  <b>{v}</b> · {pct(v, total)}" for v in vals], textposition="outside",
                           cliponaxis=False, textfont=dict(size=14, color="#111827"),
                           hovertemplate="%{y}: %{x}<extra></extra>"))
    fig.update_layout(height=40 + per * len(items), margin=dict(l=0, r=90, t=4, b=4), bargap=0.3,
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
                      xaxis=dict(visible=False, range=[0, max(vals or [1]) * 1.05]),
                      yaxis=dict(tickfont=dict(size=14, color="#111827"), ticksuffix="  "))
    return fig


def section(title: str, takeaway: str = ""):
    c = st.container(border=True)
    c.markdown(f"<p class='sec-title'>{title}</p>" + (f"<p class='takeaway'>{takeaway}</p>" if takeaway else ""),
               unsafe_allow_html=True)
    return c


def kpi(col, label: str, value: str, sub: str, tip: str = ""):
    """Number card. `tip` is a plain-English explanation shown on hover (what it means + how it is calculated)."""
    i = "<span class='i'>ⓘ</span>" if tip else ""
    t = f"<div class='tip'>{tip}</div>" if tip else ""
    col.markdown(f"<div class='kpi'><div class='label'>{label}{i}</div><div class='value'>{value}</div>"
                 f"<div class='sub'>{sub}</div>{t}</div>", unsafe_allow_html=True)


def table(df: pd.DataFrame, **kw):
    """Full-height tables: no scroll boxes inside the scrolling page (Baymard)."""
    st.dataframe(df, hide_index=True, width="stretch", height=35 * (len(df) + 1) + 3, **kw)


# ------------------------------------------------------------------------------------------------
# Header, dataset choice, upload
# ------------------------------------------------------------------------------------------------
if "upload_rows" not in st.session_state:
    st.session_state.upload_rows = None
    st.session_state.upload_name = None

st.markdown("<h1>Aster Vale Pain Clinic · Enrollment Call Outcomes</h1>", unsafe_allow_html=True)
sample = load_sample()
options = [f"Sample · {len(sample)} labelled calls"]
if st.session_state.upload_rows:
    options.append(f"Your upload · {len(st.session_state.upload_rows)} calls")
top_l, top_r = st.columns([3, 2])
with top_l:
    choice = st.segmented_control("Dataset", options, default=options[-1], label_visibility="collapsed") or options[0]
with top_r, st.expander("📤 Upload more data to see the analysis"):
    st.markdown("<span class='note'>JSONL or CSV with <code>transcript_id, call_started_at, timezone, transcript"
                "</code>. If the file also has the four label columns, the Model quality tab scores it. A new "
                "dashboard is built for your file.</span>", unsafe_allow_html=True)
    up = st.file_uploader("Upload transcripts", type=["jsonl", "csv"], label_visibility="collapsed")
    if up is not None:
        try:
            raw = parse_upload(up)
            n_up = len(read_jsonl(json.dumps(r) for r in raw))
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read this file: {e}")
            n_up = 0
        if n_up and st.button(f"Analyse {calls(n_up)} (≈{max(1, round(n_up * 0.75))} min on Groq free tier)",
                              type="primary"):
            if not get_api_key():
                st.error("GROQ_API_KEY is not configured on the server (.env or hosting secrets).")
            else:
                bar = st.progress(0.0, text="Starting…")
                st.session_state.upload_rows = analyse_upload(raw, bar)
                st.session_state.upload_name = up.name
                st.rerun()

is_upload = choice.startswith("Your upload")
rows: list[dict] = st.session_state.upload_rows if is_upload else sample
if not rows:
    st.info("No results yet. Upload a file above.")
    st.stop()
N = len(rows)
gold_rows = [r for r in rows if r.get("gold")]
G = len(gold_rows)
runs = Counter(r.get("run", "") for r in rows)
dates = sorted(datetime.fromisoformat(r["call_started_at"]).date() for r in rows)
st.markdown(f"<p class='scope'><b>{calls(N)}</b> made {dates[0]:%d %b} – {dates[-1]:%d %b %Y} · source: "
            f"{st.session_state.upload_name if is_upload else 'assignment sheet “50 Labeled Calls”'} · labels are "
            f"<b>AI-predicted</b> by Groq <code>{MODEL_ID}</code> and checked by the written policy rules</p>",
            unsafe_allow_html=True)

tab_mgr, tab_model, tab_calls = st.tabs(["📋 Manager view", "🔬 Model quality", "🔎 Call explorer"])


def correct(r: dict, f: str) -> bool:
    ok = r[f] == r["gold"][f]
    return ok and (f != "sub_disposition" or r["disposition"] == r["gold"]["disposition"])


# ================================================================================================
# MANAGER VIEW
# ================================================================================================
with tab_mgr:
    dc = Counter(r["disposition"] for r in rows)
    booked = dc["Appointment Scheduled"]
    not_booked = [r for r in rows if r["disposition"] != "Appointment Scheduled"]
    reasons = Counter(r["sub_disposition"] for r in not_booked if r["disposition"] == "Follow-up Required")
    top_reason = reasons.most_common(1)[0] if reasons else ("—", 0)
    lost = dc["Not Interested"] + dc["Invalid Lead"] + dc["Unable to Assess"]

    q1, _ = st.columns([1, 3])
    as_of = q1.date_input("Queue as of", value=dates[-1], help="Callbacks are classed as overdue / due today "
                          "relative to this date. Defaults to the last call date in the data.")
    cb = []
    for r in rows:
        if r.get("next_call_at"):
            when = datetime.fromisoformat(r["next_call_at"])
            delta = (when.date() - as_of).days
            status = ("Overdue" if delta < 0 else "Due today" if delta == 0 else
                      "Next 7 days" if delta <= 7 else "Later")
            cb.append({**r, "_when": when, "_status": status})
    cb.sort(key=lambda x: x["_when"])
    qc = Counter(x["_status"] for x in cb)

    st.markdown(
        f"<div class='headline'><b>{pct(booked, N)} of calls booked a consultation</b> ({booked} of {N}). "
        f"Top reason open leads have not booked: <b>{top_reason[0]}</b> ({top_reason[1]} of {dc['Follow-up Required']}). "
        f"As of {as_of:%d %b}: <b>{qc['Overdue']} callbacks overdue</b>, {qc['Due today']} due today.</div>",
        unsafe_allow_html=True)

    k = st.columns(4)
    kpi(k[0], "Booked (predicted)", f"{booked} of {N}", f"{pct(booked, N)} ended with a confirmed slot",
        f"<b>What it means:</b> calls where the person agreed a consultation <b>date and time</b> with the agent."
        f"<br><b>How we count it:</b> our system read each call and decided the outcome. {booked} of the {N} calls "
        f"ended with a booking, so {booked} ÷ {N} = {pct(booked, N)}.<br><b>Why “predicted”:</b> these labels come "
        "from our AI system, not from a person, so they can be wrong.")
    kpi(k[1], f"Callbacks as of {as_of:%d %b}", f"{qc['Overdue']} overdue",
        f"{qc['Due today']} due today · {qc['Next 7 days'] + qc['Later']} upcoming · {len(cb)} total",
        f"<b>What it means:</b> people the team must phone again. {len(cb)} calls have a planned callback time."
        f"<br><b>How we count it:</b> we compare each callback time with the “Queue as of” date "
        f"({as_of:%d %b}). Before that date = <b>overdue</b> ({qc['Overdue']}); on that date = <b>due today</b> "
        f"({qc['Due today']}); after it = <b>upcoming</b> ({qc['Next 7 days'] + qc['Later']}).")
    kpi(k[2], "Top blocker", top_reason[0], f"{top_reason[1]} of {dc['Follow-up Required']} open leads",
        f"<b>What it means:</b> the most common reason people did not book yet.<br><b>How we count it:</b> for "
        f"every call without a booking, the system picks the customer's main reason (e.g. travel, family, cost). "
        f"<b>{top_reason[0]}</b> was the reason in {top_reason[1]} of the {dc['Follow-up Required']} open leads "
        "(calls that ended in follow-up needed).")
    kpi(k[3], "Lost (predicted)", f"{lost} of {N}", f"{dc['Not Interested']} not interested · "
        f"{dc['Invalid Lead']} invalid · {dc['Unable to Assess']} unclear",
        "<b>What it means:</b> calls with no chance of a booking for now.<br><b>How we count it:</b> we add up "
        f"people who clearly said no ({dc['Not Interested']}), wrong or invalid contacts ({dc['Invalid Lead']}), "
        f"and calls too broken to judge ({dc['Unable to Assess']}). Total {lost} of {N}.")
    st.write("")

    # Funnel: each step is a subset of the one above (outcome + lead quality in one view)
    unusable = dc["Invalid Lead"] + dc["Unable to Assess"]
    valid = N - unusable
    open_ = valid - dc["Not Interested"]
    keen = sum(1 for r in rows if r["disposition"] in ("Appointment Scheduled", "Follow-up Required")
               and r["lead_quality"] in ("High", "Medium"))
    steps = [("Calls made", N, NEUTRAL)]
    if unusable:
        steps.append(("Valid leads", valid, "#64748B"))
    steps += [("Did not say no", open_, "#94A3B8"), ("Genuinely interested", keen, OUTCOME["Follow-up Required"]),
              ("Booked a consultation", booked, OUTCOME["Appointment Scheduled"])]
    with section("From call to booking", f"Of {calls(N)}, <b>{booked} booked</b> ({pct(booked, N)}). "
                 f"{keen - booked} more are genuinely interested but not booked yet: that is the follow-up "
                 f"opportunity."):
        fig = go.Figure(go.Funnel(
            y=[s for s, _, _ in steps], x=[v for _, v, _ in steps], marker=dict(color=[c for *_, c in steps]),
            texttemplate="<b>%{value}</b> calls · %{percentInitial:.0%}", textfont=dict(size=15, color="white"),
            connector=dict(fillcolor="#E5E7EB"), hovertemplate="%{y}: %{value} calls<extra></extra>"))
        fig.update_layout(height=70 + 64 * len(steps), margin=dict(l=0, r=0, t=4, b=4), font=dict(size=14),
                          yaxis=dict(tickfont=dict(size=16, color="#111827"), ticksuffix="  "),
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, config={"displayModeBar": False})
        drops = []
        if unusable:
            drops.append(f"{unusable} invalid or unusable")
        drops += [f"{dc['Not Interested']} said no", f"{open_ - keen} barely interested (Low or unclear)",
                  f"{keen - booked} interested but blocked or undecided (follow-up)"]
        st.caption("Dropped between steps: " + " · ".join(drops) + ". Percentages are out of all calls. "
                   "Genuinely interested = lead quality High or Medium: High means committed to a consultation "
                   "0–6 days after the call with nothing blocking; Medium means interested but a blocker or "
                   "unconfirmed visit remains.")

    # ---- Why calls did not book: open follow-ups only (denominator = follow-ups), grouped into themes ----
    follow_rows = [r for r in rows if r["disposition"] == "Follow-up Required"]
    F = len(follow_rows)
    if F:
        rs = sorted(reasons.items(), key=lambda kv: -kv[1])
        theme_rows = []
        for theme, members in THEMES.items():
            parts = [(m, reasons[m]) for m in members if reasons[m]]
            if parts:
                theme_rows.append((theme, sum(v for _, v in parts), parts))
        theme_rows.sort(key=lambda t: -t[1])
        caveat = ""
        if gold_rows:
            acc = sum(correct(r, "sub_disposition") for r in gold_rows) / G
            caveat = f"Reasons are AI-classified; on {calls(G)} with an answer key they matched {acc:.0%} of the time. "
        with section("Why open leads have not booked yet",
                     f"<b>{rs[0][0]}</b> is the #1 reason: {rs[0][1]} of {F} open leads ({pct(rs[0][1], F)})."):
            shades = ["#92400E", "#D97706", "#F59E0B", "#FCD34D"]  # amber family = follow-up, darkest = biggest
            pc, lc_ = st.columns([1, 1.1])
            with pc:
                fig = go.Figure(go.Pie(
                    labels=[t for t, _, _ in theme_rows], values=[n for _, n, _ in theme_rows], hole=0.55,
                    sort=False, direction="clockwise", rotation=0, marker=dict(colors=shades[:len(theme_rows)],
                                                                            line=dict(color="white", width=2)),
                    texttemplate="<b>%{value}</b><br>%{percent:.0%}", textposition="inside",
                    textfont=dict(size=15), insidetextorientation="horizontal",
                    hovertemplate="%{label}: %{value} of " + str(F) + " open leads (%{percent:.0%})<extra></extra>"))
                fig.add_annotation(text=f"<b>{F}</b><br>open leads", showarrow=False, font=dict(size=18, color="#111827"))
                fig.update_layout(height=300, margin=dict(l=0, r=0, t=6, b=6), showlegend=False,
                                  paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, config={"displayModeBar": False})
            with lc_:
                for (t, n, parts), colr in zip(theme_rows, shades):
                    inner = " · ".join(f"{m} <b>{v}</b>" for m, v in parts)
                    st.markdown(f"<div style='border-left:8px solid {colr}; padding:6px 12px; margin:10px 0; "
                                f"background:#fff; border-radius:8px'><div style='font-size:1.02rem'>{THEME_ICON[t]} "
                                f"<b>{t}</b> · {n} of {F} ({pct(n, F)})</div><div class='note' style='font-size:.9rem'>"
                                f"{inner}</div></div>", unsafe_allow_html=True)
            ni = Counter(r["sub_disposition"] for r in rows if r["disposition"] == "Not Interested")
            extra = f"Not counted here: {sum(ni.values())} people who said no. " if ni else ""
            st.caption(f"Open leads = calls that ended in “follow-up needed” ({F}). Each slice is a theme; the list beside "
                       f"it shows the exact reasons inside each theme. {extra}{caveat}")

            st.markdown("**Top 3 reasons, in the customer's words, and what to try**")
            qcols = st.columns(3)
            for col, (reason, n) in zip(qcols, rs[:3]):
                cands = {}
                for r in follow_rows:
                    ev = r.get("evidence") or {}
                    qt = (ev.get("blocker_quote") or ev.get("vague_postponement_quote") or "").strip()
                    if r["sub_disposition"] == reason and qt and qt not in cands:
                        verified = bool(r.get("gold")) and r["gold"]["sub_disposition"] == reason
                        # prefer calls whose reason is confirmed by the answer key, then clear ~90-character quotes
                        cands[qt] = (not verified, abs(len(qt) - 90), r["transcript_id"])
                best = sorted(cands.items(), key=lambda kv: kv[1][:2])[:2]
                body = "".join(f"<div class='quote'>“{q[:180]}” <span class='note'>· {meta[2]}</span></div>"
                               for q, meta in best) or "<span class='note'>No quote extracted.</span>"
                tip = PLAYBOOK.get(reason, "")
                col.markdown(f"<div class='qcard'><div class='t'>{reason} · {n} of {F}</div>{body}"
                             + (f"<div class='try'>💡 <b>What to try:</b> {tip}</div>" if tip else "") + "</div>",
                             unsafe_allow_html=True)
            st.caption("“What to try” lines are suggestions drawn from what agents offered in these calls "
                       "(e.g. prepay discount, senior-doctor slot), not measured results.")

    with section("Callback queue", f"<b>{len(cb)} people to call back.</b> As of {as_of:%a %d %b %Y}: "
                 f"<b>{qc['Overdue']} overdue</b>, <b>{qc['Due today']} due today</b>, {qc['Next 7 days']} in the "
                 f"next 7 days, {qc['Later']} later."):
        if cb:
            # 1) One status strip: the whole bar = every callback, split by urgency
            fig = go.Figure()
            for stt in QUEUE:
                if qc[stt]:
                    fig.add_bar(y=[""], x=[qc[stt]], orientation="h", name=stt, marker_color=QUEUE[stt],
                                text=f"<b>{stt}</b>  {qc[stt]}", textposition="inside", insidetextanchor="middle",
                                textfont=dict(size=15, color="white" if stt in ("Overdue", "Due today") else "#111827"),
                                hovertemplate=f"{stt}: {qc[stt]} of {len(cb)}<extra></extra>")
            fig.update_layout(barmode="stack", height=70, margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
                              xaxis=dict(visible=False, range=[0, len(cb)]), yaxis=dict(visible=False), bargap=0,
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, config={"displayModeBar": False}, key="queue_strip")

            # 2) Compact workload-by-day chart on a real date axis
            with st.expander("Callbacks per day (workload calendar)"):
                per_day = Counter((x["_when"].date(), x["_status"]) for x in cb)
                fig = go.Figure()
                for stt in QUEUE:
                    ds = sorted(d for d, s_ in per_day if s_ == stt)
                    if ds:
                        fig.add_bar(x=ds, y=[per_day[(d, stt)] for d in ds], name=stt, marker_color=QUEUE[stt],
                                    text=[per_day[(d, stt)] for d in ds], textposition="outside", cliponaxis=False,
                                    hovertemplate="%{x|%a %d %b}: %{y}<extra>" + stt + "</extra>")
                fig.update_layout(barmode="stack", height=200, margin=dict(l=0, r=0, t=24, b=0),
                                  plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                  yaxis=dict(visible=False), xaxis=dict(type="date", tickformat="%d %b"),
                                  legend=dict(orientation="h", y=1.2, traceorder="normal"))
                st.plotly_chart(fig, config={"displayModeBar": False}, key="queue_days")

            # 3) The list, split by urgency into tabs (no repeated status column)
            def queue_table(items: list[dict], overdue: bool) -> pd.DataFrame:
                out = []
                for x in items:
                    days = (x["_when"].date() - as_of).days
                    when_rel = (f"{-days} day{'s' if -days != 1 else ''} late" if days < 0 else
                                "today" if days == 0 else f"in {days} day{'s' if days != 1 else ''}")
                    out.append({"Call back at": nice_dt(x["next_call_at"]), "When": when_rel,
                                "Call": x["transcript_id"], "Reason": x["sub_disposition"],
                                "Lead": x["lead_quality"], "Time set by": basis_short(x)})
                return pd.DataFrame(out)

            tabs = st.tabs([f"{stt} · {qc[stt]}" for stt in QUEUE])
            for tab, stt in zip(tabs, QUEUE):
                with tab:
                    items = [x for x in cb if x["_status"] == stt]
                    if items:
                        table(queue_table(items, stt == "Overdue"))
                    else:
                        st.caption("Nothing here.")
            all_df = pd.DataFrame([{"Status": x["_status"], **queue_table([x], False).iloc[0].to_dict()} for x in cb])
            st.download_button("⬇ Download the full callback list (CSV)", all_df.to_csv(index=False),
                               file_name="callbacks.csv")
        st.caption("“Time set by” — Agreed time: both sides agreed a time · Agreed day: a day was agreed, no time, so "
                   "11:00 · Vague → +3 days: customer said e.g. “I'll let you know”, so call date + 3 days at 11:00 · "
                   "No next step → next day: nothing agreed, so call date + 1 day at 11:00 (policy rules F3–F8).")

    appts = []
    for r in rows:
        ev = evidence(r)
        if r["disposition"] == "Appointment Scheduled" and ev and ev.appointment_when:
            booked_on = parse_call_time(r["call_started_at"])
            slot = resolve(ev.appointment_when, booked_on)
            if not slot:
                continue
            ahead = (slot.date() - booked_on.date()).days
            remind = slot.date() - timedelta(days=1)
            if ahead == 0:
                stage, remind_txt = "Same-day booking", "not needed (same day)"
            elif remind <= booked_on.date():
                stage, remind_txt = ("Appointment passed" if slot.date() < as_of else
                                     "Today's appointments" if slot.date() == as_of else "Upcoming"), "booked the day before"
            else:
                remind_txt = f"{remind:%a %d %b}"
                stage = None
            if stage is None or stage == "Same-day booking":
                stage = ("Appointment passed" if slot.date() < as_of else "Today's appointments" if slot.date() == as_of
                         else "Remind today" if remind == as_of else "Upcoming")
            appts.append({"_stage": stage, "_d": slot, "Appointment": nice_dt(slot.isoformat()), "Remind on": remind_txt,
                          "Call": r["transcript_id"], "Booked": f"{ahead} day{'s' if ahead != 1 else ''} ahead"
                          if ahead else "same day", "Agreed in the call as": f"“{ev.appointment_when.phrase}”",
                          "Type": r["sub_disposition"]})
    if appts:
        appts.sort(key=lambda a: a["_d"])
        stages = ["Remind today", "Today's appointments", "Upcoming", "Appointment passed"]
        sc_ = Counter(a["_stage"] for a in appts)
        n_today = sc_["Today's appointments"]
        rt = sc_["Remind today"]
        lead_txt = (f"<b>{rt} reminder call{'s' if rt != 1 else ''} due today</b>" if rt
                    else "no reminder calls due today")
        with section("Appointments to confirm",
                     f"<b>{calls(len(appts))} booked.</b> As of {as_of:%d %b}: {lead_txt}, {n_today} appointment"
                     f"{'s' if n_today != 1 else ''} today, {sc_['Upcoming']} upcoming, "
                     f"{sc_['Appointment passed']} already passed (check they attended)."):
            show_type = len({a["Type"] for a in appts}) > 1
            tabs = st.tabs([f"{s_} · {sc_[s_]}" for s_ in stages])
            for tab, s_ in zip(tabs, stages):
                with tab:
                    items = [a for a in appts if a["_stage"] == s_]
                    if items:
                        df_ = pd.DataFrame(items).drop(columns=["_stage", "_d"] + ([] if show_type else ["Type"]))
                        table(df_)
                    else:
                        st.caption("Nothing here.")
            st.caption("Why: calling patients the day before their appointment cut no-shows from 26% to 19% in a "
                       "randomized trial (JABFM 2001). Same-day bookings need no reminder; for passed appointments, "
                       "check whether the patient attended.")

# ================================================================================================
# MODEL QUALITY
# ================================================================================================
with tab_model:
    run_txt = ", ".join(f"{k} ({calls(v)})" for k, v in runs.items())
    st.info(f"**Model** `{MODEL_ID}` on Groq · temperature 0 · seed 7 · reasoning effort medium · strict JSON schema "
            f"· current prompt **{PROMPT_VERSION}**. Labels on this page come from: {run_txt}. "
            "The LLM only extracts evidence; Python applies every policy rule.")
    if not gold_rows:
        st.warning("This dataset has no answer-key columns, so accuracy cannot be computed. Upload a file that also "
                   "contains disposition, sub_disposition, lead_quality and next_call_at.")
    else:
        fc = {f: sum(correct(r, f) for r in gold_rows) for f in FIELDS}
        pts = sum(fc[f] / G * WEIGHTS[f] for f in FIELDS)
        exact = sum(all(correct(r, f) for f in FIELDS) for r in gold_rows)
        wrong = 4 * G - sum(fc.values())
        k = st.columns(3)
        parts = " + ".join(f"{fc[f] / G * WEIGHTS[f]:.1f}" for f in FIELDS)
        kpi(k[0], "Weighted score", f"{pts:.1f} / 70", f"assignment weights · training set ({calls(G)} given)",
            "<b>What it means:</b> the assignment's scoring formula (the “Evaluation weights” row) applied to the "
            "50 labelled calls in the sheet. It is a development score, not the private evaluation score."
            "<br><b>How it is calculated:</b> every call has 4 predicted labels. Each is compared with the answer key "
            "in the “50 Labeled Calls” tab. Weights: Outcome 15, Reason 20, Lead quality 15, Next call time 20 "
            "(70 in total). Each field earns its weight × the share of calls it got right.<br><b>Example:</b> "
            f"Reason was right on {fc['sub_disposition']} of {G} calls → {fc['sub_disposition']}/{G} × 20 = "
            f"{fc['sub_disposition'] / G * 20:.1f} points. Adding all four: {parts} = <b>{pts:.1f}</b>.")
        kpi(k[1], "Calls fully correct", f"{exact} of {G}", f"{pct(exact, G)} have all four labels right",
            "<b>What it means:</b> calls where <b>all 4 answers</b> (outcome, reason, lead quality, next call time) "
            "match the correct answers.<br><b>How we count it:</b> if even one of the 4 is wrong, the call does not "
            f"count. {exact} of {G} calls had all 4 right, so {exact} ÷ {G} = {pct(exact, G)}. This is the strictest "
            "measure.")
        kpi(k[2], "Wrong labels", f"{wrong} of {4 * G}", f"across {calls(G - exact)}",
            f"<b>What it means:</b> how many single answers were wrong.<br><b>How we count it:</b> {G} calls × 4 "
            f"answers each = {4 * G} answers. {wrong} of them do not match the correct answer. They sit on "
            f"{calls(G - exact)} (one call can have more than one wrong answer). Every wrong answer is listed in "
            "the table further down this page.")
        earned = {f: fc[f] / G * WEIGHTS[f] for f in FIELDS}
        worst = max(FIELDS, key=lambda f: WEIGHTS[f] - earned[f])
        st.markdown("<p class='takeaway' style='margin-top:10px'><b>Per field:</b> "
                    + " · ".join(f"{FIELD_NAMES[f].split(' (')[0]} {fc[f]}/{G} ({pct(fc[f], G)})" for f in FIELDS)
                    + f". Most missed points: <b>{FIELD_NAMES[worst].split(' (')[0]}</b> "
                    f"({WEIGHTS[worst] - earned[worst]:.1f} of {70 - pts:.1f}).</p>", unsafe_allow_html=True)
        if not is_upload:
            st.caption("Honest estimate: these calls were also used to write the prompt, so this score is optimistic. "
                       "The cleanest unseen-data number we have is 58.6 / 70 (prompt v2 on 18 calls not used while "
                       "writing it)." + (" See ERROR_ANALYSIS.md." if (ROOT / "ERROR_ANALYSIS.md").exists() else ""))
        st.write("")
        with section("Where the model confuses labels", "Rows = answer key, columns = our label. The diagonal "
                     "(correct) is greyed so the mistakes stand out."):
            fld = st.selectbox("Field", ["sub_disposition", "disposition", "lead_quality"],
                               format_func=lambda f: FIELD_NAMES[f])
            data = [r for r in gold_rows if fld != "sub_disposition" or r["disposition"] == r["gold"]["disposition"]]
            labs = sorted({r["gold"][fld] for r in data} | {r[fld] for r in data},
                          key=lambda s: -sum(r["gold"][fld] == s for r in data))
            m = [[sum(r["gold"][fld] == g and r[fld] == p for r in data) for p in labs] for g in labs]
            z = [[(v if i != j and v else None) for j, v in enumerate(row)] for i, row in enumerate(m)]
            fig = go.Figure(go.Heatmap(z=z, x=labs, y=labs, colorscale=[[0, "#BFDBFE"], [1, "#1E3A8A"]],
                                       showscale=False, hoverongaps=False,
                                       hovertemplate="key %{y} → ours %{x}: %{z}<extra></extra>"))
            fig.add_trace(go.Scatter(x=[p for _ in labs for p in labs], y=[g for g in labs for _ in labs],
                                     mode="text", hoverinfo="skip",
                                     text=[(str(v) if v else "") for row in m for v in row],
                                     textfont=dict(size=13, color="#111827")))
            fig.update_layout(height=140 + 34 * len(labs), margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
                              plot_bgcolor="#F1F5F9", xaxis=dict(side="top", tickangle=-30, title="Our label"),
                              yaxis=dict(autorange="reversed", title="Answer key"))
            st.plotly_chart(fig, config={"displayModeBar": False})
            st.caption("Grey cells with a number = correct; blue cells = mistakes (darker = more).")

        notes = json.loads((ROOT / "data/error_notes.json").read_text()) if (
            not is_upload and (ROOT / "data/error_notes.json").exists()) else {}

        def show(f: str, v):
            return (nice_dt(v) if v else "no callback") if f == "next_call_at" else v

        errs = [{"Call": r["transcript_id"], "Field": FIELD_NAMES[f], "Answer key": show(f, r["gold"][f]),
                 "Our label": show(f, r[f]), "Points lost": f"−{WEIGHTS[f] / G:.2f}",
                 "Likely cause": notes.get(r["transcript_id"], "—")}
                for r in gold_rows for f in FIELDS if not correct(r, f)]
        if errs:
            with section(f"All {len(errs)} wrong labels", "A wrong label = for that call and field, our answer "
                         "differs from the answer key in the “50 Labeled Calls” tab. Every row is shown."):
                st.table(pd.DataFrame(errs).set_index("Call"))  # static table: wraps long causes, no inner scroll

    log_path = ROOT / "data/run_log.json"
    if log_path.exists() and not is_upload:
        log = json.loads(log_path.read_text())
        with section("What we changed and what it did", "Every prompt or rule change, the calls it was measured "
                     "on, and the score against the answer key. Rule changes re-score saved evidence (no new LLM calls)."):
            st.table(pd.DataFrame([{"Step": it["step"], "Type": it["kind"], "Measured on": it["calls"],
                                    "Score": it["score"], "All 4 right": it["exact"], "What changed": it["change"]}
                                   for it in log["iterations"]]).set_index("Step"))
        u = log["usage"]
        n_calls = sum(x["calls"] for x in u)
        pin, pout = log["price_per_million_usd"]["input"], log["price_per_million_usd"]["output"]
        tin, tout = sum(x["prompt_tokens"] for x in u), sum(x["completion_tokens"] for x in u)
        secs = sum(x["runtime_s"] for x in u)
        cost = tin / 1e6 * pin + tout / 1e6 * pout
        with section("Runtime, tokens and cost", f"Measured over {calls(n_calls)} sent to Groq "
                     f"<code>{MODEL_ID}</code> during development (cached repeats excluded)."):
            k = st.columns(4)
            kpi(k[0], "Tokens per call", f"{(tin + tout) / n_calls / 1000:.1f}K",
                f"{tin / n_calls / 1000:.1f}K in · {tout / n_calls / 1000:.1f}K out",
                "<b>What it means:</b> tokens are small pieces of text (about ¾ of a word). The AI service charges "
                "by tokens.<br><b>How we calculate it:</b> Groq tells us how many tokens each request used. "
                f"We sent {n_calls} calls: {tin:,} tokens in (instructions + transcript) and {tout:,} out (the answer). "
                f"Total ÷ {n_calls} = {(tin + tout) / n_calls / 1000:.1f}K per call.")
            kpi(k[1], "Cost per call", f"${cost / n_calls:.4f}", f"at Groq list price ${pin}/M in, ${pout}/M out",
                f"<b>What it means:</b> what it costs to label one phone call.<br><b>How we calculate it:</b> "
                f"Groq's price is ${pin} per million tokens in and ${pout} per million out. "
                f"{tin:,} in × ${pin}/M + {tout:,} out × ${pout}/M = ${cost:.4f} for {n_calls} calls → "
                f"${cost / n_calls:.4f} each. We used the free tier, so we actually paid $0.")
            kpi(k[2], "Cost for 50 calls", f"${cost / n_calls * 50:.3f}", "the size of the hidden evaluation set",
                f"<b>What it means:</b> the estimated cost of labelling 50 calls, the size of the private evaluation set.<br><b>How "
                f"we calculate it:</b> cost per call (${cost / n_calls:.4f}) × 50 calls = ${cost / n_calls * 50:.3f}.")
            kpi(k[3], "Time per call", f"{secs / n_calls:.0f} s", "free tier, 1 at a time (8K tokens/min limit)",
                f"<b>What it means:</b> how long one call takes to label.<br><b>How we calculate it:</b> total running "
                f"time ({secs / 60:.0f} min) ÷ {n_calls} calls. Most of this is waiting: the free tier lets us send "
                "only about 8,000 tokens per minute, so we must pause between calls. On a paid tier it takes a few "
                "seconds per call.")
            st.write("")
            st.table(pd.DataFrame([{"Run": x["run"], "Calls": x["calls"], "Runtime": f"{x['runtime_s'] / 60:.1f} min",
                                    "Input tokens": f"{x['prompt_tokens']:,}", "Output tokens": f"{x['completion_tokens']:,}",
                                    "Cost (USD)": f"${x['prompt_tokens'] / 1e6 * pin + x['completion_tokens'] / 1e6 * pout:.4f}"}
                                   for x in u]).set_index("Run"))
            st.caption("Time per call is dominated by the free tier's rate limit; on a paid Groq tier the same calls take a "
                       "few seconds each. Cost is computed from the token counts Groq returned for each request.")

    with section("Task 2 · Caller and patient gender (no AI model)", "Rule-based: the agent's form of address "
                 "(sir / ma'am), self-references (“my husband”), stated first names, and the patient relation "
                 "(“my mother”). Shown only in aggregate."):
        gx = [predict_gender(r["transcript"]) for r in rows]
        g1, g2 = st.columns([1, 1.15])
        with g1:
            st.markdown(f"**Gender mix** · each bar = all {calls(N)}")
            fig = go.Figure()
            for gname in ("Male", "Female", "Multiple", "Unknown"):
                vals = [sum(x[k].capitalize() == gname for x in gx) for k in ("caller_gender", "patient_gender")]
                if any(vals):
                    fig.add_bar(y=["Callers", "Patients"], x=vals, orientation="h", name=gname,
                                marker_color=GENDER[gname], text=[f"{v} · {pct(v, N)}" if v >= 3 else "" for v in vals],
                                textposition="inside", insidetextanchor="middle",
                                textfont=dict(color="white", size=14), hovertemplate=f"{gname}: %{{x}}<extra></extra>")
            fig.update_layout(barmode="stack", height=210, margin=dict(l=0, r=0, t=4, b=4), bargap=0.35,
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                              xaxis=dict(visible=False), yaxis=dict(autorange="reversed",
                                                                    tickfont=dict(size=15, color="#111827")),
                              legend=dict(orientation="h", y=-0.12, traceorder="normal"))
            st.plotly_chart(fig, config={"displayModeBar": False}, key="g_mix")

        def who(x: dict) -> str:
            c = {"male": "A man", "female": "A woman"}.get(x["caller_gender"], "Someone")
            if x["patient_is"] == "caller":
                return f"{c}, for {'himself' if x['caller_gender'] == 'male' else 'herself' if x['caller_gender'] == 'female' else 'themself'}"
            if x["patient_gender"] == "multiple":
                return f"{c}, for two relatives"
            return f"{c}, for a {'female' if x['patient_gender'] == 'female' else 'male' if x['patient_gender'] == 'male' else ''} relative".replace("a  relative", "a relative")

        wc = Counter(who(x) for x in gx).most_common()
        fem = sum(x["patient_gender"] == "female" for x in gx)
        fem_self = sum(x["patient_gender"] == "female" and x["patient_is"] == "caller" for x in gx)
        with g2:
            st.markdown("**Who called for whom**")
            st.plotly_chart(hbar(wc, [NEUTRAL] * len(wc), N, per=34), config={"displayModeBar": False}, key="g_who")
        if fem:
            st.markdown(f"<p class='takeaway'><b>{fem} patients are women, but only {fem_self} called for themselves</b>: "
                        f"the rest reached the clinic through a relative. This links to the “Family Discussion” blocker.</p>",
                        unsafe_allow_html=True)
        gg = ROOT / "data/gold_gender.csv"
        if gg.exists() and not is_upload:
            hand = pd.read_csv(gg, dtype=str).set_index("transcript_id")["caller_gender"].to_dict()
            pred = {r["transcript_id"]: predict_gender(r["transcript"])["caller_gender"] for r in rows}
            cov = [t for t in hand if pred.get(t) not in (None, "unknown")]
            ok = sum(pred[t] == hand[t] for t in cov)
            st.caption(f"Caller gender vs hand labels: {ok} of {len(cov)} correct, {len(hand) - len(cov)} abstained. "
                       "Hand labels were made from the same kind of cues, so this checks consistency, not an "
                       "independent benchmark. Unknown = no reliable cue, so the detector abstains. Multiple = two "
                       "patients of different genders.")

# ================================================================================================
# CALL EXPLORER
# ================================================================================================
with tab_calls:
    f1, f2, f3, f4 = st.columns([1, 1, 1, 1.2])
    fo = f1.multiselect("Outcome", list(OUTCOME), format_func=lambda d: OUTCOME_SHORT[d])
    fr = f2.multiselect("Reason", sorted({r["sub_disposition"] for r in rows}))
    fl = f3.multiselect("Lead quality", list(LEAD))
    search = f4.text_input("Search call ID or transcript text", placeholder="e.g. AVT-041 or insurance")
    only_wrong = st.checkbox("Only calls with a wrong label", disabled=not gold_rows)
    view = [r for r in rows if (not fo or r["disposition"] in fo) and (not fr or r["sub_disposition"] in fr)
            and (not fl or r["lead_quality"] in fl)
            and (not search or search.lower() in (r["transcript_id"] + r["transcript"]).lower())
            and (not only_wrong or (r.get("gold") and not all(correct(r, f) for f in FIELDS)))]
    left, right = st.columns([1.05, 1.25])
    with left:
        st.caption(f"Showing {len(view)} of {calls(N)} · click a row")
        tdf = pd.DataFrame([{"Call": r["transcript_id"], "Date": nice_dt(r["call_started_at"]),
                             "Outcome": OUTCOME_SHORT[r["disposition"]], "Reason": r["sub_disposition"],
                             "Lead": r["lead_quality"],
                             **({"Key": "✓" if all(correct(r, f) for f in FIELDS) else "✗"} if r.get("gold") else {})}
                            for r in view])
        sel = st.dataframe(tdf, hide_index=True, width="stretch", height=560, on_select="rerun",
                           selection_mode="single-row", key="explorer")
        st.download_button("⬇ Download these calls (CSV)", tdf.to_csv(index=False), file_name="calls.csv")
        st.download_button("⬇ Download predictions (JSONL)",
                           "".join(json.dumps({"transcript_id": r["transcript_id"], **{f: r[f] for f in FIELDS}}) + "\n"
                                   for r in rows), file_name="predictions.jsonl")
    with right:
        picked = sel.selection.rows[0] if sel and sel.selection.rows else (0 if view else None)
        if picked is None:
            st.info("No calls match these filters.")
        else:
            r = view[picked]
            st.markdown(f"#### {r['transcript_id']} · {nice_dt(r['call_started_at'], year=True)}")
            st.markdown(
                f"<span class='chip' style='background:{OUTCOME[r['disposition']]}'>{OUTCOME_SHORT[r['disposition']]}</span>"
                f"<span class='chip' style='background:#334155'>{r['sub_disposition']}</span>"
                f"<span class='chip' style='background:{LEAD[r['lead_quality']]}'>Lead: {r['lead_quality']}</span>"
                f"<span class='chip' style='background:{NEUTRAL}'>Next call: {nice_dt(r['next_call_at'])}</span>",
                unsafe_allow_html=True)
            if r.get("gold"):
                table(pd.DataFrame([{"Field": FIELD_NAMES[f],
                                     "Answer key": nice_dt(r["gold"][f]) if f == "next_call_at" else r["gold"][f],
                                     "Our label": nice_dt(r[f]) if f == "next_call_at" else r[f],
                                     "": "✓" if correct(r, f) else "✗"} for f in FIELDS]))
            ev = r.get("evidence") or {}
            st.markdown("**Evidence quoted from the transcript**")
            shown = False
            for label, k2 in [("Appointment", "appointment_quote"), ("Blocker", "blocker_quote"),
                              ("Refusal", "refusal_quote"), ("Postponement", "vague_postponement_quote"),
                              ("Validity", "validity_quote")]:
                if ev.get(k2):
                    shown = True
                    st.markdown(f"<div class='quote'><b>{label}:</b> “{ev[k2]}”</div>", unsafe_allow_html=True)
            for c in ev.get("callbacks") or []:
                shown = True
                st.markdown(f"<div class='quote'><b>Callback</b> · proposed by {c['proposer']} · "
                            f"{'acknowledged' if c['acknowledged_by_other_party'] else 'not acknowledged'} · "
                            f"{c['status']}: “{c['quote']}”</div>", unsafe_allow_html=True)
            if not shown:
                st.caption("No quotes extracted.")
            if r.get("next_call_at"):
                st.caption(f"Next call basis: {basis(r)}")
            with st.expander("Policy rules applied"):
                st.code("\n".join(r.get("trace") or ["—"]), language="text")
            gd = predict_gender(r["transcript"])
            st.caption(f"Gender (rule-based): caller {gd['caller_gender']} · patient {gd['patient_gender']} "
                       f"({gd['patient_is']})")
            with st.expander("Transcript", expanded=True):
                st.text(r["transcript"])
