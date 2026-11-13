# Aster Vale Call Intelligence

Labels pain-clinic enrollment calls with **disposition, sub-disposition, lead quality and next callback
time** (policy `assignment-v2`), and infers **caller/patient gender without any transformer model**.
A Streamlit dashboard sits on top of the same code for interactive review.

```
transcript ──► Groq · openai/gpt-oss-120b ──► Evidence JSON ──► deterministic Python policy ──► 4 labels
               (reads the call, quotes evidence,   (strict schema)   (precedence D6, PAIR, tie-breaks,
                reports *spoken* time phrases)                          LQ order, callback rules F1–F12)
```

**Why this split:** the LLM is good at reading messy, translated Hinglish dialogue; it is bad at applying
precedence tables and calendar arithmetic consistently. So the LLM answers small evidence questions
(*was a slot mutually confirmed? what is the customer's final blocker? which callbacks were acknowledged and
what exact words set the time?*) and Python makes every decision. Every prediction carries a rule trace.

---

## Data

The assignment transcripts and answer key are **not committed** (they belong to the hiring team). Export the
"50 Labeled Calls" tab as CSV, then:

```bash
python scripts/prepare_data.py "50 Labeled Calls.csv"   # -> data/labeled_50.jsonl, data/inputs_50.jsonl
```

## Quick start (graded runner)

```bash
# Python 3.11–3.13
python -m venv .venv && source .venv/bin/activate
pip install -e .
export GROQ_API_KEY=gsk_...            # or put it in .env (see .env.example)

aster-vale run --input data/inputs_50.jsonl --output predictions.jsonl
```

That is the single command: **input JSONL → prediction JSONL**, one object per input, exactly five keys:

```json
{"transcript_id":"AVT-002","disposition":"Appointment Scheduled","sub_disposition":"Clinic Visit","lead_quality":"High","next_call_at":null}
```

Input records need `transcript_id, call_started_at, timezone, transcript`; any other columns (e.g. gold
labels) are ignored and never reach the model.

| Option | Default | |
|---|---|---|
| `--provider` | `groq` | graded provider. `cerebras`/`nvidia` exist for development only (see *Model choice*) |
| `--model` | `openai/gpt-oss-120b` | |
| `--reasoning-effort` | `medium` | |
| `--concurrency` | `2` | use `1` on the Groq free tier |
| `--evidence-out FILE` | off | also write extracted evidence + rule trace per call |
| `--no-cache` | off | responses are cached in `.cache/llm_cache.sqlite` keyed by provider+model+prompt version+input |

Other commands:

```bash
aster-vale evaluate --pred predictions.jsonl --gold data/labeled_50.jsonl --show-errors   # per-field accuracy
aster-vale rescore  --evidence ev.jsonl --input data/inputs_50.jsonl --output p.jsonl     # re-apply rules, no LLM
aster-vale gender   --input data/inputs_50.jsonl --output gender.jsonl                    # rule-based gender
pytest -q                                                                                 # 29 tests
streamlit run app/streamlit_app.py                                                        # dashboard (pip install -e ".[web]")
docker compose up app                                                                     # dashboard in Docker
```

## Generation settings and reliability

| Setting | Value |
|---|---|
| Model | `openai/gpt-oss-120b` on Groq (exact ID) |
| temperature / seed | `0` / `7` |
| reasoning_effort | `medium`, `include_reasoning=false` |
| max_completion_tokens | 6000 |
| Output format | `response_format = json_schema` with `strict: true` (schema generated from `core/schemas.py::Evidence`); automatic fallback to `json_object` + schema-in-prompt if a provider rejects strict mode |
| Validation | Pydantic v2 (`extra="forbid"`); a schema-invalid reply is retried |
| Timeout | 90 s per request |
| Retries | up to 5 attempts (tenacity) on 429 / 5xx / timeouts / connection errors / invalid JSON; honours `retry-after`, else exponential backoff 2→60 s |
| Never drop a record | if a call still fails, a conservative valid record is emitted (`Follow-up Required / Further Discussion / Medium / call date +1 day 11:00`) and logged as `fallback` |
| Input hygiene | unparseable input lines are skipped with a log line; duplicate IDs keep the first occurrence; output order = input order |

**Groq free-tier note:** `gpt-oss-120b` on the free tier is limited to ~8K tokens/minute and ~200K
tokens/day. One call is ~6K tokens, so a free key processes about one call per 45 s and ~33 calls per day.
The runner waits and retries automatically; a Developer-tier key runs the 50 calls in a few minutes.

## Policy implementation (deterministic, `core/rules.py`, `core/callback.py`)

- **Disposition precedence (D6):** Invalid Lead → Not Interested → Appointment Scheduled → Follow-up
  Required → Unable to Assess. Under-15 is the only policy exclusion; distance alone is never invalid.
- **Sub-disposition:** only children allowed by the PAIR table; Do Not Contact beats Explicit Refusal;
  Rescheduled beats modality; follow-up tie-break Family → Cost → Travel → Current Treatment → Treatment
  Concern → Specific Callback → Information Requested → Vague Deferral → Further Discussion.
- **Lead quality order:** Not Qualified → Not Assessable → Low → High (confirmed slot on day offset 0–6, no
  blocker) → Medium.
- **next_call_at:** the LLM returns a structured *reading* of each spoken time (`kind`, weekday + this/next,
  day/month, hour as spoken, am/pm, part of day). Python resolves it against `call_started_at`:
  suppression (F2); last acknowledged, non-cancelled callback (F3/F9); tomorrow/parso/in N days, weekday
  semantics, next week = Monday (F4); 11:00 default, 1–7 → PM, morning 10 / afternoon 15 / evening 18 /
  lunch 13 (F5); year roll-over and past-time rejection (F6); `+3 days 11:00` for vague postponement,
  including acknowledged event-only callbacks like "next week after Dussehra" (F7/F12); `+1 day 11:00`
  otherwise; null for appointments (F8); minute/hour intervals keep seconds (F10); month clamping (F11).
  Output is always `YYYY-MM-DDTHH:MM:SS+05:30` or JSON `null`.

All boundary examples EX1–EX6 and each F-rule are unit tests in `tests/test_callback.py`.

## Model choice

The brief requires an open model served by **Groq**. Of Groq's current production models, `openai/gpt-oss-120b`
is the strongest reasoning model with strict JSON-schema output and a 131K context (Llama-3.3-70B, Qwen3-32B,
Kimi-K2 have been retired from Groq's free tier). `openai/gpt-oss-20b` is the cheaper backup (`--model`).

Development constraint: the Groq free tier (~200K tokens/day) cannot fit repeated full runs. The code
therefore supports other OpenAI-compatible providers via `--provider` purely for development iteration.
All reported numbers below come from **Groq `openai/gpt-oss-120b`**, the graded configuration.

## Results

See [`ERROR_ANALYSIS.md`](ERROR_ANALYSIS.md) for per-field accuracy, failure patterns, the iteration log,
runtime, token usage and cost. Predictions for all 50 supplied calls (`predictions_50.jsonl`) come from one
run of the frozen prompt v3; that run is in progress (throttled by the Groq free-tier daily token limit) and the
file will be added when it completes. See ERROR_ANALYSIS.md §1 for the current status.

## Task 2 · Gender without transformers (`core/gender.py`)

Weighted evidence voting (signed log-odds, + male / − female), separately for the **caller** and the
**patient**:

1. **Agent forms of address**, recency-weighted: `sir / bhaiya / Mr` vs `ma'am / madam / didi / Mrs`.
   Only *agent* turns count: customers also say "sir/ma'am" — to the agent.
2. **Self-kinship:** "my husband" → female caller, "my wife" → male caller; "I am Renu's husband" → male
   caller, female patient.
3. **Stated first name** (from "my name is…", replies to "your good name?", or the agent addressing them)
   matched to an Indian first-name lexicon, with a weak suffix back-off.
4. **Patient relation:** "my father/son/…" → male patient, "my mother/daughter/…" → female; both parents →
   `multiple`; otherwise the patient is assumed to be the caller.
5. |score| < 1 → `unknown` (abstain rather than guess).

Result on the 50 calls vs hand labels (`data/gold_gender.csv`): **50/50 covered, 50/50 correct** after lowering the abstain threshold to 0.5 so a single explicit agent "Sir" counts; previously the one
abstention (AVT-042) contains a single "Sir". Caveat: the hand labels were made by reviewing the same
kinds of cues (agent address, self-introductions), so this measures consistency on clear cases rather
than an independent benchmark. Known limits: speaker-label swaps, two people on speakerphone, agents
mis-addressing callers, and patients referred to without a kinship word.

## Dashboard

`streamlit run app/streamlit_app.py` → paste a transcript, upload JSONL/CSV, or load the saved 50-call run.
KPIs, disposition / lead-quality / sub-disposition / callback-timeline bar charts (no pie charts: ~20
sub-labels and ordinal lead quality read far better as sorted or ordered bars), per-call evidence quotes and
rule trace beside the transcript, accuracy and confusion matrix when gold labels are present, and gender.
API keys typed into the UI are used only for that session and never written to disk.

## Layout

```
src/aster_vale/core/  schemas.py  prompts.py  llm.py  pipeline.py  rules.py  callback.py  evaluate.py  gender.py
src/aster_vale/cli.py app/streamlit_app.py  tests/  data/  runs/ (local run artefacts, git-ignored)
```

Dependencies are pinned in `pyproject.toml` (openai 1.109.1, pydantic 2.11.9, tenacity 9.1.2,
python-dotenv 1.1.1; web extra: streamlit 1.50.0, plotly 6.3.0, pandas 2.3.3).
