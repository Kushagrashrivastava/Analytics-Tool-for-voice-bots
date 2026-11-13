"""End-to-end processing of one record or a batch. Never raises per record; output order = input order."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Iterable

from .callback import parse_call_time
from .llm import EvidenceExtractor
from .prompts import USER_TEMPLATE
from .rules import decide, fallback_prediction
from .schemas import Evidence, InputRecord, Prediction

log = logging.getLogger("aster_vale.pipeline")


@dataclass
class RecordResult:
    prediction: Prediction
    evidence: Evidence | None = None
    trace: list[str] = field(default_factory=list)
    status: str = "ok"  # ok | fallback
    error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached: bool = False
    latency_s: float = 0.0


def build_user_message(rec: InputRecord) -> str:
    ref = parse_call_time(rec.call_started_at, rec.timezone)
    return USER_TEMPLATE.format(
        call_local=ref.strftime("%Y-%m-%d %H:%M"), weekday=ref.strftime("%A"), timezone=rec.timezone,
        transcript=rec.transcript.strip(),
    )


def process_record(rec: InputRecord, extractor: EvidenceExtractor) -> RecordResult:
    ref = parse_call_time(rec.call_started_at, rec.timezone)
    try:
        res = extractor.extract(build_user_message(rec))
        d = decide(rec.transcript_id, res.evidence, ref, rec.transcript)
        return RecordResult(d.prediction, res.evidence, d.trace, "ok", None, res.prompt_tokens,
                            res.completion_tokens, res.cached, res.latency_s)
    except Exception as e:  # noqa: BLE001 - a record must never be dropped
        log.error("%s: extraction failed, using fallback: %r", rec.transcript_id, e)
        return RecordResult(fallback_prediction(rec.transcript_id, ref), status="fallback", error=repr(e))


def process_batch(records: list[InputRecord], extractor: EvidenceExtractor, concurrency: int = 2,
                  on_done: Callable[[int, RecordResult], None] | None = None) -> list[RecordResult]:
    results: list[RecordResult | None] = [None] * len(records)

    def work(i: int) -> None:
        results[i] = process_record(records[i], extractor)
        if on_done:
            on_done(i, results[i])

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        list(pool.map(work, range(len(records))))
    return results  # type: ignore[return-value]


def read_jsonl(lines: Iterable[str]) -> list[InputRecord]:
    """Parse input JSONL; duplicate IDs keep the first occurrence."""
    out, seen = [], set()
    for n, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = InputRecord.model_validate(json.loads(line))
        except Exception as e:  # noqa: BLE001
            log.error("line %d skipped: unparseable input record: %s", n, e)
            continue
        if rec.transcript_id in seen:
            log.warning("duplicate transcript_id %s ignored", rec.transcript_id)
            continue
        seen.add(rec.transcript_id)
        out.append(rec)
    return out
