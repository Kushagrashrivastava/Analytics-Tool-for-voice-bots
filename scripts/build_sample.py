"""Assemble data/sample_results.jsonl for the dashboard from saved evidence files (no LLM calls).

For each call the newest evidence file listed first wins; the current deterministic rules are re-applied.
Usage: python scripts/build_sample.py runs/final_ev.jsonl [older_ev.jsonl ...]
"""

import json
import sys
from pathlib import Path

from aster_vale.core.callback import parse_call_time
from aster_vale.core.gender import predict_gender
from aster_vale.core.rules import decide
from aster_vale.core.schemas import Evidence

ROOT = Path(__file__).resolve().parents[1]
gold = {json.loads(x)["transcript_id"]: json.loads(x) for x in (ROOT / "data/labeled_50.jsonl").read_text().splitlines()}
evidence: dict[str, tuple[dict, str]] = {}
for path in sys.argv[1:]:
    version = Path(path).stem.replace("_ev", "")
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if row.get("evidence") and row["transcript_id"] not in evidence:
            evidence[row["transcript_id"]] = (row["evidence"], version)

out = []
for tid, g in gold.items():
    if tid not in evidence:
        continue
    ev, version = evidence[tid]
    d = decide(tid, Evidence.model_validate(ev), parse_call_time(g["call_started_at"], g["timezone"]), g["transcript"])
    out.append({**d.prediction.model_dump(), "call_started_at": g["call_started_at"], "transcript": g["transcript"],
                "evidence": ev, "trace": d.trace, "run": version,
                "gold": {k: g[k] for k in ("disposition", "sub_disposition", "lead_quality", "next_call_at")},
                "gender": predict_gender(g["transcript"])})
(ROOT / "data/sample_results.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
print(f"wrote {len(out)} calls; runs: { {v: sum(r['run'] == v for r in out) for v in {r['run'] for r in out}} }")
