"""Scoring that mirrors the assignment's scoring contract."""

from __future__ import annotations

from collections import Counter

FIELDS = ("disposition", "sub_disposition", "lead_quality", "next_call_at")
WEIGHTS = {"disposition": 15, "sub_disposition": 20, "lead_quality": 15, "next_call_at": 20}


def _norm(v):
    return None if v in (None, "null", "") else v


def score(preds: list[dict], gold: list[dict]) -> dict:
    p_by = {p["transcript_id"]: p for p in preds}
    n = len(gold)
    correct = Counter()
    exact = 0
    mismatches = []
    confusion = {f: Counter() for f in FIELDS}
    for g in gold:
        p = p_by.get(g["transcript_id"], {})
        ok = {}
        for f in FIELDS:
            gv, pv = _norm(g.get(f)), _norm(p.get(f))
            hit = gv == pv
            if f == "sub_disposition":  # credit only with the correct parent disposition
                hit = hit and _norm(g.get("disposition")) == _norm(p.get("disposition"))
            ok[f] = hit
            correct[f] += hit
            confusion[f][(gv, pv)] += 1
        if all(ok.values()):
            exact += 1
        else:
            mismatches.append({
                "transcript_id": g["transcript_id"],
                **{f: {"gold": _norm(g.get(f)), "pred": _norm(p.get(f))} for f in FIELDS if not ok[f]},
            })
    acc = {f: correct[f] / n if n else 0.0 for f in FIELDS}
    return {
        "n": n,
        "accuracy": acc,
        "exact_match": exact / n if n else 0.0,
        "weighted_points_of_70": sum(acc[f] * WEIGHTS[f] for f in FIELDS),
        "mismatches": mismatches,
        "confusion": {f: [[g, p, c] for (g, p), c in confusion[f].items()] for f in FIELDS},
    }
