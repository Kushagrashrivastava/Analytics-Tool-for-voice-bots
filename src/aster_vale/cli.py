"""Command line: `aster-vale run`, `aster-vale evaluate`, `aster-vale gender`."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PRICE_PER_M = {"openai/gpt-oss-120b": (0.15, 0.60), "openai/gpt-oss-20b": (0.075, 0.30)}  # USD, Groq list price


def cmd_run(a) -> int:
    from .core.llm import EvidenceExtractor, LLMConfig
    from .core.pipeline import process_batch, read_jsonl

    records = read_jsonl(Path(a.input).read_text(encoding="utf-8").splitlines())
    cfg = LLMConfig(provider=a.provider, model=a.model, reasoning_effort=a.reasoning_effort,
                    cache_path=None if a.no_cache else a.cache)
    try:
        extractor = EvidenceExtractor(cfg)
    except RuntimeError as e:
        print(f"error: {e}. Export it or put it in .env (see .env.example).", file=sys.stderr)
        return 2
    t0 = time.monotonic()

    def progress(i, r):
        flag = "cache" if r.cached else r.status
        print(f"[{i + 1}/{len(records)}] {r.prediction.transcript_id} {flag} -> "
              f"{r.prediction.disposition} / {r.prediction.sub_disposition}", file=sys.stderr)

    results = process_batch(records, extractor, concurrency=a.concurrency, on_done=progress)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(r.prediction.model_dump_json() + "\n")
    if a.evidence_out:
        with open(a.evidence_out, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps({"transcript_id": r.prediction.transcript_id, "status": r.status,
                                    "error": r.error, "trace": r.trace,
                                    "evidence": r.evidence.model_dump() if r.evidence else None},
                                   ensure_ascii=False) + "\n")
    pt = sum(r.prompt_tokens for r in results)
    ct = sum(r.completion_tokens for r in results)
    pin, pout = PRICE_PER_M.get(cfg.model_id, (0.15, 0.60))
    stats = {"records": len(results), "fallbacks": sum(r.status != "ok" for r in results),
             "cached": sum(r.cached for r in results), "provider": cfg.provider, "model": cfg.model_id,
             "runtime_s": round(time.monotonic() - t0, 1), "prompt_tokens": pt, "completion_tokens": ct,
             "est_cost_usd_at_groq_price": round(pt / 1e6 * pin + ct / 1e6 * pout, 4)}
    print(json.dumps(stats), file=sys.stderr)
    return 0


def cmd_evaluate(a) -> int:
    from .core.evaluate import score

    load = lambda p: [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]  # noqa: E731
    gold = load(a.gold)
    if a.ids:
        keep = set(Path(a.ids).read_text().split())
        gold = [g for g in gold if g["transcript_id"] in keep]
    res = score(load(a.pred), gold)
    print(f"n={res['n']}  exact_match={res['exact_match']:.1%}  points={res['weighted_points_of_70']:.1f}/70")
    for f, v in res["accuracy"].items():
        print(f"  {f:16s} {v:.1%}")
    if a.show_errors:
        for m in res["mismatches"]:
            print(json.dumps(m, ensure_ascii=False))
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


def cmd_rescore(a) -> int:
    """Re-apply the deterministic policy to saved evidence (no LLM calls)."""
    from .core.callback import parse_call_time
    from .core.pipeline import read_jsonl
    from .core.rules import decide
    from .core.schemas import Evidence

    recs = {r.transcript_id: r for r in read_jsonl(Path(a.input).read_text(encoding="utf-8").splitlines())}
    n = 0
    with open(a.output, "w", encoding="utf-8") as f:
        for line in Path(a.evidence).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            rec = recs.get(row["transcript_id"])
            if rec is None or not row.get("evidence"):
                continue
            d = decide(rec.transcript_id, Evidence.model_validate(row["evidence"]),
                       parse_call_time(rec.call_started_at, rec.timezone), rec.transcript)
            f.write(d.prediction.model_dump_json() + "\n")
            n += 1
    print(f"rescored {n} records -> {a.output}", file=sys.stderr)
    return 0


def cmd_gender(a) -> int:
    from .core.gender import predict_gender
    from .core.pipeline import read_jsonl

    records = read_jsonl(Path(a.input).read_text(encoding="utf-8").splitlines())
    with open(a.output, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({"transcript_id": r.transcript_id, **predict_gender(r.transcript)},
                               ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} rows to {a.output}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="aster-vale")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="label transcripts: input JSONL -> prediction JSONL")
    r.add_argument("--input", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--provider", default="groq", choices=["groq", "cerebras", "nvidia"])
    r.add_argument("--model", default=None, help="override model id (default openai/gpt-oss-120b on Groq)")
    r.add_argument("--reasoning-effort", default="medium", choices=["low", "medium", "high"])
    r.add_argument("--concurrency", type=int, default=2)
    r.add_argument("--cache", default=".cache/llm_cache.sqlite")
    r.add_argument("--no-cache", action="store_true")
    r.add_argument("--evidence-out", default=None, help="optional JSONL with extracted evidence + rule trace")
    r.set_defaults(fn=cmd_run)

    e = sub.add_parser("evaluate", help="score predictions against gold labels")
    e.add_argument("--pred", required=True)
    e.add_argument("--gold", required=True)
    e.add_argument("--ids", default=None, help="optional file of transcript_ids to restrict scoring")
    e.add_argument("--show-errors", action="store_true")
    e.add_argument("--json-out", default=None)
    e.set_defaults(fn=cmd_evaluate)

    rs = sub.add_parser("rescore", help="re-apply policy rules to saved evidence JSONL (no LLM)")
    rs.add_argument("--evidence", required=True)
    rs.add_argument("--input", required=True)
    rs.add_argument("--output", required=True)
    rs.set_defaults(fn=cmd_rescore)

    g = sub.add_parser("gender", help="rule-based caller/patient gender (no transformers)")
    g.add_argument("--input", required=True)
    g.add_argument("--output", required=True)
    g.set_defaults(fn=cmd_gender)

    a = p.parse_args(argv)
    from dotenv import load_dotenv

    load_dotenv()  # picks up GROQ_API_KEY etc. from ./.env if present; real env vars take precedence
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s", stream=sys.stderr)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
