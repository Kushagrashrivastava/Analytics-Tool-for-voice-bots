"""Convert a CSV export of the "50 Labeled Calls" tab into the JSONL files the runner and dashboard use.

Usage: python scripts/prepare_data.py path/to/50_Labeled_Calls.csv
Writes data/labeled_50.jsonl (inputs + answer key) and data/inputs_50.jsonl (input fields only).
The sheet's literal "null" becomes JSON null.
"""

import csv
import json
import sys
from pathlib import Path

INPUT_FIELDS = ("transcript_id", "call_started_at", "timezone", "transcript")

src = Path(sys.argv[1])
out = Path(__file__).resolve().parents[1] / "data"
out.mkdir(exist_ok=True)
rows = [{k: (None if v == "null" else v) for k, v in r.items()} for r in csv.DictReader(src.open(encoding="utf-8"))]
with open(out / "labeled_50.jsonl", "w", encoding="utf-8") as g, open(out / "inputs_50.jsonl", "w", encoding="utf-8") as i:
    for r in rows:
        g.write(json.dumps(r, ensure_ascii=False) + "\n")
        i.write(json.dumps({k: r[k] for k in INPUT_FIELDS}, ensure_ascii=False) + "\n")
print(f"wrote {len(rows)} records to {out}/labeled_50.jsonl and inputs_50.jsonl")
