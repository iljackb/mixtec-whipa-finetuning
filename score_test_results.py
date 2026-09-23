"""
Score a fine-tuned model's test predictions using WhIPA's OWN STIPA_METRICS
class (code/scripts/metrics.py) -- not a reimplementation, their exact PER/PFER
computation.

Fixes from the previous version:
  - No more hand-copying predictions out of test_whipa.py's terminal output
    into a hardcoded RESULTS list. This now reads directly from the CSV that
    test_whipa.py's --output-csv writes (wav_file, token_n, start, end, orth,
    predicted, gold_raw, gold_normalized).
  - Saves a results log (per-token PER/PFER + the run's mean) to a CSV via
    --output-log, so numbers can be pulled straight into a report/writeup
    instead of re-typing them from the terminal.
  - Rows where test_whipa.py recorded an inference error (predicted starts
    with "[inference error:") are skipped from scoring (they'd distort PER/
    PFER with a nonsense comparison), but are still counted and reported
    separately so failures aren't silently dropped from the record.

Run from inside whipa/code/ (needs panphon + the scripts/ package on path).

Usage:
    cd code
    python3 score_test_results.py \
        --input-csv ../test_results.csv \
        --output-log ../test_scores.csv \
        --model-name lowhipa-mixtec-v2
"""

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

from scripts.metrics import STIPA_METRICS

INPUT_FIELDNAMES = [
    "wav_file", "token_n", "start", "end", "orth",
    "predicted", "gold_raw", "gold_normalized",
]
LOG_FIELDNAMES = [
    "wav_file", "token_n", "orth", "predicted", "gold_normalized", "per", "pfer",
]


def load_results(input_csv: Path):
    rows = []
    with open(input_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", type=str, default="../test_results.csv",
                     help="CSV produced by test_whipa.py's --output-csv")
    ap.add_argument("--output-log", type=str, default="../test_scores.csv",
                     help="Where to save the per-token PER/PFER log plus a "
                          "trailing summary line, for pulling numbers into a "
                          "report. Set to '' to disable and only print.")
    ap.add_argument("--model-name", type=str, default="",
                     help="Optional label recorded in the printed/saved summary "
                          "(e.g. 'lowhipa-mixtec-v2'), so scores from different "
                          "training runs don't get mixed up later.")
    args = ap.parse_args()

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise SystemExit(
            f"Input CSV not found: {input_path}\n"
            f"Run test_whipa.py first with --output-csv to produce it."
        )

    rows = load_results(input_path)
    if not rows:
        raise SystemExit(f"No rows found in {input_path}")

    eval_metrics = STIPA_METRICS()

    scored_rows = []
    error_rows = []

    per_list = []
    pfer_list = []

    label = f" ({args.model_name})" if args.model_name else ""
    print(f"Scoring {len(rows)} token(s) from {input_path}{label}")
    print(f"{'wav_file':22} {'tok':4} {'PER%':>8} {'PFER%':>8}")

    for row in rows:
        predicted = row["predicted"]
        gold = row["gold_normalized"]

        if predicted.startswith("[inference error:"):
            error_rows.append(row)
            continue

        m = eval_metrics.compute_all(pred=predicted, gold=gold, char_based=False)
        per_list.append(m["per"])
        pfer_list.append(m["pfer"])

        print(f"{row['wav_file']:22} {row['token_n']:>4} {m['per']:8.1f} {m['pfer']:8.1f}")

        scored_rows.append({
            "wav_file": row["wav_file"],
            "token_n": row["token_n"],
            "orth": row["orth"],
            "predicted": predicted,
            "gold_normalized": gold,
            "per": round(m["per"], 2),
            "pfer": round(m["pfer"], 2),
        })

    if not per_list:
        raise SystemExit("No scoreable rows (all were inference errors) -- nothing to report.")

    mean_per = sum(per_list) / len(per_list)
    mean_pfer = sum(pfer_list) / len(pfer_list)

    print(f"\nScored {len(scored_rows)} token(s); {len(error_rows)} skipped (inference errors)")
    print(f"Mean PER:  {mean_per:.1f}%")
    print(f"Mean PFER: {mean_pfer:.1f}%")

    if args.output_log:
        out_path = Path(args.output_log)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_FIELDNAMES)
            writer.writeheader()
            writer.writerows(scored_rows)
            writer.writerow({})  # blank separator before the summary line
            writer.writerow({
                "wav_file": "MEAN",
                "token_n": f"n={len(scored_rows)} scored, {len(error_rows)} errors",
                "orth": args.model_name,
                "predicted": "",
                "gold_normalized": "",
                "per": round(mean_per, 2),
                "pfer": round(mean_pfer, 2),
            })
        print(f"\nSaved results log to {out_path}")
        print(f"(run at {datetime.now(timezone.utc).isoformat(timespec='seconds')})")


if __name__ == "__main__":
    main()
