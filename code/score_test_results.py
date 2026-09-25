"""
Score a fine-tuned model's test predictions using WhIPA's OWN STIPA_METRICS
class (code/scripts/metrics.py) -- not a reimplementation, their exact PER/PFER
computation.

Fixes from the previous version:
  - Splits results into two tracks -- single-word tokens vs. multi-word
    (phrase-level) tokens -- based on the "n_words" column test_whipa.py now
    records, and reports/saves a separate mean PER/PFER for each instead of
    one mixed mean. A phrase's PER/PFER is computed over the whole cropped
    <u> span exactly as before (test_whipa.py's extraction/transcription is
    unchanged) -- this only changes how the RESULTS get grouped for
    reporting. CSVs from before this change have no "n_words" column; those
    rows are treated as single-word (accurate for the original test set,
    which was single-word-only).

Fixes from the version before that:
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

LOG_FIELDNAMES = [
    "wav_file", "token_n", "orth", "n_words", "track",
    "predicted", "gold_normalized", "per", "pfer",
]


def load_results(input_csv: Path):
    rows = []
    with open(input_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def track_for(row) -> str:
    """"single" if the token is one word, "phrase" if multi-word. Rows from
    before the n_words column existed default to "single" -- accurate for
    the original test set, which was single-word-only."""
    n_words = int(row.get("n_words") or 1)
    return "single" if n_words == 1 else "phrase"


def summarize(label: str, rows: list) -> dict | None:
    per_list = [r["per"] for r in rows]
    pfer_list = [r["pfer"] for r in rows]
    if not per_list:
        return None
    mean_per = sum(per_list) / len(per_list)
    mean_pfer = sum(pfer_list) / len(pfer_list)
    print(f"\n{label}: n={len(rows)}")
    print(f"  Mean PER:  {mean_per:.1f}%")
    print(f"  Mean PFER: {mean_pfer:.1f}%")
    return {"n": len(rows), "mean_per": mean_per, "mean_pfer": mean_pfer}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", type=str, default="../test_results.csv",
                     help="CSV produced by test_whipa.py's --output-csv")
    ap.add_argument("--output-log", type=str, default="../test_scores.csv",
                     help="Where to save the per-token PER/PFER log plus "
                          "trailing summary lines (one per track), for "
                          "pulling numbers into a report. Set to '' to "
                          "disable and only print.")
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

    label = f" ({args.model_name})" if args.model_name else ""
    print(f"Scoring {len(rows)} token(s) from {input_path}{label}")
    print(f"{'wav_file':22} {'tok':4} {'track':7} {'PER%':>8} {'PFER%':>8}")

    for row in rows:
        predicted = row["predicted"]
        gold = row["gold_normalized"]
        track = track_for(row)
        n_words = int(row.get("n_words") or 1)

        if predicted.startswith("[inference error:"):
            error_rows.append(row)
            continue

        m = eval_metrics.compute_all(pred=predicted, gold=gold, char_based=False)

        print(f"{row['wav_file']:22} {row['token_n']:>4} {track:7} {m['per']:8.1f} {m['pfer']:8.1f}")

        scored_rows.append({
            "wav_file": row["wav_file"],
            "token_n": row["token_n"],
            "orth": row["orth"],
            "n_words": n_words,
            "track": track,
            "predicted": predicted,
            "gold_normalized": gold,
            "per": round(m["per"], 2),
            "pfer": round(m["pfer"], 2),
        })

    if not scored_rows:
        raise SystemExit("No scoreable rows (all were inference errors) -- nothing to report.")

    single_rows = [r for r in scored_rows if r["track"] == "single"]
    phrase_rows = [r for r in scored_rows if r["track"] == "phrase"]

    print(f"\nScored {len(scored_rows)} token(s) total; {len(error_rows)} skipped (inference errors)")
    single_summary = summarize("Single-word track", single_rows)
    phrase_summary = summarize("Phrase-level track", phrase_rows)

    if args.output_log:
        out_path = Path(args.output_log)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_FIELDNAMES)
            writer.writeheader()
            writer.writerows(scored_rows)
            writer.writerow({})  # blank separator before the summary lines

            if single_summary:
                writer.writerow({
                    "wav_file": "MEAN_SINGLE_WORD",
                    "token_n": f"n={single_summary['n']} scored, {len(error_rows)} errors",
                    "orth": args.model_name,
                    "n_words": "1",
                    "track": "single",
                    "predicted": "", "gold_normalized": "",
                    "per": round(single_summary["mean_per"], 2),
                    "pfer": round(single_summary["mean_pfer"], 2),
                })
            if phrase_summary:
                writer.writerow({
                    "wav_file": "MEAN_PHRASE_LEVEL",
                    "token_n": f"n={phrase_summary['n']} scored",
                    "orth": args.model_name,
                    "n_words": ">1",
                    "track": "phrase",
                    "predicted": "", "gold_normalized": "",
                    "per": round(phrase_summary["mean_per"], 2),
                    "pfer": round(phrase_summary["mean_pfer"], 2),
                })
        print(f"\nSaved results log to {out_path}")
        print(f"(run at {datetime.now(timezone.utc).isoformat(timespec='seconds')})")


if __name__ == "__main__":
    main()
