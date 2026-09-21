"""
Combine any number of extraction-script output CSVs into one manifest, ready
for verify_audio_paths.py and build_finetune_dataset.py.

Accepts a flexible, arbitrary-length list of input files (from any folder --
paths are just passed as given, no assumption they share a directory), rather
than fixed named slots per source type. This is meant to scale as more
sources are added over time (e.g. additional AILLA recordings beyond
MYUC-1042), without needing another code change each time.

Each input file must already carry its own "source_corpus" column (every
extraction script in this pipeline -- extract_finetune_data.py,
extract_finetune_data_sentences.py, extract_finetune_data_myuc.py -- sets
this itself), so this script doesn't need to be told what to call each
source; it just reads and preserves whatever's already there.

Usage:
    python3 combine_manifests.py \
        "/path/to/SIL_docs/Aprendamos-2018/speech_transcriptions/finetune_review.csv" \
        "/path/to/SIL_docs/Aprendamos-2018/speech_transcriptions/finetune_review_sentences.csv" \
        "/path/to/misc_sources/Jerry_Guillem_Mixtec/bees/finetune_review_myuc.csv" \
        --output finetune_manifest_combined.csv
"""

import argparse
import csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_files", nargs="+",
                     help="Any number of extraction-script output CSVs, from any folder")
    ap.add_argument("--output", type=str, default="finetune_manifest_combined.csv")
    args = ap.parse_args()

    rows = []
    fieldnames = None

    for fname in args.input_files:
        try:
            with open(fname, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    print(f"WARNING: {fname} appears empty, skipping")
                    continue
                if fieldnames is None:
                    fieldnames = reader.fieldnames
                for row in reader:
                    if "source_corpus" not in row or not row["source_corpus"]:
                        print(f"WARNING: {fname} has a row with no source_corpus label -- "
                              f"check that this file was produced by an up-to-date extraction script")
                        row["source_corpus"] = row.get("source_corpus") or "unlabeled"
                    rows.append(row)
        except FileNotFoundError:
            print(f"WARNING: {fname} not found, skipping")

    if not rows:
        print("No rows found in any input file -- nothing to combine.")
        return

    # Fieldnames might differ slightly across sources (e.g. myuc rows don't
    # have every column the others do) -- union them all, preserving first-
    # seen order, so no column silently gets dropped.
    all_fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                all_fields.append(key)
                seen.add(key)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Combined manifest: {len(rows)} total tokens -> {args.output}")
    label_counts = {}
    for row in rows:
        label = row["source_corpus"]
        label_counts[label] = label_counts.get(label, 0) + 1
    for label, count in label_counts.items():
        print(f"  {label}: {count} tokens")


if __name__ == "__main__":
    main()
