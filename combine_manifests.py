"""
Combine the two extraction scripts' output CSVs into one manifest, ready for
verify_audio_paths.py and build_finetune_dataset.py.

This step existed only as one-off inline code earlier in this project's
development and was never saved as a standalone script -- this fills that
gap so the documented pipeline can actually be followed start to finish.

Usage:
    python3 combine_manifests.py \
        --single-word finetune_review.csv \
        --sentence-level finetune_review_sentences.csv \
        --output finetune_manifest_combined.csv
"""

import argparse
import csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single-word", type=str, default="finetune_review.csv",
                     help="Output of extract_finetune_data.py")
    ap.add_argument("--sentence-level", type=str, default="finetune_review_sentences.csv",
                     help="Output of extract_finetune_data_sentences.py")
    ap.add_argument("--output", type=str, default="finetune_manifest_combined.csv")
    args = ap.parse_args()

    sources = [
        (args.single_word, "single-word"),
        (args.sentence_level, "sentence-level"),
    ]

    rows = []
    fieldnames = None

    for fname, label in sources:
        try:
            with open(fname, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                for row in reader:
                    row["source_corpus"] = label
                    rows.append(row)
        except FileNotFoundError:
            print(f"WARNING: {fname} not found, skipping ({label} corpus will be absent)")

    if not rows:
        print("No rows found in either input file -- nothing to combine.")
        return

    out_fields = fieldnames + ["source_corpus"]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Combined manifest: {len(rows)} total tokens -> {args.output}")
    for fname, label in sources:
        count = sum(1 for r in rows if r["source_corpus"] == label)
        print(f"  {label}: {count} tokens (from {fname})")


if __name__ == "__main__":
    main()
