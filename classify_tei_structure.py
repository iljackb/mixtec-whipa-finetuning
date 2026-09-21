"""
Classify TEI/XML files by their ACTUAL <u>/<w>/@synch structure, rather than
by filename convention -- the naming-based FILES lists in
extract_finetune_data_sentences.py (and the ADJ_* assumption in
extract_finetune_data.py) can silently misroute files whose name doesn't
match their real structure (e.g. _Los_Sonidos_del_mixteco.xml is named like
a sentence file but is structurally single-word; be_at_home_PST_3s-inf_01_02_03_TS.xml
is structurally a sentence file but isn't named like one at all).

For every <u> in a file, this looks at the orthography <seg> (notation="orth"
or notation="orth-ucsb") inside it and counts:
  - how many <w> elements it has
  - whether those <w>s each carry their OWN, distinct @synch pair, or share
    one @synch / have none at all

A file is classified per-<u>, then given one overall label:
  - "single-word"      : every <u> has <=1 <w>, or all <w>s in a <u> share
                          identical/absent @synch (no independent timing)
  - "sentence"          : at least one <u> has >1 <w> with genuinely distinct
                          @synch pairs per word (real per-word timing)
  - "whole-utterance"   : <u> has start/end of its own but no <w> elements at
                          all inside the orth seg (MYUC-1042 style)
  - "mixed"             : file contains more than one of the above patterns
                          across its own <u>s -- needs a human look, this
                          script won't guess which extraction script applies

This is READ-ONLY -- it doesn't touch or move any files, it just reports.

Usage:
    python3 classify_tei_structure.py <folder> [--recursive] [--tsv OUTPUT_FILE] [--append]

    Example:
        # single corpus root, all subfolders nested under it:
        python3 classify_tei_structure.py /path/to/Mixtepec_Mixtec --recursive --tsv classification.tsv

        # a second, unrelated root -- append to the same TSV rather than
        # overwriting, so results accumulate across runs:
        python3 classify_tei_structure.py /path/to/mixtec-whipa-finetuning --recursive --tsv classification.tsv --append
"""

import sys
import glob
import os
import csv
from lxml import etree

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def classify_u(u_elem):
    """Return one of 'single-word', 'sentence', 'whole-utterance' for one <u>."""
    # The orthography <seg> isn't reliably identifiable by @notation alone --
    # Aprendamos/Leccion files carry NO notation attribute at all on their
    # orth seg (just xml:lang="mix" and @synch), while ADJ_*/be_at_home_*
    # style files use notation="orth" and MYUC-1042-style files use
    # notation="orth-ucsb". The one thing that's consistent across all of
    # them: the orth seg is never notation="ipa". So: take every direct
    # <seg> child of <u> that ISN'T notation="ipa", and use whichever one
    # actually contains <w> elements (there should be at most one candidate
    # in practice, but guard against multiple by picking the one with the
    # most <w> children).
    candidates = [
        seg for seg in u_elem.findall("tei:seg", TEI_NS)
        if seg.get("notation") != "ipa"
    ]

    orth_seg = None
    best_word_count = -1
    for seg in candidates:
        w_count = len(seg.findall("tei:w", TEI_NS))
        if w_count > best_word_count:
            best_word_count = w_count
            orth_seg = seg

    if orth_seg is None:
        # No non-IPA seg at all inside this <u> -- can't classify from
        # structure; treat as whole-utterance (nothing at word level to
        # extract anyway).
        return "whole-utterance"

    words = orth_seg.findall("tei:w", TEI_NS)

    if not words:
        return "whole-utterance"

    # Check for the presence of ANY per-word timing FIRST, before looking at
    # word count -- a source with categorically no word-level timing (like
    # MYUC-1042) can still have plenty of genuinely one-word utterances, and
    # those must still classify as whole-utterance, not single-word, since
    # "single-word" implies "extractable via per-word @synch", which doesn't
    # exist here regardless of how many <w>s there are.
    synchs = [w.get("synch") for w in words]
    if all(s is None for s in synchs):
        return "whole-utterance"

    if len(words) == 1:
        return "single-word"

    distinct_synchs = set(synchs)
    if len(distinct_synchs) == len(words) and None not in distinct_synchs:
        return "sentence"

    # Multiple <w>s but they share synch values or some are missing --
    # ambiguous; call it single-word-ish (no independently-timed extraction
    # is possible) but this is worth a human glance.
    return "single-word"


def classify_file(path):
    try:
        tree = etree.parse(path)
    except Exception as e:
        return None, f"PARSE ERROR: {e}"

    root = tree.getroot()
    us = root.findall(".//tei:u", TEI_NS)

    if not us:
        return None, "no <u> elements found"

    labels = {classify_u(u) for u in us}

    if len(labels) == 1:
        return labels.pop(), f"{len(us)} <u> elements, consistent structure"
    else:
        return "mixed", f"{len(us)} <u> elements, mixed structure: {sorted(labels)}"


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 classify_tei_structure.py <folder> [--recursive] [--tsv OUTPUT_FILE] [--append]")
        sys.exit(1)

    folder = sys.argv[1]
    args = sys.argv[2:]
    recursive = "--recursive" in args
    append = "--append" in args
    tsv_path = None
    if "--tsv" in args:
        tsv_idx = args.index("--tsv")
        if tsv_idx + 1 >= len(args):
            print("--tsv requires an output file path")
            sys.exit(1)
        tsv_path = args[tsv_idx + 1]

    pattern = "**/*.xml" if recursive else "*.xml"
    all_files = sorted(glob.glob(os.path.join(folder, pattern), recursive=recursive))

    # Skip metadata records -- these are per-file metadata sidecars, not
    # transcription content, and have no <u>/<w> structure to classify.
    files = [f for f in all_files if not os.path.basename(f).lower().endswith("metadata.xml")]
    skipped = len(all_files) - len(files)
    if skipped:
        print(f"Skipping {skipped} *metadata.xml file(s)")

    if not files:
        print(f"No .xml files found in {folder} (recursive={recursive})")
        return

    # Use absolute paths in every record -- when appending across multiple
    # unrelated corpus roots, a path relative to just one of them would be
    # ambiguous/misleading in the combined TSV.
    rows = []  # (abs_path, label, detail)
    by_label = {}
    for path in files:
        abs_path = os.path.abspath(path)
        label, detail = classify_file(path)
        by_label.setdefault(label, []).append((path, detail))
        rows.append((abs_path, label if label else "UNCLASSIFIABLE", detail))

    for label in ["single-word", "sentence", "whole-utterance", "mixed", None]:
        if label not in by_label:
            continue
        header = label if label else "UNCLASSIFIABLE"
        print(f"\n=== {header} ({len(by_label[label])} files) ===")
        for path, detail in by_label[label]:
            print(f"  {os.path.relpath(path, folder):60} {detail}")

    print(f"\nTotal files scanned: {len(files)} (root: {os.path.abspath(folder)})")
    print("\nSuggested script mapping:")
    print("  single-word     -> extract_finetune_data.py")
    print("  sentence        -> extract_finetune_data_sentences.py")
    print("  whole-utterance -> extract_finetune_data_myuc.py (or similar whole-utterance extractor)")
    print("  mixed           -> needs manual review; a single file mixing structures")
    print("                     may need per-<u> splitting rather than a single script")

    if tsv_path:
        file_exists = os.path.exists(tsv_path)
        mode = "a" if append and file_exists else "w"
        with open(tsv_path, mode, newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            if mode == "w":
                writer.writerow(["abs_path", "label", "detail"])
            writer.writerows(rows)
        verb = "Appended" if mode == "a" else "Wrote"
        print(f"\n{verb} {len(rows)} rows to {tsv_path}")


if __name__ == "__main__":
    main()
