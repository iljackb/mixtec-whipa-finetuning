"""
Parse the space-aligned transcription text format used in this researcher's
archive (e.g. MYUC-1042.txt) into a clean, structured intermediate form,
ready for conversion into this project's TEI/XML conventions.

FORMAT OBSERVED:
  - First 2 lines: source file path, export date/time (metadata, not data)
  - Blank line
  - Repeated blocks, each a set of lines in the form:
        TierType@SpeakerCode<spaces>Content
    Recognized TierTypes: Transcription, Translation-Eng, Translation-Spn,
    Notes (optional, not always present), TC (timecode range, no @speaker
    suffix -- applies to the whole block regardless of speaker)
  - Blocks separated by one or more blank lines

FILTERING:
  Only blocks belonging to one of the given speaker code(s) (default: just
  "GML") are kept; --speaker accepts a comma-separated list. All other
  speakers' blocks are discarded entirely.

TIMECODE FORMAT: "HH:MM:SS.mmm - HH:MM:SS.mmm" -- converted to seconds
(float) for both start and end, matching how this project's other TEI files
represent timing.

Usage:
    python3 parse_researcher_archive.py MYUC-1042.txt --speaker GML
    python3 parse_researcher_archive.py MYUC-1038.txt --speaker Interviewer,Sp2,HVL

NOTE (2026-09-28): --speaker now accepts a comma-separated list of codes to
keep (originally single-speaker only, extended for MYUC-1038 which has
Mixtec content split across multiple speaker/interviewer tiers). Any block
whose speaker code is not in the list is still discarded, same as before.
This does NOT filter by language -- a kept speaker's block is kept whether
it is Mixtec or Spanish (e.g. consent-section code-switching); that is left
for manual review in the output CSV, per project convention of flagging/
leaving decisions for review rather than silently guessing.
"""

import argparse
import re
import json
from pathlib import Path

TIME_RE = re.compile(r"(\d+):(\d+):(\d+)\.(\d+)")
LINE_RE = re.compile(r"^(\S+)\s+(.*)$")


def timecode_to_seconds(tc: str) -> float:
    m = TIME_RE.match(tc.strip())
    if not m:
        raise ValueError(f"Unrecognized timecode format: {tc!r}")
    h, m_, s, ms = m.groups()
    return int(h) * 3600 + int(m_) * 60 + int(s) + int(ms) / (10 ** len(ms))


def parse_file(path: Path, speakers: set):
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    # Skip the first metadata lines (file path, date) up to the first blank line
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip() == "":
            start_idx = i + 1
            break

    # Split remaining lines into blocks separated by blank lines
    blocks = []
    current_block = []
    for line in lines[start_idx:]:
        if line.strip() == "":
            if current_block:
                blocks.append(current_block)
                current_block = []
        else:
            current_block.append(line.rstrip("\n"))
    if current_block:
        blocks.append(current_block)

    entries = []
    skipped_other_speaker = 0

    for block in blocks:
        parsed = {}
        block_speaker = None
        tc = None

        for line in block:
            m = LINE_RE.match(line)
            if not m:
                continue
            tier_full, content = m.groups()
            content = content.strip()

            if tier_full == "TC":
                tc = content
                continue

            if "@" not in tier_full:
                continue  # unrecognized line, skip

            tier_type, tier_speaker = tier_full.split("@", 1)
            block_speaker = tier_speaker  # all non-TC lines in a block share the same speaker

            if tier_type == "Transcription":
                parsed["orth"] = content
            elif tier_type == "Translation-Eng":
                parsed["eng"] = content
            elif tier_type == "Translation-Spn":
                parsed["spn"] = content
            elif tier_type == "Notes":
                parsed["notes"] = content

        if block_speaker not in speakers:
            skipped_other_speaker += 1
            continue

        if not parsed.get("orth") or not tc:
            continue  # incomplete block, skip

        start_str, end_str = [s.strip() for s in tc.split("-")]
        parsed["start"] = timecode_to_seconds(start_str)
        parsed["end"] = timecode_to_seconds(end_str)
        entries.append(parsed)

    return entries, skipped_other_speaker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_file", type=str)
    ap.add_argument("--speaker", type=str, default="GML",
                     help="Comma-separated speaker code(s) to keep (default: GML). "
                          "All other speakers' blocks are discarded. Does not filter "
                          "by language -- a kept speaker's Spanish blocks (e.g. "
                          "consent-section code-switching) are kept too; review/remove "
                          "those manually in the output CSV.")
    ap.add_argument("--output", type=str, default=None,
                     help="Optional: write parsed result to this JSON file for inspection")
    args = ap.parse_args()

    speakers = {s.strip() for s in args.speaker.split(",") if s.strip()}
    entries, skipped = parse_file(Path(args.input_file), speakers)

    print(f"Parsed {len(entries)} entries for speaker(s) {sorted(speakers)}")
    print(f"Skipped {skipped} entries from other speakers (interviewer turns, etc.)")
    print(f"\nFirst 3 entries:")
    for e in entries[:3]:
        print(f"  [{e['start']:.3f}-{e['end']:.3f}] {e}")

    has_notes = sum(1 for e in entries if "notes" in e)
    print(f"\nEntries with notes: {has_notes} / {len(entries)}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        print(f"\nWrote full parsed output to {args.output}")


if __name__ == "__main__":
    main()
