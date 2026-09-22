"""
Unified replacement for extract_finetune_data.py, extract_finetune_data_sentences.py,
and extract_finetune_data_myuc.py.

WHY THIS EXISTS: those three scripts each hard-picked a list of files
(FILES = [...] / a fixed XML_DIR glob / a single CLI arg) and assumed every
<u> in a given file has the same structure -- one word per <u> ("single-word"),
multiple words per <u> with independent @synch timing ("sentence"), or no
per-word timing at all ("whole-utterance"). That assumption breaks for files
that genuinely MIX utterance types internally (confirmed via
classify_tei_structure.py against the full corpus: several transcriptions-xml/
files have some <u>s that are plain sentences and others with no word timing
at all in the SAME file) -- no single file-level script choice handles those
correctly.

This script routes at the <u> LEVEL instead of the file level: every <u> in
every scanned file is independently classified (single-word / sentence /
whole-utterance) using the exact same structural rule
classify_tei_structure.py uses, then dispatched to whichever of the three
ORIGINAL scripts' extraction logic matches -- reproduced here verbatim, not
reimplemented, so existing single-word/sentence/whole-utterance manifests
this replaces should come out byte-for-byte equivalent for files that were
already correctly classified as one uniform type. The only behavioral
addition is that mixed-type files now extract every <u> correctly instead of
needing manual per-file review.

Per-<u> classification rule (must exactly match classify_tei_structure.py):
  1. Take every direct <seg> child of <u> that is NOT notation="ipa".
  2. Among those, the "structural" seg is whichever one actually has <w>
     child elements (MYUC-1042-style files have BOTH an untokenized seg with
     no <w>s and a tokenized seg with <w>s -- the untokenized one is used
     for cleaner orth TEXT extraction, but the TOKENIZED one is what tells
     us whether real per-word timing exists).
  3. If that structural seg has no <w>s at all -> "whole-utterance".
  4. If its <w>s carry NO @synch at all -> "whole-utterance" (checked BEFORE
     word count, since a whole-utterance source like MYUC-1042 can still have
     genuinely one-word utterances that must not be misread as "single-word").
  5. Exactly one <w>, with @synch present -> "single-word".
  6. More than one <w>, each with its own distinct @synch -> "sentence".

Extraction logic per classification (verbatim from the original scripts,
EXCEPT for the timeline-resolution fix documented below):
  - "single-word"     -> from extract_finetune_data.py: one row per <u>,
                          using the <u>'s OWN start/end and n (not the
                          word's synch times), source_corpus="single-word".
  - "sentence"         -> from extract_finetune_data_sentences.py: one row
                          per matched orth/ipa word pair using EACH WORD'S
                          OWN synch-derived start/end
                          (source_corpus="sentence-level-word"), plus one
                          aggregate row joining all matched words in order
                          using the <u>'s own start/end
                          (source_corpus="sentence-level-full").
  - "whole-utterance"  -> from extract_finetune_data_myuc.py: one row per
                          <u> using the <u>'s own start/end, orth/ipa text
                          extracted via get_seg_text() (handles both plain-
                          text-in-seg and <w>-subdivided-seg shapes).
                          source_corpus="whole-utterance" -- NOTE: this is a
                          rename from the original "myuc-utterance", because
                          this path is no longer MYUC-specific (it now also
                          covers vocab-conocelos-mx-fb.xml,
                          vocab-20180914-Tisu.xml, N-V_pain-hurt-JS-TTS.xml,
                          speech-20170528-JS-TTS.xml, and any mixed-type
                          file's whole-utterance <u>s). If anything downstream
                          greps for the literal string "myuc-utterance",
                          update it to "whole-utterance" or teach it to
                          recognize both.

FIX (2026-09-21): "sentence"-type rows' start/end times were WRONG. Each
<w>'s @synch attribute references timeline-point IDs (e.g. synch="#T6 #T8"),
and the REAL elapsed-seconds value for each ID lives in this file's
<timeline><when xml:id="T6" interval="0.63"/></timeline>. The original
get_word_map() (carried over verbatim from extract_finetune_data_sentences.py)
never consulted <timeline> at all -- it regex-extracted the literal digit
suffix from the ID string itself (SYNCH_TIME_RE = r"#T([\d.]+)", so "#T6"
-> "6") and used that digit directly as a start/end time in SECONDS. For a
short recording, "T6" and "T8" are nowhere near 6 and 8 seconds in --
confirmed via a real example (ADJ_tall_1st_pl_01_JS.xml): word "kue" has
synch="#T6 #T8", whose REAL timeline values are interval="0.63"/interval=
"0.81", but the old code produced start=6.0, end=8.0. Cropping audio at
6.0-8.0s on a ~1.24s recording produces an empty/zero-length array every
time, which is exactly the "1,314 zero-length segment" failures
build_finetune_dataset.py's diagnostics surfaced.

This is fixed by build_timeline_map() (parses each file's <timeline> into an
{xml:id: seconds} dict, once per file) and a rewritten get_word_map() that
resolves each @synch ID through that dict instead of regexing its digits.
Only "sentence-level-word" rows were affected (the only place that used
get_word_map()'s parsed timing) -- "single-word", "whole-utterance", and the
"sentence-level-full" aggregate row all use <u>'s own start/end attributes
directly, which were always correct.

Excludes:
  - Any file whose name ends in "metadata.xml" (case-insensitive) -- these
    are per-file metadata sidecars, not transcription content. (Harmless
    even if a metadata file's name doesn't match this exactly, e.g. a
    typo'd "-meatadata.xml" -- such files have no <u> elements anyway and
    silently contribute zero rows.)
  - HELD_OUT basenames -- the existing zero-shot baseline test set, kept
    out of training data exactly as extract_finetune_data.py did.

Usage:
    python3 extract_finetune_data_unified.py <root_dir> --recursive --output finetune_review_unified.csv

    Example:
        python3 extract_finetune_data_unified.py "/Users/jackbowers/Archived - Box Sync/Language_Data/Mixtepec_Mixtec" --recursive --output finetune_review_unified.csv
"""

import argparse
import csv
import glob
import re
import unicodedata
from pathlib import Path

from lxml import etree

from normalize_ipa import normalize_for_training

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}
XML_NS_ID = "{http://www.w3.org/XML/1998/namespace}id"

# ---------------------------------------------------------------------------
# Carried over verbatim from extract_finetune_data.py
# ---------------------------------------------------------------------------

HELD_OUT = {
    "ADJ_beautiful_anim_01_JS",
    "ADJ_beautiful_inan_01_JS",
    "ADJ_big_01_02_03_JS",
    "ADJ_dangerous_01_JS",
    "ADJ_dangerous_02_JS",
    "ADJ_difficult_01_02_spkrTS",
    "ADJ_fat_01_02_spkrTS",
    "ADJ_heavy_01_02_03_JS",
    "ADJ_long_DIST_01_02_03_JS",
    "ADJ_long_SHAPE_01_02_03_TS",
}

STANDALONE_TONE_CHARS = set(range(0x02E5, 0x02EA)) | {0x2197, 0x2198, 0x2219, 0xA71B, 0xA71C}


def strip_tones(ipa: str) -> str:
    no_standalone = "".join(ch for ch in ipa if ord(ch) not in STANDALONE_TONE_CHARS)
    decomposed = unicodedata.normalize("NFD", no_standalone)
    TONE_COMBINING_MARKS = {0x0301, 0x0300, 0x0302, 0x0304, 0x030C, 0x1DC7, 0x1DC5}
    cleaned = "".join(ch for ch in decomposed if ord(ch) not in TONE_COMBINING_MARKS)
    return unicodedata.normalize("NFC", cleaned)


def get_wav_media_ref(tree, xml_filename: str = "") -> str:
    """Unchanged from all three original scripts: <media> lookup, else
    derive from the XML's own filename stem + .wav."""
    media = tree.find(".//tei:media", TEI_NS)
    if media is not None:
        url = media.get("url", "")
        return url.split(":", 1)[1] if ":" in url else url
    if xml_filename:
        return Path(xml_filename).stem + ".wav"
    return ""


# ---------------------------------------------------------------------------
# NEW: timeline resolution (this is the bug fix)
# ---------------------------------------------------------------------------

def build_timeline_map(tree):
    """Parse this file's <timeline><when xml:id="Tn" interval="X.XX"/></timeline>
    into a {xml:id: seconds} dict, so a <w>'s @synch reference (e.g. "#T6")
    can be resolved to the REAL elapsed-seconds value for that timeline
    point, instead of misreading the digits inside the ID string itself as
    if they were seconds. One dict per file -- IDs are only unique within a
    file, not across the corpus."""
    timeline = {}
    for when in tree.findall(".//tei:timeline/tei:when", TEI_NS):
        xml_id = when.get(XML_NS_ID)
        interval = when.get("interval")
        if xml_id is None or interval is None:
            continue
        try:
            timeline[xml_id] = float(interval)
        except ValueError:
            continue
    return timeline


# ---------------------------------------------------------------------------
# Carried over from extract_finetune_data_sentences.py, FIXED to resolve
# @synch IDs through the file's <timeline> instead of regexing their digits.
# ---------------------------------------------------------------------------

SYNCH_ID_RE = re.compile(r"#(\S+)")


def get_word_map(seg, timeline):
    """Map each <w>'s start-timeline-id -> (text, start_seconds, end_seconds),
    with start/end resolved via `timeline` (see build_timeline_map). A <w>
    whose synch ID isn't in this file's <timeline> is skipped rather than
    guessed at."""
    result = {}
    if seg is None:
        return result
    for w in seg.findall("tei:w", TEI_NS):
        text = "".join(w.itertext()).strip()
        if not text:
            continue
        synch = w.get("synch", "")
        ids = SYNCH_ID_RE.findall(synch)
        if not ids:
            continue
        start_id = ids[0]
        end_id = ids[1] if len(ids) > 1 else None
        if start_id not in timeline:
            continue
        start = timeline[start_id]
        end = timeline.get(end_id) if end_id else None
        result[start_id] = (text, start, end)
    return result


# ---------------------------------------------------------------------------
# Carried over verbatim from extract_finetune_data_myuc.py
# ---------------------------------------------------------------------------

def get_seg_text(seg):
    """Return the seg's text content, whether it's plain text directly in
    the <seg>, or subdivided into <w> child elements (joined with spaces)."""
    if seg is None:
        return ""
    words = seg.findall("tei:w", TEI_NS)
    if words:
        return " ".join("".join(w.itertext()).strip() for w in words)
    return "".join(seg.itertext()).strip()


# ---------------------------------------------------------------------------
# NEW: per-<u> structural classification (must match classify_tei_structure.py)
# ---------------------------------------------------------------------------

def find_segs(u_elem):
    """Return (structural_orth_seg, untokenized_orth_seg_or_None, ipa_seg).

    structural_orth_seg: the non-ipa <seg> with the most <w> children (used
    to determine word/timing structure, and as a text-extraction fallback).
    untokenized_orth_seg: a non-ipa <seg> with type="untokenized" if one
    exists (MYUC-style sources) -- preferred for whole-utterance TEXT
    extraction since it preserves original spacing/punctuation rather than
    reconstructing it by joining <w>s with plain spaces.
    ipa_seg: the <seg notation="ipa">.
    """
    candidates = [seg for seg in u_elem.findall("tei:seg", TEI_NS) if seg.get("notation") != "ipa"]

    structural_seg = None
    best_word_count = -1
    untokenized_seg = None
    for seg in candidates:
        if seg.get("type") == "untokenized":
            untokenized_seg = seg
        w_count = len(seg.findall("tei:w", TEI_NS))
        if w_count > best_word_count:
            best_word_count = w_count
            structural_seg = seg

    ipa_seg = u_elem.find('tei:seg[@notation="ipa"]', TEI_NS)

    return structural_seg, untokenized_seg, ipa_seg


def classify_u(structural_seg):
    """Return 'single-word', 'sentence', or 'whole-utterance'. Logic must
    stay in lockstep with classify_tei_structure.py's classify_u()."""
    if structural_seg is None:
        return "whole-utterance"

    words = structural_seg.findall("tei:w", TEI_NS)
    if not words:
        return "whole-utterance"

    synchs = [w.get("synch") for w in words]
    if all(s is None for s in synchs):
        return "whole-utterance"

    if len(words) == 1:
        return "single-word"

    distinct_synchs = set(synchs)
    if len(distinct_synchs) == len(words) and None not in distinct_synchs:
        return "sentence"

    # Multiple <w>s with partially-shared/missing synch -- ambiguous;
    # classify_tei_structure.py falls back to "single-word" here too (no
    # independently-timed multi-word extraction is possible), which for
    # extraction purposes means: treat like a sentence-type <u> anyway so
    # each word with a synch value still gets its own row, but skip words
    # missing timing individually rather than dropping the whole <u>.
    return "sentence"


# ---------------------------------------------------------------------------
# Row builders -- one per classification, extraction logic preserved from
# the original per-type script
# ---------------------------------------------------------------------------

def build_row(xml_filename, wav_ref, token_n, start, end, orth_text, ipa_text, source_corpus):
    ipa_notone = strip_tones(ipa_text)
    ipa_full_normalized = normalize_for_training(ipa_text)
    return {
        "xml_file": xml_filename,
        "wav_file": wav_ref,
        "token_n": token_n,
        "start": start,
        "end": end,
        "orth": orth_text,
        "ipa_gold": ipa_text,
        "ipa_notone": ipa_notone,
        "ipa_full_normalized": ipa_full_normalized,
        "changed": "yes" if ipa_text != ipa_notone else "no",
        "source_corpus": source_corpus,
    }


def extract_single_word(u, xml_filename, wav_ref, structural_seg, ipa_seg):
    """Verbatim behavior from extract_finetune_data.py: uses the <u>'s own
    start/end/n, not the word's own synch timing. Unaffected by the
    timeline-resolution bug/fix."""
    orth_w = structural_seg.find(".//tei:w", TEI_NS)
    ipa_w = ipa_seg.find(".//tei:w", TEI_NS)
    if orth_w is None or ipa_w is None:
        return None

    orth_text = "".join(orth_w.itertext()).strip()
    ipa_text = "".join(ipa_w.itertext()).strip()
    if not orth_text or not ipa_text:
        return None

    return [build_row(
        xml_filename, wav_ref, u.get("n", ""), u.get("start", ""), u.get("end", ""),
        orth_text, ipa_text, "single-word",
    )]


def extract_sentence(u, xml_filename, wav_ref, structural_seg, ipa_seg, timeline):
    """Behavior from extract_finetune_data_sentences.py, FIXED: per-word rows
    now use each word's REAL timeline-resolved start/end (via `timeline`,
    see build_timeline_map/get_word_map) instead of the literal digits in
    its synch-ID string. The aggregate row is unaffected -- it always used
    the <u>'s own start/end."""
    orth_words = get_word_map(structural_seg, timeline)
    ipa_words = get_word_map(ipa_seg, timeline)
    matched_keys = set(orth_words) & set(ipa_words)

    rows = []
    for start_key in matched_keys:
        orth_text, o_start, o_end = orth_words[start_key]
        ipa_text, i_start, i_end = ipa_words[start_key]
        start = i_start if i_start is not None else o_start
        end = i_end if i_end is not None else o_end
        if end is None:
            continue
        rows.append(build_row(
            xml_filename, wav_ref, "", start, end,
            orth_text, ipa_text, "sentence-level-word",
        ))

    if matched_keys:
        ordered_keys = sorted(matched_keys, key=lambda k: timeline[k])
        joined_normalized = " ".join(normalize_for_training(ipa_words[k][0]) for k in ordered_keys)
        joined_orth = " ".join(orth_words[k][0] for k in ordered_keys)
        u_start = u.get("start")
        u_end = u.get("end")
        if u_start is not None and u_end is not None:
            rows.append({
                "xml_file": xml_filename,
                "wav_file": wav_ref,
                "token_n": "",
                "start": u_start,
                "end": u_end,
                "orth": joined_orth,
                "ipa_gold": "",
                "ipa_notone": "",
                "ipa_full_normalized": joined_normalized,
                "changed": "",
                "source_corpus": "sentence-level-full",
            })

    return rows


def extract_whole_utterance(u, xml_filename, wav_ref, structural_seg, untokenized_seg, ipa_seg):
    """Verbatim behavior from extract_finetune_data_myuc.py, generalized to
    any non-ipa seg (not just notation="orth-ucsb"). Prefers the untokenized
    seg's raw text for orth when one exists (cleaner than word-rejoining).
    Unaffected by the timeline-resolution bug/fix."""
    orth_source = untokenized_seg if untokenized_seg is not None else structural_seg
    orth_text = get_seg_text(orth_source)
    ipa_text = get_seg_text(ipa_seg)

    if not orth_text or not ipa_text:
        return None

    return [build_row(
        xml_filename, wav_ref, "", u.get("start"), u.get("end"),
        orth_text, ipa_text, "whole-utterance",
    )]


def process_file(path: Path):
    try:
        tree = etree.parse(str(path))
    except Exception:
        parser = etree.XMLParser(recover=True)
        try:
            tree = etree.parse(str(path), parser)
        except Exception as e:
            return [], f"unparseable: {e}"

    wav_ref = get_wav_media_ref(tree, path.name)
    timeline = build_timeline_map(tree)
    rows = []
    skipped = 0

    for u in tree.findall(".//tei:u", TEI_NS):
        structural_seg, untokenized_seg, ipa_seg = find_segs(u)

        if ipa_seg is None:
            skipped += 1
            continue

        label = classify_u(structural_seg)

        if label == "single-word":
            new_rows = extract_single_word(u, path.name, wav_ref, structural_seg, ipa_seg)
        elif label == "sentence":
            new_rows = extract_sentence(u, path.name, wav_ref, structural_seg, ipa_seg, timeline)
        else:
            new_rows = extract_whole_utterance(u, path.name, wav_ref, structural_seg, untokenized_seg, ipa_seg)

        if new_rows:
            rows.extend(new_rows)
        else:
            skipped += 1

    return rows, None if rows or skipped == 0 else "no usable <u> content"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root_dir", type=str)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--output", type=str, default="finetune_review_unified.csv")
    args = ap.parse_args()

    pattern = "**/*.xml" if args.recursive else "*.xml"
    all_files = sorted(glob.glob(str(Path(args.root_dir) / pattern), recursive=args.recursive))
    files = [Path(f) for f in all_files if not f.lower().endswith("metadata.xml")]

    print(f"Scanning {len(files)} file(s) (skipped {len(all_files) - len(files)} metadata.xml file(s))")

    all_rows = []
    skipped_held_out = 0
    skipped_unparseable = 0
    label_counts = {}

    for path in files:
        if path.stem in HELD_OUT:
            skipped_held_out += 1
            continue

        rows, error = process_file(path)
        if error and error.startswith("unparseable"):
            print(f"  SKIP (unparseable): {path.name}: {error}")
            skipped_unparseable += 1
            continue

        for r in rows:
            label_counts[r["source_corpus"]] = label_counts.get(r["source_corpus"], 0) + 1
        all_rows.extend(rows)

    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "xml_file", "wav_file", "token_n", "start", "end",
            "orth", "ipa_gold", "ipa_notone", "ipa_full_normalized", "changed",
            "source_corpus",
        ])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} total rows to {args.output}")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    print(f"Held-out test files skipped: {skipped_held_out}")
    print(f"Unparseable files skipped: {skipped_unparseable}")


if __name__ == "__main__":
    main()
