"""
Extract training-ready rows from a whole-utterance-only TEI source (e.g.
MYUC-1042.xml, produced by myuc_to_tei.py), for combining into the project's
manifest via combine_manifests.py.

WHY THIS IS A SEPARATE SCRIPT, not an extension of the other two:
extract_finetune_data.py and extract_finetune_data_sentences.py both work by
finding and matching individual <w> elements with real per-word timing
(@synch). This source has no real per-word timing at all -- only the whole
<u>'s own start/end exists. Even if <w> elements are present here (for
corpus-structure consistency / future forced-alignment readiness), they do
NOT carry independent timing, so they must NOT be extracted as separate
training rows -- doing so would pair the identical whole-utterance audio
with multiple different, shorter, mutually-contradictory training targets.

This script therefore always extracts exactly ONE training example per <u>,
using the <u>'s own start/end. It is robust to BOTH of the following shapes
for the orth-ucsb/ipa <seg> elements:
  - plain text directly inside the <seg> (current myuc_to_tei.py output)
  - text subdivided into <w> child elements (a possible future revision) --
    in this case, the <w> elements' text is joined back together with
    spaces, reconstructing the exact same whole-utterance string either way.

Usage: 
    python3 extract_finetune_data_myuc.py "/Users/jackbowers/Archived - Box Sync/Language_Data/Mixtepec_Mixtec/misc-sources/Jerry_Guillem_Mixtec/bees/MYUC-1042.xml" --output "/Users/jackbowers/Archived - Box Sync/Language_Data/Mixtepec_Mixtec/misc-sources/Jerry_Guillem_Mixtec/bees/finetune_review_myuc.csv"

> outputs file to /bees folder in Mixtepec_Mixtec project dir not whipa (so that will be where i need to retrieve it)
"""

import argparse
import csv
from pathlib import Path
from lxml import etree

from normalize_ipa import normalize_for_training, strip_tones

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def get_seg_text(seg):
    """Return the seg's text content, whether it's plain text directly in
    the <seg>, or subdivided into <w> child elements (joined with spaces --
    reconstructs the identical string either way)."""
    words = seg.findall("tei:w", TEI_NS)
    if words:
        return " ".join("".join(w.itertext()).strip() for w in words)
    return "".join(seg.itertext()).strip()


def get_wav_ref(tree, xml_filename):
    media = tree.find(".//tei:media", TEI_NS)
    if media is not None:
        url = media.get("url", "")
        return url.split(":", 1)[1] if ":" in url else url
    return Path(xml_filename).stem + ".wav"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xml_file", type=str)
    ap.add_argument("--output", type=str, default="finetune_review_myuc.csv")
    args = ap.parse_args()

    xml_path = Path(args.xml_file)

    try:
        tree = etree.parse(str(xml_path))
    except Exception:
        parser = etree.XMLParser(recover=True)
        tree = etree.parse(str(xml_path), parser)

    wav_ref = get_wav_ref(tree, xml_path.name)

    rows = []
    skipped_no_orth_or_ipa = 0

    for u in tree.findall(".//tei:u", TEI_NS):
        start = u.get("start")
        end = u.get("end")

        orth_seg = None
        ipa_seg = None
        for seg in u.findall("tei:seg", TEI_NS):
            notation = seg.get("notation")
            if notation == "orth-ucsb":
                orth_seg = seg
            elif notation == "ipa":
                ipa_seg = seg

        if orth_seg is None or ipa_seg is None:
            skipped_no_orth_or_ipa += 1
            continue

        orth_text = get_seg_text(orth_seg)
        ipa_text = get_seg_text(ipa_seg)

        if not orth_text or not ipa_text:
            skipped_no_orth_or_ipa += 1
            continue

        ipa_notone = strip_tones(ipa_text)
        ipa_full_normalized = normalize_for_training(ipa_text)

        rows.append({
            "xml_file": xml_path.name,
            "wav_file": wav_ref,
            "token_n": "",  # not applicable -- whole-utterance row, no word-level token index
            "start": start,
            "end": end,
            "orth": orth_text,
            "ipa_gold": ipa_text,
            "ipa_notone": ipa_notone,
            "ipa_full_normalized": ipa_full_normalized,
            "changed": "yes" if ipa_text != ipa_notone else "no",
            "source_corpus": "myuc-utterance",
        })

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "xml_file", "wav_file", "token_n", "start", "end",
            "orth", "ipa_gold", "ipa_notone", "ipa_full_normalized", "changed",
            "source_corpus",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} whole-utterance rows to {args.output}")
    print(f"Skipped (missing orth-ucsb or ipa seg): {skipped_no_orth_or_ipa}")


if __name__ == "__main__":
    main()
