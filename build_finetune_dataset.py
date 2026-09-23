"""
Build a fine-tuning dataset for WhIPA/LoWhIPA from finetune_manifest_combined.csv.

DESIGN, revised after inspecting your actual code/scripts/whipa_utils.py directly:

  Rather than reimplementing WhIPA's own audio-feature-extraction and label-
  tokenization logic by hand (risking a subtle mismatch with their actual
  behavior -- e.g. their prepare_dataset_ipa() calls
  tokenizer.encode_plus(batch["ipa"], add_special_tokens=True), and we can't
  be fully sure what special-token handling that applies without risk of
  guessing wrong), this script does the MINIMUM work itself:

    1. Reads the manifest (one row per token).
    2. Finds each row's .wav file (recursive search across directories you give it).
    3. Loads + resamples to 16kHz (Whisper's hard requirement) + crops to the
       token's start/end time.
    4. Packages this into a HuggingFace datasets.Dataset with EXACTLY the raw
       columns their own code expects as INPUT to prepare_dataset_ipa():
         - "audio": {"array": ..., "sampling_rate": 16000}
         - "ipa":   the normalized IPA target string (confirmed via direct
                    inspection of scripts/whipa_utils.py -- prepare_dataset_ipa
                    reads batch["ipa"] literally, not "text" or "ipa_target")
    5. Saves this to disk.

  It does NOT precompute input_features or labels. That conversion should be
  done by importing and calling THEIR OWN prep_dataset()/prepare_dataset_ipa()
  functions directly (see run_prep_dataset.py, generated alongside this
  script), so the actual feature-extraction and tokenization is guaranteed
  identical to what fine_tune.py itself would produce -- not a reimplementation.

  Rows whose training target (the --ipa-column, default "ipa_full_normalized")
  is empty/blank after stripping whitespace are skipped -- an empty string as
  a training label doesn't just mean "no data", it actively trains the model
  to expect silence/no-output for a real audio segment, which is worse than
  omitting the row entirely. This is a generic guard, not specific to any one
  file -- it will also catch any future row where normalization happens to
  strip everything from the gold transcription (e.g. a token whose only
  annotated content was a tone mark).

Usage:
    python3 build_finetune_dataset.py finetune_review_unified.csv \
        --search-dir "/path/to/media/speech-mix" \
        --search-dir "/path/to/SIL_docs/Aprendamos-2018" \
        --output-dir ./whipa_raw_dataset

Requires: datasets, soundfile, scipy, numpy (already installed per earlier
whipa setup steps in this project). Does NOT require transformers/torch for
this step -- those are only needed for the subsequent prep_dataset() call.
"""

import argparse
import csv
from collections import Counter
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from datasets import Dataset

TARGET_SR = 16000


def build_wav_index(search_dirs):
    """Same approach as verify_audio_paths.py: map filename (lowercase) -> full
    path, searched recursively, so we don't need to know the exact subfolder
    each audio file lives in."""
    index = {}
    for d in search_dirs:
        d = Path(d)
        if not d.exists():
            print(f"WARNING: search dir not found, skipping: {d}")
            continue
        for wav_path in d.rglob("*.wav"):
            index[wav_path.name.lower()] = wav_path
    return index


def name_variants(name: str):
    """Same fix as verify_audio_paths.py -- some recordings use a space where
    others use an underscore (e.g. 'Leccion 01.wav' vs 'Leccion_01.wav')."""
    return {name, name.replace("_", " "), name.replace(" ", "_")}


def find_wav_path(wav_file: str, wav_index: dict):
    wav_name = Path(wav_file).name.lower()
    for variant in name_variants(wav_name):
        if variant in wav_index:
            return wav_index[variant]
    return None


def resample_to_16k(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    """Whisper's feature extractor requires exactly 16kHz -- this is the same
    resampling approach used in test_whipa.py, needed because your recordings
    come in at various native rates (44100, 96000, etc.)."""
    if orig_sr == TARGET_SR:
        return audio
    g = gcd(orig_sr, TARGET_SR)
    up, down = TARGET_SR // g, orig_sr // g
    return resample_poly(audio, up, down)


def load_and_crop(wav_path: Path, start: float, end: float):
    """Returns (segment, diagnostics). diagnostics is a dict with the values
    used to compute the crop, so a zero-length result can be explained rather
    than just silently counted."""
    audio, sr = sf.read(str(wav_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # downmix stereo to mono
    audio = resample_to_16k(audio, sr)
    sr = TARGET_SR

    start_f = float(start)
    end_f = float(end)
    start_sample = int(start_f * sr)
    end_sample = int(end_f * sr)
    segment = audio[start_sample:end_sample]

    diagnostics = {
        "audio_total_samples": len(audio),
        "start_f": start_f,
        "end_f": end_f,
        "start_sample": start_sample,
        "end_sample": end_sample,
    }
    return np.ascontiguousarray(segment, dtype=np.float32), diagnostics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest_csv", type=str)
    ap.add_argument("--search-dir", action="append", required=True, dest="search_dirs",
                     help="Directory to search recursively for .wav files (repeatable)")
    ap.add_argument("--output-dir", type=str, default="./whipa_raw_dataset")
    ap.add_argument("--ipa-column", type=str, default="ipa_full_normalized",
                     help="Which manifest column to use as the training target "
                          "(default: the fully-normalized column from normalize_ipa.py). "
                          "Gets saved under the literal column name 'ipa', matching "
                          "what prepare_dataset_ipa() in whipa_utils.py expects.")
    ap.add_argument("--skip-report", type=str, default="skipped_rows_report.csv",
                     help="Where to write a CSV of every skipped row (reason, xml_file, "
                          "source_corpus, start/end, and computed sample indices) for "
                          "diagnosing why rows were dropped. Set to '' to disable.")
    args = ap.parse_args()

    print("Indexing .wav files in search directories...")
    wav_index = build_wav_index(args.search_dirs)
    print(f"Found {len(wav_index)} unique .wav filenames.\n")

    dataset_rows = []
    skipped_no_audio = 0
    skipped_zero_length = 0
    skipped_exception = 0
    skipped_empty_target = 0
    skip_report_rows = []
    zero_length_by_corpus = Counter()

    with open(args.manifest_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            xml_file = row.get("xml_file", "")
            source_corpus = row.get("source_corpus", "")

            ipa_target = row.get(args.ipa_column, "").strip()
            if not ipa_target:
                skipped_empty_target += 1
                skip_report_rows.append({
                    "reason": "empty_target", "xml_file": xml_file,
                    "source_corpus": source_corpus, "wav_file": row.get("wav_file", ""),
                    "start": row.get("start", ""), "end": row.get("end", ""),
                })
                continue

            wav_file = row.get("wav_file", "").strip()
            wav_path = find_wav_path(wav_file, wav_index)

            if wav_path is None:
                skipped_no_audio += 1
                skip_report_rows.append({
                    "reason": "no_audio", "xml_file": xml_file,
                    "source_corpus": source_corpus, "wav_file": wav_file,
                    "start": row.get("start", ""), "end": row.get("end", ""),
                })
                continue

            try:
                segment, diag = load_and_crop(wav_path, row["start"], row["end"])
                if len(segment) == 0:
                    skipped_zero_length += 1
                    zero_length_by_corpus[source_corpus] += 1
                    skip_report_rows.append({
                        "reason": "zero_length_segment", "xml_file": xml_file,
                        "source_corpus": source_corpus, "wav_file": wav_file,
                        "start": diag["start_f"], "end": diag["end_f"],
                        "start_sample": diag["start_sample"], "end_sample": diag["end_sample"],
                        "audio_total_samples": diag["audio_total_samples"],
                    })
                    continue

                dataset_rows.append({
                    # Raw audio -- prepare_dataset_ipa() reads audio["array"]
                    # and audio["sampling_rate"] itself and runs it through
                    # the processor; we do NOT precompute input_features here.
                    "audio": {"array": segment, "sampling_rate": TARGET_SR},

                    # MUST be named literally "ipa" -- confirmed by directly
                    # reading prepare_dataset_ipa() in scripts/whipa_utils.py:
                    # batch["labels"] = tokenizer.encode_plus(batch["ipa"], ...)
                    "ipa": ipa_target,

                    # Bookkeeping columns -- not read by WhIPA's own code, but
                    # useful for your own debugging/filtering later. Safe to
                    # leave in; prepare_dataset_ipa() only reads "audio"/"ipa".
                    "xml_file": xml_file,
                    "wav_file": wav_file,
                    "orth": row.get("orth", ""),
                    "source_corpus": source_corpus,
                })

            except Exception as e:
                print(f"  ERROR on row {i} ({wav_file}): {e}")
                skipped_exception += 1
                skip_report_rows.append({
                    "reason": "exception", "xml_file": xml_file,
                    "source_corpus": source_corpus, "wav_file": wav_file,
                    "start": row.get("start", ""), "end": row.get("end", ""),
                    "error": str(e),
                })
                continue

    print(f"\nBuilt {len(dataset_rows)} raw training examples")
    print(f"  Skipped (empty training target after normalization): {skipped_empty_target}")
    print(f"  Skipped (audio not found): {skipped_no_audio}")
    print(f"  Skipped (zero-length segment after crop): {skipped_zero_length}")
    print(f"  Skipped (processing exception): {skipped_exception}")

    if zero_length_by_corpus:
        print("\n  Zero-length skips by source_corpus:")
        for corpus, count in zero_length_by_corpus.most_common():
            print(f"    {corpus}: {count}")

    if args.skip_report and skip_report_rows:
        fieldnames = ["reason", "xml_file", "source_corpus", "wav_file",
                      "start", "end", "start_sample", "end_sample",
                      "audio_total_samples", "error"]
        with open(args.skip_report, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in skip_report_rows:
                writer.writerow(r)
        print(f"\n  Full skip report ({len(skip_report_rows)} rows) written to {args.skip_report}")

    dataset = Dataset.from_list(dataset_rows)
    dataset.save_to_disk(args.output_dir)
    print(f"\nSaved RAW dataset (audio + ipa text, not yet tokenized/featurized) to {args.output_dir}")
    print(f"Next step: run run_prep_dataset.py on this output to apply WhIPA's own")
    print(f"prep_dataset()/prepare_dataset_ipa() functions and produce the final")
    print(f"training-ready dataset (with input_features + labels).")


if __name__ == "__main__":
    main()
