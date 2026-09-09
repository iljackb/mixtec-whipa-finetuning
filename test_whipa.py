"""
WhIPA test v4: segments each .wav file using the <u start="..." end="..."> boundaries
in its matching TEI XML, auto-resamples audio to 16kHz (Whisper's required rate),
extracts mel features via WHIPA's own processor, runs transcribe_ipa() on each
cropped token, and prints predicted IPA next to that token's own gold IPA.

Fixes from v3:
  - Actually resamples to 16kHz instead of just warning about mismatched rates
    (your files came in at 44100Hz/96000Hz; Whisper's feature extractor requires
    exactly 16000Hz).

Usage:
    cd /path/to/whipa
    python3 test_whipa.py --audio_dir test_data --xml_dir test_data --model jshrdt/whipa-large-cv --base_model_name openai/whisper-large-v2

Requires: pip3 install scipy   (only needed for the resampling step)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from lxml import etree

try:
    from scipy.signal import resample_poly
    from math import gcd
except ImportError:
    print("scipy is required for automatic resampling. Run: pip3 install scipy")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))
from deploy import WHIPA  # noqa: E402

# Apply the SAME normalization pipeline used to build training targets, so
# the gold shown here matches what the model was actually trained to predict
# (tone-stripped, affricates tie-barred, vowel-length doubled, creakiness
# rule applied) -- comparing raw un-normalized gold against a fine-tuned
# model's output isn't a fair or meaningful comparison.
from normalize_ipa import normalize_for_training  # noqa: E402

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}
TARGET_SR = 16000


def extract_tokens(xml_path: Path):
    """
    Return a list of dicts, one per <u> element (one per token/repetition):
      {"n": "1", "start": 0.0, "end": 1.19, "orth": "ka'nu", "ipa": "ka˥ʔnũ˧"}
    """
    tree = etree.parse(str(xml_path))
    tokens = []

    for u in tree.findall(".//tei:u", TEI_NS):
        start = float(u.get("start", 0.0))
        end = float(u.get("end", 0.0))
        n = u.get("n", "?")

        orth_seg = u.find(".//tei:seg[@notation='orth']", TEI_NS)
        ipa_seg = u.find(".//tei:seg[@notation='ipa']", TEI_NS)

        orth = " ".join("".join(w.itertext()).strip()
                         for w in orth_seg.findall(".//tei:w", TEI_NS)) if orth_seg is not None else ""
        ipa = " ".join("".join(w.itertext()).strip()
                        for w in ipa_seg.findall(".//tei:w", TEI_NS)) if ipa_seg is not None else ""

        tokens.append({"n": n, "start": start, "end": end, "orth": orth, "ipa": ipa})

    return tokens


def crop_audio(audio: np.ndarray, sr: int, start_sec: float, end_sec: float) -> np.ndarray:
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    return audio[start_sample:end_sample]


def resample_to_target(audio: np.ndarray, orig_sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    g = gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    return resample_poly(audio, up, down)


def load_audio(wav_path: Path):
    audio, sr = sf.read(str(wav_path))
    if audio.ndim > 1:
        print(f"  NOTE: {wav_path.name} has {audio.shape[1]} channels; downmixing to mono.")
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        print(f"  NOTE: {wav_path.name} is {sr}Hz; resampling to {TARGET_SR}Hz.")
        audio = resample_to_target(audio, sr, TARGET_SR)
        sr = TARGET_SR
    return audio, sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", type=str, default="test_data")
    ap.add_argument("--xml_dir", type=str, default="test_data")
    ap.add_argument("--model", type=str, default="jshrdt/whipa-large-cv")
    ap.add_argument("--base_model_name", type=str, default="openai/whisper-large-v2")
    ap.add_argument("--lora", action="store_true", help="Set if using a lowhipa-* (LoRA) checkpoint")
    args = ap.parse_args()

    audio_dir = Path(args.audio_dir)
    xml_dir = Path(args.xml_dir)

    wav_files = sorted(audio_dir.glob("*.wav"))
    if not wav_files:
        print(f"No .wav files found in {audio_dir}")
        sys.exit(1)

    print(f"Loading model: {args.model} (base: {args.base_model_name}, lora={args.lora}) ...")
    whipa = WHIPA(model_path=args.model, base_model_name=args.base_model_name, lora=args.lora)

    for wav_path in wav_files:
        xml_path = xml_dir / (wav_path.stem + ".xml")
        if not xml_path.exists():
            print(f"File: {wav_path.name}\n  [no matching XML found, skipping]\n")
            continue

        audio, sr = load_audio(wav_path)
        tokens = extract_tokens(xml_path)

        if not tokens:
            print(f"File: {wav_path.name}\n  [no <u> tokens found in XML, skipping]\n")
            continue

        print(f"File: {wav_path.name}  ({len(tokens)} token{'s' if len(tokens) != 1 else ''})")
        for tok in tokens:
            segment = crop_audio(audio, sr, tok["start"], tok["end"])
            segment = np.ascontiguousarray(segment, dtype=np.float32)

            try:
                input_features = whipa.processor(
                    segment, sampling_rate=sr, return_tensors="np"
                ).input_features[0]
                sample = {
                    "input_features": input_features,
                    "audio": {"array": segment},
                }
                prediction = whipa.transcribe_ipa(sample, verbose=True)
            except Exception as e:
                prediction = f"[inference error: {e}]"

            print(f"  Token {tok['n']} [{tok['start']:.2f}-{tok['end']:.2f}s]  orth: {tok['orth']}")
            print(f"    Predicted: {prediction}")
            print(f"    Gold (raw):        {tok['ipa']}")
            print(f"    Gold (normalized): {normalize_for_training(tok['ipa'])}")
        print()


if __name__ == "__main__":
    main()
