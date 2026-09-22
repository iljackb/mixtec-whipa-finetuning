# mixtec-whipa-finetuning

Custom pipeline for fine-tuning [WhIPA](https://github.com/jshrdt/whipa) (a Whisper-based speech-to-IPA (STIPA) model) on Mixtepec Mixtec (Sa'an Savi), an under-resourced Otomanguean language with no prior representation in WhIPA's training data.

This repo is the ASR/phonetic-transcription tooling side of a larger language documentation project. The linguistic corpus itself (TEI/XML transcriptions, dictionary, paradigms) lives separately at **[iljackb/Mixtepec_Mixtec](https://github.com/iljackb/Mixtepec_Mixtec)**: see that repo's `ASR-finetuning/` folder for versioned result reports and methodology documentation tied to specific training runs.

## Before you start

If you're trying to fine-tune WhIPA on your own custom corpus, read **[docs/IMPLEMENTATION_NOTES.md](docs/IMPLEMENTATION_NOTES.md)** first. It documents several undocumented requirements, known bugs in the current codebase, and a non-CUDA (Apple Silicon) training path that aren't covered anywhere in WhIPA's own README.


## What's in here

- `code/`: a clone of the upstream [jshrdt/whipa](https://github.com/jshrdt/whipa) package, with a handful of bug fixes applied (see commit history; several of its dependencies have been removed/renamed in current `transformers` releases since it was written).

- Custom pipeline scripts (project root): none of these exist in upstream WhIPA, and they are specific to this project's Mixtepec Mixtec corpus and orthographic conventions, not a generic, reusable toolkit for fine-tuning WhIPA on any language (see [docs/IMPLEMENTATION_NOTES.md](docs/IMPLEMENTATION_NOTES.md) for the general, transferable gaps/lessons instead). They handle everything needed to go from an existing TEI/XML transcription corpus to a WhIPA-ready training dataset, run in this order:

  - `extract_finetune_data_unified.py`: extracts token-level orthography, gold IPA, and audio timestamps from the whole corpus in one pass, classifying and dispatching **each `<u>` independently** (not per-file) so a single file can freely mix single-word, sentence, and whole-utterance utterances and still extract correctly. Internally uses `normalize_ipa.py` to clean up inconsistent legacy IPA notation, tone marking, vowel length, affricates, and creakiness into consistent training targets. Replaces the three older per-format scripts formerly in the repo root (`extract_finetune_data.py`, `extract_finetune_data_sentences.py`, `extract_finetune_data_myuc.py`), which routed at the file level and couldn't handle files mixing utterance types -- moved to `legacy-scripts/` for historical reference; see `docs/IMPLEMENTATION_NOTES.md` for a confirmed bug in one of them.

  - `verify_audio_paths.py`: cross-references expected audio filenames against what's actually on disk.

  - `build_finetune_dataset.py`: crops/resamples audio and builds a raw `datasets.Dataset`.

  - `run_prep_dataset.py`: applies WhIPA's own feature-extraction/tokenization functions to that dataset.

  - `train_lora_mps.py`: LoRA fine-tuning script adapted for Apple Silicon (MPS), since WhIPA's own training script's PEFT path is CUDA-only (8-bit quantization via `bitsandbytes`).

  - `test_whipa.py`: inference/testing against a held-out set.

  - `score_test_results.py`: evaluation via WhIPA's own `STIPA_METRICS` (PER/PFER), scoring the predictions `test_whipa.py` just produced.

- Diagnostic / as-needed scripts (project root): not part of the sequential pipeline above -- run these on demand, not on every corpus run.

  - `classify_tei_structure.py`: scans a corpus directory and classifies every `<u>` by its actual structure (single-word / sentence / whole-utterance), independent of filename convention. Read-only, reports only -- doesn't feed into extraction (its per-`<u>` classification logic is now built directly into `extract_finetune_data_unified.py`, which classifies inline as it extracts). Useful as a quick look-before-you-extract pass on a new or changed corpus subtree, or to spot files that fail to parse.

  - `dedupe_xml_ids.py`: fixes "ID already defined" parse errors by regenerating duplicate `xml:id`/`id` values within a file. Most files won't have this problem -- only run it against a file that actually fails to parse with that error.

## Why a separate repo

The corpus (`Mixtepec_Mixtec`) and this tooling are different in kind: one is linguistic data, the other is a Python codebase with its own dependencies and development workflow: so they're kept as separate, cross-linked repos rather than combined.

## Status

First working fine-tuned checkpoint (`lowhipa-mixtec-v1`) trained on 1,076 tokens; results and methodology documented in `Mixtepec_Mixtec/ASR-finetuning/`.

## Outputs (not committed to git -- see .gitignore)

Running the full pipeline produces three generated directories, each regenerable from
the manifest + audio + code, so none are version-controlled:

- `whipa_raw_dataset/` -- `build_finetune_dataset.py`'s output. Raw cropped/resampled
  audio + normalized IPA text, one row per training example, not yet tokenized.

- `whipa_prepped_dataset/` -- `run_prep_dataset.py`'s output. The same rows after
  WhIPA's own `prep_dataset()` adds `input_features` (mel-spectrogram) and `labels`
  (tokenized IPA); still carries the original raw-dataset columns alongside the new
  ones, since `prep_dataset()` adds rather than replaces columns.

- `models/<checkpoint-name>/` -- `train_lora_mps.py`'s output. Periodic training
  checkpoints, TensorBoard logs, and the final model (LoRA adapter merged into the
  base model, plus tokenizer/processor) once training completes.

### Note:
This system was implemented (and debugged) with AI assistance from Claude
