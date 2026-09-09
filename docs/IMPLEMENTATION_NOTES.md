# Implementation Notes: Fine-Tuning WhIPA on a Custom Corpus

This document records what was and wasn't covered by WhIPA's own documentation when adapting it to fine-tune on a custom corpus (in this case, a TEI/XML corpus of Mixtepec Mixtec time aligned transcriptions originally from Praat TextGrid), plus a list of bugs found and fixed in the process. The goal is to save time for anyone else trying to fine-tune WhIPA on their own data, since several of the gaps and bugs below aren't specific to this particular corpus, and would likely affect any custom fine-tuning attempt.

## Provenance legend

- **DOCUMENTED**: directly specified in WhIPA's README or the accompanying paper, used as intended.
- **SOURCE-DERIVED**: not documented in the README/paper at all, but discoverable by reading WhIPA's actual source code. The functionality exists and works as intended, but a user following only the README would not know it was needed.
- **BUG FIX**: a code change required just to get already-documented functionality working, due to the codebase being written against older versions of its dependencies (mainly `transformers`). Not a design gap, a compatibility problem with current library versions.
- **CUSTOM**: built from scratch, with no equivalent in WhIPA's code or documentation, because the task (working with an arbitrary custom corpus, rather than WhIPA's own benchmark corpora) is outside what WhIPA was designed to handle.
- **HARDWARE ADAPTATION**: a substitution for a piece of WhIPA's own code that is CUDA-only and doesn't run on Apple Silicon.

## Additional crucial information needed to implemnt the WHIPA (not explained in the documentation)

While the README documents inference with an existing pretrained checkpoint reasonably well (loading a `WHIPA` object, calling `transcribe_ipa()`), it does not document the following crucial information essential to implementation:

- The exact dataset schema `prep_dataset()` / `prepare_dataset_ipa()` expects. **The raw text column must be literally named `"ipa"`, and `"labels"` must already be tokenized (via a direct tokenizer call), not raw text.** This had to be found by reading `scripts/whipa_utils.py` directly.
- How to fine-tune on a custom, non-benchmark dataset at all. `fine_tune.py`'s `__main__` block is built entirely around a config-driven system for loading WhIPA's own named corpora (CommonVoice, ASC, THCHS-30, Sanna), **with no documented path for bringing your own dataset.**
- The original documentation lacks any Apple Silicon / non-CUDA guidance. The LoRA loading path is hardcoded to CUDA-only 8-bit quantization via `bitsandbytes` (**see LoRA fine-tuning on non-CUDA hardware below**).
- **A dependency list.** No requirements.txt exists; the package/module dependency set had to be discovered by running the code and installing packages as import errors surfaced. Several of these (epitran, textgrid, mecab-python3, unidic, panphon) are pulled in as top-level imports in scripts/whipa_utils.py, even though the specific functions needed for a training run (prep_dataset, prepare_dataset_ipa) never actually use them; Python still requires the whole module's imports to succeed before it will hand you any one function from it. A compiled requirements.txt is included in this repo (see  [requirements.txt](https://github.com/iljackb/mixtec-whipa-finetuning/blob/main/requirements.txt)).

Additionally, several things that should have worked per the documented API didn't, because the codebase predates current versions of `transformers`. These are listed as bug fixes below. They aren't a documentation gap but outdated versioning blocks the process at various points, and they would affect any user regardless of how carefully they read the README.

## Bugs found and fixed (all BUG FIX)

1. `deploy.py`: missing `import torch`, used inside `transcribe_ipa()` but never imported at module level.
2. `deploy.py`: `transcribe_ipa()` referenced an undefined variable `whipa` (should have been `self`), apparently a leftover from whatever script or notebook this method was originally developed in.
3. `deploy.py`'s `__init__` parameter is `base_model_name`, not `base_model` as the README's usage example shows. A README/code mismatch rather than a bug in the code itself, but causes the same practical blocker.
4. `scripts/whipa_utils.py`: `prepare_dataset_ipa()` called `tokenizer.encode_plus(...)`, a method removed in current `transformers` releases. Fixed by calling the tokenizer directly (`tokenizer(...)`), which returns the same `.input_ids` attribute.
5. `scripts/loader.py`: top-level import of `TRANSFORMERS_CACHE` from `transformers`, a constant removed in current releases. Fixed by dropping it from the import line (confirmed unused by any function actually needed for a training run).
6. Any custom training script that reuses `fine_tune.py`'s own `Seq2SeqTrainingArguments(...)` call verbatim will hit `overwrite_output_dir` being rejected, since that parameter has also been removed from current `transformers`.

## Data pipeline stages that had no WhIPA counterpart (all CUSTOM)

WhIPA has no concept of an arbitrary custom corpus format i.e. the system's data-loading layer only knows how to fetch and parse a small, fixed list of specific, named datasets it was built around, not any general-purpose ingestion mechanism at all (i.e. there's no generic "point me at a folder of audio+transcription pairs in some standard shape.." option anywhere in the pipeline). Thus, the following had to be built from scratch:


- **Normalization of gold transcription targets.** If your corpus has any internal notation inconsistency (different tone marking conventions, inconsistent vowel length notation, etc.), this needs to be resolved into consistent training targets yourself; WhIPA has no normalization step of its own. For this project's specific conventions and needs this is handled by [normalize_ipa.py](https://github.com/iljackb/mixtec-whipa-finetuning/blob/main/normalize_ipa.py) however each project would need to modify to their own specific customs and needs.

- **Verification that expected audio files actually exist and are locatable on disk.** This step doesn't exist anywhere in WhIPA's own pipeline, since it never ingests a corpus by matching filenames in the first place. Building it yourself is unavoidable for any real corpus, which may have some filename inconsistencies between what a transcription file references and what's actually saved (e.g. transcriptions exist but paired audio files are missing). This preprocessing step is handled by: [verify_audio_paths.py](https://github.com/iljackb/mixtec-whipa-finetuning/blob/main/verify_audio_paths.py)

- **Cropping audio to the exact segment needed for each training example**, and resampling it to 16kHz (a hard Whisper requirement). WhIPA has no mechanism to accept a long audio file plus a set of timestamps directly; you have to crop each segment out yourself ahead of time and hand it pre-cut short clips. This preprocessing step is handled by [build_finetune_dataset.py](https://github.com/iljackb/mixtec-whipa-finetuning/blob/main/build_finetune_dataset.py)

## Dataset construction and feature/label preparation

- **CUSTOM**: a script that crops/resamples audio per token and builds a `datasets.Dataset` containing just the raw `audio` array and the `"ipa"` text column, deliberately not precomputing features or labels itself.
- **SOURCE-DERIVED**: rather than reimplementing WhIPA's tokenization and feature extraction, it's better to import and call their actual `prep_dataset()` / `prepare_dataset_ipa()` functions directly against this raw dataset. This guarantees identical preprocessing to what `fine_tune.py` itself would produce, and avoids subtly mismatching their tokenization logic (in particular, whatever special-token handling their tokenizer setup applies).

## LoRA fine-tuning on non-CUDA hardware

A standalone training script was built by reading `fine_tune.py` and `scripts/loader.py` end to end and reusing as much of their actual logic as possible, substituting only the one CUDA-specific step:

- **DOCUMENTED, reused directly**: `DataCollatorSpeechSeq2SeqWithPadding` and `SavePeftModelCallback`, both importable directly from `fine_tune.py` without triggering its CLI logic (they're plain class definitions sitting outside the `if __name__ == "__main__":` guard).
- **DOCUMENTED, reused directly**: `add_ipa()` (special `<|ip|>` token setup and embedding resize) from `scripts/loader.py`, and the LoRA hyperparameters (r=32, alpha=64, dropout=0.05, target modules limited to the decoder's `q_proj`/`v_proj`), copied verbatim from their own `LoraConfig` construction.
- **HARDWARE ADAPTATION**: WhIPA's own PEFT branch loads the base model via `BitsAndBytesConfig(load_in_8bit=True), device_map="auto"`, which is CUDA-only. On Apple Silicon, load the model in full precision and move it explicitly to the `mps` device instead.
- **CUSTOM**: training arguments modeled on `fine_tune.py`'s own documented CPU-default fallback values, since MPS is closer in practice to "no CUDA" than to a full CUDA GPU for HuggingFace Trainer's purposes. `fp16` should be disabled; MPS support for mixed precision has historically been inconsistent. Also worth measuring evaluation overhead directly: on a 1.5B parameter model, a single evaluation pass took roughly 100 seconds in testing, meaning the library's own default `eval_steps=50` can add hours of pure evaluation overhead to a full run. Increasing this interval is a large, easy time saving with no effect on model quality.
- **CUSTOM, found via debugging**: if training appears to hang silently for several minutes before any progress bar appears, check whether the dataset still carries unused columns (raw audio arrays, bookkeeping fields) alongside `input_features`/`labels`. Combined with `remove_unused_columns=False` (itself required for PEFT), the Trainer can spend a long time handling large unused columns it never actually needs. Dropping everything except `input_features` and `labels` before constructing the Trainer resolved this.

## Evaluation

- **DOCUMENTED, reused directly**: WhIPA's own `STIPA_METRICS` class (`code/scripts/metrics.py`) for PER/PFER scoring. This is the one part of the pipeline that the README describes completely and accurately; use `compute_all()` directly rather than reimplementing phone-level edit distance or feature-based scoring.
- **CUSTOM**: when comparing predictions against gold transcriptions, make sure the gold text is passed through the exact same normalization pipeline used to build training targets. Comparing raw, un-normalized gold against a fine-tuned model's output will produce a misleading accuracy figure, since the model is being penalized for not producing details (like tone marks) it was never trained to produce.

## Summary

The modeling core of WhIPA (LoRA configuration, tokenization, feature extraction, evaluation metrics) is solid and works correctly once its undocumented schema requirements are discovered. The entire data pipeline needed to go from an arbitrary existing corpus to something WhIPA can consume has no equivalent in the package at all and needs to be built per project. If you're planning to fine-tune WhIPA on your own data, budget time for that data pipeline separately from the actual model fine-tuning step; in this project, it was a substantially larger share of total effort than the fine-tuning itself.
