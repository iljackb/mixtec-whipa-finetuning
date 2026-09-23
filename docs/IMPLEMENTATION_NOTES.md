# Implementation Notes: Fine-Tuning WhIPA on a Custom Corpus

This document records what's actually required to fine-tune WhIPA on a custom corpus (in this case, a TEI/XML corpus of Mixtepec Mixtec time-aligned transcriptions originally from Praat TextGrid), plus a list of bugs found and fixed in the process. The goal is to save time for anyone else trying to fine-tune WhIPA on their own data, since most of what's below isn't specific to this particular corpus and would likely affect any custom fine-tuning attempt.

## How this is organized

- **Bug fixes**: code changes required to get WhIPA's existing functionality working, due to the codebase being written against older versions of its dependencies (mainly `transformers`). Compatibility problems with current library versions, not design gaps.

- **Custom pipeline components**: built from scratch, with no equivalent in WhIPA's code, because the task (working with an arbitrary custom corpus, rather than WhIPA's own benchmark corpora) is outside what WhIPA was designed to handle.

- **Hardware adaptation**: a substitution for a piece of WhIPA's own code that is CUDA-only and doesn't run on Apple Silicon.

## What you need to know before implementing

- The exact dataset schema `prep_dataset()` / `prepare_dataset_ipa()` expects: **the raw text column must be literally named `"ipa"`, and `"labels"` must already be tokenized** (via a direct tokenizer call), not raw text. Found by reading `scripts/whipa_utils.py` directly.

- How to fine-tune on a custom, non-benchmark dataset at all: `fine_tune.py`'s `__main__` block is built entirely around a config-driven system for loading WhIPA's own named corpora (CommonVoice, ASC, THCHS-30, Sanna). To fine-tune on your own dataset, reuse `fine_tune.py`'s reusable pieces (`DataCollatorSpeechSeq2SeqWithPadding`, `SavePeftModelCallback`, `add_ipa()`, the LoRA config) directly in your own training script rather than trying to route a custom dataset through its CLI.

- Apple Silicon / non-CUDA setup: the LoRA loading path is hardcoded to CUDA-only 8-bit quantization via `bitsandbytes` (see **LoRA fine-tuning on non-CUDA hardware** below for the substitution).

- **Dependencies**: no `requirements.txt` ships with the package; the module dependency set was discovered by running the code and installing packages as import errors surfaced. Several of these (epitran, textgrid, mecab-python3, unidic, panphon) are pulled in as top-level imports in `scripts/whipa_utils.py`, even though the specific functions needed for a training run (`prep_dataset`, `prepare_dataset_ipa`) never actually use them — Python still requires the whole module's imports to succeed before it will hand you any one function from it. A compiled `requirements.txt` is included in this repo (see [requirements.txt](https://github.com/iljackb/mixtec-whipa-finetuning/blob/main/requirements.txt)).

Additionally, several things needed direct code changes because the codebase predates current versions of `transformers`. These are listed as bug fixes below — they'd affect any user regardless of how carefully they read the README, since outdated versioning blocks the process at various points.

## Bugs found and fixed

1. `deploy.py`: missing `import torch`, used inside `transcribe_ipa()` but never imported at module level.

2. `deploy.py`: `transcribe_ipa()` referenced an undefined variable `whipa` (should have been `self`), apparently a leftover from whatever script or notebook this method was originally developed in.

3. `deploy.py`'s `__init__` parameter is `base_model_name`, not `base_model`. Passing the wrong keyword raises a `TypeError`.

4. `scripts/whipa_utils.py`: `prepare_dataset_ipa()` called `tokenizer.encode_plus(...)`, a method removed in current `transformers` releases. Fixed by calling the tokenizer directly (`tokenizer(...)`), which returns the same `.input_ids` attribute.

5. `scripts/loader.py`: top-level import of `TRANSFORMERS_CACHE` from `transformers`, a constant removed in current releases. Fixed by dropping it from the import line (confirmed unused by any function actually needed for a training run).

6. Any custom training script that reuses `fine_tune.py`'s own `Seq2SeqTrainingArguments(...)` call verbatim will hit `overwrite_output_dir` being rejected, since that parameter has also been removed from current `transformers`.

## Building a data pipeline for a custom corpus

WhIPA's data-loading layer only knows how to fetch and parse a small, fixed list of specific named datasets — there's no generic "point me at a folder of audio+transcription pairs" option anywhere in the pipeline. So a full custom data pipeline needs to be built for any corpus that isn't one of WhIPA's own benchmarks:

- **Extraction of token-level orthography, gold IPA, and audio timestamps from the source corpus format** (in this case TEI/XML, but the same extraction/parsing work is needed for any custom format). This project's source corpus has three structurally different utterance types needing separate handling: single-word-per-utterance, multi-word-per-utterance with real word-level timing ("sentence"), and whole-utterance-only sources with no word-level timing at all (extracting one training example per utterance rather than per word). This was originally handled by three separate scripts, each assuming a given file was uniformly one type — which broke for files that genuinely mix utterance types internally (confirmed against this corpus: some files have some `<u>`s that are plain sentences and others with no word-level timing at all, in the same file). [extract_finetune_data_unified.py](https://github.com/iljackb/mixtec-whipa-finetuning/blob/main/extract_finetune_data_unified.py) replaces them: it classifies and dispatches **each utterance independently** (not per-file) using the same structural rule the three original scripts encoded, so a single file can freely mix all three types and still extract correctly in one pass. The three original per-type scripts are kept in `legacy-scripts/` for reference.

- **Normalization of gold transcription targets.** Any corpus with internal notation inconsistency (different tone marking conventions, inconsistent vowel length notation, etc.) needs this resolved into consistent training targets, since WhIPA has no normalization step of its own. For this project's specific conventions and needs this is handled by [normalize_ipa.py](https://github.com/iljackb/mixtec-whipa-finetuning/blob/main/normalize_ipa.py) — each project would need to adapt this to its own orthographic customs.

- **Combining multiple extraction scripts' output into one manifest.** This was originally handled by a separate `combine_manifests.py` script, since retired — `extract_finetune_data_unified.py` now performs this merge internally by classifying and dispatching each `<u>` independently across the whole corpus in a single pass, so there's no longer a separate combining step.

- **Verification that expected audio files actually exist and are locatable on disk.** Any real corpus may have filename inconsistencies between what a transcription file references and what's actually saved (e.g. transcriptions exist but paired audio files are missing), so this check is unavoidable to build. Handled by [verify_audio_paths.py](https://github.com/iljackb/mixtec-whipa-finetuning/blob/main/verify_audio_paths.py)

- **Cropping audio to the exact segment needed for each training example**, and resampling it to 16kHz (a hard Whisper requirement). WhIPA has no mechanism to accept a long audio file plus a set of timestamps directly — each segment needs to be cropped out ahead of time and handed over as pre-cut short clips. Handled by [build_finetune_dataset.py](https://github.com/iljackb/mixtec-whipa-finetuning/blob/main/build_finetune_dataset.py)

Note on reusing these specific scripts versus reusing the lesson: the scripts referenced above (and the corpus's own normalization rules they encode) are specific to one project's corpus, its own orthographic conventions, and its own TEI structure. They are not a generic, plug-and-play toolkit for fine-tuning WhIPA on a different language. If your source corpus uses a different orthography, a different notation for tone or length, or a different file format entirely, you will need to write your own equivalent extraction and normalization layer for your own data, following the same general shape (extract token-level orthography/IPA/timestamps, normalize inconsistent notation, verify audio, crop and build the dataset) rather than adapting these files directly. Even within this one project, three separate extraction scripts were needed for three structurally different source formats found in the same corpus, which is itself a useful data point: expect the extraction/normalization layer to be genuinely project-specific work, not a one-time cost you can borrow from this project's implementation.

## Dataset construction and feature/label preparation

- **Custom**: a script that crops/resamples audio per token and builds a `datasets.Dataset` containing just the raw `audio` array and the `"ipa"` text column, deliberately not precomputing features or labels itself.

- Rather than reimplementing WhIPA's tokenization and feature extraction, it's better to import and call their actual `prep_dataset()` / `prepare_dataset_ipa()` functions directly against this raw dataset. This guarantees identical preprocessing to what `fine_tune.py` itself would produce, and avoids subtly mismatching their tokenization logic (in particular, whatever special-token handling their tokenizer setup applies).

## LoRA fine-tuning on non-CUDA hardware

A standalone training script was built by reading `fine_tune.py` and `scripts/loader.py` end to end and reusing as much of their actual logic as possible, substituting only the one CUDA-specific step:

- Reused directly: `DataCollatorSpeechSeq2SeqWithPadding` and `SavePeftModelCallback`, both importable directly from `fine_tune.py` without triggering its CLI logic (they're plain class definitions sitting outside the `if __name__ == "__main__":` guard).

- Reused directly: `add_ipa()` (special `<|ip|>` token setup and embedding resize) from `scripts/loader.py`, and the LoRA hyperparameters (r=32, alpha=64, dropout=0.05, target modules limited to the decoder's `q_proj`/`v_proj`), copied verbatim from their own `LoraConfig` construction.

- **Hardware adaptation**: WhIPA's own PEFT branch loads the base model via `BitsAndBytesConfig(load_in_8bit=True), device_map="auto"`, which is CUDA-only. On Apple Silicon, load the model in full precision and move it explicitly to the `mps` device instead.

- Training arguments modeled on `fine_tune.py`'s own documented CPU-default fallback values, since MPS is closer in practice to "no CUDA" than to a full CUDA GPU for HuggingFace Trainer's purposes. `fp16` should be disabled; MPS support for mixed precision has historically been inconsistent. Also worth measuring evaluation overhead directly: on a 1.5B parameter model, a single evaluation pass took roughly 100 seconds in testing, meaning the library's own default `eval_steps=50` can add hours of pure evaluation overhead to a full run. Increasing this interval is a large, easy time saving with no effect on model quality.

- Found via debugging: if training appears to hang silently for several minutes before any progress bar appears, check whether the dataset still carries unused columns (raw audio arrays, bookkeeping fields) alongside `input_features`/`labels`. Combined with `remove_unused_columns=False` (itself required for PEFT), the Trainer can spend a long time handling large unused columns it never actually needs. Dropping everything except `input_features` and `labels` before constructing the Trainer resolved this.

- **Environment quirk (macOS)**: `train_lora_mps.py` can abort immediately with `OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.` This happens when two installed packages (commonly `torch` and `numpy`/`scipy`) each bundle their own copy of the OpenMP runtime — not a bug in this project's code or a sign of a bad dataset/config. Workaround: run with `KMP_DUPLICATE_LIB_OK=TRUE` set, e.g. `KMP_DUPLICATE_LIB_OK=TRUE python3 train_lora_mps.py ...`. This is a well-known, generally benign false alarm for a single-machine PyTorch+NumPy stack on Apple Silicon; if training completes and losses look sane, the duplicate-runtime warning wasn't a real problem.

## Evaluation

- WhIPA's own `STIPA_METRICS` class (`code/scripts/metrics.py`) handles PER/PFER scoring correctly as-is; use `compute_all()` directly rather than reimplementing phone-level edit distance or feature-based scoring.

- When comparing predictions against gold transcriptions, make sure the gold text is passed through the exact same normalization pipeline used to build training targets. Comparing raw, un-normalized gold against a fine-tuned model's output will produce a misleading accuracy figure, since the model is being penalized for not producing details (like tone marks) it was never trained to produce.

## Summary

The modeling core of WhIPA (LoRA configuration, tokenization, feature extraction, evaluation metrics) is solid and works correctly once its schema requirements and custom-dataset path are worked out. The entire data pipeline needed to go from an arbitrary existing corpus to something WhIPA can consume has no equivalent in the package at all and needs to be built per project. If you're planning to fine-tune WhIPA on your own data, budget time for that data pipeline separately from the actual model fine-tuning step; in this project, it was a substantially larger share of total effort than the fine-tuning itself.
