# mixtec-whipa-finetuning

Custom pipeline for fine-tuning [WhIPA](https://github.com/jshrdt/whipa) (a Whisper-based speech-to-IPA (STIPA) model) on Mixtepec Mixtec (Sà'án Sàvǐ), an under-resourced Otomanguean language with no prior representation in WhIPA's training data.

This repo is the ASR/phonetic-transcription tooling side of a larger language documentation project. The linguistic corpus itself (TEI/XML transcriptions, dictionary, paradigms) lives separately at **[iljackb/Mixtepec_Mixtec](https://github.com/iljackb/Mixtepec_Mixtec)**: see that repo's `ASR-finetuning/` folder for versioned result reports and methodology documentation tied to specific training runs.

## Before you start

If you're trying to fine-tune WhIPA on your own custom corpus, read **[docs/IMPLEMENTATION_NOTES.md](docs/IMPLEMENTATION_NOTES.md)** first. It documents several undocumented requirements, known bugs in the current codebase, and a non-CUDA (Apple Silicon) training path that aren't covered anywhere in WhIPA's own README.


## What's in here

- `code/`: a clone of the upstream [jshrdt/whipa](https://github.com/jshrdt/whipa) package, with a handful of bug fixes applied (see commit history; several of its dependencies have been removed/renamed in current `transformers` releases since it was written). One change affects results: `code/deploy.py`'s decoding rate limiter now uses exact clip duration (see [Evaluation](#evaluation)).
- Custom pipeline scripts (project root): none of these exist in upstream WhIPA, and they are specific to this project's Mixtepec Mixtec corpus and orthographic conventions, not a generic, reusable toolkit for fine-tuning WhIPA on any language (see [docs/IMPLEMENTATION_NOTES.md](docs/IMPLEMENTATION_NOTES.md) for the general, transferable gaps/lessons instead). They handle everything needed to go from an existing TEI/XML transcription corpus to a WhIPA-ready training dataset, run in this order:
  - **Where to run them:** `run_prep_dataset.py`, `train_lora_mps.py` and `score_test_results.py` live in `code/` and are run from there (`train_lora_mps.py` with `PYTHONPATH=.`); `test_whipa.py` is run from the project root.
  - `classify_tei_structure.py`: scans a corpus directory and classifies every `<u>` by its actual structure (single-word / sentence / whole-utterance), independent of filename convention. Read-only diagnostic -- run this first on any new or changed corpus subtree to see what's there before extracting from it, and to catch files that fail to parse.
  - `dedupe_xml_ids.py`: fixes "ID already defined" parse errors by regenerating duplicate `xml:id`/`id` values within a file. Run against anything `classify_tei_structure.py` flags as a parse error before extraction.
  - `extract_finetune_data_unified.py`: extracts token-level orthography, gold IPA, and audio timestamps from the whole corpus in one pass, classifying and dispatching **each `<u>` independently** (not per-file) so a single file can freely mix single-word, sentence, and whole-utterance utterances and still extract correctly. Internally uses `normalize_ipa.py` to clean up inconsistent legacy IPA notation, tone marking, vowel length, affricates, and creakiness into consistent training targets. Replaces the three older per-format scripts (`extract_finetune_data.py`, `extract_finetune_data_sentences.py`, `extract_finetune_data_myuc.py`), which routed at the file level and couldn't handle files mixing utterance types -- kept in the repo for reference/diffing during the transition, but no longer the primary extraction path.
  - `verify_audio_paths.py`: cross-references expected audio filenames against what's actually on disk.
  - `build_finetune_dataset.py`: crops/resamples audio and builds a raw `datasets.Dataset`.
  - `run_prep_dataset.py`: applies WhIPA's own feature-extraction/tokenization functions to that dataset.
  - `train_lora_mps.py`: LoRA fine-tuning script adapted for Apple Silicon (MPS), since WhIPA's own training script's PEFT path is CUDA-only (8-bit quantization via `bitsandbytes`).
  - `summarize_training.py`: summarizes a finished fine-tuning run from the Trainer's saved `trainer_state.json` (in the `checkpoint-*` folders of the model output directory): the eval-loss curve per evaluation step, whether the best and final checkpoints coincide, and overfitting checks. Prints a ready-to-paste Markdown section for the results report; `--md` also saves it (e.g. into the model folder), `--csv` saves the curve. Run it on each new model directory right after training, e.g. `python3 summarize_training.py models/lowhipa-mixtec-v3`.
  - `test_whipa.py`: inference/testing against a held-out set. Writes one predictions CSV per model (`--output-csv`; the output folder is created if missing).
  - `score_test_results.py`: evaluation via WhIPA's own `STIPA_METRICS` (PER/PFER), scoring the predictions `test_whipa.py` just produced and appending the scores to a shared log (see [Evaluation](#evaluation)).

## Status

- First working fine-tuned checkpoint (`lowhipa-mixtec-v1`) trained on 1,076 tokens;

- Second working fine-tuned checkpoint (`lowhipa-mixtec-v2`) trained on 3,321 tokens;

- Third checkpoint (`lowhipa-mixtec-v3`) trained on 3,629 tokens (3,266 train / 363 dev, 4,085 steps, final eval loss 0.6516). On the 19-token held-out set it is comparable to v2 (phrase-level PER 12.2% vs. 15.6%; single-word PER 23.8% vs. 24.4%), and both are far ahead of zero-shot (phrase-level PER 58.2%);

See detailed documentation, latest results as well as status of the fine tuning and testing in [WhIPA Fine-Tuning on Mixtepec Mixtec: Preliminary Results Report (v3)](https://github.com/iljackb/mixtec-whipa-finetuning/blob/main/docs/Preliminary_Results_Report-v3-20261022.md). Earlier reports are in `docs/archive/reports/`.

## Evaluation

**Running and scoring a model** (use identical arguments for every model you compare; the saved v1-v3 models are full checkpoints and are tested without `--lora`):

```
python3 test_whipa.py --audio_dir test_data --xml_dir test_data \
    --model models/lowhipa-mixtec-v3 --base_model_name openai/whisper-large-v2 \
    --output-csv test_results/v3/test_results.csv

cd code
python3 score_test_results.py --input-csv ../test_results/v3/test_results.csv \
    --output-log ../test_results/test_scores.csv --model-name lowhipa-mixtec-v3
```

**Layout of `test_results/`:**
- `<model>/test_results.csv`: predictions for one model (`zeroshot`, `v1`, `v2`, `v3`, ...).
- `test_scores.csv`: one shared score log for all models. Runs rescored after the 2026-10-22 rate-limiter fix carry a `-ratefix` suffix in the model name.
- `archive-old-ratelimit/`: results produced before that fix. **Not comparable** to anything newer; kept for audit.

**Rate limiter (`code/deploy.py`).** Upstream WhIPA caps predicted phones at 20 per second of audio but rounds the clip length to whole seconds, so any clip under 0.5 s got a cap of 1 phone and a correct prediction was cut to a single character. As of 2026-10-22 the cap uses the exact duration with a floor of 5 phones. This deviates from the published method and changed exactly one test token (`na'nu`, 0.47 s).

**Test set.**
- The test files live in `test_data/`, a directory the training scripts never read; the file list is in `test_holdout.txt`.
- For future runs, decide the held-out set *before* extraction and dataset build: move whole recordings (audio + XML) into the test directory, not individual utterances. Aim for about 100+ tokens, mostly utterances from field recordings plus a separate single-word track, covering several speakers, and report scores per speaker.
- The current 19-token set is too small to resolve differences of a point or two between models.

**Scorer warnings.** `score_test_results.py` prints a set of symbols (for example the combining cedilla and the voiced diacritic U+032C) when it has no feature values for them; how those affect PFER is unverified.

## Before launching a long training run

Training takes about 36 hours, so check these first:

1. `build_finetune_dataset.py` row count matches expectations.
2. **`run_prep_dataset.py` was rerun after the latest build** and its output count matches (v3: 3,629). Skipping this trains on a stale dataset.
3. At launch the trainer prints `Train: N | Dev: M`; check N + M against the prepared count.
4. Expected steps = ceil(N / batch size) x epochs (v3: 817 x 5 = 4,085). Stop the run if the trainer's total differs.

## Known caveats

- The train/dev split in `train_lora_mps.py` is random by utterance (`train_test_split`, seed 42), so dev loss is optimistic and not comparable between runs. Compare models on PER/PFER over the held-out set instead.
- Targets are currently toneless; tone is planned for a later run, scored with and without tones.
- Nasalization and vowel length vary by speaker and word, and the gold transcription and the audio can disagree; see section 5.2 of the v3 report.

## Repo hygiene

`.gitignore` excludes model weights and checkpoints (`models/`, `checkpoint-*/`, `*.safetensors`, ...), prepared and raw datasets (`*.arrow`, `whipa_prepped_dataset*/`, `whipa_raw_dataset*/`) and large archive audio. Before pushing, check that no large file is in the unpushed commits:

```
git rev-list --objects origin/main..HEAD | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | sort -k3 -n -r | head -5
```

### Note:
This system was implemented (and debugged) with AI assistance from Claude
