# WhIPA Fine-Tuning on Mixtepec Mixtec: Preliminary Results Report

**Author:** Jack Bowers
**Date:** 2026-09-09 (updated 2026-09-24)
**Model:** LoRA fine-tune of `openai/whisper-large-v2`, using the WhIPA/LoWhIPA framework (Suchardt et al., 2025 EMNLP; code: github.com/jshrdt/whipa)
**Checkpoint identifiers:** `lowhipa-mixtec-v1`, `lowhipa-mixtec-v2`

---

## 1. Objective

Evaluate whether LoRA fine-tuning of a Whisper-based speech-to-IPA (STIPA) model, using a small existing corpus of Mixtepec Mixtec phonetic transcriptions, improves automatic phonetic transcription accuracy relative to the same base model's zero-shot performance on this language. Mixtepec Mixtec is not represented in any of WhIPA's original training data (CommonVoice languages, Arabic Speech Corpus, THCHS-30 Mandarin), nor in any typologically similar training language.

---

## 2. Data

## 2.1 Training corpus

**v1 training pool** (pre-truncation-fix; predates integration of the AILLA "Documentation of Mixtepec Mixtec" collection⁴):

| Source | Tokens | Description |
|---|---|---|
| `transcriptions-xml/` | 818 | Individually elicited words/short phrases, TEI/XML |
| SIL "Aprendamos" materials (Lección 01–05) | 264 | Word-level tokens from sentence-level elicitation recordings |
| **Total** | **1,082** (1,076 with resolvable audio) | 6 tokens excluded: audio file not locatable |

**v2 training pool** (corrected extraction via `extract_finetune_data_unified.py`, includes AILLA):

| Source | Tokens (raw extraction) | Description |
|---|---|---|
| Mixtepec Mixtec Language Resources (Bowers, Salazar & Salazar, 2019)¹ — `transcriptions-xml/` | 2,513 | Elicited words/phrases/sentences; corrected extraction now captures multi-word sentences at word level (previously truncated) |
| SIL "Aprendamos" materials (Lecciónes 01–05, 07–10, 12)² | 564 | Word-level tokens from sentence-level elicitation recordings |
| AILLA "I work with bees" item (Martínez López et al., 2022)³, from the broader "Documentation of Mixtepec Mixtec" collection⁴ — `MYUC-1042` | 269 | Continuous natural-speech recording (not elicited) orthographically transcribed & annotated by Salazar, Belmar Viernes, Campbell et al.; IPA transcriptions added by Jack Bowers |
| **Total (raw extraction)** | **3,346** | |
| **Total training tokens (resolvable audio, confirmed via dataset row count)** | **3,321** | 2,988 train / 333 dev (90/10 split) |

¹ Bowers, J., Salazar, J., & Salazar, T. (2019). *Mixtepec Mixtec Language Resources* (V4) [Data set]. Harvard Dataverse. https://doi.org/10.7910/DVN/BF2VNK
² Includes Lección 06 (stray duplicate of 05, excluded) and Lección 11 (unfinished, not yet transcribed) — neither contributes tokens.
³ Martínez López, G. (Speaker), Salazar, J. (Transcriber/Translator/Annotator), Belmar Viernes, G. (Transcriber/Annotator/Researcher/Editor), Aguilar, V. (Recorder/Interviewer), Salazar, C. (Interviewer/Interpreter), López Santiago, D. (Illustrator), & Campbell, E. (Transcriber). (2022). *I work with bees. Documentation of Mixtepec Mixtec* [Data set]. The Archive of the Indigenous Languages of Latin America (AILLA). PID Set 27418. https://www.ailla.utexas.org/sets/27418/ (Accessed 25 September 2026.)
⁴ Salazar, J., & Belmar Viernes, G. *Documentation of Mixtepec Mixtec* [Data collection]. The Archive of the Indigenous Languages of Latin America (AILLA). PID Collection 2123. https://www.ailla.utexas.org/ (Accessed 25 September 2026.)

**Note (added 2026-09-21):** the 818-token v1-era `transcriptions-xml/` figure above was extracted with a script since found to truncate multi-word utterances to their first word. ~441 of those 818 tokens were likely mismatched audio/text pairs rather than genuine single-word elicitations. This is corrected in the v2 pool above (2,513 tokens, word-level extraction of multi-word sentences).

**Note (2026-09-25):** the transcription added to the training set deriving from the "I work with bees" recording (`MYUC-1042`) differs from the rest of the corpus. The rest of the dataset is consistently segmented in the TEI/XML by complete sentence, phrase, or word, with each lexical item labeled as its own `<w>` element carrying its own time alignment. This content, however, was originally transcribed in larger chunks, and because it comes from natural, casual speech, it wasn't always segmented into complete sentences. When integrating it into the TEI/XML, I kept these chunks as-is, with time alignment coming only from the original, almost exclusively multi-word segment boundaries, rather than per-word. Structurally, this means these `<u>` elements carry no per-word `<w synch>` timing, only a single utterance-level span; `extract_finetune_data_unified.py` accordingly classifies them as `"whole-utterance"` rather than `"sentence"` type. This adds meaningful diversity to the training data: continuous, naturally paced speech rather than isolated elicited tokens.

Training/dev split (v1): 968 / 108 (90/10 random split, seed=42), performed on the combined 1,076-token pool.

## 2.2 Held-out test set

**Refreshed 2026-09-25.** The original 10-file set (2.2, pre-2026-09-24) was found to be substantially echo/reverb-contaminated. Six files were confirmed unsuitable for evaluation and moved into the training pool instead (echo degrades eval precision but not training robustness); the remaining 4 were kept, and 4 newly recorded, higher-quality files were added.

**Current set: 8 files, split into two evaluation tracks:**

| Track | Files | Tokens |
|---|---|---|
| Single-word | `ADJ_dangerous_01_JS`, `ADJ_dangerous_02_JS`, `ADJ_difficult_01_02_spkrTS`, `ADJ_fat_01_02_spkrTS`, + single-word `<u>`s embedded in the 4 new files | 8 |
| Phrase-level | Multi-word `<u>`s in `190630_0048-hand-cactus`, `190630_0051-na'nu-ka'nu`, `190710_0257-ate-eat-will-eat`, `190710_0270-tell-the-truth` | 11 |

**Files removed from test use (kept in training):** `ADJ_beautiful_anim_01_JS`, `ADJ_beautiful_inan_01_JS`, `ADJ_big_01_02_03_JS`, `ADJ_heavy_01_02_03_JS`, `ADJ_long_DIST_01_02_03_JS`, `ADJ_long_SHAPE_01_02_03_TS`.

**Known limitation:** n=8/n=11 is small; standard error on the single-word track is roughly ±17 points at current sample size. Treat these as preliminary, not benchmark-grade, until the set is expanded in the next fine tuning round (target ~20+ per track for a tighter estimate).

### 2.3 Training-target normalization

Gold IPA transcriptions in the source corpus follow several inconsistent notational conventions (see Section 5 and the accompanying "IPA Transcription Guidelines" document for full detail). Before use as training targets, all gold IPA strings were normalized via a fixed pipeline (`normalize_ipa.py`):

1. **Tone-stripping**: all tone/suprasegmental marks removed (Chao tone letters, contour arrows, indeterminate-tone marker, downstep, combining tone diacritics). Segmental features (nasalization, length, glottal stop, dental diacritics) explicitly preserved.
2. **Affricate normalization**: precomposed ligatures (ʧ, ʤ, ʦ, ʣ) converted to plain two-character sequences (tʃ, dʒ, ts, dz). No tie-bar insertion (see Section 6, decision reversal).
3. **Vowel-length normalization**: `Vː` (vowel + IPA length mark) converted to `VV` (doubled vowel letter), correctly handling nasalized/diacritic-bearing vowels (e.g. `ɛ̃ː` → `ɛ̃ɛ̃`, not `ɛɛ̃`).
4. **Creakiness normalization**: creaky-voice diacritic (U+0330) stripped when adjacent to `ʔ` (redundant coarticulatory effect); retained otherwise. Confirmed via direct consultation that creaky voice in this corpus occurs only adjacent to `/ʔ/`, never independently.

Tone is deliberately excluded from this training pass; tone modeling is scoped as a separate, later phase.

---

## 3. Method

### 3.1 Pipeline overview

1. **Extraction**: TEI/XML transcriptions parsed to extract token-level orthography, gold IPA, and precise start/end audio timestamps. Two extraction paths were required due to differing source-corpus structures (single-word-per-utterance vs. multi-word-per-utterance).
2. **Audio verification**: cross-referenced every token's expected audio filename against actual files on disk across multiple candidate directories, resolving naming inconsistencies (space vs. underscore separators, missing `<media>` references).
3. **Dataset construction**: audio cropped to each token's exact time span, resampled to 16kHz (Whisper's required input rate), packaged into a HuggingFace `datasets.Dataset` with the raw `audio` array and normalized `ipa` text column.
4. **Feature/label preparation**: WhIPA's own `prep_dataset()`/`prepare_dataset_ipa()` functions (from `scripts/whipa_utils.py`) applied directly, producing precomputed Whisper mel-spectrogram features and tokenized label sequences.
5. **LoRA fine-tuning**: base `whisper-large-v2` loaded in full precision (no quantization), special `<|ip|>` IPA-language token added and embeddings resized, LoRA adapter applied to decoder `q_proj`/`v_proj` modules (r=32, alpha=64, dropout=0.05 — identical to WhIPA's own published configuration), trained via HuggingFace `Seq2SeqTrainer`.

### 3.2 Hardware and training configuration — `lowhipa-mixtec-v1`

- **Hardware**: Apple Silicon M5 (MPS backend, no CUDA GPU)
- **Base model**: `openai/whisper-large-v2` (1,553,792,000 total parameters)
- **Trainable parameters**: 10,485,760 (0.6748% of total, via LoRA)
- **Epochs**: 3
- **Batch size**: 2 (per device)
- **Learning rate**: 1e-5
- **Eval/save interval**: every 242 steps
- **Total steps**: 1,452
- **Precision**: full (fp32); fp16 disabled due to inconsistent MPS support
- **Total training time**: 3 hours 13 minutes (11,610 seconds)

### 3.3 Training dynamics: `lowhipa-mixtec-v1`

| Step (approx.) | Train loss | Eval loss |
|---|---|---|
| Start (step ~1) | 7.119 | — |
| ~50% (epoch ~1.5) | ~1.0–1.3 | ~1.0 |
| End (step 1452, epoch 3) | 2.58 (running avg.) | 1.008 |

Loss decreased steadily and substantially across training (final eval loss of 1.008, down from an initial training loss of 7.119).

### 3.4 Training configuration: `lowhipa-mixtec-v2`

- **Hardware**: Apple Silicon M5 (MPS backend, no CUDA GPU)
- **Training pool**: 3,321 tokens (2,988 train / 333 dev, 90/10 split) — see Section 2.1
- **Epochs**: 5
- **Batch size**: 4 (per device) — confirmed by step-count math: 3,735 steps ÷ 5 epochs = 747 steps/epoch; 2,988 train examples ÷ 747 ≈ 4.0
- **Learning rate**: 1e-5 (script default; consistent with the observed decay curve)
- **Total steps**: 3,735
- **Final train loss**: 1.132
- **Final eval loss**: 0.5586
- **Total training time**: ~27 hours (97,620s reported `train_runtime`)
- **Precision**: full (fp32)

Final eval loss (0.5586) is noticeably lower than v1's (1.008), consistent with the larger, corrected training pool. See Section 3.5 for the full loss curve.


## 3.5 Training dynamics: `lowhipa-mixtec-v2`

| Step | Epoch | Train loss (nearby) | Eval loss |
|---|---|---|---|
| 1 | 0.00 | 5.18 | — |
| 242 | 0.32 | — | 3.7323 |
| 484 | 0.65 | — | 1.8906 |
| 726 | 0.97 | — | 1.1182 |
| 968 | 1.30 | — | 0.8944 |
| 1210 | 1.62 | — | 0.7741 |
| 1452 | 1.94 | — | 0.7116 |
| 1694 | 2.27 | — | 0.6753 |
| 1936 | 2.59 | — | 0.6449 |
| 2178 | 2.92 | — | 0.6177 |
| 2420 | 3.24 | — | 0.6017 |
| 2662 | 3.56 | — | 0.5868 |
| 2904 | 3.89 | — | 0.5753 |
| 3146 | 4.21 | — | 0.5640 |
| 3388 | 4.54 | — | 0.5614 |
| 3630 | 4.86 | — | 0.5590 |
| 3735 (end) | 5.00 | 1.132 (final running avg) | **0.5586** |

Eval loss decreases smoothly and monotonically throughout with no overfitting inflection, and the curve is already flattening by epoch 4, suggesting 5 epochs was close to sufficient for this pool size rather than leaving headroom on the table.

---

> **⚠️ Test-set quality addendum — superseded 2026-09-25:** the original 20-token
> held-out set (most of it echo/reverb-contaminated) has been replaced. Six
> confirmed-bad files were moved into the training pool; four clean files were
> kept and four new, higher-quality recordings were added, split into a
> single-word track (n=8) and a new phrase-level track (n=11). See Section 2.2
> and the current results in Section 4.

## 4. Evaluation results

Scored with WhIPA's own `STIPA_METRICS` (PER = Phone Error Rate, PFER = Phone Feature Error Rate). Current numbers use the refreshed test set (Section 2.2).

### Single-word track (n=8)

| | Zero-shot | `lowhipa-mixtec-v1` | `lowhipa-mixtec-v2` |
|---|---|---|---|
| Mean PER | 59.58% | 62.29% | **31.88%** |
| Mean PFER | 25.02% | 34.47% | **19.72%** |

### Phrase-level track (n=11)

| | Zero-shot | `lowhipa-mixtec-v1` | `lowhipa-mixtec-v2` |
|---|---|---|---|
| Mean PER | 58.21% | 64.82% | **15.65%** |
| Mean PFER | 19.06% | 53.10% | **3.97%** |

`lowhipa-mixtec-v1` underperforming zero-shot on both tracks is further evidence for the documented v1 training-data corruption bug (Section 2.1), not a sign fine-tuning itself is harmful. `lowhipa-mixtec-v2`, trained on the corrected/expanded pool, improves substantially over zero-shot on every metric, with the largest gains on phrase-level PFER.

**Note:** one single-word token (`naꞌnu`, a 0.47s clip) produced an identical degenerate one-character prediction across all three models, possibly likely due to clip length; included above for transparency but inflates the single-word track roughly equally across all three runs.

**Known limitation:** n=8/n=11 is a small test set (standard error on the single-word track is roughly ±17 points at this sample size). Treat these as preliminary; target ~20+ tokens per track in the next round for a tighter estimate.

Per-token results for all three runs are in `test_results/{zeroshot,v1,v2}/test_scores.csv`.

One evaluation-pipeline note for reproducibility: an earlier version of `score_test_results.py` used a hardcoded list of predictions from a single past run rather than reading `test_whipa.py`'s CSV output, which meant re-running it against a *different* checkpoint's predictions silently re-scored the same stale data every time. This was caught and fixed on 2026-09-24; all numbers above come from the corrected script.

> **Historical note (superseded 2026-09-25):** results below used the original 20-token test set later found to be substantially echo-contaminated. Kept for audit-trail purposes only — not for citation.
>
> | | Zero-shot | v1 | v2 |
> |---|---|---|---|
> | Mean PER | 69.1% | 44.5% | 29.6% |
> | Mean PFER | 27.6% | 22.1% | 9.5% |
---

## 8. References

Bowers, J., Salazar, J., & Salazar, T. (2019). *Mixtepec Mixtec Language Resources* (V4) [Data set]. Harvard Dataverse. https://doi.org/10.7910/DVN/BF2VNK

Martínez López, G. (Speaker), Salazar, J. (Transcriber/Translator/Annotator), Belmar Viernes, G. (Transcriber/Annotator/Researcher/Editor), Aguilar, V. (Recorder/Interviewer), Salazar, C. (Interviewer/Interpreter), López Santiago, D. (Illustrator), & Campbell, E. (Transcriber). (2022). *I work with bees. Documentation of Mixtepec Mixtec* [Data set]. The Archive of the Indigenous Languages of Latin America (AILLA). PID Set 27418. https://www.ailla.utexas.org/sets/27418/

Suchardt, J. L., El-Shazli, H., & Cassotti, P. (2025). Towards Language-Agnostic STIPA: Universal Phonetic Transcription to Support Language Documentation at Scale. In *Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing*, pages 31411–31427, Suzhou, China. Association for Computational Linguistics. https://doi.org/10.18653/v1/2025.emnlp-main.1600