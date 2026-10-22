# WhIPA Fine-Tuning on Mixtepec Mixtec (Sà'án Sàvǐ): Preliminary Results Report (v3)

**Author:** Jack Bowers

**Date:** 2026-10-22

**Model:** LoRA fine-tune of `openai/whisper-large-v2`, using the WhIPA/LoWhIPA framework (Suchardt et al., 2025 EMNLP; code: github.com/jshrdt/whipa)

**Checkpoint identifiers:**
- `lowhipa-mixtec-v1` (2026/09/15)
- `lowhipa-mixtec-v2` (2026/09/23)
- `lowhipa-mixtec-v3` (2026/10/22)

**Supersedes:** `Preliminary_Results_Report-v2-20260923.md` (to be moved to `docs/archive/`).

---

## 1. Objective and status

The objective is unchanged from the v2 report: evaluate whether LoRA fine-tuning of a Whisper-based speech-to-IPA (STIPA) model on a small corpus of Mixtepec Mixtec phonetic transcriptions improves automatic phonetic transcription relative to the same base model's zero-shot performance, with the longer-term aim of a human-in-the-loop transcription workflow for the backlog of untranscribed recordings. Neither Mixtepec Mixtec nor any other Otomanguean language appears in WhIPA's original training data.

### 1.1 Status overview

- `lowhipa-mixtec-v3` was trained on 3,629 tokens (v2: 3,321), the first run to include the newly integrated field-recording material.
- On the 19-token held-out set, v3 and v2 are **comparable**. v3 is nominally ahead on PER (single-word 23.8% vs 24.4%; phrase 12.2% vs 15.6%) and level on PFER (single-word 9.4% vs 9.8%; phrase 4.2% vs 4.0%). The difference comes from two tokens and is within what a test set this small can produce (Section 4.3).
- Both fine-tuned models are far ahead of zero-shot: v3 reduces PER by 60.1% (single-word) and 79.0% (phrase) and PFER by 48.1% and 78.0% relative to zero-shot.
- **All results in this report were regenerated after a change to the decoding rate limiter** (Section 4.1). Numbers in the v2 report are not directly comparable to the numbers here; the old figures are kept in Section 4.2 for audit.
- Three remaining error sources are not fixed by more training data of the same kind: nasalization and vowel length that differ between the gold and the audio, symbols the scorer has no feature values for, and the size of the test set (Section 5).

---

## 2. Data

### 2.1 Training corpus (v3)

- **Total training tokens (resolvable audio, confirmed via dataset row count): 3,629**, split **3,266 train / 363 dev** (90/10 random split by utterance, seed 42).
- Net change from v2: +308 tokens (3,321 → 3,629).
- Sources are those of v2 (Mixtepec Mixtec Language Resources `transcriptions-xml/`; SIL "Aprendamos" Lecciones 01–05, 07–10, 12; AILLA "I work with bees" item, `MYUC-1042`) plus newly integrated field-recording material (including `MYUC-1038`, which was not in the v2 pool). By the author's estimate, field recordings now make up the majority of training tokens.
- `MYUC-1042` was re-processed for this run: utterance start times that produced zero-length or negative-length segments were corrected, and its audio file, which was an MP3 stored with a `.wav` extension, was replaced with a true WAV. Rows previously lost to these defects are now included.
- **[TODO: add per-source token counts, as in the v2 report's Section 2.1 table.]**

The `MYUC` field recordings differ structurally from the elicited material (continuous natural speech, utterance-level rather than per-word alignment); see the v2 report's note of 2026-09-25, which still applies.

**Prep-step incident (resolved).** A first v3 attempt was trained on a stale prepared dataset of 3,321 examples because `run_prep_dataset.py` had not been rerun after the rebuild. That run was deleted. The version name `v3` was reused for the correct retrain, so version numbers stay sequential with no gaps. The checks now run before every long job are: raw row count, prepared row count (3,629), the train/dev counts printed by the trainer (3,266 / 363), and the total step count (4,085).

### 2.2 Held-out test set

Unchanged from the v2 report: **19 tokens** from 8 files, in two tracks.

| Track | Files | Tokens |
|---|---|---|
| Single-word | `ADJ_dangerous_01_JS`, `ADJ_dangerous_02_JS`, `ADJ_difficult_01_02_spkrTS`, `ADJ_fat_01_02_spkrTS`, + single-word `<u>`s embedded in the 4 field files | 8 |
| Phrase-level | Multi-word `<u>`s in `190630_0048-hand-cactus`, `190630_0051-na'nu-ka'nu`, `190710_0257-ate-eat-will-eat`, `190710_0270-tell-the-truth` | 11 |

The test files are kept out of training by being in a separate directory (`test_data/`) that the training scripts never read; the file list is recorded in `test_holdout.txt`. **None of the new v3 material was held out**, so this test set is the old one and none of the new field-recording utterances are in it. It also does not resemble the training mix: training is now mostly field-recording utterances, while the test set is mostly short elicitation files.

### 2.3 Training-target normalization

Unchanged (`normalize_ipa.py`): tone-stripping, affricate normalization (no tie-bar insertion at this stage), vowel-length normalization (`Vː` → `VV`, correct for nasalized vowels, e.g. `ɛ̃ː` → `ɛ̃ɛ̃`), and creakiness normalization (creaky-voice diacritic removed adjacent to `ʔ`). Segmental features, including nasalization and length, are preserved in the targets. **Tone is excluded from this training pass**; the plan is to add tonal targets in a later run via modified extraction and normalization, and to score accuracy both with and without tones.

---

## 3. Method

### 3.1 Pipeline

As in the v2 report, with the pipeline now documented step by step in `README.md`: `classify_tei_structure.py` → `dedupe_xml_ids.py` (only on parse errors) → `extract_finetune_data_unified.py` → `verify_audio_paths.py` → `build_finetune_dataset.py` → `run_prep_dataset.py` → `train_lora_mps.py` → `summarize_training.py` → `test_whipa.py` → `score_test_results.py`. WhIPA's own `prep_dataset()` is applied unchanged. LoRA configuration is identical to v2 (r=32, alpha=64, dropout 0.05, decoder `q_proj`/`v_proj`, encoder frozen, `<|ip|>` token added).

### 3.2 Training configuration: `lowhipa-mixtec-v3`

- **Hardware**: Apple Silicon M5 (MPS backend, no CUDA GPU)
- **Training pool**: 3,629 tokens (3,266 train / 363 dev)
- **Epochs**: 5
- **Batch size**: 4 (per device)
- **Learning rate**: 1e-5
- **Warmup**: 10 steps
- **Total steps**: 4,085 (817 steps per epoch)
- **Eval/save interval**: every 242 steps; `load_best_model_at_end=True`, no early stopping
- **Final eval loss**: 0.6516
- **Final train loss**: **[TODO: from `summarize_training.py` output]**
- **Total training time**: 36 hours 2 minutes
- **Precision**: full (fp32); gradient checkpointing enabled

The Hugging Face trainer state, summarized by `summarize_training.py` (`python3 summarize_training.py ../models/lowhipa-mixtec-v3 --md v3_dynamics.md`), is in the model directory's `training_summary.md`.

### 3.3 Training dynamics: `lowhipa-mixtec-v3`

**[TODO: paste the Step | Epoch | Train loss | Eval loss table and the automatic-checks lines from `training_summary.md`.]**

**Eval loss is not comparable between v2 and v3.** v2's final eval loss (0.5586) was measured on a different dev split (333 examples) than v3's (0.6516, 363 examples that include the new field-recording utterances). Dev loss also measures optimistically here: the 90/10 split in `train_lora_mps.py` is a random split by utterance (`dataset.train_test_split(test_size=0.1, seed=42)`), so utterances from the same recording appear on both sides. It is useful for watching a single run converge, not for comparing runs or for judging generalization. Model comparison in this report uses only PER and PFER on the identical held-out tokens.

---

## 4. Evaluation results

Scored with WhIPA's own `STIPA_METRICS` (PER = Phone Error Rate, PFER = Phone Feature Error Rate; PFER is the more phonologically informative of the two because it credits near-misses in voicing, place and manner).

### 4.1 Change to the decoding rate limiter

WhIPA's `deploy.py` guards against runaway or looping output by capping the number of predicted phones per clip at 20 phones per second of audio. As shipped, the cap is computed as `max(round(seconds) * 20.0, 1)`. Because `round()` rounds the clip length to whole seconds, any clip under 0.5 s counts as 0 seconds and receives a cap of **1 phone**. A prediction over the cap triggers a fallback chain (other beam widths, a length penalty) and, as a last resort, truncates the output string to the cap (`[:rate_limit]`).

This affected exactly one test token, `190630_0051-na'nu-ka'nu`, token 1 (0.47 s, gold `nãʔnũ`). In every model the first prediction exceeded the 1-phone cap and was replaced by the fallback with `n`, giving 80.0% PER on every model (the v2 report noted this identical one-character prediction and suggested clip length as the cause). With the cap corrected the token scores 80.0% (zero-shot), 60.0% (v1), 20.0% (v2) and 40.0% (v3) PER, so the artifact inflated the single-word track mean by between 0 and 7.5 points depending on the model.

**Fix (2026-10-22), a deviation from the published WhIPA code:** the cap now uses the exact clip duration and a floor of 5 phones:

```python
rate_limit = max(int(len(sample["audio"]["array"]) / 16_000 * max_phones_per_sec_rate), 5)
```

For `na'nu` this gives a cap of 9 phones (0.47 s), and the fallback no longer fires. The fix is inference-only: **no model was retrained.** All four models were re-run through `test_whipa.py` on the same 19 tokens with otherwise identical arguments and rescored. Every other prediction was unchanged; only the `na'nu` token differs. The 5-phone floor is a judgment call, not a tuned value. Other changes to `deploy.py`: a one-line fix of a variable reference (`whipa.model.device` → `self.model.device`) made earlier.

Results from before the fix are kept in `test_results/archive-old-ratelimit/`; the rescored results are `test_results/{zeroshot,v1,v2,v3}/`. In `test_scores.csv`, rescored runs carry a `-ratefix` suffix in the model name.

### 4.2 Results (corrected limiter)

**Single-word track (n=8)**

| | Zero-shot | `v1` | `v2` | `v3` |
|---|---|---|---|---|
| Mean PER | 59.6% | 59.8% | 24.4% | **23.8%** |
| Mean PFER | 18.1% | 27.4% | 9.8% | **9.4%** |

**Phrase-level track (n=11)**

| | Zero-shot | `v1` | `v2` | `v3` |
|---|---|---|---|---|
| Mean PER | 58.2% | 64.8% | 15.6% | **12.2%** |
| Mean PFER | 19.1% | 53.1% | **4.0%** | 4.2% |

**Relative reduction vs. zero-shot**

| | v2 | v3 |
|---|---|---|
| Single-word PER | 59.1% | 60.1% |
| Single-word PFER | 45.9% | 48.1% |
| Phrase PER | 73.2% | 79.0% |
| Phrase PFER | 79.1% | 78.0% |

`v1` did not improve on zero-shot (the v1 training pool was corrupted by an extraction bug, as documented in the v2 report), so it is shown for the record only.

**Effect of the limiter fix (audit trail).** Only the single-word track changed:

| Single-word track | Zero-shot | `v1` | `v2` | `v3` |
|---|---|---|---|---|
| PER, before fix | 59.58% | 62.29% | 31.88% | 28.8% |
| PER, after fix | 59.6% | 59.8% | 24.4% | 23.8% |
| PFER, before fix | 25.02% | 34.47% | 19.72% | 19.2% |
| PFER, after fix | 18.1% | 27.4% | 9.8% | 9.4% |

Phrase-track figures are unchanged by the fix. The pre-fix `v3` figures were computed in this round from the same predictions (single-word 28.8 / 19.2, phrase 12.2 / 4.2).

### 4.3 v2 vs. v3, token by token

Of 19 tokens, 13 score identically in both PER and PFER. The tokens that differ:

| Token | v2 PER / PFER | v3 PER / PFER |
|---|---|---|
| `190710_0270-tell-the-truth` token 2 | 30.8 / 10.6 | **0.0 / 0.0** |
| `190710_0270-tell-the-truth` token 3 | 13.3 / 2.2 | 6.7 / 6.7 |
| `190710_0270-tell-the-truth` token 4 | 9.1 / 0.4 | 9.1 / 9.1 |
| `ADJ_fat_01_02_spkrTS` token 2 | 25.0 / 4.2 | **0.0 / 0.0** |
| `190630_0051-na'nu-ka'nu` token 1 | **20.0 / 0.8** | 40.0 / 1.7 |
| `190710_0257-ate-eat-will-eat` token 2 | 9.1 / 1.1 | 9.1 / 0.9 |

v3's phrase-track PER lead comes from `tell-the-truth` tokens 2 and 3; its single-word lead comes from `fat` token 2, partly offset by `na'nu`. Its PFER is slightly worse on `tell-the-truth` tokens 3 and 4, where its remaining errors are a space inside a word (`kunda kani`), a shortened vowel (`nda` for `ndaa`) and an inserted glottal stop (`ndaʔi`). v2 over-predicted nasals in this utterance (`nĩndai`, `nindãʔi`, `kũːnda`) where the gold has none, and v3 stopped doing so; but v3 also dropped the first nasal of `na'nu` (`nãʔnu` in v2, `naʔnu` in v3, gold `nãʔnũ`). **The accurate summary is that v2 and v3 are comparable on this set**, not that v3 is better.

---

## 5. Analysis and limitations

### 5.1 What the remaining errors are

The hardest tokens are identical across v2 and v3 and are not helped by more data of the same kind:

- **`ADJ_dangerous_01_JS` / `_02_JS` (`xeen`), 66.7% PER / 36.1% PFER each.** Gold `ʃɛ̃ɛ̃`; both models predict `ʃɛː`. These two tokens, which are the same word, contribute about 17 of v3's 23.8 points of single-word PER. Scored with nasalization and length collapsed, they would be near 0.
- **`190710_0257-ate-eat-will-eat` tokens 5 and 6** (`kũk̬at͡si na staa`): 41.7 / 17.7 and 25.0 / 9.0. Both involve the voiced-diacritic symbol (Section 5.3).
- **`190630_0048-hand-cactus`** (16.7 / 1.4): the model emits a stray combining cedilla (`tuşit͡ʃi` for `tuʃit͡ʃi`).

### 5.2 Nasalization and length

Two observations, both from the author's own listening, which should be treated as open until the checks below are done:

1. **One speaker [TODO: name the speaker/recordings] frequently does not nasalize words that are normally nasalized, and does not articulate normally long vowels as long.** Where the gold marks the expected nasal or length, the model, which follows the audio, disagrees. This adds variance to both training targets and test scoring: the same word can have a nasal/long target in one place and a plain one in another.
2. **`xee`.** The standard orthography writes `xee` without a nasal. The test speaker (`_JS`) nasalizes it, which is why the test gold, transcribed by ear, has `ʃɛ̃ɛ̃` (the test file lists the orthography as `xeen`). The author re-listened on 2026-10-22 and confirmed the nasal is present in the test audio. The zero-shot model also produced a nasal here (`ʂɛ̃` for `_01`), so the base model can hear it; the fine-tuned models do not predict it.

As the author notes, Mixtec speakers do not use the orthography as a pronunciation guide (it is hardly used in written form at all), so a spelling without a nasal says nothing about whether speakers nasalize. The question for the training data is therefore not what the orthography says but how the training targets were produced. `orth_to_ipa.py` generates a draft IPA from the written form, which contains a nasal only where the spelling does; the draft is then audited by ear. **Open check:** whether that audit covered nasalization consistently for the field-recording material. A quick way to test it: count training rows whose target contains a combining tilde, and listen to a sample of training clips of `xee` (`grep -i "xee" finetune_review_unified.csv`). **[TODO: record the outcome.]** If the training audio is nasal but the targets are not, the model learned the lexical spelling rather than the acoustics, and a nasal audit of the orthography-derived rows is the right fix. If the training speakers do not nasalize it, the model reflects those speakers, and generalization to a speaker who does is a speaker-coverage limitation.

A scoring variant that collapses nasalization and length (nasal tildes removed, `ː` and doubled vowels treated alike) would separate how much of the remaining error is this kind of convention mismatch. It has not been implemented; reporting the strict and collapsed scores together is planned.

### 5.3 Scorer: symbols without feature values

`score_test_results.py` prints warnings for symbols it has no feature values for: the combining cedilla (U+0327), produced by v2 and v3 on `hand-cactus` (`tuşit͡ʃi`), and the voiced diacritic (U+032C), present in the gold `k̬` of `ate-eat-will-eat` tokens 5 and 6. How the scorer treats these symbols in PFER has not been verified, so PFER on the affected tokens (notably `ate-eat-will-eat` tokens 5 and 6) may be distorted.

### 5.4 Test-set size and composition

n = 8 single-word and n = 11 phrase tokens. One token moves a track mean by several points (the `na'nu` limiter artifact alone moved the single-word mean by about 5 points), and the single-word standard error was estimated at roughly ±17 points in the v2 report. Differences of a point or two between v2 and v3 should not be interpreted. The set also does not match the training mix (Section 2.2). The phrase track (field recordings of the main speakers) is closer to the intended use; the single-word track (elicitation files, other speakers, 96 kHz recordings) probably measures a different condition. The single-word-vs-phrase gap is mostly arithmetic and one word: PER divides by length, so one error in a three-phone word costs 33% but only about 5% in a twenty-phone phrase, and the repeated `xeen` token supplies most of the single-word error.

### 5.5 Dev set

See Section 3.3: the random utterance-level split makes dev loss optimistic and not comparable between runs.

---

## 6. Next steps

1. **Hold out a proper test set before the next dataset build.** Move whole recordings (not individual utterances) into the test directory *before* running extraction and build. Target about 100 or more tokens, mostly utterances from field recordings with a smaller single-word track, covering several speakers including the one who does not nasalize or lengthen, with scores reported per speaker. Keep the file list in `test_holdout.txt`.
2. **Group the dev split by recording** in `train_lora_mps.py` (needs a recording-ID column in the prepared dataset) if dev loss is going to be used to choose between runs.
3. **Resolve the nasalization check** (Section 5.2) and, if the targets are missing nasals the audio has, audit them. This is the leading candidate for the single change in v4: corrected data with the new held-out set, everything else unchanged.
4. **Add a collapsed nasal/length scoring variant** and decide how the scorer should treat symbols without feature values.
5. **Later, one change at a time:** tonal targets (new `keep_tones` option and an `ipa_tonal_normalized` column, scored with and without tones); a learning-rate experiment (1e-4 or 5e-4, vs. the published LoRA setting of 1e-3), as a separate run so the effect can be attributed.
6. Back up `models/lowhipa-mixtec-v3/`; the intermediate `checkpoint-*` folders other than `checkpoint-4085` can be removed to save space.

---

## 7. References

Bowers, J., Salazar, J., & Salazar, T. (2019). *Mixtepec Mixtec Language Resources* (V4) [Data set]. Harvard Dataverse. https://doi.org/10.7910/DVN/BF2VNK

Martínez López, G. (Speaker), Salazar, J. (Transcriber/Translator/Annotator), Belmar Viernes, G. (Transcriber/Annotator/Researcher/Editor), Aguilar, V. (Recorder/Interviewer), Salazar, C. (Interviewer/Interpreter), López Santiago, D. (Illustrator), & Campbell, E. (Transcriber). (2022). *I work with bees. Documentation of Mixtepec Mixtec* [Data set]. The Archive of the Indigenous Languages of Latin America (AILLA). PID Set 27418. https://www.ailla.utexas.org/sets/27418/

Suchardt, J. L., El-Shazli, H., & Cassotti, P. (2025). Towards Language-Agnostic STIPA: Universal Phonetic Transcription to Support Language Documentation at Scale. In *Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing*, pages 31411–31427, Suzhou, China. Association for Computational Linguistics. https://doi.org/10.18653/v1/2025.emnlp-main.1600
