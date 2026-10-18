# AILLA / Researcher-Archive Source Processing Pipeline

**Purpose**: documents the preprocessing pipeline used to bring a raw AILLA
(or similar researcher-archive) source — a space-aligned `.txt` transcription
export, not yet TEI/XML — into this project's corpus. This pipeline is
**upstream of and separate from** the fine-tuning extraction pipeline
(`extract_finetune_data_unified.py`, etc.): it produces a human-reviewed
draft, then TEI/XML, not training data directly. A source only enters
`extract_finetune_data_unified.py`'s scope once it exists as TEI/XML.

Last updated: 2026-10-18

---

## When to use this

Use this pipeline when integrating a new source that arrives as a raw
space-aligned transcription `.txt` export (the format researchers on this
project use when exporting from ELAN), rather than as TEI/XML. As of this
writing, two AILLA items have gone through it: `MYUC-1042` ("I work with
bees") and `MYUC-1038` ("How the ancestors used to celebrate").

If a source already exists as TEI/XML (e.g. anything from
`transcriptions-xml/` or the SIL "Aprendamos" materials), this pipeline does
not apply — those go through the existing Praat/XSLT conversion path
(`praat2tei-Mex2019-claude.xsl` / `praat2tei-sil-claude.xsl`) instead.

---

## Pipeline overview

```
raw .txt (ELAN export)
      |
      v
parse_researcher_archive.py  --speaker <code[,code...]>
      |
      v
<name>_parsed.json   (intermediate: orth/eng/spn/notes + start/end seconds)
      |
      v
orth_to_ipa.py
      |
      v
<name>_draft_ipa_review.csv   (draft_ipa + review flags)
      |
      v
MANUAL REVIEW (Jack) -- audit by ear, fill final_ipa, delete non-target-language rows
      |
      v
<name>_final review CSV  (draft_ipa column finalized as final_ipa)
      |
      v
myuc_to_tei.py  <csv> <contributors.txt> [output.xml] [wav]
      |
      v
<name>.xml   (TEI/XML, ready for the rest of the corpus pipeline)
```

Scripts: `parse_researcher_archive.py` and `orth_to_ipa.py` live at the top
level of `~/Code/Projects/whipa/`. `myuc_to_tei.py` was not in the repo as of
2026-10-18 (never committed; see Open items) — commit it alongside the others.

Transcription conventions applied during manual review are documented in
`IPA_Transcription_Guidelines.md` (esp. §9, and §6 for pending items).

---

## Step 1: `parse_researcher_archive.py`

Parses the space-aligned transcription format into a clean intermediate
JSON, ready for IPA conversion.

**Format assumed** (see the script's own docstring for full detail):
- First 2 lines: source file path + export date/time (metadata, skipped).
- Blocks separated by blank lines, each block a set of
  `TierType@SpeakerCode<spaces>Content` lines. Recognized tier types:
  `Transcription`, `Translation-Eng`, `Translation-Spn`, `Notes` (optional),
  and `TC` (timecode range, no `@speaker` suffix — applies to the whole
  block).
- Timecodes in `HH:MM:SS.mmm - HH:MM:SS.mmm` format, converted to seconds.

**Usage:**
```
python3 parse_researcher_archive.py <source>.txt --speaker <code[,code...]> [--output <name>_parsed.json]
```

**Filtering — by speaker code only, not by language:**
`--speaker` takes a comma-separated list of speaker codes to keep; every
block whose code isn't in that list is discarded. This was originally
single-speaker only (built for MYUC-1042, one main speaker); it was extended
2026-09-28 for MYUC-1038, which splits Mixtec content across multiple tiers
(the primary speaker plus one or more interviewers who also speak Mixtec).

**Important**: this filters by speaker code, not by language. A kept
speaker's blocks are kept whether the content is Mixtec or Spanish (e.g. an
interviewer's tier that switches to Spanish during the consent/setup
section at the start of a recording still ends up in the output). This is
deliberate — same "flag/leave for review rather than silently guess"
convention used everywhere else in this project (cf.
`IPA_Transcription_Guidelines.md`'s "do not guess" notes, `orth_to_ipa.py`'s
unmapped-char flags). Spanish code-switched rows are easy to spot in the
output CSV (they get piled up with irrelevant `unmapped char` flags for
letters that never occur in Mixtec orthography, e.g. `b`, `d`, `q`) and
should be deleted manually during review, not auto-filtered.

**Example — MYUC-1038** (Mixtec content split across the main speaker HVL
plus two interviewer-side tiers, `Interviewer` and `Sp2`; a third tier,
`Interviewer2`, was confirmed Spanish-only and excluded entirely):
```
python3 parse_researcher_archive.py MYUC-1038.txt --speaker "Interviewer,Sp2,HVL" --output myuc1038_parsed.json
```

---

## Step 2: `orth_to_ipa.py`

Rule-based orthography → draft IPA converter (see the script's own
docstring for the full list of confirmed rules and what's deliberately left
as a tentative/flagged default, e.g. `t` → `t̪`).

**Usage:**
```
python3 orth_to_ipa.py <name>_parsed.json [--output <name>_draft_ipa_review.csv]
```
If `--output` is omitted, the output filename is derived from the input
(`<name>_parsed.json` → `<name>_draft_ipa_review.csv`). Generalized
2026-09-28 from a version hardcoded to `myuc1042_parsed.json` /
`myuc1042_draft_ipa_review.csv`.

Output columns: `start,end,orth,draft_ipa,eng,spn,notes,flags`. This is a
**draft for review, not gold IPA** — every row should be checked before
being treated as a training target, same as `finetune_review_unified.csv`
elsewhere in this project.

**Whole-word exceptions (`WORD_EXCEPTIONS`)**: checked first, on an exact
whole-word match, before any general rule. Current entries (2026-10-18):
`kuê` → `wɛ̂` (k fully elided), `nùǔ` → `nũ̀ũ̌` ("face"; always nasalized,
tones/length usually reduced, set as the initial form to modify on analysis
of the actual utterance), `ka` → `k̬a` (whole word only; partially voiced k —
not applied to `ka` inside longer words). See `IPA_Transcription_Guidelines.md`
§9d.

**Known limits — handle before running a new source:**
- **Hyphens are NOT yet disambiguated.** The script treats every hyphen as a
  clitic boundary (convert each piece, join with a space). That is wrong for
  bound-morpheme markers written with hyphens by the UCSB/other-group
  convention (e.g. `ìǐ-à`, `-i`), which are phonologically fused and should
  be joined, not spaced; reducing enclitics such as `-ña` (→ [a]) are a third
  case. Resolve the morpheme/enclitic rule (see `IPA_Transcription_Guidelines.md`
  §9e) **before** running any new source that contains hyphenated forms.
- **`r` is deliberately unmapped.** Orthographic `r` is realized as either
  trill [r] or tap [ɾ], decided case-by-case by ear, so it has no entry in
  `CONSONANT_MAP` and shows up as an `unmapped char` flag to be filled in
  during review (§9b).
- **Cross-word coarticulation is not modeled.** Rules like glide formation
  across a word boundary (lexical /kú ìǐ ɲà/ realized [kwìǐã]) are written by
  hand during review, as one continuous string with no internal space (§9c).

---

## Step 3: manual review (by ear)

The draft CSV is audited row by row, and the finalized IPA goes in a
`final_ipa` column (this is the column name `myuc_to_tei.py` reads; the
`draft_ipa` column from Step 2 is not read). Optional extra column
`Notes (Jack)` carries Jack's own notes into the TEI. Conventions to apply
(full detail and rationale in `IPA_Transcription_Guidelines.md` §9 and §6):

- Transcribe what is actually heard, not what the orthography says.
- Spacing in the IPA follows real phonetic breaks, not lexical ones: no space
  for non-lexical pauses or cross-word coarticulation (§9c).
- Elongated vowel with rising/falling tone from a transitional pause: global
  ↗/↘, original diacritic dropped (`tsíǒ` → `[tsíooo↗]`) (§9a).
- Never trim the `orth` column to match a trimmed `final_ipa`. If an unclear
  leading/trailing word is dropped from the IPA, keep the full orth and name
  the dropped word(s) in the notes (§9f).
- A stray non-lexical sound between two words (e.g. an isolated `[s]`): no
  mechanism yet (§6). Current policy is to skip the utterance, or keep only
  the clean preceding or following portion.
- Delete non-target-language rows (e.g. Spanish consent/setup speech).

---

## Step 4: `myuc_to_tei.py`

Builds the final TEI/XML from the audited review CSV plus a
`contributors.txt`. Written for MYUC-1042; **the header template is
file-specific and must be edited for each new source** (see below).

**Usage:**
```
python3 myuc_to_tei.py <final_review.csv> <contributors.txt> [output.xml] [wav_file]
```
The last two arguments default to `MYUC-1042.xml` / `MYUC-1042.wav`; always
pass your own for a new source so the 1042 output isn't overwritten.

**Input CSV columns read**: `start`, `end`, `orth`, `final_ipa`, `eng`, `spn`,
`notes` (UCSB team's notes), `Notes (Jack)`. `flags` and `draft_ipa` are
ignored. Timestamps are re-normalized to exactly 3 decimal places.

**Output structure** (per row): `<u n xml:id="u-{start}" start end>` containing
`<seg xml:lang="mix" notation="orth-ucsb">` (original orthography),
`<seg xml:lang="mix" notation="ipa">` (finalized IPA — this exact `notation`
value is how the extraction scripts find the IPA tier, so don't change it),
and a `<spanGrp type="annotations">` with English/Spanish translations and
notes (UCSB notes `resp="#JS #GVB #EC"`, Jack's notes `resp="#JB"`).

**Editing the file-specific header** — `build_header()`:
- `<title>` lines (generic, English, Mixtec) in `titleStmt`, and the three
  `bibl` titles (English, Spanish, Mixtec)
- AILLA id (`<idno type="AILLA">`) and the `<ptr target>` set URL
- publication `<date>` and recording/creation date
- `placeName` block (town, municipality, region, country)
- `keywords` (genre, content type)
- the licence line in `publicationStmt`

**Contributors**: one line per person in `contributors.txt`, formatted
`name<TAB>role1, role2`; `build_resp_stmts()` turns each into a `<respStmt>`.
Jack's own `JB` respStmt is hardcoded in `titleStmt`. To attribute a
contributor's notes via `@resp`, add their name and initials to the
`RESP_INITIALS` dict at the top of the script (currently JS, GVB, EC, JB).

**Caution — two versions exist.** A variant that adds tokenized `<w>`
elements to the orth and IPA segs was iterated on but is not finished: it
calls `orth.split(r"[\s\-]+", orth.split())` where `re.split(r"[\s\-]+", orth)`
is intended, and has `type="untokenized notation="orth-ucsb"` with a missing
closing quote (malformed XML). Use the non-tokenized version, or fix both
before using the tokenized one.

---

## The intermediate `<name>_parsed.json`

Fully regenerable from the source `.txt` by re-running Step 1 — it carries
no information the `.txt` doesn't already have. Not required to be kept
once a file's IPA review is finished and it's been converted to TEI/XML;
useful to keep only as a shortcut while actively iterating on IPA review
(lets you re-run Step 2 without re-parsing). Not intended to be committed
long-term.

---

## Open items

- **`myuc_to_tei.py` is not in the repo** (never committed as of 2026-10-18),
  and none of the three MYUC scripts were documented before this file. Commit
  it, and decide which of its two variants is canonical.
- **Hyphen disambiguation** (morpheme vs. enclitic vs. reducing enclitic) must
  be resolved before the next source containing hyphenated forms goes through
  `orth_to_ipa.py` — see Step 2 and `IPA_Transcription_Guidelines.md` §9e.
- **Header template is hand-edited per source.** If more sources go through
  Step 4, consider moving the file-specific header fields (titles, AILLA id,
  dates, place, keywords) out of `build_header()` into an input file, the way
  contributors already are. Also generalize the `MYUC-1042` defaults.
- **No automated language detection.** Code-switched Spanish rows inside an
  otherwise-Mixtec speaker tier are left in the output for manual deletion
  (see Step 1 above) rather than heuristically filtered — deliberate, not an
  oversight.
- **Speaker identity is not resolved at Steps 1–2.** `parse_researcher_archive.py`
  keeps whatever speaker codes it's told to, without checking those codes
  against the source's AILLA citation metadata (transcriber/interviewer
  names). That reconciliation happens at TEI conversion (Step 4, via
  `contributors.txt` and `RESP_INITIALS`).
