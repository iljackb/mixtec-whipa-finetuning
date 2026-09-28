# AILLA / Researcher-Archive Source Processing Pipeline

**Purpose**: documents the preprocessing pipeline used to bring a raw AILLA
(or similar researcher-archive) source — a space-aligned `.txt` transcription
export, not yet TEI/XML — into this project's corpus. This pipeline is
**upstream of and separate from** the fine-tuning extraction pipeline
(`extract_finetune_data_unified.py`, etc.): it produces a human-reviewed
draft, not training data directly. A source only enters
`extract_finetune_data_unified.py`'s scope once it exists as TEI/XML.

Last updated: 2026-09-28

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
MANUAL REVIEW (Jack) -- fix/verify draft_ipa, delete non-target-language rows
      |
      v
[not yet scripted] -- integration into TEI/XML
```

Both scripts live at the top level of `~/Code/Projects/whipa/`.

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

- **No scripted TEI/XML integration step yet.** The draft CSV → TEI
  conversion for these sources is still manual. If this pipeline gets used
  for more sources, that's the next piece worth scripting.
- **No automated language detection.** Code-switched Spanish rows inside an
  otherwise-Mixtec speaker tier are left in the output for manual deletion
  (see Step 1 above) rather than heuristically filtered — deliberate, not an
  oversight.
- **Speaker identity is not resolved at this stage.** `parse_researcher_archive.py`
  keeps whatever speaker codes it's told to, without checking those codes
  against the source's AILLA citation metadata (transcriber/interviewer
  names). That reconciliation happens later, at TEI conversion.
