"""
Consolidated IPA normalization for WhIPA fine-tuning training targets.

Combines all normalization rules into one module, so both extraction scripts
apply identical, consistent normalization. Operates ONLY on derived training-
target strings -- never touches archival TEI source files.

Rules implemented (see IPA_Transcription_Guidelines.md for full rationale):
  1. Whitespace collapsing -- strips embedded newlines/indentation artifacts
     from multi-line XML source (e.g. a <w> split across lines via multiple
     <m> children), collapses any internal whitespace run to a single space.
  2. Tone-stripping (strip_tones) -- already existed, included here for
     completeness.
  3. Affricate tie-bar standardization -- tʃ/ʧ -> t͡ʃ, dʒ/ʤ -> d͡ʒ, ts -> t͡s
     (confirmed via corpus scan: 84x tʃ, 32x dʒ, 91x ts, 6x ʧ, 17x ʤ; 0 tɕ/dʑ/etc found)
  4. General vowel-length normalization -- Vː -> VV (doubles the vowel+diacritic
     cluster, e.g. nasalized ɛ̃ː -> ɛ̃ɛ̃, not just the bare vowel)
  5. Dental diacritic removal -- not phonologically contrastive in this language.
  6. Rare incidental marks removal -- one-off marks confirmed non-systematic.
  7. Creakiness heuristic -- strips creaky-voice diacritic (U+0330) when adjacent
     to ʔ (redundant, non-phonological per VʔV coarticulation); keeps it when NOT
     adjacent to ʔ (informative -- likely marking a reduced/deleted glottal stop)
  8. Aspiration removal -- strips superscript ʰ (U+02B0); not phonologically
     significant for this corpus's training targets.
  9. Prenasal normalization -- ⁿ (U+207F superscript n) -> plain 'n', per
     decision to represent prenasals as plain n + following consonant rather
     than superscript, matching the newer transcription convention.

  NOTE: a blanket case-normalization (lowercasing) step was tried and then
  REMOVED (2026-09-21) -- it masked stray uppercase Latin letters that are
  useful as a visual signal for finding remaining SAMPA vestiges during
  manual corpus cleanup (e.g. "saːL" -> "saal" hid the fact that an "L"
  needed fixing at the source). Stray-uppercase typos like this should be
  found and fixed in the archival XML directly, not normalized away here.
"""

import re
import unicodedata

# ---------------------------------------------------------------------------
# 0. Whitespace collapsing
# ---------------------------------------------------------------------------
def normalize_whitespace(ipa: str) -> str:
    """
    Collapse any run of whitespace (including literal newlines/indentation
    pulled in from multi-line XML, e.g. a <w> whose text is split across
    several <m> children on separate lines) into a single space, then strip
    leading/trailing whitespace. Safe for both single-word strings (result
    has no internal whitespace left, since there shouldn't be any) and
    multi-word aggregate strings (intentional word-separating spaces are
    preserved, just normalized to exactly one space).
    """
    return re.sub(r"\s+", " ", ipa).strip()


# ---------------------------------------------------------------------------
# 1. Tone-stripping
# ---------------------------------------------------------------------------
# Standalone (non-combining) tone/pitch marks: Chao tone letters (˥˦˧˨˩),
# contour arrows, indeterminate-tone marker, and up/downstep modifier letters.
# NOTE: both up-arrow (U+A71B) and down-arrow (U+A71C) must be listed --
# an earlier version of this set only included U+A71C (down), silently
# leaving every up-arrow (U+A71B) instance in normalized output.
STANDALONE_TONE_CHARS = set(range(0x02E5, 0x02EA)) | {0x2197, 0x2198, 0x2219, 0xA71B, 0xA71C}
TONE_COMBINING_MARKS = {0x0301, 0x0300, 0x0302, 0x0304, 0x030C, 0x1DC7, 0x1DC5}


def strip_tones(ipa: str) -> str:
    no_standalone = "".join(ch for ch in ipa if ord(ch) not in STANDALONE_TONE_CHARS)
    decomposed = unicodedata.normalize("NFD", no_standalone)
    cleaned = "".join(ch for ch in decomposed if ord(ch) not in TONE_COMBINING_MARKS)
    return unicodedata.normalize("NFC", cleaned)


# ---------------------------------------------------------------------------
# 2. Affricate tie-bar standardization
# ---------------------------------------------------------------------------
TIE_BAR = "͡"  # combining double inverted breve

# Order matters: ligatures first (single char -> two chars + tie bar),
# then plain two-char sequences (insert tie bar between them).
LIGATURE_MAP = {
    "ʧ": "t" + TIE_BAR + "ʃ",
    "ʤ": "d" + TIE_BAR + "ʒ",
    "ʦ": "t" + TIE_BAR + "s",
    "ʣ": "d" + TIE_BAR + "z",
}

PLAIN_SEQUENCE_MAP = {
    "tʃ": "t" + TIE_BAR + "ʃ",
    "dʒ": "d" + TIE_BAR + "ʒ",
    "ts": "t" + TIE_BAR + "s",
    "dz": "d" + TIE_BAR + "z",
}


def standardize_affricates(ipa: str) -> str:
    for ligature, replacement in LIGATURE_MAP.items():
        ipa = ipa.replace(ligature, replacement)
    for plain, replacement in PLAIN_SEQUENCE_MAP.items():
        # skip if already tie-barred (avoid double-inserting)
        already_tied = plain[0] + TIE_BAR + plain[1]
        ipa = ipa.replace(already_tied, "")  # temp placeholder to protect existing tie-barred forms
        ipa = ipa.replace(plain, replacement)
        ipa = ipa.replace("", already_tied)
    return ipa


# ---------------------------------------------------------------------------
# 3. General vowel-length normalization: Vː -> VV
# ---------------------------------------------------------------------------
VOWELS = "aeiouɛɔɨɯ"
LENGTH_MARK = "ː"  # ː

# Matches a base vowel plus any combining diacritics (e.g. nasalization tilde),
# followed by the length mark -- so nasalized/other-marked long vowels double
# correctly (ɛ̃ː -> ɛ̃ɛ̃), not just the bare vowel.
LONG_VOWEL_RE = re.compile(f"([{VOWELS}][̀-ͯ]*)ː")


def normalize_vowel_length(ipa: str) -> str:
    return LONG_VOWEL_RE.sub(lambda m: m.group(1) * 2, ipa)


# ---------------------------------------------------------------------------
# 4. Creakiness heuristic / dental / rare marks
# ---------------------------------------------------------------------------
CREAKY_MARK = "̰"  # combining tilde below
DENTAL_DIACRITIC = "̪"  # combining bridge below
RARE_INCIDENTAL_MARKS = {0x0329, 0x0339}  # vertical line below, right half ring below -- 1 occurrence each in the corpus, removed for normalization


def normalize_dental_diacritic(ipa: str) -> str:
    """
    Strip the dental diacritic entirely. Unlike creakiness (which is
    conditionally meaningful, see normalize_creakiness below), dental
    articulation is not phonologically contrastive in this language --
    confirmed via direct consultation. Always stripped, no adjacency
    condition needed.
    """
    decomposed = unicodedata.normalize("NFD", ipa)
    cleaned = decomposed.replace(DENTAL_DIACRITIC, "")
    return unicodedata.normalize("NFC", cleaned)


def normalize_rare_incidental_marks(ipa: str) -> str:
    """
    Strip a small set of combining marks that occur only once each across the
    whole corpus (syllabic marker, less-rounded marker) -- confirmed
    incidental/non-systematic rather than a regular phonological pattern
    (contrast with COMBINING CARON BELOW, U+032C, which is kept: that one is
    systematic, marking a regular partially-voiced post-nasal /k/ variant).
    """
    decomposed = unicodedata.normalize("NFD", ipa)
    cleaned = "".join(ch for ch in decomposed if ord(ch) not in RARE_INCIDENTAL_MARKS)
    return unicodedata.normalize("NFC", cleaned)


def normalize_creakiness(ipa: str) -> str:
    """
    Strip creaky-voice diacritic when adjacent (within 1 char) to ʔ (redundant,
    since the glottal stop is already fully articulated/marked). Keep it
    otherwise (likely marking a reduced/deleted glottal stop -- informative).
    """
    decomposed = unicodedata.normalize("NFD", ipa)
    chars = list(decomposed)
    result = []
    i = 0
    while i < len(chars):
        ch = chars[i]
        if ch == CREAKY_MARK:
            # look at immediate neighbors (1 char before/after in the decomposed stream)
            prev_ch = result[-1] if result else ""
            next_ch = chars[i + 1] if i + 1 < len(chars) else ""
            if prev_ch == "ʔ" or next_ch == "ʔ":
                i += 1
                continue  # drop it -- redundant
            else:
                result.append(ch)  # keep it -- informative
        else:
            result.append(ch)
        i += 1
    return unicodedata.normalize("NFC", "".join(result))


# ---------------------------------------------------------------------------
# 5. Aspiration removal
# ---------------------------------------------------------------------------
ASPIRATION_MARK = "ʰ"  # ʰ MODIFIER LETTER SMALL H


def strip_aspiration(ipa: str) -> str:
    """
    Strip superscript aspiration (ʰ), e.g. kʰ -> k. Not phonologically
    significant for this corpus's training targets.
    """
    return ipa.replace(ASPIRATION_MARK, "")


# ---------------------------------------------------------------------------
# 6. Prenasal normalization
# ---------------------------------------------------------------------------
SUPERSCRIPT_N = "ⁿ"  # ⁿ SUPERSCRIPT LATIN SMALL LETTER N


def normalize_prenasal(ipa: str) -> str:
    """
    Normalize superscript prenasal marking (ⁿ) to plain 'n', e.g. ⁿd -> nd.
    Matches the newer transcription convention (plain n + following
    consonant) rather than the older superscript convention, so both are
    represented identically in training targets.
    """
    return ipa.replace(SUPERSCRIPT_N, "n")


# ---------------------------------------------------------------------------
# Combined pipeline
# ---------------------------------------------------------------------------
def normalize_for_training(ipa: str) -> str:
    """Apply all rules in sequence to produce a final ASR training target."""
    ipa = normalize_whitespace(ipa)
    ipa = strip_tones(ipa)
    ipa = standardize_affricates(ipa)
    ipa = normalize_vowel_length(ipa)
    ipa = normalize_dental_diacritic(ipa)
    ipa = normalize_rare_incidental_marks(ipa)
    ipa = normalize_creakiness(ipa)
    ipa = strip_aspiration(ipa)
    ipa = normalize_prenasal(ipa)
    return ipa
