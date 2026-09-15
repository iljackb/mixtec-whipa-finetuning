"""
Rule-based orthography -> IPA converter for the Mixtepec Mixtec orthography
used in the AILLA MYUC-1042 source (and presumably future sources following
the same practical orthography).

RULES IMPLEMENTED, IN ORDER (all confirmed directly, not guessed):
  1. Hyphen = clitic boundary. Split into separate pieces, convert each
     independently, join final IPA with a SPACE (not a hyphen) -- clitics
     are phonologically/prosodically independent words, per direct
     confirmation, and should be treated identically to any other word
     boundary in the training target.
  2. "ku" + {a,e,i,o} (NOT separated by an apostrophe/glottal stop, NOT
     followed by "u" itself) -> "kw" + vowel.
  3. "ia" -> "ja" (general rule, not conditioned on a specific morphological
     environment -- applies everywhere "ia" occurs).
  4. Apostrophe -> ʔ (standard convention throughout this project's corpus).
  5. Nasal vowels: "Vn" (single vowel + n, WORD-FINAL only per the
     orthography's own rule) -> nasalized vowel, tone diacritic preserved.
     Checked BEFORE plain vowel+tone mapping (longer pattern first).
  6. Plain vowel + tone diacritic -> IPA vowel + same tone diacritic
     (identity mapping, confirmed vowel inventory: a i u o e -> a i u o ɛ).
  7. Consonants: only mappings CONFIRMED against real corpus evidence are
     included (ch->tʃ, x->ʃ, ts->ts, s->s, v->v, k->k, n->n). "t" is left as
     a TENTATIVE default (t->t̪) since real corpus evidence showed a genuine
     ambiguity (t̪ in one confirmed pair, d̪ in another) -- every word
     containing "t" is flagged in the output for direct review, not silently
     resolved.

ANYTHING NOT COVERED by rules 1-7 (e.g. p, m, l, r, w, y, ñ -- consonants not
yet confirmed) is passed through UNCHANGED, and the containing word is
flagged in the output so gaps are immediately visible during review, rather
than silently guessed at.

This produces a DRAFT for review, not a final transcription -- matching the
same review-CSV pattern used throughout this project (finetune_review.csv,
etc.). Every entry should be checked before being treated as gold.
"""

import re
import json
import csv
import unicodedata

# ---------------------------------------------------------------------------
# Vowel + tone dict (oral) -- confirmed inventory: a i u o e (-> ɛ)
# ---------------------------------------------------------------------------
TONES = {"": "", "\u0301": "\u0301", "\u0300": "\u0300", "\u030C": "\u030C", "\u0302": "\u0302"}

ORAL_VOWEL_MAP = {}
for orth_v, ipa_v in {"a": "a", "i": "i", "u": "u", "o": "o", "e": "ɛ"}.items():
    for tone in TONES:
        key = unicodedata.normalize("NFC", orth_v + tone)
        val = unicodedata.normalize("NFC", ipa_v + tone)
        ORAL_VOWEL_MAP[key] = val

# ---------------------------------------------------------------------------
# Nasal vowel + tone dict -- word-final "Vn" -> nasalized vowel
# ---------------------------------------------------------------------------
NASAL_TILDE = "\u0303"
VOWELS = "aeiou"
NASAL_VOWEL_MAP = {}
for orth_v, ipa_v in {"a": "a", "i": "i", "u": "u", "o": "o", "e": "ɛ"}.items():
    for tone in TONES:
        orth_key = unicodedata.normalize("NFC", orth_v + tone) + "n"
        ipa_val = unicodedata.normalize("NFC", ipa_v + NASAL_TILDE + tone)
        NASAL_VOWEL_MAP[orth_key] = ipa_val

# ---------------------------------------------------------------------------
# Confirmed consonant mappings only
# ---------------------------------------------------------------------------
CONSONANT_MAP = {
    "nch": "ndʒ",  # prenasalization triggers voicing: NOT the naive n+ch = ntʃ
    "nk": "ŋk",    # prenasal assimilates to velar place before a velar consonant
    "ch": "tʃ",
    "ts": "ts",
    "x": "ʃ",
    "s": "s",
    "v": "v",
    "k": "k",
    "n": "n",
    "y": "j",
    "t": "t̪",  # TENTATIVE -- flagged in output, see module docstring
}
# NOTE: "ñ" -> "ɲ" is NOT handled here. It's protected from NFD decomposition
# and substituted directly in convert_word() before this map is ever
# consulted, since "ñ" decomposes to bare "n" + a floating combining tilde,
# which would otherwise make this entry unreachable and risk interfering
# with nasal-vowel matching.

APOSTROPHE = "\u0294"  # ʔ

# Both regexes operate on NFD-DECOMPOSED text (see convert_word), and
# explicitly capture any combining marks (tone diacritics) that follow the
# triggering vowel, so a toned vowel like "à" (which decomposes to "a" +
# COMBINING GRAVE ACCENT) is matched and its tone preserved -- not just a
# bare, untoned vowel.
KU_GLIDE_RE = re.compile(r"ku([aeio][\u0300-\u036F]*)")
IA_GLIDE_RE = re.compile(r"i([\u0300-\u036F]*)a([\u0300-\u036F]*)")


ELLIPSIS_CHARS = ("...", "\u2026")  # literal three dots, or the single-char ellipsis symbol

# Whole-word lexical exceptions: known irregular/suppletive pronunciations
# that don't follow the general phonological rules above -- checked FIRST,
# before any other rule, on an exact whole-word match.
WORD_EXCEPTIONS = {
    "kuê": "wɛ̂",  # k fully elided here (unlike the general ku-glide rule,
                   # which would otherwise give "kwɛ̂" -- confirmed this
                   # specific word drops the k entirely, not just glides it)
}


def handle_extra_long(word: str):
    """
    Detect word-final expressive lengthening marked by ellipsis in the
    orthography (e.g. "kuu...", "kuun..."), and convert to a FIXED,
    consistent representation: exactly 3 copies of the vowel (nasalized if
    the lengthening is on a nasal vowel), regardless of how many vowel
    letters appear in the orthography or how long the audio suggests. This
    is a deliberate normalization decision (2026-09-10): the exact duration
    of expressive lengthening is not phonemically contrastive, so
    consistency across the corpus matters more than attempting to encode a
    continuous, non-contrastive scale.
    Returns (stem, extra_long_ipa_suffix) if a match is found, else (word, None).
    """
    stripped = None
    for ell in ELLIPSIS_CHARS:
        if word.endswith(ell):
            stripped = word[:-len(ell)]
            break
    if stripped is None:
        return word, None

    decomposed = unicodedata.normalize("NFD", stripped)

    # Nasal case: strip a trailing word-final "n" marker first, if present.
    is_nasal = decomposed.endswith("n")
    body = decomposed[:-1] if is_nasal else decomposed

    if not body:
        return word, None

    # Scan backward from the end of `body`: first consume any trailing
    # combining tone marks (belongs to the last vowel occurrence), then the
    # base vowel itself, then keep consuming any further immediately-
    # preceding occurrences of that SAME base vowel (with or without their
    # own tone marks) -- this correctly handles a run of 1, 2, or more
    # identical vowel letters before the ellipsis, not just a single one.
    search_end = len(body)  # captured BEFORE any decrementing -- this is the
                             # correct end boundary for the last vowel+tone
                             # unit; using idx+1 instead (a bug caught via
                             # direct testing) incorrectly excluded the tone
                             # mark itself whenever one was present.
    idx = len(body) - 1
    while idx >= 0 and 0x0300 <= ord(body[idx]) <= 0x036F:
        idx -= 1
    if idx < 0 or body[idx] not in VOWELS:
        return word, None  # doesn't end in a recognizable vowel -- leave unchanged

    last_vowel_tone_end = search_end
    base_vowel = body[idx]
    run_start = idx  # will move backward as we consume more matching vowels

    while True:
        # try to consume one more occurrence of the same base vowel
        # (with its own optional tone marks) immediately before run_start
        j = run_start - 1
        # skip that occurrence's tone marks (if any) walking backward
        tone_end = run_start
        while j >= 0 and 0x0300 <= ord(body[j]) <= 0x036F:
            j -= 1
        if j >= 0 and body[j] == base_vowel:
            run_start = j
        else:
            break

    vowel_run = body[run_start:last_vowel_tone_end]  # the whole extended-vowel span
    stem = body[:run_start]

    # Use the LAST occurrence's tone (closest to the ellipsis) as the tone
    # for the resulting tripled output -- most representative of the actual
    # trailing-off pitch.
    last_occurrence = body[idx:last_vowel_tone_end]
    vowel_nfc = unicodedata.normalize("NFC", last_occurrence)
    if vowel_nfc not in ORAL_VOWEL_MAP:
        return word, None

    base_ipa = ORAL_VOWEL_MAP[vowel_nfc]
    if is_nasal:
        base_ipa = unicodedata.normalize("NFC", base_ipa[0] + NASAL_TILDE + base_ipa[1:])

    return unicodedata.normalize("NFC", stem), base_ipa * 3


def convert_word(word: str):
    """Convert a single orthographic word (no hyphens) to a draft IPA form.
    Returns (ipa_string, flags) where flags notes anything needing review."""
    flags = []

    # Whole-word lexical exceptions checked FIRST, before any general rule.
    if word in WORD_EXCEPTIONS:
        return WORD_EXCEPTIONS[word], flags

    # Handle expressive lengthening (ellipsis) FIRST, before any other rule --
    # strips the ellipsis and the extended vowel, processes the remaining
    # stem normally through the rest of this function, then appends the
    # fixed-length (3x) IPA suffix directly.
    extra_long_suffix = None
    word, extra_long_suffix = handle_extra_long(word)

    # Protect "ñ" from NFD decomposition BEFORE anything else -- ñ decomposes
    # to bare "n" + a separate combining tilde, which would otherwise (a)
    # make our "ñ"->"ɲ" consonant rule unreachable (only bare "n" is ever
    # seen post-decomposition) and (b) risk the orphaned tilde interacting
    # with nearby nasal-vowel matching. Use a Private Use Area placeholder,
    # restore to "ɲ" at the very end.
    NY_PLACEHOLDER = "\uE001"
    text = word.replace("ñ", NY_PLACEHOLDER).replace("Ñ", NY_PLACEHOLDER)

    # Rule: apostrophe -> glottal stop. Handles every common variant someone
    # might actually type or that appears in source orthography: plain
    # keyboard apostrophe, modifier letter apostrophe, the saltillo (the
    # standard glottal-stop character in many Mesoamerican orthographies,
    # already used throughout this project's own source material), and the
    # curly/smart quote a word processor might auto-substitute.
    for variant in ["'", "\u02BC", "\uA78C", "\u2019"]:
        text = text.replace(variant, APOSTROPHE)

    # Decompose HERE, once, before any tone-mark-sensitive string rule runs.
    # Everything from this point on (ku-glide, ia-glide, and the main
    # character-scan loop) operates on this same decomposed form -- fixes an
    # earlier bug where the ku-glide regex only matched bare, untoned vowels
    # because it ran on not-yet-decomposed (often precomposed-accented) text.
    text = unicodedata.normalize("NFD", text)

    # Rule: ku + vowel (+ any tone marks) -> kw + vowel (+ its tone marks),
    # NOT triggered before "u" itself (excluded from the character class) or
    # across a glottal stop (glottal stop isn't a vowel, so it can't match here).
    text = KU_GLIDE_RE.sub(r"kw\1", text)

    # Rule: "ia" -> "ja", tone-mark aware -- matches i(+tone) + a(+tone),
    # preserving whichever tone mark(s) were actually present.
    text = IA_GLIDE_RE.sub(r"j\1a\2", text)

    # Longest-match-first tokenization over what remains: try nasal vowel
    # patterns (longer), then plain vowel+tone, then consonant map, then
    # pass through unchanged (flagged).
    result = []
    i = 0
    chars = list(text)

    while i < len(chars):
        matched = False

        # Try DOUBLED nasal vowel first (VVn, word-final -> BOTH vowels
        # nasalized, e.g. "saan" -> "sãã", not just the second/last one).
        # Must be checked BEFORE the single-vowel nasal check below, since
        # that check would otherwise only catch the second vowel (the one
        # actually adjacent to "n") and miss the first one entirely.
        # Try longest first: tone on both vowels (5), tone on one (4), no tone (3).
        for length in (5, 4, 3):
            if i + length == len(chars):  # must reach the exact end of the word
                span = chars[i:i + length]
                # span must be: vowel1, [tone1], vowel2, [tone2], "n" -- with
                # vowel1 == vowel2 (same vowel doubled) and "n" literally last
                if span[-1] == "n" and len(span) >= 3:
                    # find where vowel2 starts: everything before "n" except
                    # trailing tone marks belongs to vowel2's tone; split by
                    # locating the two base vowel characters
                    body = span[:-1]  # everything except the final "n"
                    base_positions = [idx for idx, ch in enumerate(body) if ch in VOWELS]
                    # CRITICAL: the first vowel must be at position 0 of this
                    # span -- otherwise a leading consonant got swept into the
                    # candidate window by coincidental length matching and
                    # would be silently discarded (real bug, caught via direct
                    # testing: "saan"/"kaan" lost their initial consonant
                    # entirely before this check was added).
                    if (len(base_positions) == 2 and base_positions[0] == 0
                            and body[base_positions[0]] == body[base_positions[1]]):
                        v1_idx, v2_idx = base_positions
                        vowel1_part = "".join(body[v1_idx:v2_idx])  # vowel1 + its tone marks
                        vowel2_part = "".join(body[v2_idx:])        # vowel2 + its tone marks
                        v1_nfc = unicodedata.normalize("NFC", vowel1_part)
                        v2_nfc = unicodedata.normalize("NFC", vowel2_part)
                        if v1_nfc in ORAL_VOWEL_MAP and v2_nfc in ORAL_VOWEL_MAP:
                            # Re-derive each vowel's IPA form, then insert nasal
                            # tilde into each (matching NASAL_VOWEL_MAP's own
                            # construction: tilde before any tone diacritic)
                            def nasalize(vowel_ipa):
                                base = vowel_ipa[0]
                                tone_marks = vowel_ipa[1:]
                                return unicodedata.normalize("NFC", base + NASAL_TILDE + tone_marks)
                            result.append(nasalize(ORAL_VOWEL_MAP[v1_nfc]))
                            result.append(nasalize(ORAL_VOWEL_MAP[v2_nfc]))
                            i += length
                            matched = True
                            break
        if matched:
            continue

        # Try nasal vowel patterns (vowel + optional tone + "n") -- ONLY
        # valid when the matched span reaches the END of the word, since
        # nasal vowels are word-final ONLY (confirmed rule). Without this
        # check, a coincidental vowel-then-"n" sequence mid-word (e.g. the
        # "n" that starts a following "ñ") gets misread as a nasal marker.
        for length in (3, 2):  # vowel+tone+n (3 codepoints) or vowel+n (2 codepoints)
            if i + length == len(chars):  # must reach the exact end of the word
                candidate = "".join(chars[i:i + length])
                candidate_nfc = unicodedata.normalize("NFC", candidate)
                if candidate_nfc in NASAL_VOWEL_MAP:
                    result.append(NASAL_VOWEL_MAP[candidate_nfc])
                    i += length
                    matched = True
                    break
        if matched:
            continue

        # Try plain vowel + tone (2 codepoints) or bare vowel (1 codepoint)
        for length in (2, 1):
            if i + length <= len(chars):
                candidate = "".join(chars[i:i + length])
                candidate_nfc = unicodedata.normalize("NFC", candidate)
                if candidate_nfc in ORAL_VOWEL_MAP:
                    result.append(ORAL_VOWEL_MAP[candidate_nfc])
                    i += length
                    matched = True
                    break
        if matched:
            continue

        # Try consonant sequences: 3-char first (e.g. "nch"), then digraphs (2), then single
        for length in (3, 2, 1):
            if i + length <= len(chars):
                candidate = "".join(chars[i:i + length]).lower()
                if candidate in CONSONANT_MAP:
                    result.append(CONSONANT_MAP[candidate])
                    if candidate == "t":
                        flags.append("t->t̪ (tentative, verify)")
                    i += length
                    matched = True
                    break
        if matched:
            continue

        # Already-converted glottal stop or unmapped character: pass through
        ch = chars[i]
        if ch != APOSTROPHE and ch.isalpha():
            flags.append(f"unmapped char: {ch!r}")
        result.append(ch)
        i += 1

    ipa = unicodedata.normalize("NFC", "".join(result))
    ipa = ipa.replace(NY_PLACEHOLDER, "ɲ")  # restore protected ñ -> ɲ
    if extra_long_suffix:
        ipa = ipa + extra_long_suffix
    return ipa, flags


def convert_utterance(orth: str):
    """Split on hyphens (clitic boundaries), convert each piece, join with space."""
    pieces = orth.split("-")
    ipa_pieces = []
    all_flags = []
    for piece in pieces:
        # Handle multi-word pieces too (space-separated words within one clitic group)
        for word in piece.split():
            ipa, flags = convert_word(word)
            ipa_pieces.append(ipa)
            all_flags.extend(flags)
    return " ".join(ipa_pieces), all_flags


def main():
    with open("myuc1042_parsed.json", encoding="utf-8") as f:
        entries = json.load(f)

    rows = []
    for e in entries:
        draft_ipa, flags = convert_utterance(e["orth"])
        rows.append({
            "start": e["start"],
            "end": e["end"],
            "orth": e["orth"],
            "draft_ipa": draft_ipa,
            "eng": e.get("eng", ""),
            "spn": e.get("spn", ""),
            "notes": e.get("notes", ""),
            "flags": "; ".join(sorted(set(flags))) if flags else "",
        })

    out_path = "myuc1042_draft_ipa_review.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["start", "end", "orth", "draft_ipa", "eng", "spn", "notes", "flags"])
        writer.writeheader()
        writer.writerows(rows)

    flagged_count = sum(1 for r in rows if r["flags"])
    print(f"Wrote {len(rows)} entries to {out_path}")
    print(f"  Entries with review flags: {flagged_count} ({flagged_count/len(rows)*100:.0f}%)")

    # Report which unmapped characters are most common, to prioritize what
    # consonant mappings are most worth getting from Jack next
    from collections import Counter
    unmapped = Counter()
    for r in rows:
        for flag in r["flags"].split("; "):
            if flag.startswith("unmapped char:"):
                unmapped[flag] += 1
    if unmapped:
        print("\nMost common unmapped characters (prioritize these):")
        for flag, count in unmapped.most_common(15):
            print(f"  {flag}: {count}")


if __name__ == "__main__":
    main()
