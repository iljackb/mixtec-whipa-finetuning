"""
Score the fine-tuned model's test predictions using WhIPA's OWN STIPA_METRICS
class (code/scripts/metrics.py) -- not a reimplementation, their exact PER/PFER
computation.

Run from inside whipa/code/ (needs panphon + the scripts/ package on path).

Usage:
    cd code
    python3 score_test_results.py
"""

from scripts.metrics import STIPA_METRICS

# ---------------------------------------------------------------------------
# TONE-INCLUSIVE SCORING -- NOT ACTIVE YET.
#
# lowhipa-mixtec-v1's training targets have tone marks stripped, so predicted
# output never contains tone regardless of what the gold has. Scoring with
# tone marks left in gold right now would just penalize the model for
# omitting something it was never trained to produce. This is future-proofing
# for whenever a v2/tone-inclusive checkpoint exists: uncomment the
# STRIP_TONES_FOR_COMPARISON toggle and the tone-inclusive block below it.
# Until then, every line under this comment does nothing -- the script runs
# exactly as it did before.
#
# from normalize_ipa import strip_tones
# import sys
# sys.path.append("..")  # normalize_ipa.py lives one level up, in whipa/
# ---------------------------------------------------------------------------

# (name, predicted, gold_normalized) -- pulled directly from the
# test_whipa.py run against lowhipa-mixtec-v1 on the 10 held-out ADJ_* files
RESULTS = [
    ("che'e",    "t͡ʃe",       "t͡ʃɛʔɛ"),
    ("vii",      "b",          "vii"),
    ("ka'nu1",   "tano",       "kaʔnũ"),
    ("ka'nu2",   "kãʔ̪n̪̪",     "kaʔnũ"),
    ("ka'nu3",   "kano",       "kaʔnũ"),
    ("xeen1",    "ʃe",         "ʃɛ̃ɛ̃"),
    ("xeen2",    "ʃe",         "ʃɛ̃ɛ̃"),
    ("nchichi1", "dʒitʃi",     "nd͡ʒit͡ʃi"),
    ("nchich2",  "ndʒitʃi",    "nd͡ʒit͡ʃi"),
    ("kochi1",   "koçi",       "kot͡ʃi"),
    ("kochi2",   "koʔçi",      "kot͡ʃi"),
    ("vee1",     "vee",        "vee"),
    ("vee2",     "be",         "vee"),
    ("vee3",     "beː",        "vee"),
    ("nani1",    "nani",       "nani"),
    ("nani2",    "nani",       "nani"),
    ("nani3",    "nani",       "nani"),
    ("kani1",    "kani",       "kani"),
    ("kani2",    "kani",       "kani"),
    ("kani3",    "kani",       "kani"),
]

eval_metrics = STIPA_METRICS()

per_list = []
pfer_list = []

# ---------------------------------------------------------------------------
# TONE-INCLUSIVE SCORING -- NOT ACTIVE YET (see block comment above).
#
# When there's a checkpoint trained on tone-inclusive targets, gold in
# RESULTS above should also switch to tone-inclusive transcriptions. At that
# point, uncomment the two lists below and the tone-inclusive compute_all()
# call inside the loop, so every row reports BOTH a tone-stripped score
# (comparable to v1, tone marks removed from both pred/gold before scoring)
# and a tone-inclusive score (gold kept as-is, pred compared with whatever
# tone marks it actually produced).
#
# per_list_tonestripped = []
# pfer_list_tonestripped = []
# ---------------------------------------------------------------------------

print(f"{'token':12} {'PER%':>8} {'PFER%':>8}")
for name, pred, gold in RESULTS:
    m = eval_metrics.compute_all(pred=pred, gold=gold, char_based=False)
    per_list.append(m["per"])
    pfer_list.append(m["pfer"])
    print(f"{name:12} {m['per']:8.1f} {m['pfer']:8.1f}")

    # -----------------------------------------------------------------------
    # TONE-INCLUSIVE SCORING -- NOT ACTIVE YET (see block comments above).
    # Dormant until gold in RESULTS actually carries tone marks; with today's
    # tone-stripped gold, this would just duplicate the numbers already
    # computed above.
    #
    # pred_stripped = strip_tones(pred)
    # gold_stripped = strip_tones(gold)
    # m_stripped = eval_metrics.compute_all(pred=pred_stripped, gold=gold_stripped, char_based=False)
    # per_list_tonestripped.append(m_stripped["per"])
    # pfer_list_tonestripped.append(m_stripped["pfer"])
    # print(f"{'  (tone-stripped)':12} {m_stripped['per']:8.1f} {m_stripped['pfer']:8.1f}")
    # -----------------------------------------------------------------------

mean_per = sum(per_list) / len(per_list)
mean_pfer = sum(pfer_list) / len(pfer_list)

print(f"\nMean PER:  {mean_per:.1f}%")
print(f"Mean PFER: {mean_pfer:.1f}%")

# ---------------------------------------------------------------------------
# TONE-INCLUSIVE SCORING -- NOT ACTIVE YET (see block comments above).
#
# mean_per_stripped = sum(per_list_tonestripped) / len(per_list_tonestripped)
# mean_pfer_stripped = sum(pfer_list_tonestripped) / len(pfer_list_tonestripped)
# print(f"Mean PER (tone-stripped):  {mean_per_stripped:.1f}%")
# print(f"Mean PFER (tone-stripped): {mean_pfer_stripped:.1f}%")
# ---------------------------------------------------------------------------
