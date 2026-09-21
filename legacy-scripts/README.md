# Legacy extraction scripts

Superseded by `extract_finetune_data_unified.py` in the repo root. Kept here for
historical reference only — do not run these to produce training data.

`extract_finetune_data.py` in particular has a confirmed bug (truncates multi-word
utterances to their first word while keeping the full utterance's audio timing);
see `docs/IMPLEMENTATION_NOTES.md` for detail.
