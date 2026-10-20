"""
Summarize a finished (or in-progress) fine-tuning run from the Hugging Face
Trainer's own saved history, so the training-dynamics section of the results
report no longer has to be assembled by hand.

Reads `trainer_state.json`, which the Trainer writes into every
`checkpoint-*` folder. It does NOT read train.log (that is just console
output); trainer_state.json holds the same numbers in structured form.

Given a model output directory (e.g. models/lowhipa-mixtec-v3) it uses the
trainer_state.json from the highest-numbered checkpoint-* folder, which has the
fullest history. You can also pass a trainer_state.json path directly.

Prints a Markdown section in the same layout as Preliminary_Results_Report.md
(configuration bullets, then a Step | Epoch | Train loss | Eval loss table),
followed by automatic checks:
  - best vs. final checkpoint (is the saved model the lowest-eval-loss one?)
  - whether eval loss ever rose (possible overfitting)
  - how much eval loss still improved over the last evaluation

Stdlib only. Run it from anywhere:

    python3 summarize_training.py ../models/lowhipa-mixtec-v3
    python3 summarize_training.py ../models/lowhipa-mixtec-v2 ../models/lowhipa-mixtec-v3
    python3 summarize_training.py ../models/lowhipa-mixtec-v3 --md v3_dynamics.md --csv v3_curve.csv

Notes / limits:
  - "Train loss" in the table is the MEAN of the logged training-loss values
    since the previous evaluation (a few logged steps each), not a single
    running average. Trainer's own "final train loss" (the running average over
    the whole run, printed at the end of training) is not stored in
    trainer_state.json until training completes, so it may be absent.
  - Total wall-clock training time is only available when the final summary
    entry (train_runtime) was written; otherwise use --runtime-hours to add it.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path


def find_state_file(path_str):
    p = Path(path_str).expanduser()
    if p.is_file():
        return p
    if not p.is_dir():
        sys.exit(f"Not found: {p}")
    direct = p / "trainer_state.json"
    candidates = []
    for ck in p.glob("checkpoint-*"):
        m = re.fullmatch(r"checkpoint-(\d+)", ck.name)
        if m and (ck / "trainer_state.json").exists():
            candidates.append((int(m.group(1)), ck / "trainer_state.json"))
    if candidates:
        return max(candidates)[1]
    if direct.exists():
        return direct
    sys.exit(f"No trainer_state.json found under {p} "
             f"(looked for checkpoint-*/trainer_state.json)")


def summarize(state_path, runtime_hours=None):
    d = json.loads(Path(state_path).read_text())
    log = d.get("log_history", [])

    evals = [e for e in log if "eval_loss" in e]
    trains = [e for e in log if "loss" in e and "eval_loss" not in e]
    if not evals:
        sys.exit(f"{state_path}: no eval entries yet (run too short?)")

    # Mean training loss between consecutive evaluations.
    rows = []
    prev_step = 0
    if trains:
        rows.append({"step": trains[0]["step"], "epoch": trains[0].get("epoch"),
                     "train": trains[0]["loss"], "eval": None, "first": True})
    for e in evals:
        in_window = [t["loss"] for t in trains if prev_step < t["step"] <= e["step"]]
        train_mean = sum(in_window) / len(in_window) if in_window else None
        rows.append({"step": e["step"], "epoch": e.get("epoch"),
                     "train": train_mean, "eval": e["eval_loss"], "first": False})
        prev_step = e["step"]

    best_ckpt = d.get("best_model_checkpoint")
    best_metric = d.get("best_metric")
    min_eval = min(evals, key=lambda e: e["eval_loss"])
    last_eval = evals[-1]
    max_steps = d.get("max_steps")
    global_step = d.get("global_step")
    epochs = d.get("num_train_epochs")
    batch = d.get("train_batch_size")

    runtime = None
    for e in log:
        if "train_runtime" in e:
            runtime = e["train_runtime"]
    if runtime is None and runtime_hours is not None:
        runtime = runtime_hours * 3600

    rises = [(a, b) for a, b in zip(evals, evals[1:]) if b["eval_loss"] > a["eval_loss"]]
    last_delta = None
    if len(evals) >= 2:
        last_delta = evals[-2]["eval_loss"] - evals[-1]["eval_loss"]

    return dict(state_path=str(state_path), rows=rows, evals=evals, trains=trains,
                best_ckpt=best_ckpt, best_metric=best_metric, min_eval=min_eval,
                last_eval=last_eval, max_steps=max_steps, global_step=global_step,
                epochs=epochs, batch=batch, runtime=runtime, rises=rises,
                last_delta=last_delta, eval_steps=d.get("eval_steps"),
                save_steps=d.get("save_steps"))


def fmt_time(seconds):
    h = int(seconds // 3600)
    m = int(round((seconds % 3600) / 60))
    return f"{h} hours {m} minutes ({int(seconds):,} seconds)"


def to_markdown(name, s):
    out = []
    out.append(f"### Training dynamics: `{name}`\n")
    out.append(f"_Source: `{s['state_path']}`_\n")
    out.append("**Configuration (from trainer state):**\n")
    out.append(f"- **Epochs**: {s['epochs']}")
    out.append(f"- **Batch size**: {s['batch']} (per device)")
    out.append(f"- **Total steps**: {s['max_steps']:,}"
               + (f" (reached step {s['global_step']:,})" if s['global_step'] != s['max_steps'] else ""))
    out.append(f"- **Eval/save interval**: every {s['eval_steps']} / {s['save_steps']} steps")
    out.append(f"- **Final eval loss**: {s['last_eval']['eval_loss']:.4f} (step {s['last_eval']['step']})")
    if s["runtime"]:
        out.append(f"- **Total training time**: {fmt_time(s['runtime'])}")
    else:
        out.append("- **Total training time**: not recorded in trainer state "
                   "(pass --runtime-hours to include it)")
    out.append("")
    out.append("| Step | Epoch | Train loss (mean since last eval) | Eval loss |")
    out.append("|---|---|---|---|")
    for r in s["rows"]:
        tr = "—" if r["train"] is None else f"{r['train']:.3f}"
        ev = "—" if r["eval"] is None else f"{r['eval']:.4f}"
        ep = "" if r["epoch"] is None else f"{r['epoch']:.2f}"
        if r["eval"] is not None and r["step"] == s["last_eval"]["step"]:
            ev = f"**{ev}**"
        out.append(f"| {r['step']} | {ep} | {tr} | {ev} |")
    out.append("")
    out.append("**Automatic checks:**\n")

    me, le = s["min_eval"], s["last_eval"]
    if me["step"] == le["step"]:
        out.append(f"- Lowest eval loss ({me['eval_loss']:.4f}) is at the **last** evaluation "
                   f"(step {me['step']}), so the final model and the best model coincide.")
    else:
        out.append(f"- Lowest eval loss ({me['eval_loss']:.4f}) was at **step {me['step']}**, "
                   f"not the last evaluation (step {le['step']}, {le['eval_loss']:.4f}). "
                   f"Check which weights were saved as the final model.")
    if s["best_ckpt"]:
        out.append(f"- Trainer's recorded best checkpoint: `{s['best_ckpt']}` "
                   f"(best_metric {s['best_metric']:.4f}).")
    if s["rises"]:
        steps = ", ".join(str(b["step"]) for _, b in s["rises"])
        out.append(f"- Eval loss **rose** at step(s) {steps}: possible overfitting "
                   f"or noisy dev set. Inspect the table.")
    else:
        out.append("- Eval loss decreased at every evaluation (no overfitting inflection on the dev set).")
    if s["last_delta"] is not None:
        out.append(f"- Improvement over the final evaluation interval: {s['last_delta']:.4f} "
                   f"({'still improving' if s['last_delta'] > 0.005 else 'flattened'}).")
    out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+",
                    help="Model output dir(s) (containing checkpoint-*/) or trainer_state.json path(s)")
    ap.add_argument("--md", type=str, default=None, help="Also write the Markdown to this file")
    ap.add_argument("--csv", type=str, default=None,
                    help="Also write the eval curve as CSV (model,step,epoch,eval_loss)")
    ap.add_argument("--runtime-hours", type=float, default=None,
                    help="Wall-clock hours, used only if the trainer state has no train_runtime")
    args = ap.parse_args()

    sections, summaries, csv_rows = [], [], []
    for p in args.paths:
        sp = find_state_file(p)
        pp = Path(p).expanduser()
        if pp.is_dir():
            name = pp.name
        elif sp.parent.name.startswith("checkpoint-"):
            name = sp.parent.parent.name
        else:
            name = sp.parent.name
        s = summarize(sp, args.runtime_hours)
        summaries.append((name, s))
        sections.append(to_markdown(name, s))
        for e in s["evals"]:
            csv_rows.append((name, e["step"], e.get("epoch"), e["eval_loss"]))

    if len(summaries) > 1:
        t = ["### Comparison across runs\n",
             "| Run | Epochs | Steps | Final eval loss | Best eval loss (step) |",
             "|---|---|---|---|---|"]
        for name, s in summaries:
            t.append(f"| `{name}` | {s['epochs']} | {s['max_steps']:,} | "
                     f"{s['last_eval']['eval_loss']:.4f} | "
                     f"{s['min_eval']['eval_loss']:.4f} ({s['min_eval']['step']}) |")
        t.append("\nNote: eval loss is only strictly comparable between runs that used "
                 "the same dev split and target format (e.g. toneless vs. tonal targets differ).\n")
        sections.append("\n".join(t))

    text = "\n".join(sections)
    print(text)
    if args.md:
        Path(args.md).write_text(text)
        print(f"\nWrote {args.md}", file=sys.stderr)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "step", "epoch", "eval_loss"])
            w.writerows(csv_rows)
        print(f"Wrote {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
