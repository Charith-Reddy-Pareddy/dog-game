"""Why does fictitious play with policy-gradient best responses not converge?

Exact fictitious play does (see `exact_fictitious_play`), so the cause must
be in the learned best responses. This compares variants side by side: how
the opponent is mixed (a different earlier policy each game, or the
state-by-state average that is actually scored) and how long each best
response trains. Each job's progress is saved as it finishes, so an
interrupted run resumes. Run it with `python3 -m doggame.soccer_fp_diagnosis`.
"""

import json
import sys
from multiprocessing import Pool

import numpy as np
import torch

from doggame.soccer import Soccer
from doggame.soccer_eval import best_response, exploitability, random_policy
from doggame.soccer_pg import train_fictitious_play

# name: (rounds, iterations of training per round, how the opponent is mixed)
VARIANTS = {
    "baseline": (60, 20, "game"),
    "state-average opponent": (60, 20, "state"),
    "longer best responses": (20, 200, "game"),
    "both": (20, 200, "state"),
}


def exact_fictitious_play(game, rounds, checkpoints=()):
    """Fictitious play with exact best responses. Returns the average
    policies and [(round, exploitability of the average)] at the checkpoints."""
    average = [random_policy(game), random_policy(game)]
    curve = []
    for t in range(1, rounds + 1):
        best0, _ = best_response(game, average[1], player=0)
        best1, _ = best_response(game, average[0], player=1)
        average = [average[0] + (best0 - average[0]) / (t + 1), average[1] + (best1 - average[1]) / (t + 1)]
        if t in checkpoints:
            curve.append((t, exploitability(game, *average)))
    return average, curve


def run_job(job):
    width, height, goal_size, variant, seed, algorithm = job
    torch.set_num_threads(1)
    game = Soccer(width, height, goal_size)
    rounds, iterations, mix = VARIANTS[variant]
    every = max(1, rounds // 4)
    curve = []

    def watch(r, history):
        if (r + 1) % every == 0:
            average = [torch.stack(h).mean(0).numpy().astype(float) for h in history]
            latest = [h[-1].numpy().astype(float) for h in history]
            curve.append((r + 1, exploitability(game, *average), exploitability(game, *latest)))

    train_fictitious_play(game, algorithm, rounds=rounds, iterations=iterations, seed=seed, on_round=watch, mix=mix)
    return {"variant": variant, "seed": seed, "algorithm": algorithm, "curve": curve}


def summarize(results):
    print("exploitability of the average policy (and of the latest policy), mean over seeds")
    for variant in VARIANTS:
        rows = [r for r in results if r["variant"] == variant]
        if rows:
            points = zip(*[r["curve"] for r in rows])
            text = "  ".join(f"r{p[0][0]}: {np.mean([x[1] for x in p]):.2f} ({np.mean([x[2] for x in p]):.2f})" for p in points)
            print(f"{variant:24s} seeds {len(rows)}  {text}")


def run(seeds=2, algorithm="reinforce", width=5, height=3, goal_size=1, workers=8, results_file="soccer_fp_diagnosis.jsonl"):
    try:
        with open(results_file) as f:
            results = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        results = []
    finished = {(r["variant"], r["seed"], r["algorithm"]) for r in results}
    todo = [(width, height, goal_size, v, s, algorithm) for v in VARIANTS for s in range(seeds) if (v, s, algorithm) not in finished]
    print(f"{len(finished)} jobs already done, {len(todo)} to run", flush=True)
    with Pool(workers) as pool, open(results_file, "a") as out:
        for done, result in enumerate(pool.imap_unordered(run_job, todo), 1):
            results.append(result)
            out.write(json.dumps(result) + "\n")
            out.flush()
            print(f"[{done}/{len(todo)}] {result['variant']} seed {result['seed']}", flush=True)
    summarize(results)


if __name__ == "__main__":
    run(*(int(x) for x in sys.argv[1:2]))
