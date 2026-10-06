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
from doggame.soccer_eval import best_response, exploitability, random_policy, visited_states
from doggame.soccer_pg import train_fictitious_play

# name: (rounds, iterations of training per round, how the opponent is mixed, random starting states)
VARIANTS = {
    "baseline": (60, 20, "game", False),
    "state-average opponent": (60, 20, "state", False),
    "longer best responses": (20, 200, "game", False),
    "both": (20, 200, "state", False),
    "random starts": (60, 20, "game", True),
    "random starts + state average": (60, 20, "state", True),
    "baseline, 200 rounds": (200, 20, "game", False),
    "random starts + state average, 200 rounds": (200, 20, "state", True),
    "random starts + state average, 400 rounds": (400, 20, "state", True),
}

RARE = 10  # a state trained on fewer times than this counts as rarely trained
PROGRESS_FILE = "soccer_fp_progress.jsonl"  # each checkpoint is saved here as it happens, so a stopped run keeps its curve


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
    rounds, iterations, mix, random_starts = VARIANTS[variant]
    every = max(1, rounds // 8)
    curve = []

    def watch(r, history):
        if (r + 1) % every == 0:
            average = [torch.stack(h).mean(0).numpy().astype(float) for h in history]
            latest = [h[-1].numpy().astype(float) for h in history]
            point = (r + 1, exploitability(game, *average), exploitability(game, *latest))
            curve.append(point)
            with open(PROGRESS_FILE, "a") as f:
                f.write(json.dumps({"variant": variant, "seed": seed, "algorithm": algorithm, "point": point}) + "\n")

    visits = torch.zeros(game.n_states, dtype=torch.long)
    average0, _ = train_fictitious_play(game, algorithm, rounds=rounds, iterations=iterations, seed=seed,
                                        on_round=watch, mix=mix, random_starts=random_starts, visits=visits)

    # the states an exact best response reaches against player 0's policy: how well were they trained?
    adversary, _ = best_response(game, average0, player=1)
    reached = visited_states(game, average0, adversary)
    visits = visits.numpy()
    return {"variant": variant, "seed": seed, "algorithm": algorithm, "curve": curve,
            "adversary_states": int(reached.sum()),
            "rarely_trained_share": float((reached & (visits < RARE)).sum() / reached.sum()),
            "untrained_share": float((reached & (visits == 0)).sum() / reached.sum())}


def summarize(results):
    print("exploitability of the average policy (and of the latest policy), mean over seeds")
    for variant in VARIANTS:
        rows = [r for r in results if r["variant"] == variant]
        if rows:
            points = zip(*[r["curve"] for r in rows])
            text = "  ".join(f"r{p[0][0]}: {np.mean([x[1] for x in p]):.2f} ({np.mean([x[2] for x in p]):.2f})" for p in points)
            rare = [r["rarely_trained_share"] for r in rows if "rarely_trained_share" in r]
            coverage = f"{np.mean(rare):.0%}" if rare else "not recorded"
            print(f"{variant:42s} seeds {len(rows)}  {text}  | best-response states trained <{RARE}x: {coverage}")


def run(seeds=2, algorithm="reinforce", variants=None, width=5, height=3, goal_size=1, workers=8,
        results_file="soccer_fp_diagnosis.jsonl"):
    """`variants` limits which variants run (all of them by default)."""
    variants = list(variants or VARIANTS)
    try:
        with open(results_file) as f:
            results = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        results = []
    finished = {(r["variant"], r["seed"], r["algorithm"]) for r in results}
    todo = [(width, height, goal_size, v, s, algorithm) for v in variants for s in range(seeds) if (v, s, algorithm) not in finished]
    print(f"{len(finished)} jobs already done, {len(todo)} to run", flush=True)
    with Pool(workers) as pool, open(results_file, "a") as out:
        for done, result in enumerate(pool.imap_unordered(run_job, todo), 1):
            results.append(result)
            out.write(json.dumps(result) + "\n")
            out.flush()
            print(f"[{done}/{len(todo)}] {result['variant']} seed {result['seed']}", flush=True)
    summarize(results)


if __name__ == "__main__":
    args = sys.argv[1:]
    run(int(args[0]) if args else 2, args[1] if len(args) > 1 else "reinforce", args[2:] or None)
