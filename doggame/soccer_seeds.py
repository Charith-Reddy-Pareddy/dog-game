"""Repeat the soccer comparison over several seeds, in parallel.

Each job trains one method (self-play or fictitious play) with one
algorithm and one seed, then scores player 0's policy the same way
`doggame.soccer_experiments` does. While training it also records the
exploitability along the way, to show whether a method is still improving.
Run it with `python3 -m doggame.soccer_seeds`.
"""

import json
import sys
from multiprocessing import Pool

import numpy as np
import torch

from doggame.soccer import Soccer
from doggame.soccer_experiments import evaluate
from doggame.soccer_eval import exploitability
from doggame.soccer_pg import ALGORITHMS, train_fictitious_play, train_self_play
from doggame.soccer_solver import solve_soccer

METHODS = ("self-play", "fictitious play")


def run_job(job):
    width, height, goal_size, algorithm, method, seed, steps, checkpoint = job
    torch.set_num_threads(1)
    game = Soccer(width, height, goal_size)
    solution = solve_soccer(game)
    curve = []

    if method == "self-play":
        def watch(i, agents):
            if (i + 1) % checkpoint == 0:
                curve.append((i + 1, exploitability(game, agents[0].table(), agents[1].table())))
        agents = train_self_play(game, algorithm, iterations=steps, seed=seed, on_iteration=watch)
        policy0, policy1 = agents[0].table(), agents[1].table()
    else:
        def watch(r, history):
            if (r + 1) % checkpoint == 0:
                average = [torch.stack(h).mean(0).numpy().astype(float) for h in history]
                curve.append((r + 1, exploitability(game, *average)))
        policy0, policy1 = train_fictitious_play(game, algorithm, rounds=steps, iterations=20, seed=seed, on_round=watch)

    result = evaluate(game, solution, policy0, policy1)
    return {"algorithm": algorithm, "method": method, "seed": seed, "curve": curve, **result}


def summarize(results):
    print("player 0, mean over seeds of wins/losses/ties per 1000 games (and the worst seed's losses)")
    print(f"{'method':28s} {'seeds':>5s} {'vs random':>14s} {'vs exact NE':>20s} {'vs exact BR':>14s} {'exploitability':>22s}")
    for algorithm in ALGORITHMS:
        for method in METHODS:
            rows = [r for r in results if r["algorithm"] == algorithm and r["method"] == method]
            if not rows:
                continue
            mean = lambda key: {k: np.mean([r[key][k] for r in rows]) for k in ("wins", "losses", "ties")}
            fmt = lambda m: f"{m['wins']:.0f}/{m['losses']:.0f}/{m['ties']:.0f}"
            nash_losses = [r["nash"]["losses"] for r in rows]
            exploit = [r["exploitability"] for r in rows]
            print(f"{algorithm + ' ' + method:28s} {len(rows):5d} {fmt(mean('random')):>14s} "
                  f"{fmt(mean('nash')) + ' (max ' + str(max(nash_losses)) + ')':>20s} {fmt(mean('best response')):>14s} "
                  f"{np.mean(exploit):.2f} ({min(exploit):.2f}-{max(exploit):.2f})".rjust(22))
    print("\nexploitability of the average policy along fictitious play (mean over seeds)")
    for algorithm in ALGORITHMS:
        rows = [r for r in results if r["algorithm"] == algorithm and r["method"] == "fictitious play"]
        if rows:
            points = zip(*[r["curve"] for r in rows])
            print(f"{algorithm:10s}", "  ".join(f"r{p[0][0]}: {np.mean([x[1] for x in p]):.2f}" for p in points))


def load_results(path):
    try:
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []


def run(seeds=5, self_play_iterations=500, fp_rounds=200, width=5, height=3, goal_size=1,
        workers=6, results_file="soccer_seeds.jsonl"):
    """Every finished job is appended to `results_file` straight away, and jobs
    already in it are skipped, so an interrupted run picks up where it stopped."""
    results = load_results(results_file)
    finished = {(r["algorithm"], r["method"], r["seed"]) for r in results}
    jobs = [(width, height, goal_size, a, "self-play", s, self_play_iterations, 100) for a in ALGORITHMS for s in range(seeds)]
    jobs += [(width, height, goal_size, a, "fictitious play", s, fp_rounds, fp_rounds // 5) for a in ALGORITHMS for s in range(seeds)]
    todo = [job for job in jobs if (job[3], job[4], job[5]) not in finished]
    print(f"{len(finished)} jobs already done, {len(todo)} to run", flush=True)

    with Pool(workers) as pool, open(results_file, "a") as out:
        for done, result in enumerate(pool.imap_unordered(run_job, todo), 1):
            results.append(result)
            out.write(json.dumps(result) + "\n")
            out.flush()
            print(f"[{done}/{len(todo)}] {result['algorithm']} {result['method']} seed {result['seed']}", flush=True)
    summarize(results)


if __name__ == "__main__":
    run(*(int(x) for x in sys.argv[1:4]))
