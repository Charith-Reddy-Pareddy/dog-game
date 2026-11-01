"""Does the average policy decline slowly because it still contains the weak
early policies?

A fictitious-play checkpoint holds every policy each player has had so far.
This scores the same snapshots under different ways of averaging them: all of
them, without the first part, only the recent part, or just the latest. If the
early policies are what drags the average, the averages that drop them should
be much less exploitable. Run it with
`python3 -m doggame.soccer_averaging soccer_fp_checkpoints/*.pt`.
"""

import sys

import numpy as np
import torch

from doggame.soccer import Soccer
from doggame.soccer_eval import exploitability

# name: the fraction of the snapshots, from the start, to drop before averaging
SCHEMES = {
    "all snapshots": 0.0,
    "without the first quarter": 0.25,
    "second half only": 0.5,
    "last quarter only": 0.75,
}


def average_from(snapshots, start):
    """The plain average of snapshots[start:], as a float64 table."""
    return torch.stack(snapshots[start:]).mean(0).numpy().astype(np.float64)


def scheme_exploitabilities(game, history):
    """Exploitability of the pair of players under each averaging scheme, plus
    the latest snapshot alone. `history` is [player 0's snapshots, player 1's]."""
    n = len(history[0])
    scores = {}
    for name, dropped in SCHEMES.items():
        start = min(int(n * dropped), n - 1)
        scores[name] = exploitability(game, average_from(history[0], start), average_from(history[1], start))
    scores["latest only"] = exploitability(game, average_from(history[0], n - 1), average_from(history[1], n - 1))
    return scores


def analyze(paths, width=5, height=3, goal_size=1, rounds=None):
    """Score each checkpoint. With `rounds`, checkpoints past that round are cut
    back to their first `rounds` rounds (checkpoints short of it are skipped)."""
    game = Soccer(width, height, goal_size)
    rows = []
    for path in paths:
        saved = torch.load(path, weights_only=False)
        history, reached = saved["history"], saved["round"]
        if rounds:
            if reached < rounds:
                continue
            history, reached = [h[:rounds + 1] for h in history], rounds  # one initial snapshot plus one per round
        scores = scheme_exploitabilities(game, history)
        rows.append((path, reached, scores))
        print(f"{path} (round {reached}): " + ", ".join(f"{k} {v:.2f}" for k, v in scores.items()))
    print("\nmean over checkpoints")
    for name in [*SCHEMES, "latest only"]:
        print(f"  {name:28s} {np.mean([r[2][name] for r in rows]):.2f}")
    return rows


if __name__ == "__main__":
    args = sys.argv[1:]
    cut = int(args[1]) if args[:1] == ["--rounds"] else None
    args = args[2:] if cut else args
    analyze(args, rounds=cut)
