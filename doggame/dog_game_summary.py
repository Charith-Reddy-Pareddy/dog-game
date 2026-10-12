"""Tables summarizing results/dog_game_studies.jsonl and results/dog_game_runs.jsonl.

Each function returns a table as a list of rows (the first row is the header),
so the report, the website and the PDF can all show the same numbers. Print
every table as Markdown with `python3 -m doggame.dog_game_summary`.
"""

import json
from collections import defaultdict

import numpy as np

from doggame.experiments import DEFAULT_CONFIGS

STUDIES = "results/dog_game_studies.jsonl"
RUNS = "results/dog_game_runs.jsonl"
CONVERGED = 0.004
DEGENERATE = 3  # the setup whose houses share a y-coordinate
ALGORITHM_NAMES = {"reinforce": "REINFORCE", "a2c": "A2C", "ppo": "PPO"}


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def study(rows, name):
    return [r for r in rows if r["study"] == name]


def fmt(value, digits=4):
    return f"{value:.{digits}f}"


def houses(config_index):
    c = DEFAULT_CONFIGS[config_index]
    return f"({c['house_red'][0]}, {c['house_red'][1]}) / ({c['house_blue'][0]}, {c['house_blue'][1]}), w = {c['w']}"


def convergence_table(runs):
    """Policy-gradient self-play over the five house setups, three layouts and several seeds."""
    cells = defaultdict(list)
    for r in runs:
        if r["kind"] == "convergence":
            cells[(r["config"], r["architecture"])].append(r["exploitability"])
    architectures = ("separate", "shared", "partial")
    table = [["House setup"] + list(architectures)]
    for config in range(len(DEFAULT_CONFIGS)):
        row = [houses(config)]
        for a in architectures:
            values = cells[(config, a)]
            row.append(f"{sum(v < CONVERGED for v in values)} of {len(values)} (median {fmt(np.median(values))})")
        table.append(row)
    return table


def convergence_counts(runs):
    """(runs under the threshold, all runs) in total and per architecture."""
    by = defaultdict(list)
    for r in runs:
        if r["kind"] == "convergence":
            by[r["architecture"]].append(r["exploitability"] < CONVERGED)
    every = [x for v in by.values() for x in v]
    return (sum(every), len(every)), {a: (sum(v), len(v)) for a, v in by.items()}


def algorithms_table(rows):
    """Share of runs under the threshold after each number of episodes, per algorithm."""
    runs = study(rows, "algorithms")
    checkpoints = sorted({int(e) for r in runs for e in r["curve"]})
    table = [["Algorithm"] + [f"{e} episodes" for e in checkpoints] + ["Median at the end"]]
    for name, label in ALGORITHM_NAMES.items():
        mine = [r for r in runs if r["key"]["algorithm"] == name]
        if not mine:
            continue
        shares = [f"{sum(r['curve'][str(e)] < CONVERGED for r in mine)} of {len(mine)}" for e in checkpoints]
        table.append([label] + shares + [fmt(np.median([r["curve"][str(checkpoints[-1])] for r in mine]))])
    return table


WALKED = 0.1  # summed distance to the two corners below this counts as having walked there (my threshold)


def walking_algorithms_table(rows, late=150):
    """Mean distance from the one-shot Nash corners on the walking game, "At N" being after N episodes of training,
    and how much of the later training (checkpoints from `late` episodes on) each algorithm spends at the corners
    (summed distance under WALKED). The paths swing, so one end-of-training number would mislead."""
    runs = study(rows, "walking")
    shown = [e for e in (50, 100, 200, 300) if any(str(e) in r["curve"] for r in runs)]
    last = str(max(int(e) for r in runs for e in r["curve"])) if runs else ""
    table = [["Algorithm", "Before training"] + [f"At {e}" for e in shown]
             + [f"Late checkpoints at the corners (from {late})", "Seeds at the corners at the end"]]
    for name, label in ALGORITHM_NAMES.items():
        mine = [r for r in runs if r["key"]["algorithm"] == name]
        if not mine:
            continue
        later = [v < WALKED for r in mine for e, v in r["curve"].items() if int(e) >= late]
        table.append([label, fmt(np.mean([r["start"] for r in mine]), 2)]
                     + [fmt(np.mean([r["curve"][str(e)] for r in mine]), 3) for e in shown]
                     + [f"{sum(later)} of {len(later)}", f"{sum(r['curve'][last] < WALKED for r in mine)} of {len(mine)}"])
    return table


def walking_share_series(rows):
    """For each algorithm, (episodes, share of seeds under the walking threshold) at every checkpoint."""
    runs = study(rows, "walking")
    series = {}
    for name in ALGORITHM_NAMES:
        mine = [r for r in runs if r["key"]["algorithm"] == name]
        if mine:
            series[name] = [(e, float(np.mean([r["curve"][str(e)] < WALKED for r in mine]))) for e in sorted(int(k) for k in mine[0]["curve"])]
    return series


def selection_outcome(red, blue):
    """Which equilibrium of the shared-y setup a learned policy is at."""
    if red[1] > 0.9 and blue[1] < 0.1:
        return "red top, blue bottom"
    if red[1] < 0.1 and blue[1] > 0.9:
        return "red bottom, blue top"
    if abs(red[1] - 0.5) < 0.1 and abs(blue[1] - 0.5) < 0.1:
        return "both at y = 0.5"
    return "other"


def selection_table(rows):
    outcomes = ("red top, blue bottom", "red bottom, blue top", "both at y = 0.5", "other")
    table = [["Layout"] + list(outcomes) + ["Median exploitability"]]
    for architecture in ("separate", "shared", "partial"):
        mine = [r for r in study(rows, "selection") if r["key"]["architecture"] == architecture]
        if not mine:
            continue
        found = [selection_outcome(r["red"], r["blue"]) for r in mine]
        table.append([architecture] + [f"{found.count(o)} of {len(mine)}" for o in outcomes] + [fmt(np.median([r["exploitability"] for r in mine]), 5)])
    return table


def inertia_table(rows):
    runs = study(rows, "inertia")
    w0s = sorted({r["key"]["w0"] for r in runs})
    table = [["House setup"] + [f"w0 = {w}" for w in w0s]]
    for config in sorted({r["key"]["config"] for r in runs}):
        row = [houses(config)]
        for w0 in w0s:
            mine = [r for r in runs if r["key"]["config"] == config and r["key"]["w0"] == w0]
            row.append(f"{sum(r['exploitability'] < CONVERGED for r in mine)} of {len(mine)} (median {fmt(np.median([r['exploitability'] for r in mine]))})")
        table.append(row)
    return table


def inertia_distances(rows):
    """Median distance to the one-shot Nash actions, per w0 (over both setups and all seeds)."""
    runs = study(rows, "inertia")
    return {w0: float(np.median([r["distance"] for r in runs if r["key"]["w0"] == w0])) for w0 in sorted({r["key"]["w0"] for r in runs})}


def pretraining_table(rows):
    runs = study(rows, "pretraining")
    checkpoints = sorted({int(e) for r in runs for e in r["curve"]})
    table = [["Starting point"] + [f"after {e} episodes" for e in checkpoints]]
    for source in (None, 0, 4):
        mine = [r for r in runs if r["key"]["source"] == source]
        if not mine:
            continue
        label = "from scratch" if source is None else f"pre-trained on setup {source + 1}"
        table.append([label] + [f"{sum(r['curve'][str(e)] < CONVERGED for r in mine)} of {len(mine)} (median {fmt(np.median([r['curve'][str(e)] for r in mine]))})" for e in checkpoints])
    return table


def stage_solver_table(rows):
    runs = study(rows, "stage_solver")
    resyncs = sorted({r["key"]["resync_every"] for r in runs})
    table = [["Fictitious-play iterations"] + [f"re-solve every {n}: Q error / gap" for n in resyncs]]
    for iterations in sorted({r["key"]["iterations"] for r in runs}):
        row = [str(iterations)]
        for n in resyncs:
            mine = [r for r in runs if r["key"]["iterations"] == iterations and r["key"]["resync_every"] == n]
            row.append(f"{fmt(np.median([r['q_error'] for r in mine]))} / {fmt(np.median([r['gap'] for r in mine]))}")
        table.append(row)
    return table


def resolution_table(rows):
    """How the exact solution of the discretized game depends on the grid. The setup whose houses share a
    y-coordinate has a continuum of equilibria, so it is reported separately instead of being counted as a miss."""
    runs = study(rows, "resolution")
    unique = [i for i in range(len(DEFAULT_CONFIGS)) if i != DEGENERATE]
    table = [["Grid", "Actions per player", "Matrix entries", f"Setups with a unique equilibrium found (of {len(unique)})",
              "Shared-y setup: distance to the solver's equilibrium", "Equilibrium is one action pair", "Seconds (median)"]]
    for resolution in sorted({r["key"]["resolution"] for r in runs}):
        mine = [r for r in runs if r["key"]["resolution"] == resolution]
        shared = [r for r in mine if r["key"]["config"] == DEGENERATE]
        found = sum(r["distance"] < 1e-6 for r in mine if r["key"]["config"] in unique)
        actions = mine[0]["actions"]
        table.append([f"{resolution} x {resolution}", str(actions), f"{actions ** 2:,}", str(found),
                      fmt(shared[0]["distance"], 2) if shared else "-", "yes" if all(r["support"] == [1, 1] for r in mine) else "no",
                      fmt(np.median([r["seconds"] for r in mine]), 3)])
    return table


def noise_table(rows):
    runs = study(rows, "noise")
    table = [["Fictitious-play iterations", "Indifference spread", "Actions counted as played", "Weight on the leftover first pick"]]
    for iterations in sorted({r["key"]["iterations"] for r in runs if r["key"]["iterations"] != "exact"}):
        mine = [r for r in runs if r["key"]["iterations"] == iterations]
        table.append([str(iterations), fmt(np.median([r["spread"] for r in mine])), str(int(np.median([r["support"] for r in mine]))),
                      fmt(np.median([1 - r["top"] for r in mine]), 5)])
    exact = [r for r in runs if r["key"]["iterations"] == "exact"]
    if exact:
        table.append(["exact (Lemke-Howson)", fmt(exact[0]["spread"]), str(exact[0]["support"]), fmt(1 - exact[0]["top"], 5)])
    return table


def walking_table(runs):
    """Mean distance of the walking players from the one-shot Nash corners, before and after training."""
    table = [["Method", "Distance before training", "Distance after training", "Seeds that improved"]]
    for method, label in (("reinforce", "REINFORCE (angle and step)"), ("nash-q", "Nash-Q (10 directions)")):
        mine = [r for r in runs if r["kind"] == "walking" and r["architecture"] == method]
        table.append([label, fmt(np.mean([r["before"] for r in mine]), 2), fmt(np.mean([r["after"] for r in mine]), 2),
                      f"{sum(r['after'] < r['before'] for r in mine)} of {len(mine)}"])
    return table


def markdown(table):
    lines = ["| " + " | ".join(table[0]) + " |", "|" + "---|" * len(table[0])]
    lines += ["| " + " | ".join(row) + " |" for row in table[1:]]
    return "\n".join(lines)


if __name__ == "__main__":
    studies, runs = load(STUDIES), load(RUNS)
    for name, table in (("convergence", convergence_table(runs)), ("algorithms", algorithms_table(studies)), ("selection", selection_table(studies)),
                        ("inertia", inertia_table(studies)), ("pretraining", pretraining_table(studies)), ("stage solver", stage_solver_table(studies)),
                        ("resolution", resolution_table(studies)), ("noise", noise_table(studies)), ("walking", walking_table(runs)), ("walking algorithms", walking_algorithms_table(studies))):
        print(f"\n### {name}\n\n{markdown(table)}")
    print("\nconvergence counts:", convergence_counts(runs))
    print("inertia median distance:", inertia_distances(studies))
