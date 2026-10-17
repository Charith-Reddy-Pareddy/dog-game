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


def _walking_row(label, runs, late=150, shown=(50, 100, 200, 300)):
    """One table row: the mean distance at a few points in training, and the time spent at the corners."""
    last = str(max(int(e) for r in runs for e in r["curve"]))
    later = [v < WALKED for r in runs for e, v in r["curve"].items() if int(e) >= late]
    return ([label, fmt(np.mean([r["start"] for r in runs]), 2)] + [fmt(np.mean([r["curve"][str(e)] for r in runs]), 3) for e in shown]
            + [f"{sum(later)} of {len(later)}", f"{sum(r['curve'][last] < WALKED for r in runs)} of {len(runs)}"])


def _walking_header(first, late=150, shown=(50, 100, 200, 300)):
    return [first, "Before training"] + [f"At {e}" for e in shown] + [f"Late checkpoints at the corners (from {late})", "Seeds at the corners at the end"]


def walking_algorithms_table(rows, late=150):
    """Mean distance from the one-shot Nash corners on the walking game, "At N" being after N episodes of training,
    and how much of the later training (checkpoints from `late` episodes on) each algorithm spends at the corners
    (summed distance under WALKED). The paths swing, so one end-of-training number would mislead."""
    runs = study(rows, "walking")
    shown = [e for e in (50, 100, 200, 300) if any(str(e) in r["curve"] for r in runs)]
    table = [_walking_header("Algorithm", late, shown)]
    for name, label in ALGORITHM_NAMES.items():
        mine = [r for r in runs if r["key"]["algorithm"] == name]
        if mine:
            table.append(_walking_row(label, mine, late, shown))
    return table


def ppo_walking_table(rows, late=150):
    """PPO on the walking game with fewer epochs or a smaller learning rate, next to the default and A2C."""
    walking = study(rows, "walking")
    variants = [r for r in study(rows, "ppo_variants") if r["key"]["game"] == "walking"]
    shown = [e for e in (50, 100, 200, 300) if any(str(e) in r["curve"] for r in walking + variants)]
    table = [_walking_header("Setting", late, shown)]
    groups = [("A2C (reference)", [r for r in walking if r["key"]["algorithm"] == "a2c"]),
              ("PPO, lr 0.01, 4 epochs (default)", [r for r in walking if r["key"]["algorithm"] == "ppo"])]
    for lr, epochs in ((0.01, 1), (0.003, 4), (0.003, 1)):
        groups.append((f"PPO, lr {lr}, {epochs} epoch{'s' if epochs > 1 else ''}", [r for r in variants if r["key"]["lr"] == lr and r["key"]["epochs"] == epochs]))
    table += [_walking_row(label, runs, late, shown) for label, runs in groups if runs]
    return table


def ppo_long_table(rows, late=450):
    """The walking game's A2C and PPO settings trained for 900 episodes, to see whether the late swings settle."""
    runs = study(rows, "ppo_long")
    shown = [e for e in (300, 600, 900) if any(str(e) in r["curve"] for r in runs)]
    table = [_walking_header("Setting", late, shown)]
    for setting, label in (("a2c", "A2C"), ("ppo4", "PPO, 4 epochs (default)"), ("ppo1", "PPO, 1 epoch")):
        mine = [r for r in runs if r["key"]["setting"] == setting]
        if mine:
            table.append(_walking_row(label, mine, late, shown))
    return table


def walking_settings_table(rows, late=150):
    """A2C on the walking game with a different maximum step or discount, next to the default (0.2 and 0.9)."""
    shown = (50, 100, 200, 300)
    default = [r for r in study(rows, "walking") if r["key"]["algorithm"] == "a2c"]
    varied = study(rows, "walking_settings")
    table = [_walking_header("Setting", late, shown)]
    for max_step, discount in ((0.1, 0.9), (0.2, 0.9), (0.4, 0.9), (0.2, 0.5), (0.2, 0.99)):
        mine = default if (max_step, discount) == (0.2, 0.9) else [r for r in varied if (r["key"]["max_step"], r["key"]["discount"]) == (max_step, discount)]
        if mine:
            table.append(_walking_row(f"max step {max_step}, discount {discount}" + (" (default)" if mine is default else ""), mine, late, shown))
    return table


def walking_houses_table(rows, late=150):
    """The three algorithms on the walking game's default house setup and on three others."""
    shown = (100, 300)
    default = {"reinforce": [], "a2c": [], "ppo": []}
    for r in study(rows, "walking"):
        default[r["key"]["algorithm"]].append(r)
    table = [["House setup", "Algorithm"] + _walking_header("", late, shown)[1:]]
    others = study(rows, "walking_houses")
    for label, source in [("default: (0.88, 0.12) / (0.3, 0.65), w = 0.5", lambda a: default[a])] + [
            (houses(i), lambda a, i=i: [r for r in others if r["key"]["setup"] == i and r["key"]["algorithm"] == a]) for i in sorted({r["key"]["setup"] for r in others})]:
        for name, algorithm in ALGORITHM_NAMES.items():
            mine = source(name)
            if mine:
                table.append([label, algorithm] + _walking_row("", mine, late, shown)[1:])
    return table


def walking_inertia_table(rows, late=150):
    """A2C on the walking game with the dog's own weight w0 at 0 (the default), 0.3 and 0.6."""
    shown = (50, 100, 200, 300)
    table = [_walking_header("Setting", late, shown)]
    groups = [("w0 = 0 (default)", [r for r in study(rows, "walking") if r["key"]["algorithm"] == "a2c"])]
    groups += [(f"w0 = {w0}", [r for r in study(rows, "walking_inertia") if r["key"]["w0"] == w0]) for w0 in (0.3, 0.6)]
    table += [_walking_row(label, runs, late, shown) for label, runs in groups if runs]
    return table


def walking_equilibrium_table(rows):
    """For the walking game: how much either player could gain by deviating when both follow the corner path."""
    runs = study(rows, "walking_equilibrium")
    table = [["House setup", "Players start", "Red's gain", "Blue's gain", "Largest gain as a share of the reward"]]
    labels = {"default": "default: (0.88, 0.12) / (0.3, 0.65), w = 0.5"}
    for r in runs:
        setup = r["key"]["setup"]
        label = labels.get(setup) or houses(setup)
        share = max(r["red"]["gain"] / abs(r["red"]["followed"]), r["blue"]["gain"] / abs(r["blue"]["followed"])) if min(abs(r["red"]["followed"]), abs(r["blue"]["followed"])) > 1e-9 else 0.0
        table.append([label, "on their houses" if r["key"]["start"] == "houses" else "on the corners",
                      f"{r['red']['gain']:.5f}", f"{r['blue']['gain']:.5f}", f"{100 * share:.2f}%"])
    return table


def walking_capacity_table(rows, late=150):
    """PPO on the walking game with 16 or 64 hidden units, and 4 epochs or 1."""
    walking = [r for r in study(rows, "walking") if r["key"]["algorithm"] == "ppo"]
    one_epoch = [r for r in study(rows, "ppo_variants") if r["key"]["game"] == "walking" and (r["key"]["lr"], r["key"]["epochs"]) == (0.01, 1)]
    bigger = study(rows, "walking_capacity")
    shown = (50, 100, 200, 300)
    groups = [("16 hidden units, 4 epochs (default)", walking), ("64 hidden units, 4 epochs", [r for r in bigger if r["key"]["epochs"] == 4]),
              ("16 hidden units, 1 epoch", one_epoch), ("64 hidden units, 1 epoch", [r for r in bigger if r["key"]["epochs"] == 1])]
    return [_walking_header("PPO setting", late, shown)] + [_walking_row(label, runs, late, shown) for label, runs in groups if runs]


def walking_ppo_inertia_table(rows, late=150):
    """PPO and A2C on the walking game with the dog's own weight w0 at 0 (the default), 0.3 and 0.6."""
    shown = (50, 100, 200, 300)
    table = [_walking_header("Setting", late, shown)]
    for algorithm, label, extra in (("ppo", "PPO", "walking_ppo_inertia"), ("a2c", "A2C", "walking_inertia")):
        default = [r for r in study(rows, "walking") if r["key"]["algorithm"] == algorithm]
        groups = [(f"{label}, w0 = 0 (default)", default)] + [(f"{label}, w0 = {w0}", [r for r in study(rows, extra) if r["key"]["w0"] == w0]) for w0 in (0.3, 0.6)]
        table += [_walking_row(name, runs, late, shown) for name, runs in groups if runs]
    return table


def walking_hard_long_table(rows, late=450):
    """The three algorithms on the hardest walking setup, trained for 900 episodes."""
    runs = study(rows, "walking_hard_long")
    shown = [e for e in (300, 600, 900) if any(str(e) in r["curve"] for r in runs)]
    table = [_walking_header("Algorithm", late, shown)]
    for name, label in ALGORITHM_NAMES.items():
        mine = [r for r in runs if r["key"]["algorithm"] == name]
        if mine:
            table.append(_walking_row(label, mine, late, shown))
    return table


def inertia_equilibrium_table(rows):
    """Main game with dog inertia: red's gain / blue's gain from the best deviation against the one-shot Nash profile."""
    runs = study(rows, "inertia_equilibrium")
    w0s = sorted({r["key"]["w0"] for r in runs})
    table = [["House setup"] + [f"w0 = {w}" for w in w0s]]
    for setup in sorted({r["key"]["setup"] for r in runs}):
        row = [houses(setup)]
        for w0 in w0s:
            r = next(x for x in runs if x["key"]["setup"] == setup and x["key"]["w0"] == w0)
            row.append(f"{r['red']['gain']:.5f} / {r['blue']['gain']:.5f}")
        table.append(row)
    return table


def mirror_p(top, bottom):
    """Two-sided binomial test that the two mirror equilibria are equally likely (they are by symmetry)."""
    from scipy.stats import binomtest
    return binomtest(top, top + bottom).pvalue if top + bottom else float("nan")


def selection_noise_table(rows):
    """Where self-play ends in the shared-y setup for each initial exploration noise (separate layout). The default
    noise comes from this study when it has run there, otherwise from the `selection` study."""
    outcomes = ("red top, blue bottom", "red bottom, blue top", "both at y = 0.5", "other")
    noise = study(rows, "selection_noise")
    default = [r for r in noise if r["key"]["log_std"] == -0.5] or [r for r in study(rows, "selection") if r["key"]["architecture"] == "separate"]
    groups = [(-0.5, default)] + [(s, [r for r in noise if r["key"]["log_std"] == s]) for s in (-2.0, -1.0, 0.0)]
    table = [["Initial noise (std)"] + list(outcomes) + ["p (mirror images equally likely)"]]
    for log_std, runs in sorted(groups):
        if runs:
            found = [selection_outcome(r["red"], r["blue"]) for r in runs]
            table.append([f"{np.exp(log_std):.2f}" + (" (default)" if log_std == -0.5 else "")] + [f"{found.count(o)} of {len(runs)}" for o in outcomes]
                         + [f"{mirror_p(found.count(outcomes[0]), found.count(outcomes[1])):.2f}"])
    return table


def selection_clip_table(rows):
    """The default selection runs, without clipping of actions and the dog, and also with an unbounded mean in y."""
    outcomes = ("red top, blue bottom", "red bottom, blue top", "both at y = 0.5", "other")
    table = [["Policy mean in y, actions and dog"] + list(outcomes) + ["Red's final y: quartiles", "Red's y outside [0, 1]"]]
    clipped = [r for r in study(rows, "selection_noise") if r["key"]["log_std"] == -0.5]
    for label, runs in (("sigmoid, clipped (default)", clipped), ("sigmoid, not clipped", study(rows, "selection_clip")),
                        ("linear, not clipped", study(rows, "selection_linear"))):
        if runs:
            found, y = [selection_outcome(r["red"], r["blue"]) for r in runs], np.array([r["red"][1] for r in runs])
            table.append([label] + [f"{found.count(o)} of {len(runs)}" for o in outcomes] + [" / ".join(f"{v:.2f}" for v in np.percentile(y, [25, 50, 75])),
                                                                                           f"{int(((y < 0) | (y > 1)).sum())} of {len(runs)}"])
    return table


def inertia_explicit_table(rows):
    """Per setup and inertia: the share of dog positions where holding the one-shot actions fails, by the closed form
    and in the dynamic program, and the explicit equilibrium's largest gap to the dynamic program (x, y)."""
    runs = study(rows, "inertia_explicit")
    w0s = sorted({r["key"]["w0"] for r in runs})
    table = [["House setup"] + [f"w0 = {w}: hold fails, theory / DP; gap to DP" for w in w0s]]
    for setup in sorted({r["key"]["setup"] for r in runs}):
        row = [houses(setup)]
        for w0 in w0s:
            r = next(x for x in runs if x["key"]["setup"] == setup and x["key"]["w0"] == w0)
            cell = lambda c: (f"{100 * r['theory'][c]:.0f}% / {100 * r['dp'][c]:.0f}%; " + (f"{r['gap'][c]:.2f}" if r["kind"][c] != "shared" else "shared"))
            row.append(f"x {cell(0)}, y {cell(1)}")
        table.append(row)
    return table


def inertia_random_table(rows):
    """The explicit feedback equilibrium on random one-coordinate games, by the kind of one-shot equilibrium."""
    runs = study(rows, "inertia_random")
    table = [["One-shot equilibrium", "Games", "Explicit policy is an equilibrium (gain < 1e-9)", "Largest one-round gain", "Holding is Markov-perfect"]]
    for kind in ("straddle", "interior", "shared"):
        mine = [r for r in runs if r["kind"] == kind]
        if mine:
            table.append([kind, str(len(mine)), f"{sum(r['gain'] < 1e-9 for r in mine)} of {len(mine)}", f"{max(r['gain'] for r in mine):.1e}",
                          f"{sum(r['holds'] for r in mine)} of {len(mine)}"])
    return table


def inertia_dp_table(rows):
    """The exact feedback equilibrium against holding the one-shot Nash actions: the largest action gap on the path from
    the starting dog position, and the share of dog positions (per coordinate) where the two differ."""
    runs = study(rows, "inertia_dp")
    w0s = sorted({r["key"]["w0"] for r in runs})
    table = [["House setup"] + [f"w0 = {w}: path gap / states differing (x, y)" for w in w0s]]
    for setup in sorted({r["key"]["setup"] for r in runs}):
        row = [houses(setup)]
        for w0 in w0s:
            r = next(x for x in runs if x["key"]["setup"] == setup and x["key"]["w0"] == w0)
            row.append(f"{r['path_gap']:.2f} / {100 * r['differs'][0]:.0f}%, {100 * r['differs'][1]:.0f}%")
        table.append(row)
    return table


def markov_potential_table(rows):
    """4-cycle sums of value changes, which are zero in every Markov potential game."""
    runs = study(rows, "markov_potential")
    table = [["Inertia w0", "Policies that ignore the dog", "Policies that react to the dog"]]
    for w0 in sorted({r["key"]["w0"] for r in runs}):
        cycle = {r["key"]["policies"]: r["cycle"] for r in runs if r["key"]["w0"] == w0}
        table.append([str(w0), f"{cycle.get('fixed', float('nan')):+.1e}", f"{cycle.get('reacting', float('nan')):+.2f}"])
    return table


def ppo_epochs_table(rows):
    """Main game: share of runs under the threshold after each number of episodes, with the gradient steps each
    algorithm takes per episode, so PPO's speed can be compared with REINFORCE and A2C on equal steps."""
    algorithms = study(rows, "algorithms")
    variants = [r for r in study(rows, "ppo_variants") if r["key"]["game"] == "main"]
    checkpoints = sorted({int(e) for r in algorithms for e in r["curve"]})
    table = [["Algorithm", "Gradient steps per episode"] + [f"{e} episodes" for e in checkpoints] + ["Median at the end"]]
    groups = [("REINFORCE", 1, [r for r in algorithms if r["key"]["algorithm"] == "reinforce"]),
              ("A2C", 1, [r for r in algorithms if r["key"]["algorithm"] == "a2c"])]
    for epochs in (1, 2):
        groups.append((f"PPO, {epochs} epoch{'s' if epochs > 1 else ''}", epochs, [r for r in variants if r["key"]["epochs"] == epochs]))
    groups.append(("PPO, 4 epochs (default)", 4, [r for r in algorithms if r["key"]["algorithm"] == "ppo"]))
    for label, steps, runs in groups:
        if runs:
            table.append([label, str(steps)] + [f"{sum(r['curve'][str(e)] < CONVERGED for r in runs)} of {len(runs)}" for e in checkpoints]
                         + [fmt(np.median([r["curve"][str(checkpoints[-1])] for r in runs]))])
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
                        ("resolution", resolution_table(studies)), ("noise", noise_table(studies)), ("walking", walking_table(runs)), ("walking algorithms", walking_algorithms_table(studies)),
                        ("ppo epochs", ppo_epochs_table(studies)), ("ppo walking", ppo_walking_table(studies)),
                        ("ppo long", ppo_long_table(studies)), ("walking settings", walking_settings_table(studies)),
                        ("walking houses", walking_houses_table(studies)), ("walking inertia", walking_inertia_table(studies)),
                        ("walking equilibrium", walking_equilibrium_table(studies)),
                        ("walking capacity", walking_capacity_table(studies)), ("walking ppo inertia", walking_ppo_inertia_table(studies)),
                        ("walking hard long", walking_hard_long_table(studies)), ("inertia equilibrium", inertia_equilibrium_table(studies)),
                        ("selection noise", selection_noise_table(studies)), ("inertia dp", inertia_dp_table(studies)),
                        ("markov potential", markov_potential_table(studies)), ("selection clip", selection_clip_table(studies)),
                        ("inertia explicit", inertia_explicit_table(studies)), ("inertia random", inertia_random_table(studies))):
        print(f"\n### {name}\n\n{markdown(table)}")
    print("\nconvergence counts:", convergence_counts(runs))
    print("inertia median distance:", inertia_distances(studies))
