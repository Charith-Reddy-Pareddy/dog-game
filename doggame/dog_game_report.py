"""Builds dog_game_results.pdf from results/dog_game_runs.jsonl (see
doggame.dog_game_study). Run with `python3 -m doggame.dog_game_report`
(needs reportlab)."""

import json
from collections import defaultdict

import numpy as np

from doggame.experiments import DEFAULT_CONFIGS
from doggame.pdf_kit import kit

RESULTS = "results/dog_game_runs.jsonl"
OUTPUT = "dog_game_results.pdf"
CONVERGED = 0.004  # exploitability below this counts as converged (my threshold)
ARCHITECTURES = ("separate", "shared", "partial")
METHODS = (("reinforce", "REINFORCE (angle and step)"), ("nash-q", "Nash-Q, 10 directions"))


def load(path=RESULTS):
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    return [r for r in rows if r["kind"] == "convergence"], [r for r in rows if r["kind"] == "walking"]


def describe(config):
    (rx, ry), (bx, by) = config["house_red"], config["house_blue"]
    return f"red ({rx}, {ry}), blue ({bx}, {by}), w = {config['w']}"


def build(path=OUTPUT, results=RESULTS):
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.graphics.shapes import Drawing, String
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.platypus import KeepTogether, SimpleDocTemplate, Spacer

    k = kit()
    convergence, walking = load(results)
    cells = defaultdict(list)
    for r in convergence:
        cells[(r["config"], r["architecture"])].append(r["exploitability"])
    seeds = len({r["seed"] for r in convergence})
    misses = [r for r in convergence if r["exploitability"] >= CONVERGED]
    by_method = {m: [r for r in walking if r["architecture"] == m] for m, _ in METHODS}
    mean = lambda m, key: np.mean([r[key] for r in by_method[m]])
    per_architecture = {a: [r["exploitability"] for r in convergence if r["architecture"] == a] for a in ARCHITECTURES}
    degenerate = [r["exploitability"] for r in convergence if r["config"] == 3]  # the two houses share a y-coordinate

    table_rows = [["House setup"] + [a for a in ARCHITECTURES]]
    for i, config in enumerate(DEFAULT_CONFIGS):
        row = [describe(config)]
        for a in ARCHITECTURES:
            values = cells[(i, a)]
            row.append(f"{sum(v < CONVERGED for v in values)} of {len(values)} (median {np.median(values):.4f})")
        table_rows.append(row)

    walk_rows = [["Method", "Distance before training", "Distance after training", "Seeds that improved"]]
    for m, label in METHODS:
        walk_rows.append([label, f"{mean(m, 'before'):.2f}", f"{mean(m, 'after'):.2f}",
                          f"{sum(r['after'] < r['before'] for r in by_method[m])} of {len(by_method[m])}"])
    seed_rows = [["Seed"] + [f"{label}: before / after" for _, label in METHODS]]
    for seed in sorted({r["seed"] for r in walking}):
        seed_rows.append([seed] + [f"{next(r for r in by_method[m] if r['seed'] == seed)['before']:.2f} / "
                                   f"{next(r for r in by_method[m] if r['seed'] == seed)['after']:.2f}" for m, _ in METHODS])

    def walking_chart():
        d = Drawing(440, 190)
        bars = VerticalBarChart()
        bars.x, bars.y, bars.width, bars.height = 45, 30, 380, 120
        bars.data = [[mean(m, "before") for m, _ in METHODS], [mean(m, "after") for m, _ in METHODS]]
        bars.categoryAxis.categoryNames = [label for _, label in METHODS]
        bars.categoryAxis.labels.fontSize = 8
        bars.valueAxis.valueMin, bars.valueAxis.valueMax, bars.valueAxis.valueStep = 0, 2.5, 0.5
        bars.valueAxis.labelTextFormat, bars.valueAxis.labels.fontSize = "%.1f", 8
        bars.valueAxis.gridStrokeColor, bars.valueAxis.visibleGrid = colors.HexColor("#dde3ea"), 1
        bars.bars[0].fillColor, bars.bars[1].fillColor = colors.HexColor("#8fb3dc"), colors.HexColor("#1f5fa8")
        bars.barLabelFormat, bars.barLabels.fontSize, bars.barLabels.nudge, bars.barLabels.boxAnchor = "%.2f", 7.5, 6, "s"
        d.add(bars)
        d.add(String(70, 175, "before training", fontSize=8, fillColor=colors.HexColor("#8fb3dc")))
        d.add(String(160, 175, "after training", fontSize=8, fillColor=colors.HexColor("#1f5fa8")))
        d.add(String(10, 160, "distance to the Nash corners", fontSize=8))
        return d

    near = [r for r in walking if r["before"] < 0.5]  # untrained policies that already started close to the corners
    closest = max((r for r in convergence if r["exploitability"] < CONVERGED), key=lambda r: r["exploitability"])
    miss_text = "; ".join(f"{r['architecture']} architecture, house setup {r['config'] + 1}, seed {r['seed']} ({r['exploitability']:.2f})" for r in misses)
    story = [
        k.h1("The dog game: policy-gradient and Nash-Q results"),
        k.small("Convergence to the analytical equilibrium, and the angle-radius (walking) version"),
        Spacer(1, 6),
        k.h2("Summary"),
        k.point(f"Policy-gradient self-play reached exploitability below {CONVERGED} in {len(convergence) - len(misses)} of {len(convergence)} runs "
                f"({len(DEFAULT_CONFIGS)} house setups x {len(ARCHITECTURES)} network architectures x {seeds} seeds, 1500 episodes each)."),
        k.point(f"The {'miss is' if len(misses) == 1 else 'misses are'}: {miss_text}. I have not tested why it is slower or whether more training would fix it."),
        k.point("The three architectures do not separate: " + ", ".join(f"{a} {sum(v < CONVERGED for v in vs)} of {len(vs)}" for a, vs in per_architecture.items()) + "."),
        k.point(f"In the angle-radius game, both REINFORCE and the 10-direction Nash-Q network move the players toward the one-shot Nash corners "
                f"(mean distance {mean('reinforce', 'before'):.2f} to {mean('reinforce', 'after'):.2f} and {mean('nash-q', 'before'):.2f} to {mean('nash-q', 'after'):.2f})."),
        k.h2("The game"),
        k.text("Two houses sit at fixed points in the unit square. Each round, both players pick a point and the dog moves to the weighted average "
               "w * red + (1 - w) * blue. Each player is rewarded by the negative squared distance from the dog to their own house, with discount 0.9. "
               "Because the domain is bounded, players overshoot toward the edge to counteract each other. The one-shot equilibrium is found by iterated best response "
               "(doggame/nash.py), and a learned policy is scored by its exploitability: how much a player could still gain by deviating, searched on a 101 x 101 grid."),
        k.h2("Does policy gradient converge?"),
        k.text(f"REINFORCE self-play, {seeds} seeds (0-{seeds - 1}) for each setup and architecture. Each cell is the number of runs under {CONVERGED}, with the median exploitability."),
        k.table(table_rows, [2.4 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch]),
        Spacer(1, 6),
        k.point(f"The {CONVERGED} threshold is my choice. The closest passing run is {closest['exploitability']:.4f} ({closest['architecture']} architecture, "
                f"house setup {closest['config'] + 1}, seed {closest['seed']}), so the count of passing runs depends on where that line sits."),
        k.point(f"House setup 4 has both houses at y = 0.5, so the y-axis of the stage game is degenerate and many y-picks are equally good. All {len(degenerate)} runs "
                f"there have exploitability of at most {max(degenerate):.4f}, so whichever equilibrium they found is not exploitable."),
        k.h2("Angle-radius version: walking instead of jumping"),
        k.text("Each player picks an angle and a step length (up to 0.2) and walks from where it stands. The score is the sum of both players' distances from the "
               "one-shot Nash corners at the end of a game, averaged over 10 sampled games, on one house setup (red (0.88, 0.12), blue (0.3, 0.65), w = 0.5). "
               f"REINFORCE trains for 300 episodes and the Nash-Q network for 60, over {len({r['seed'] for r in walking})} seeds."),
        KeepTogether([walking_chart(), k.small("Mean distance over the seeds, before and after training."), k.table(walk_rows, [2.2 * inch, 1.6 * inch, 1.6 * inch, 1.5 * inch])]),
        Spacer(1, 8),
        KeepTogether([k.text("Per seed:"), k.table(seed_rows, [0.7 * inch, 2.6 * inch, 2.6 * inch])]),
        Spacer(1, 6),
        k.h2("Limits"),
        k.point("The corners are the equilibrium of the one-shot game. The walking game itself has not been solved, so this shows the players heading to the one-shot "
                "corners, not that they play an equilibrium of the walking game."),
        k.point("Walking was checked on one house setup only. The Nash-Q network sees the exact stage-game solution at every target, so it is not independent of the solver."),
        *[k.point(f"The untrained {dict(METHODS)[r['architecture']]} policy of seed {r['seed']} already started close to the corners ({r['before']:.2f}), "
                  "so that run shows little.") for r in near],
        k.point(f"{seeds} seeds, fixed training lengths and learning rates (0.01 for REINFORCE, 0.001 for Nash-Q), none tuned for these runs."),
        k.point("The reading of the angle-radius action as a move from the player's own position, and of the dog weights, are open questions (see ASSUMPTIONS.md)."),
        k.small("Raw numbers are in results/dog_game_runs.jsonl. The runs are doggame/dog_game_study.py."),
    ]
    SimpleDocTemplate(path, pagesize=letter, leftMargin=0.8 * inch, rightMargin=0.8 * inch, topMargin=0.7 * inch,
                      bottomMargin=0.7 * inch, title="The dog game: policy-gradient and Nash-Q results", author="").build(story)


if __name__ == "__main__":
    build()
