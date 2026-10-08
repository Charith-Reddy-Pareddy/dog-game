"""Builds soccer_fictitious_play_results.pdf from the files in results/.

Run with `python3 -m doggame.soccer_report` (needs reportlab). The numbers come
from `results/soccer_fp_diagnosis_runs.jsonl` (the training curves) and from
`results/soccer_averaging_round_400.txt` and `..._800.txt` (the output of
`python3 -m doggame.soccer_averaging --rounds N`).
"""

import json
import re

import numpy as np

from doggame.pdf_kit import kit

RESULTS = "results"
OUTPUT = "soccer_fictitious_play_results.pdf"
SCHEMES = ["all snapshots", "without the first quarter", "second half only", "last quarter only", "latest only"]
LABELS = ["all snapshots (the average policy)", "without the first quarter", "second half only", "last quarter only", "latest snapshot only"]
FIRST_RUN = "random starts + state average, 400 rounds"
SECOND_RUN = "random starts + state average, 800 rounds"


def load_curves(path, variant):
    """{seed: {round: (average policy, latest policy)}} for one variant of the diagnosis."""
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    return {r["seed"]: {c[0]: (c[1], c[2]) for c in r["curve"]} for r in rows if r["variant"] == variant}


def parse_averaging(text):
    """Reads the output of soccer_averaging: ({seed: {scheme: value}}, {scheme: mean over seeds})."""
    per_seed, means = {}, {}
    for line in text.splitlines():
        seed = re.search(r"_(\d+)_ppo\.pt \(round", line)
        if seed:
            per_seed[int(seed.group(1))] = {n: float(v) for n, v in re.findall(r"(all snapshots|without the first quarter|second half only|last quarter only|latest only) ([\d.]+)", line)}
            continue
        mean = re.match(r"\s+(.+?)\s+([\d.]+)$", line)
        if mean and mean.group(1) in SCHEMES:
            means[mean.group(1)] = float(mean.group(2))
    return per_seed, means


def mean_curve(curves, which):
    rounds = sorted(next(iter(curves.values())))
    return [(r, float(np.mean([curves[s][r][which] for s in curves]))) for r in rounds]


def build(path=OUTPUT, results=RESULTS):
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.graphics.charts.lineplots import LinePlot
    from reportlab.graphics.shapes import Drawing, String
    from reportlab.graphics.widgets.markers import makeMarker
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.platypus import KeepTogether, SimpleDocTemplate, Spacer

    runs = f"{results}/soccer_fp_diagnosis_runs.jsonl"
    first, second = load_curves(runs, FIRST_RUN), load_curves(runs, SECOND_RUN)
    seeds400, mean400 = parse_averaging(open(f"{results}/soccer_averaging_round_400.txt").read())
    seeds800, mean800 = parse_averaging(open(f"{results}/soccer_averaging_round_800.txt").read())

    average = lambda curves, r, which=0: [curves[s][r][which] for s in curves]
    spread = lambda values: f"{np.mean(values):.2f} ({min(values):.2f}-{max(values):.2f})"
    all16 = average(first, 400) + average(second, 400)
    worst = max(seeds800, key=lambda s: seeds800[s]["all snapshots"])
    others = [v["all snapshots"] for s, v in seeds800.items() if s != worst]
    helped = sum(v["without the first quarter"] < v["all snapshots"] for v in seeds800.values())
    hurt = [s for s, v in sorted(seeds800.items()) if v["without the first quarter"] > v["all snapshots"]]
    n = len(seeds800)
    example = max(hurt, key=lambda s: seeds800[s]["last quarter only"])

    k = kit()
    text, point, table, small = k.text, k.point, k.table, k.small

    def line_chart():
        d = Drawing(440, 230)
        plot = LinePlot()
        plot.x, plot.y, plot.width, plot.height = 45, 35, 380, 160
        plot.data = [mean_curve(first, 0), mean_curve(second, 0), mean_curve(second, 1)]
        names = ["average policy, seeds 0-7", "average policy, seeds 8-15", "latest policy, seeds 8-15"]
        for i, c in enumerate(["#1f5fa8", "#c0392b", "#7f8c8d"]):
            plot.lines[i].strokeColor, plot.lines[i].strokeWidth = colors.HexColor(c), 1.8
            plot.lines[i].symbol = makeMarker("FilledCircle", size=3.5, fillColor=colors.HexColor(c), strokeColor=colors.HexColor(c))
            d.add(String(70 + i * 120, 215, names[i], fontSize=7.5, fillColor=colors.HexColor(c)))
        plot.lines[2].strokeDashArray = [3, 2]
        plot.xValueAxis.valueMin, plot.xValueAxis.valueMax, plot.xValueAxis.valueStep = 0, 800, 100
        plot.yValueAxis.valueMin, plot.yValueAxis.valueMax, plot.yValueAxis.valueStep = 0, 0.6, 0.1
        plot.yValueAxis.labelTextFormat = "%.1f"
        plot.xValueAxis.labels.fontSize = plot.yValueAxis.labels.fontSize = 8
        plot.yValueAxis.gridStrokeColor, plot.yValueAxis.visibleGrid = colors.HexColor("#dde3ea"), 1
        d.add(plot)
        d.add(String(235, 10, "round", fontSize=8, textAnchor="middle"))
        d.add(String(10, 205, "exploitability", fontSize=8))
        return d

    def bar_chart():
        d = Drawing(440, 210)
        bars = VerticalBarChart()
        bars.x, bars.y, bars.width, bars.height = 45, 40, 380, 135
        bars.data = [[mean400[s] for s in SCHEMES], [mean800[s] for s in SCHEMES]]
        bars.categoryAxis.categoryNames = ["all snapshots", "without first quarter", "second half", "last quarter", "latest"]
        bars.categoryAxis.labels.fontSize = 7.5
        bars.valueAxis.valueMin, bars.valueAxis.valueMax, bars.valueAxis.valueStep = 0, 0.3, 0.05
        bars.valueAxis.labelTextFormat, bars.valueAxis.labels.fontSize = "%.2f", 8
        bars.valueAxis.gridStrokeColor, bars.valueAxis.visibleGrid = colors.HexColor("#dde3ea"), 1
        bars.bars[0].fillColor, bars.bars[1].fillColor = colors.HexColor("#8fb3dc"), colors.HexColor("#1f5fa8")
        bars.barLabelFormat, bars.barLabels.fontSize, bars.barLabels.nudge, bars.barLabels.boxAnchor = "%.2f", 7, 6, "s"
        d.add(bars)
        d.add(String(70, 195, "at round 400", fontSize=8, fillColor=colors.HexColor("#8fb3dc")))
        d.add(String(160, 195, "at round 800", fontSize=8, fillColor=colors.HexColor("#1f5fa8")))
        d.add(String(10, 180, "exploitability", fontSize=8))
        return d

    long_run = [["Round", "Average policy", "Latest policy", "Seeds with latest under 0.05"]] + [
        [r, spread(average(second, r)), spread(average(second, r, 1)), f"{sum(v < 0.05 for v in average(second, r, 1))} of {len(second)}"]
        for r in range(100, 801, 100)]
    per_seed = [["Seed", "All", "Without first quarter", "Second half", "Last quarter", "Latest"]] + [
        [s] + [f"{seeds800[s][name]:.2f}" for name in SCHEMES] for s in sorted(seeds800)]
    mean_table = [["Way of averaging", "At round 400", "At round 800"]] + [
        [label, f"{mean400[s]:.2f}", f"{mean800[s]:.2f}"] for label, s in zip(LABELS, SCHEMES)]
    hurt_seeds = " and ".join(map(str, hurt))

    story = [
        k.h1("Fictitious play with policy gradient on soccer"),
        small("Longer runs, more seeds, and whether the early policies slow the average"),
        Spacer(1, 6),
        k.h2("Summary"),
        point(f"PPO fictitious play (random starting states, state-average opponent) on the 5x3 soccer board, run to 800 rounds on seeds 8-15. "
              f"The exploitability of the average policy falls from {np.mean(average(second, 100)):.2f} at round 100 to {np.mean(average(second, 800)):.2f} at round 800 "
              f"(mean over {len(second)} seeds). It has not reached 0."),
        point(f"{sum(v <= 0.22 for v in average(second, 800))} of the {len(second)} seeds end at 0.22 or lower; seed {worst} stays at {seeds800[worst]['all snapshots']:.2f}. "
              f"Without seed {worst} the mean is {np.mean(others):.2f}."),
        point("Testing the guess that the average declines slowly because it still contains the weak early policies: the guess is partly right. "
              f"Dropping more of the early history lowers the mean exploitability, but the first quarter alone explains only a small part "
              f"({mean800['all snapshots']:.2f} to {mean800['without the first quarter']:.2f} at round 800), and for some seeds dropping early policies makes the average worse."),
        k.h2("Setup"),
        text("Soccer is a deterministic two-player zero-sum grid game, solved exactly by minimax value iteration (discount 0.9, 100-step limit counted as a tie). "
             "Exploitability is the exact best-response gain against a policy, so 0 is an equilibrium. PPO learns a best response to the state-by-state average of the "
             "opponent's earlier policies, from random starting states, 20 iterations of 256 games per round. The average policy is the plain average of every "
             "snapshot so far. Seeds 0-7 were run to 400 rounds earlier; seeds 8-15 were run to 800 rounds here."),
        k.h2("Longer run"),
        table(long_run, [0.7 * inch, 1.9 * inch, 1.9 * inch, 1.9 * inch]),
        small("Exploitability, mean over seeds 8-15 with the range in brackets."),
        Spacer(1, 4),
        line_chart(),
        small(f"Mean exploitability by round. The average policy keeps falling after round 400 "
              f"({np.mean(average(second, 400)):.2f} to {np.mean(average(second, 800)):.2f}), about 0.015 per 100 rounds."),
        point(f"Over all {len(all16)} seeds the average policy at round 400 is {np.mean(all16):.2f} on average ({min(all16):.2f}-{max(all16):.2f}), "
              f"and {sum(v < 0.05 for v in all16)} of the {len(all16)} are under 0.05."),
        point(f"The latest policy is below 0.05 for {sum(v < 0.05 for v in average(second, 800, 1))} of {len(second)} seeds at round 800, but earlier seeds showed it can jump "
              "(seed 0 went from 0.00 to 0.60), so fictitious play's claim is about the average policy."),
        k.h2("Do the early policies slow the average?"),
        text("Each checkpoint stores every snapshot so far, so the same snapshots can be averaged in different ways: all of them, without the first quarter, "
             "the second half only, the last quarter only, or just the latest. If the early policies were the main drag, the averages that drop them would be much less exploitable."),
        KeepTogether([bar_chart(),
                      small(f"Mean exploitability over the {n} seeds under each way of averaging."),
                      table(mean_table, [3.0 * inch, 1.5 * inch, 1.5 * inch])]),
        Spacer(1, 8),
        KeepTogether([text("At round 800, per seed:"),
                      table(per_seed, [0.6 * inch, 0.9 * inch, 1.5 * inch, 1.1 * inch, 1.1 * inch, 0.9 * inch])]),
        Spacer(1, 6),
        point("The mean falls steadily as more of the early history is dropped, at both rounds, which matches the guess in direction."),
        point(f"The first quarter alone accounts for a small part: {mean400['all snapshots']:.2f} to {mean400['without the first quarter']:.2f} at round 400 and "
              f"{mean800['all snapshots']:.2f} to {mean800['without the first quarter']:.2f} at round 800. "
              f"At round 800 dropping it helps {helped} of the {n} seeds and hurts {len(hurt)} (seeds {hurt_seeds})."),
        point(f"It does not explain every seed. In seeds {hurt_seeds} the average gets worse when the early policies are dropped "
              f"(seed {example}: {seeds800[example]['all snapshots']:.2f} for all snapshots, {seeds800[example]['last quarter only']:.2f} for the last quarter). "
              f"Seed {worst} is poor under every way of averaging, so something other than the early policies is wrong there."),
        k.h2("What this does and does not show"),
        point("It shows that the early policies add to the exploitability of the average policy, in most seeds."),
        point(f"It does not test the explanation that they fade like 1 over the number of rounds, and it does not say why seed {worst} is stuck."),
        point(f"It is one checkpoint per seed, {n} seeds, one small board and one set of settings. The learning rate and entropy weight were tuned on seed 0."),
        small("Raw numbers are in results/soccer_fp_diagnosis_runs.jsonl, results/soccer_averaging_round_400.txt and results/soccer_averaging_round_800.txt. "
              "The averaging test is doggame/soccer_averaging.py."),
    ]
    SimpleDocTemplate(path, pagesize=letter, leftMargin=0.8 * inch, rightMargin=0.8 * inch, topMargin=0.7 * inch,
                      bottomMargin=0.7 * inch, title="Fictitious play with policy gradient on soccer", author="").build(story)


if __name__ == "__main__":
    build()
