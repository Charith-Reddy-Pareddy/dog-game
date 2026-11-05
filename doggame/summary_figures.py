"""The nine figures of the key-results summary (doggame.summary_pdf), as fig1.png and fig3.png ... fig10.png (fig8.png
is the summary's Fig. 2, fig9.png its Fig. 7 and fig10.png its Fig. 9), and two more
for the paper: fig8 (paths of the plain and compensated receivers) and fig9 (the welfare weight of Proposition 12).
fig11, fig12 and fig13 show the dynamic game with two influencers under averaging: where the dog moves next from every
position with paths and reports (fig11), each influencer's equilibrium response at every position (fig12), and the
distance to the rest point by round (fig13). They are not part of the summary; draw them with fig11(path) and so on.

Figures 1 and 2 solve small games directly (doggame.mechanisms); the others read the saved runs:
Fig. 3 study mechanisms_wide, Fig. 4 mechanisms_phantoms and mechanisms_compensated_design, Fig. 5
mechanisms_skewed_seven, Fig. 6 mechanisms_wide and mechanisms_compensated, Fig. 7
results/dog_game_trajectories.json (doggame.dog_game_trajectories).

Run with `python3 -m doggame.summary_figures [output directory]`.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from doggame import dog_game_summary as S
from doggame import mechanisms as M
from doggame.inertia_theory import target

OUTPUT = "build/summary"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, FRAME, GRID, GREY = "#0b0b0b", "#52514e", "#a9a8a3", "#e4e3df", "#8a8984"
CHART = {"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
         "legend.frameon": False, "font.family": "DejaVu Sans"}
SQUARE = {"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": FRAME}


def aims_of(rule, targets, weights, w0):
    """What each influencer aims the next position at, per coordinate. Under the median its own target (Proposition 20);
    under averaging the equilibrium targets."""
    return [targets[:, c] if rule == "median" else M.target_equilibrium(rule, targets[:, c], weights, w0)[0] for c in range(2)]


def simulate(rule, targets, start, w0, rounds):
    """Dog path and every player's actions per round in the square, each coordinate solved on its own."""
    weights = np.full(len(targets), 1 / len(targets))
    aims = aims_of(rule, targets, weights, w0)
    dog, actions = [np.array(start, float)], []
    for _ in range(rounds):
        a = np.stack([M.reports(rule, dog[-1][c], aims[c], weights, w0) for c in range(2)], 1)
        actions.append(a)
        dog.append(w0 * dog[-1] + (1 - w0) * (a.mean(0) if rule == "mean" else np.median(a, 0)))
    return np.array(dog), np.array(actions)


def arrows(ax, path, color, alpha=1.0, size=7):
    ax.plot(path[:, 0], path[:, 1], color=color, lw=0.9, alpha=0.45 * alpha, zorder=2)
    for p, q in zip(path[:-1], path[1:]):
        if np.hypot(*(q - p)) > 1e-4:
            ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9, alpha=alpha, mutation_scale=size,
                                                           shrinkA=0, shrinkB=0), zorder=3)


def square(ax, title):
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, ec=FRAME, lw=1.5, zorder=1))
    ax.set_xlim(-0.06, 1.06), ax.set_ylim(-0.06, 1.06), ax.set_aspect("equal")
    ax.set_xticks([0, 0.5, 1]), ax.set_yticks([0, 0.5, 1])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=INK2, length=3)
    ax.set_title(title, fontsize=9, color=INK)


def ends(ax, path, color, marker="o", size=70):
    """Hollow marker at the start of a path, filled at the end."""
    ax.scatter(*path[0], s=size, marker=marker, facecolors="white", edgecolors=color, lw=1.8 if marker == "o" else 1.6, zorder=7 if marker == "D" else 5)
    ax.scatter(*path[-1], s=size, marker=marker, color=color, zorder=7 if marker == "D" else 5)


def panel(ax, targets, dog, actions, title, mean_target=True):
    square(ax, title)
    for i, t in enumerate(targets):
        a = actions[:, i] + (i - (len(targets) - 1) / 2) * 0.012  # players sharing a corner stay visible
        arrows(ax, a, COLORS[i], alpha=0.8)
        ends(ax, a, COLORS[i])
        ax.scatter(*t, s=230, marker="*", facecolors="white", edgecolors=COLORS[i], lw=1.8, zorder=6)
    arrows(ax, dog, INK, size=6)
    ends(ax, dog, INK, marker="D")
    if mean_target:
        ax.scatter(*targets.mean(0), s=60, marker="P", color=GREY, zorder=4)


def players(n):
    items = []
    for i in range(n):  # filled column by column: each player above its target
        items += [Line2D([], [], marker="o", ls="", color=COLORS[i], ms=8, label=f"player {i + 1}"),
                  Line2D([], [], marker="*", ls="", mfc="white", mec=COLORS[i], mew=1.5, ms=12, label=f"target {i + 1}")]
    return items + [Line2D([], [], marker="D", ls="", color=INK, ms=7, label="dog")]


def fig1(out):
    """Two players in BIG with a persistent dog."""
    with plt.rc_context(SQUARE):
        targets = np.array([[0.35, 0.65], [0.7, 0.3]])
        dog, actions = simulate("mean", targets, (0.05, 0.05), 0.8, 25)
        fig, ax = plt.subplots(figsize=(4.6, 5.0))
        panel(ax, targets, dog, actions, "t1 (0.35, 0.65)   t2 (0.70, 0.30)   ·   inertia 0.8   ·   25 rounds")
        fig.legend(handles=players(2) + [Line2D([], [], marker="P", ls="", color=GREY, ms=8, label="mean target (planner)")],
                   loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.995), frameon=False, fontsize=8.5, columnspacing=1.4, handletextpad=0.4)
        fig.text(0.5, 0.012, "hollow = start   ·   filled = end   ·   arrows: one per round that moved", ha="center", fontsize=7.5, color=INK2)
        fig.tight_layout(rect=(0, 0.05, 1, 0.9))
        fig.savefig(out, dpi=220), plt.close(fig)


def fig3(out, rows):
    """Share of games in which the median receiver loses less, by inertia (study mechanisms_wide)."""
    wide = S.study(rows, "mechanisms_wide")
    with plt.rc_context(CHART):
        fig, ax = plt.subplots(figsize=(7.2, 2.6))
        for n, color in zip((3, 5, 7), COLORS):
            w0s = sorted({r["key"]["w0"] for r in wide if r["key"]["n"] == n})
            share = [100 * np.mean([r["median"]["loss"] < r["mean"]["loss"] for r in wide if r["key"]["n"] == n and r["key"]["w0"] == w]) for w in w0s]
            ax.plot(w0s, share, color=color, lw=2, marker="o", ms=4, label=f"{n} influencers")
        ax.axhline(50, color=INK2, lw=0.8, ls="--")
        ax.set(xlabel="Inertia w$_0$", ylabel="Games where median\nreceiver loses less (%)", ylim=(0, 100))
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(out, dpi=220), plt.close(fig)


def fig4(out, rows):
    """Extra loss of phantoms (p, 1 - p) over the best at each inertia, three players: plain receivers at their
    equilibria (mechanisms_phantoms) and compensated ones with truthful reports (mechanisms_compensated_design)."""
    plain = [r for r in S.study(rows, "mechanisms_phantoms") if r["key"]["n"] == 3]
    comp = {r["key"]["w0"]: r for r in S.study(rows, "mechanisms_compensated_design") if r["key"]["n"] == 3 and r["key"]["discount"] == 0.9}
    with plt.rc_context(CHART):
        fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 2.9), sharey=True)
        ps = sorted({r["key"]["phantoms"][0] for r in plain})
        for w0, color in zip((0.0, 0.6, 0.9, 0.95), COLORS):
            mean = np.array([np.mean([r["loss"] for r in plain if r["key"]["w0"] == w0 and r["key"]["phantoms"][0] == p]) for p in ps])
            qs, m = np.array([shape[0] for shape in comp[w0]["shapes"][:-1]]), np.array(comp[w0]["mean"][:-1])  # last shape: exact averaging
            for ax, x, y in ((a, ps, mean), (b, qs, m)):
                excess = 100 * (y / y.min() - 1)
                ax.plot(x, excess, color=color, lw=2, label=f"inertia {w0:g}")
                i = int(y.argmin())
                ax.scatter(x[i], excess[i], s=55, color=color, edgecolors="white", linewidths=1.5, zorder=5)
        for ax, title in ((a, "Plain receiver (equilibrium)"), (b, "Compensated receiver (truthful)")):
            for x, name in ((0.0, "median"), (1 / 3, "averaging")):
                ax.axvline(x, color=INK2, lw=0.8, ls="--")
                ax.text(x + 0.006, 7.7, name, fontsize=7.5, color=INK2, va="top", bbox=dict(facecolor="white", edgecolor="none", pad=1, alpha=0.85))
            ax.set(xlim=(-0.01, 0.51), ylim=(0, 8), xlabel="Fixed points (p, 1 - p)")
            ax.set_title(title, fontsize=9, loc="left", color=INK)
        a.set_ylabel("Extra loss over the best\nreceiver at that inertia (%)")
        b.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), title="minimum marked", title_fontsize=8)
        fig.tight_layout()
        fig.savefig(out, dpi=220), plt.close(fig)


def fig5(out, rows, n=7):
    """Left: the slow-receiver criterion E[(h_(j) - mean target)^2] by order statistic j (Proposition 8), by sampling.
    Right: 5th-smallest minus median receiver, game by game (mechanisms_skewed_seven). Returns the right panel's
    differences."""
    rng, js = np.random.default_rng(0), np.arange(1, n + 1)
    loss = {(tuple(r["key"]["phantoms"]), r["key"]["draw"]): r["loss"] for r in S.study(rows, "mechanisms_skewed_seven")}
    best, median = M.order_statistic_phantoms(n, 5), M.order_statistic_phantoms(n, 4)
    diff = np.array(sorted(loss[best, d] - loss[median, d] for (ph, d) in loss if ph == best and (median, d) in loss))
    with plt.rc_context(CHART):
        fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 2.9), gridspec_kw={"width_ratios": [1, 1.15]})
        samples = (("uniform targets", rng.uniform(size=(400_000, n))), ("skewed targets, Beta(0.1, 1)", rng.beta(0.1, 1, (400_000, n))))
        for (name, sample), color in zip(samples, COLORS):
            mse = ((np.sort(sample, 1) - sample.mean(1, keepdims=True)) ** 2).mean(0)
            a.plot(js, mse, color=color, lw=2, marker="o", ms=4, label=name)
            i = int(mse.argmin())
            a.scatter(js[i], mse[i], s=70, color=color, edgecolors="white", linewidths=1.5, zorder=5)
        a.set_xticks(js, ["1st", "2nd", "3rd", "4th\n(median)", "5th", "6th", "7th"])
        a.set(xlabel="Receiver: moves toward the j-th smallest target", ylabel="Mean squared distance\nto the mean target")
        a.set_title("Theory: best slow receiver (minimum marked)", fontsize=9, loc="left", color=INK)
        a.legend(loc="upper center", fontsize=7.5)
        b.bar(np.arange(len(diff)), diff, width=0.8, color=np.where(diff < 0, COLORS[1], "#b9b8b3"))
        b.axhline(0, color=INK2, lw=0.8)
        b.axhline(diff.mean(), color=INK, lw=1.2)
        b.text(len(diff) - 1, diff.mean() - 0.004, f"mean {diff.mean():.4f}", ha="right", va="top", fontsize=7.5, color=INK)
        b.set_xticks([])
        b.set(xlabel=f"{len(diff)} random games, sorted ({(diff < 0).sum()} below zero)", ylabel="Loss of 5th smallest\nminus median receiver")
        b.set_title("Dynamic game: 7 influencers, inertia 0.99", fontsize=9, loc="left", color=INK)
        fig.tight_layout()
        fig.savefig(out, dpi=220), plt.close(fig)
    return diff


def fig6(out, rows):
    """Mean extra loss over the planner: both plain receivers at their equilibria (mechanisms_wide) and the
    compensated averaging receiver with truthful reports (mechanisms_compensated), on the same draws."""
    plain = {(r["key"]["n"], r["key"]["w0"], r["key"]["draw"]): r for r in S.study(rows, "mechanisms_wide")}
    comp = S.study(rows, "mechanisms_compensated")
    w0s = sorted({r["key"]["w0"] for r in comp})
    series = [("averaging receiver (BIG)", COLORS[0], lambda c, p: p["mean"]["loss"]),
              ("median receiver", COLORS[1], lambda c, p: p["median"]["loss"]),
              ("compensated averaging, truthful", COLORS[2], lambda c, p: c["averaging"])]
    with plt.rc_context(CHART):
        fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.7), sharey=True)
        for ax, n in zip(axes, (3, 5, 7)):
            for name, color, get in series:
                y = []
                for w0 in w0s:
                    pairs = [(c, plain[n, w0, c["key"]["draw"]]) for c in comp if (c["key"]["n"], c["key"]["w0"]) == (n, w0)]
                    y.append(100 * (np.mean([get(c, p) for c, p in pairs]) / np.mean([p["planner"] for c, p in pairs]) - 1))
                ax.plot(w0s, y, color=color, lw=2, marker="o", ms=3.5, label=name)
            ax.set_title(f"{n} influencers", fontsize=9, loc="left", color=INK)
            ax.set_xlabel("Inertia w$_0$")
        axes[0].set_ylabel("Extra loss over\nthe planner (%)")
        axes[0].set_ylim(0, None)
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 1.02))
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        fig.savefig(out, dpi=220), plt.close(fig)


def fig7(out, path="results/dog_game_trajectories.json"):
    """REINFORCE self-play in the walking game, seed 0, before and after training."""
    d = json.loads(Path(path).read_text())
    houses, nash = [np.array(d["house_red"]), np.array(d["house_blue"])], [np.array(d["nash_red"]), np.array(d["nash_blue"])]
    with plt.rc_context(SQUARE):
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.3))
        for ax, phase, title in zip(axes, ("before", "after"), ("Before training", "After REINFORCE self-play (300 episodes)")):
            run = d["seeds"][0]["reinforce"][phase]
            square(ax, title)
            for i, key in enumerate(("red", "blue")):
                walk = np.array(run[key])
                arrows(ax, walk, COLORS[i], size=7)
                ends(ax, walk, COLORS[i])
                ax.scatter(*houses[i], s=230, marker="*", facecolors="white", edgecolors=COLORS[i], lw=1.8, zorder=6)
                ax.scatter(*nash[i], s=170, marker="s", facecolors="none", edgecolors=INK2, lw=1.2, ls="--", zorder=4)
            dog = np.array(run["dog"])
            arrows(ax, dog, INK, size=6)
            ends(ax, dog, INK, marker="D")
        fig.legend(handles=players(2) + [Line2D([], [], marker="s", ls="", mfc="none", mec=INK2, ms=9, label="one-shot Nash corner")],
                   loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=8.5, columnspacing=1.4, handletextpad=0.4)
        fig.text(0.5, 0.012, "t1 (0.88, 0.12)  t2 (0.30, 0.65)  ·  step length 0.2  ·  15 steps  ·  seed 0  ·  hollow = start, filled = end, arrows: one per step that moved",
                 ha="center", fontsize=7.2, color=INK2)
        fig.tight_layout(rect=(0, 0.06, 1, 0.86))
        fig.savefig(out, dpi=220), plt.close(fig)


def fig8(out, w0=0.9, rounds=30):
    """Three influencers under BIG's averaging and the median receiver at their equilibria, the
    compensated averaging receiver with truthful reports, and the planner. Returns each one's rounds to come within
    0.01 of where it ends."""
    targets, start = np.array([[0.15, 0.7], [0.3, 0.2], [0.9, 0.55]]), (0.95, 0.95)
    averaging = (1 / 3, 2 / 3)
    runs = [("BIG's averaging receiver", *simulate("mean", targets, start, w0, rounds)),
            ("Median receiver", *simulate("median", targets, start, w0, rounds))]
    for name, step in (("Compensated averaging, truthful reports", lambda x, c: M.compensated(x, targets[None, :, c], averaging, w0)),
                       ("Planner", lambda x, c: M.planner_step(x, targets[:, c], w0))):
        dog = [np.array(start, float)]
        for _ in range(rounds):
            dog.append(np.array([step(np.array([dog[-1][c]]), c)[0] for c in range(2)]))
        runs.append((name, np.array(dog), np.repeat(targets[None], rounds, 0) if name != "Planner" else None))
    found = []
    with plt.rc_context(SQUARE):
        fig, axes = plt.subplots(2, 2, figsize=(7.4, 8.3))
        for ax, (name, dog, actions) in zip(axes.flat, runs):
            found.append(next((k for k in range(1, len(dog)) if np.abs(dog[k] - dog[-1]).max() < 0.01), len(dog) - 1))
            title = f"{name}\nwithin 0.01 of its end point after {found[-1]} rounds"
            if actions is not None:
                panel(ax, targets, dog, actions, title)
                continue
            square(ax, title)  # the planner uses no reports
            for i, t in enumerate(targets):
                ax.scatter(*t, s=230, marker="*", facecolors="white", edgecolors=COLORS[i], lw=1.8, zorder=6)
            arrows(ax, dog, INK, size=6)
            ends(ax, dog, INK, marker="D")
        influencers = players(3)[:-1]  # the paper says influencer and receiver, not player and dog
        for h in influencers:
            h.set_label(h.get_label().replace("player", "influencer"))
        receiver = Line2D([], [], marker="D", ls="", color=INK, ms=7, label="receiver")
        fig.legend(handles=influencers + [receiver, Line2D([], [], marker="P", ls="", color=GREY, ms=8, label="mean target (planner)")],
                   loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=8.5, columnspacing=1.4, handletextpad=0.4)
        fig.text(0.5, 0.008, f"t1 (0.15, 0.70)  t2 (0.30, 0.20)  t3 (0.90, 0.55)  ·  inertia {w0}  ·  {rounds} rounds  ·  circles: reports (on the targets when truthful)",
                 ha="center", fontsize=7.2, color=INK2)
        fig.tight_layout(rect=(0, 0.02, 1, 0.92), h_pad=2.5)
        fig.savefig(out, dpi=220), plt.close(fig)
    return found


def fig9(out, discount=0.9):
    """Left: the weight W(u) (1 - g) of Proposition 12 for evenly spread starts. Right: the extra loss over the planner
    of a compensated receiver with end point s, for the x-coordinates of fig8's targets."""
    u, houses = np.linspace(0, 1, 401), np.array([0.15, 0.3, 0.9])
    mean = houses.mean()
    with plt.rc_context(CHART):
        fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 2.9))
        for w0, color in zip((0.0, 0.6, 0.9, 0.97), COLORS):
            weight = M.reach_weight(u, w0, discount)
            a.plot(u, weight * (1 - discount), color=color, lw=2, label=f"inertia {w0:g}")
            gap = 6 * np.concatenate([[0], np.cumsum(((u - mean) * weight)[1:] * np.diff(u))])
            gap -= np.interp(mean, u, gap)
            b.plot(u, gap, color=color, lw=2)
        for x, name, side in ((np.median(houses), "median receiver", -1), (1 / 3, "averaging receiver", 1), (mean, "planner", 1)):
            b.axvline(x, color=INK2, lw=0.8, ls="--")
            b.text(x + 0.012 * side, b.get_ylim()[1] * 0.95, name, fontsize=7, color=INK2, va="top", ha="left" if side > 0 else "right",
                   rotation=90 if side > 0 else 0)
        a.set(xlabel="Point u", ylabel=r"W(u) (1 - $\delta$)", ylim=(0, 1.05))
        a.set_title("Discounted share of rounds in which u is reachable", fontsize=9, loc="left", color=INK)
        a.legend(loc="lower center", fontsize=7.5)
        b.set(xlabel="End point s", ylabel="Extra loss over the planner")
        b.set_title("Targets 0.15, 0.30, 0.90", fontsize=9, loc="left", color=INK)
        fig.tight_layout()
        fig.savefig(out, dpi=220), plt.close(fig)


def fig10(out, rows):
    """Continuous-move Nash-Q in the walking game (study continuous_q_tools, seed 0): greedy equilibrium paths from the
    houses (solid) and with each player starting in the other's corner (faint), one panel per move encoding."""
    found = {r["key"]["encoding"]: r["trajectories"] for r in S.study(rows, "continuous_q_tools")}
    with plt.rc_context(SQUARE):
        fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.2))
        for ax, (encoding, name) in zip(axes, (("cartesian", "Cartesian moves (vx, vy)"), ("polar", "Polar moves (r, \u03b8)"), ("square", "Square moves"))):
            square(ax, name)
            for start, alpha in (("each in the other's corner", 0.35), ("from the houses", 1.0)):
                path = np.array(found[encoding][start])  # rounds x (red, blue, dog) x 2
                for i in range(2):
                    arrows(ax, path[:, i], COLORS[i], alpha=alpha, size=6)
                    ends(ax, path[:, i], COLORS[i], size=45)
            houses = np.array(found[encoding]["from the houses"][0][:2])
            for i, h in enumerate(houses):
                ax.scatter(*h, s=200, marker="*", facecolors="white", edgecolors=COLORS[i], lw=1.6, zorder=6)
        fig.legend(handles=players(2)[:-1], loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=8.5,
                   columnspacing=1.4, handletextpad=0.4)
        fig.text(0.5, 0.012, "step length 0.2  ·  15 steps  ·  seed 0  ·  solid: from the houses  ·  faint: each starts in the other's corner",
                 ha="center", fontsize=7.2, color=INK2)
        fig.tight_layout(rect=(0, 0.05, 1, 0.88))
        fig.savefig(out, dpi=220), plt.close(fig)


TWO_TARGETS, TWO_WEIGHTS, PHANTOM, TWO_START = np.array([[0.3, 0.6], [0.7, 0.95]]), np.full(2, 0.5), 0.5, (0.95, 0.05)


def two_influencers(w0, discount=0.9):
    """The game of fig11 to fig13. In x the targets straddle the phantom 1/2, so each influencer aims at the closed form of
    Proposition 2. In y influencer 1 is the pivot and aims at its target, and influencer 2 holds the top edge (Proposition
    18). Returns the aims of each coordinate and the next position from an array of positions (n, 2)."""
    aims = [np.array([target(t, PHANTOM, w0, discount) for t in TWO_TARGETS[:, 0]]), np.array([TWO_TARGETS[0, 1], 1.0])]
    return aims, lambda p: np.stack([M.step("mean", p[:, c], aims[c], TWO_WEIGHTS, w0) for c in range(2)], 1)


def two_path(w0, start, rounds):
    """The dog's positions (rounds + 1, 2) and the reports (rounds, 2, 2) from a start."""
    aims, move = two_influencers(w0)
    dog = [np.array([start], dtype=float)]
    for _ in range(rounds):
        dog.append(move(dog[-1]))
    dog = np.concatenate(dog)
    return dog, np.stack([[M.reports("mean", p[c], aims[c], TWO_WEIGHTS, w0) for c in range(2)] for p in dog[:-1]]).transpose(0, 2, 1)


def phantom_lines(ax):
    for line in (([PHANTOM, PHANTOM], [0, 1]), ([0, 1], [PHANTOM, PHANTOM])):
        ax.plot(*line, color=COLORS[2], lw=1.0, ls=(0, (4, 3)), zorder=1.5)


def fig11(out, w0=0.8, rounds=25):
    """Where the dog moves next from every position, with paths from the corners (left), and one equilibrium path with every
    report (right). The dog lands exactly on y = 0.6 (Proposition 18), while x only approaches the phantom 1/2. Returns, per
    coordinate, the round in which the path lands exactly on its end point (None if it never does)."""
    grid, corners = np.linspace(0.03, 0.97, 15), np.array([[0.02, 0.02], [0.98, 0.02], [0.02, 0.98], [0.98, 0.98]])
    _, move = two_influencers(w0)
    px, py = np.meshgrid(grid, grid)
    nxt = move(np.stack([px.ravel(), py.ravel()], 1)).reshape(*px.shape, 2)
    u, v = nxt[..., 0] - px, nxt[..., 1] - py
    speed = np.hypot(u, v)
    with plt.rc_context(SQUARE):
        fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 4.0))
        square(a, f"Where the dog moves next\n(largest step {speed.max():.2f} per round)")
        phantom_lines(a)
        a.quiver(px, py, u, v, speed, cmap="Greys", clim=(-0.3 * speed.max(), speed.max()), angles="xy", scale_units="xy",
                 scale=speed.max() / (0.9 * (grid[1] - grid[0])), width=0.004, headwidth=4, headlength=4.5, zorder=2)
        paths = [corners]
        for _ in range(60):
            paths.append(move(paths[-1]))
        paths = np.array(paths)
        for i in range(len(corners)):
            a.plot(paths[:, i, 0], paths[:, i, 1], color=INK, lw=1.1, zorder=3)
        a.scatter(PHANTOM, TWO_TARGETS[0, 1], s=55, marker="D", color=INK, zorder=7)
        for i, t in enumerate(TWO_TARGETS):
            a.scatter(*t, s=200, marker="*", facecolors="white", edgecolors=COLORS[i], lw=1.6, zorder=6)
        dog, actions = two_path(w0, TWO_START, rounds)
        phantom_lines(b)
        panel(b, TWO_TARGETS, dog, actions, f"One path, {rounds} rounds,\nwith every report", mean_target=False)
        handles = players(2)[:-1]
        for h in handles:
            h.set_label(h.get_label().replace("player", "influencer"))
        handles += [Line2D([], [], marker="D", ls="", color=INK, ms=7, label="dog (filled: end)"),
                    Line2D([], [], color=COLORS[2], lw=1.1, ls=(0, (4, 3)), label="phantom (1/2)")]
        fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=7.8,
                   columnspacing=1.4, handletextpad=0.3)
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        fig.savefig(out, dpi=220), plt.close(fig)
    return tuple(next((t for t in range(len(dog)) if abs(dog[t, c] - end) < 1e-12), None)
                 for c, end in enumerate((PHANTOM, TWO_TARGETS[0, 1])))


def fig12(out, w0=0.8):
    """The influencers' response vector fields: at each dog position, an arrow toward the report each influencer makes there
    in equilibrium (fixed length; darker = the report is farther away), and the reports as circles. Returns, per influencer,
    the number of distinct reports."""
    grid, counts = np.linspace(0.05, 0.95, 11), []
    x, y = np.meshgrid(grid, grid)
    aims, _ = two_influencers(w0)
    reports = [np.array([M.reports("mean", z, aims[c], TWO_WEIGHTS, w0) for z in grid]) for c in range(2)]  # (position, influencer)
    with plt.rc_context(SQUARE):
        fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.5))
        for i, ax in enumerate(axes):
            square(ax, f"Influencer {i + 1}'s report")
            rx, ry = np.meshgrid(reports[0][:, i], reports[1][:, i])
            far = np.hypot(rx - x, ry - y)
            ax.quiver(x, y, (rx - x) / far, (ry - y) / far, far, cmap="Greys", clim=(-0.5, 1.4), angles="xy", scale_units="xy",
                      scale=1 / (0.8 * (grid[1] - grid[0])), width=0.005, headwidth=3.5, headlength=4, zorder=3)
            ax.scatter(rx, ry, s=16, facecolors="white", edgecolors=COLORS[i], lw=1.1, zorder=5, clip_on=False)
            ax.scatter(*TWO_TARGETS[i], s=200, marker="*", facecolors="white", edgecolors=COLORS[i], lw=1.6, zorder=6)
            counts.append(len(np.unique(np.round(np.stack([rx.ravel(), ry.ravel()], 1), 6), axis=0)))
        fig.tight_layout()
        fig.savefig(out, dpi=220), plt.close(fig)
    return counts


def fig13(out, w0=0.8, rounds=25):
    """Distance to the rest point by round, on a log scale, on the path of fig11. Returns the round in which y lands and the
    distance left in x at the last round."""
    dog, _ = two_path(w0, TWO_START, rounds)
    gap = np.abs(dog - np.array([PHANTOM, TWO_TARGETS[0, 1]]))
    land = int(np.argmax(gap[:, 1] < 1e-12))
    with plt.rc_context(CHART):
        fig, ax = plt.subplots(figsize=(6.2, 2.7))
        r = np.arange(rounds + 1)
        ax.semilogy(r, gap[:, 0], "-o", ms=3.5, lw=2, color=COLORS[0], label="x: rest point 1/2, a phantom")
        ax.semilogy(r[:land], gap[:land, 1], "-o", ms=3.5, lw=2, color=COLORS[1], label="y: rest point 0.6, a target (pivot)")
        ax.plot([land - 1, land], [gap[land - 1, 1], 1e-4], ":", lw=2, color=COLORS[1])
        ax.plot([land], [1.1e-4], marker="v", ms=8, color=COLORS[1], clip_on=False)
        ax.annotate(f"lands exactly in round {land}", (land + 0.5, 1.6e-4), fontsize=8.5, color=INK)
        ax.set_ylim(1e-4, 1.1), ax.set_xlim(0, rounds)
        ax.set_xlabel("Round"), ax.set_ylabel("Distance to rest point")
        ax.legend(loc="upper right", fontsize=8.5)
        fig.tight_layout()
        fig.savefig(out, dpi=220), plt.close(fig)
    return land, float(gap[rounds, 0])


def paper(out="paper/figures", results=S.STUDIES):
    """The paper's figures, as vector PDFs."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = S.load(results)
    fig3(out / "reversal.pdf", rows)
    fig4(out / "design.pdf", rows)
    fig8(out / "compensated_paths.pdf")
    fig9(out / "weight.pdf")
    return out


def build(out=OUTPUT, results=S.STUDIES):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = S.load(results)
    fig1(out / "fig1.png")
    fig8(out / "fig8.png")
    for name, draw in (("fig3", fig3), ("fig4", fig4), ("fig5", fig5), ("fig6", fig6)):
        draw(out / f"{name}.png", rows)
    fig7(out / "fig7.png")
    fig9(out / "fig9.png")
    fig10(out / "fig10.png", rows)
    return out


if __name__ == "__main__":
    print(build(*sys.argv[1:]))
