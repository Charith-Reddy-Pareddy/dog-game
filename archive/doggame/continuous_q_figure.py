"""Draws docs/continuous_q_trajectories.svg: greedy equilibrium paths of the continuous Nash-Q networks
(study ``continuous_q_tools``) from four starting positions, one panel per move encoding.

Run with `python3 -m doggame.continuous_q_figure`.
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from doggame import dog_game_summary as S
from doggame.dog_game_study import WALK_HOUSES

OUTPUT = "docs/continuous_q_trajectories.svg"


def build(out=OUTPUT):
    found = {r["key"]["encoding"]: r for r in S.study(S.load(S.STUDIES), "continuous_q_tools")}
    runs = {e: found[e] for e in ("cartesian", "polar", "square") if e in found}
    fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 4.4), squeeze=False)
    colours = plt.cm.viridis(np.linspace(0, 0.85, 4))
    for ax, (encoding, run) in zip(axes[0], runs.items()):
        for (name, path), colour in zip(run["trajectories"].items(), colours):
            red, blue = np.array([p[0] for p in path]), np.array([p[1] for p in path])
            ax.plot(red[:, 0], red[:, 1], "-o", ms=2.5, color=colour, label=name)
            ax.plot(blue[:, 0], blue[:, 1], "--s", ms=2.5, color=colour)
        for house, mark in zip(WALK_HOUSES, ("#c0392b", "#2e6da4")):
            ax.plot(*house, "*", ms=14, color=mark)
        ax.set(xlim=(-0.03, 1.03), ylim=(-0.03, 1.03), aspect="equal", title=encoding, xticks=[0, 0.5, 1], yticks=[0, 0.5, 1])
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Greedy equilibrium paths from four starts: red solid, blue dashed; stars are the houses", fontsize=11)
    fig.savefig(out, bbox_inches="tight"), plt.close(fig)
    return out


if __name__ == "__main__":
    print(build(*sys.argv[1:]))
