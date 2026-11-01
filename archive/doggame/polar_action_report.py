"""Build the concise polar-policy update, its figures, and public-site graph."""

import html
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from doggame.polar_action_study import (
    ALGORITHMS, CHECKPOINTS, EVAL_SAMPLES, EPISODES, OUTPUT, REPRESENTATIONS,
    THRESHOLD, _read,
)

ROOT = Path(__file__).resolve().parents[1]
MARKDOWN = ROOT / "PG_RESEARCH_UPDATE.md"
PDF = ROOT / "docs" / "PG_RESEARCH_UPDATE.pdf"
SITE_FIGURE = ROOT / "docs" / "polar_action_learning.svg"
ACTION_FIGURE = ROOT / "docs" / "polar_action_means.svg"
ROBUSTNESS_FIGURE = ROOT / "docs" / "polar_action_robustness.svg"
ASSUMPTIONS = ROOT / "ASSUMPTIONS.md"
COLORS = {
    "cartesian_gaussian_clipped": "#3568a8",
    "cartesian_beta": "#d18b25",
    "polar_beta": "#168477",
}
LABELS = {
    "cartesian_gaussian_clipped": "Cartesian Gaussian (clipped)",
    "cartesian_beta": "Cartesian Beta",
    "polar_beta": "Polar Beta x von Mises",
}
ALG_LABELS = {"reinforce": "REINFORCE", "a2c": "A2C", "ppo": "PPO"}


def load_complete(path=OUTPUT):
    rows = [r for r in _read(path) if r.get("study") == "polar_action_pg"]
    keys = {(r["key"]["representation"], r["key"]["algorithm"], r["key"]["seed"]) for r in rows}
    expected = {(repr_, alg, seed) for repr_ in REPRESENTATIONS for alg in ALGORITHMS for seed in range(10)}
    missing = expected - keys
    if missing:
        raise RuntimeError(f"Study incomplete: {len(missing)} of 90 runs remain; earliest missing key: {sorted(missing)[0]}")
    return rows


def _group(rows):
    return {
        (repr_, alg): sorted(
            (r for r in rows if r["key"]["representation"] == repr_ and r["key"]["algorithm"] == alg),
            key=lambda r: r["key"]["seed"],
        )
        for repr_ in REPRESENTATIONS for alg in ALGORITHMS
    }


def _summary(rows):
    grouped = _group(rows)
    stats = {}
    for rep in REPRESENTATIONS:
        for alg in ALGORITHMS:
            vals = np.array([r["curve"][str(EPISODES)]["expected_exploitability"] for r in grouped[(rep, alg)]])
            distances = np.array([r["curve"][str(EPISODES)]["mean_action_distance"] for r in grouped[(rep, alg)]])
            boundary = np.array([
                (r["curve"][str(EPISODES)]["red_boundary_mass"] + r["curve"][str(EPISODES)]["blue_boundary_mass"]) / 2
                for r in grouped[(rep, alg)]
            ])
            variance = np.array([
                (r["curve"][str(EPISODES)]["red_var"] + r["curve"][str(EPISODES)]["blue_var"]) / 2
                for r in grouped[(rep, alg)]
            ])
            stats[(rep, alg)] = {
                "n": len(vals), "mean": float(vals.mean()), "sd": float(vals.std(ddof=1)),
                "median": float(np.median(vals)), "converged": int((vals < THRESHOLD).sum()),
                "distance": float(distances.mean()), "variance": float(variance.mean()),
                "boundary": float(boundary.mean()),
            }
    paired = {}
    for alg in ALGORITHMS:
        polar = {r["key"]["seed"]: r["curve"][str(EPISODES)]["expected_exploitability"] for r in grouped[("polar_beta", alg)]}
        for control in ("cartesian_beta", "cartesian_gaussian_clipped"):
            baseline = {r["key"]["seed"]: r["curve"][str(EPISODES)]["expected_exploitability"] for r in grouped[(control, alg)]}
            delta = np.array([polar[s] - baseline[s] for s in sorted(polar)])
            from scipy.stats import t
            margin = float(t.ppf(0.975, len(delta) - 1) * delta.std(ddof=1) / np.sqrt(len(delta)))
            paired[(alg, control)] = (float(delta.mean()), float(delta.mean() - margin), float(delta.mean() + margin))
    return grouped, stats, paired


def load_robustness_complete(path="results/polar_action_pg_robustness.jsonl"):
    rows = [r for r in _read(path) if r.get("study") == "polar_action_pg_robustness"]
    keys = {(r["key"]["representation"], r["key"]["seed"], r["key"]["config"]) for r in rows}
    expected = {(rep, seed, config) for rep in ("cartesian_beta", "polar_beta")
                for seed in range(10) for config in range(1, 5)}
    missing = expected - keys
    if missing:
        raise RuntimeError(f"Robustness study incomplete: {len(missing)} of 80 runs remain; earliest missing key: {sorted(missing)[0]}")
    if len(keys) != len(expected):
        raise RuntimeError("Robustness study has duplicate run keys")
    return rows


def _robustness_summary(rows):
    from scipy.stats import t
    from doggame.experiments import DEFAULT_CONFIGS

    output = []
    for config_id in range(1, 5):
        config_rows = [r for r in rows if r["key"]["config"] == config_id]
        by_rep = {
            rep: {r["key"]["seed"]: r["curve"][str(EPISODES)]["expected_exploitability"]
                  for r in config_rows if r["key"]["representation"] == rep}
            for rep in ("cartesian_beta", "polar_beta")
        }
        paired = np.array([by_rep["polar_beta"][seed] - by_rep["cartesian_beta"][seed]
                           for seed in range(10)])
        margin = float(t.ppf(0.975, len(paired)-1) * paired.std(ddof=1) / np.sqrt(len(paired)))
        cfg = DEFAULT_CONFIGS[config_id]
        output.append({
            "config": config_id,
            "label": f"Red {tuple(cfg['house_red'])} / blue {tuple(cfg['house_blue'])}, w={cfg['w']}",
            "cartesian_mean": float(np.mean(list(by_rep["cartesian_beta"].values()))),
            "cartesian_sd": float(np.std(list(by_rep["cartesian_beta"].values()), ddof=1)),
            "polar_mean": float(np.mean(list(by_rep["polar_beta"].values()))),
            "polar_sd": float(np.std(list(by_rep["polar_beta"].values()), ddof=1)),
            "delta": float(paired.mean()), "low": float(paired.mean()-margin),
            "high": float(paired.mean()+margin), "polar_better": int((paired < 0).sum()),
        })
    return output


def _chart_svg(rows, width=1120, height=430):
    grouped, _, _ = _summary(rows)
    left, panel_w, panel_gap = 72, 326, 28
    top, bottom = 65, height - 88
    y_min = 1e-3
    max_value = max(r["curve"][str(e)]["expected_exploitability"] for r in rows for e in CHECKPOINTS)
    y_max = max(1.0, 10 ** np.ceil(np.log10(max_value)))
    log_min, log_max = np.log10(y_min), np.log10(y_max)
    def y_map(value):
        value = np.clip(value, y_min, y_max)
        return bottom - (np.log10(value) - log_min) / (log_max - log_min) * (bottom - top)
    def x_map(panel, episode):
        x0 = left + panel * (panel_w + panel_gap)
        return x0 + 38 + (episode / EPISODES) * (panel_w - 48)

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
           '<title id="title">Expected one-shot deviation gain across policy-gradient training</title>',
           '<desc id="desc">Median across ten seeds with interquartile whiskers, comparing Cartesian clipped Gaussian, Cartesian Beta, and polar Beta with von Mises angle policies.</desc>',
           '<rect width="100%" height="100%" fill="white"/>']
    for panel, alg in enumerate(ALGORITHMS):
        x0 = left + panel * (panel_w + panel_gap)
        x1 = x0 + panel_w - 5
        svg.append(f'<text x="{(x0+x1)/2:.1f}" y="30" text-anchor="middle" font-size="17" font-weight="700">{ALG_LABELS[alg]}</text>')
        for tick in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
            if tick > y_max:
                continue
            y = y_map(tick)
            svg.append(f'<line x1="{x0+38}" x2="{x1}" y1="{y:.1f}" y2="{y:.1f}" stroke="#d7dde3"/>')
            if panel == 0:
                svg.append(f'<text x="{x0+30}" y="{y+4:.1f}" text-anchor="end" font-size="11" fill="#525b64">{tick:g}</text>')
        for episode in CHECKPOINTS:
            x = x_map(panel, episode)
            svg.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{bottom}" stroke="#f0f2f4"/>')
            svg.append(f'<text x="{x:.1f}" y="{bottom+20}" text-anchor="middle" font-size="10" fill="#525b64">{episode}</text>')
        threshold_y = y_map(THRESHOLD)
        svg.append(f'<line x1="{x0+38}" x2="{x1}" y1="{threshold_y:.1f}" y2="{threshold_y:.1f}" stroke="#8a5b9e" stroke-dasharray="5 4"/>')
        for rep in REPRESENTATIONS:
            color = COLORS[rep]
            group = grouped[(rep, alg)]
            medians, points = [], []
            for episode in CHECKPOINTS:
                vals = np.array([r["curve"][str(episode)]["expected_exploitability"] for r in group])
                q1, median, q3 = np.quantile(vals, [0.25, 0.5, 0.75])
                x, y1, ym, y3 = x_map(panel, episode), y_map(q1), y_map(median), y_map(q3)
                svg.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y3:.1f}" y2="{y1:.1f}" stroke="{color}" stroke-width="2" opacity="0.55"/>')
                svg.append(f'<line x1="{x-4:.1f}" x2="{x+4:.1f}" y1="{y3:.1f}" y2="{y3:.1f}" stroke="{color}" opacity="0.55"/>')
                svg.append(f'<line x1="{x-4:.1f}" x2="{x+4:.1f}" y1="{y1:.1f}" y2="{y1:.1f}" stroke="{color}" opacity="0.55"/>')
                points.append((x, ym))
            path = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            svg.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round"/>')
            for x, y in points:
                svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}"/>')
        svg.append(f'<line x1="{x0+38}" x2="{x1}" y1="{bottom}" y2="{bottom}" stroke="#69727b"/>')
        svg.append(f'<line x1="{x0+38}" x2="{x0+38}" y1="{top}" y2="{bottom}" stroke="#69727b"/>')
    legend_x = 126
    for i, rep in enumerate(REPRESENTATIONS):
        x = legend_x + i * 300
        svg.append(f'<line x1="{x}" x2="{x+28}" y1="{height-25}" y2="{height-25}" stroke="{COLORS[rep]}" stroke-width="3"/>')
        svg.append(f'<text x="{x+36}" y="{height-21}" font-size="12" fill="#27313a">{html.escape(LABELS[rep])}</text>')
    svg.append(f'<line x1="965" x2="991" y1="{height-25}" y2="{height-25}" stroke="#8a5b9e" stroke-dasharray="5 4"/>')
    svg.append(f'<text x="997" y="{height-21}" font-size="10" fill="#525b64">0.004 legacy cutoff</text>')
    svg.append(f'<text x="{width/2}" y="{height-2}" text-anchor="middle" font-size="11" fill="#525b64">Training episodes; y-axis is logarithmic; whiskers show the 25th–75th percentiles.</text>')
    svg.append('</svg>')
    return "\n".join(svg)


def _action_means_svg(rows, width=1120, height=455):
    """Plot seed-level estimated action means against the analytical corners."""
    from doggame.env import DogGameEnv
    from doggame.experiments import DEFAULT_CONFIGS
    from doggame.nash import solve_stage_nash

    env = DogGameEnv(**DEFAULT_CONFIGS[0], discount=0.9)
    nash_red, nash_blue, _ = solve_stage_nash(env.house_red, env.house_blue, env.w, env.domain)
    grouped = _group(rows)
    left, panel_w, gap = 36, 330, 30
    square, top = 250, 92
    colors = {key: value for key, value in COLORS.items()}
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
           '<title id="title">Learned mean actions in the original dog game</title>',
           '<desc id="desc">At episode 1500, each marker is the 8192-sample estimated action mean for one of ten training seeds, evaluated at the reset state. Circles are red-player means, diamonds blue-player means, and stars show the analytical Nash actions.</desc>',
           '<rect width="100%" height="100%" fill="white"/>']
    def star(cx, cy, outer=8, inner=3.6):
        points = []
        for i in range(10):
            angle = -np.pi / 2 + i * np.pi / 5
            radius = outer if i % 2 == 0 else inner
            points.append(f'{cx+radius*np.cos(angle):.1f},{cy+radius*np.sin(angle):.1f}')
        return ' '.join(points)
    for panel, alg in enumerate(ALGORITHMS):
        x0 = left + panel * (panel_w + gap)
        y0 = top
        scale = square
        svg.append(f'<text x="{x0+scale/2}" y="34" text-anchor="middle" font-size="17" font-weight="700">{ALG_LABELS[alg]}</text>')
        svg.append(f'<rect x="{x0}" y="{y0}" width="{scale}" height="{scale}" fill="#fbfcfd" stroke="#69727b"/>')
        for tick in (0, .25, .5, .75, 1):
            x, y = x0 + tick * scale, y0 + (1-tick) * scale
            svg.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y0}" y2="{y0+scale}" stroke="#e2e6e9"/>')
            svg.append(f'<line x1="{x0}" x2="{x0+scale}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e2e6e9"/>')
            svg.append(f'<text x="{x:.1f}" y="{y0+scale+17}" text-anchor="middle" font-size="10" fill="#525b64">{tick:g}</text>')
            svg.append(f'<text x="{x0-7}" y="{y+3:.1f}" text-anchor="end" font-size="10" fill="#525b64">{tick:g}</text>')
        for rep in REPRESENTATIONS:
            color = colors[rep]
            for row in grouped[(rep, alg)]:
                final = row["curve"][str(EPISODES)]
                for key, shape in (("red_mean", "circle"), ("blue_mean", "diamond")):
                    mx, my = final[key]
                    cx, cy = x0 + mx*scale, y0 + (1-my)*scale
                    if shape == "circle":
                        svg.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.6" fill="{color}" fill-opacity="0.75"/>')
                    else:
                        svg.append(f'<polygon points="{cx:.1f},{cy-4.2:.1f} {cx+4.2:.1f},{cy:.1f} {cx:.1f},{cy+4.2:.1f} {cx-4.2:.1f},{cy:.1f}" fill="{color}" fill-opacity="0.75"/>')
        for action in (nash_red, nash_blue):
            cx, cy = x0 + action[0]*scale, y0 + (1-action[1])*scale
            svg.append(f'<polygon points="{star(cx,cy)}" fill="#252525" stroke="white" stroke-width="1.2"/>')
    legend_y = height - 43
    for i, rep in enumerate(REPRESENTATIONS):
        x = 65 + i*300
        svg.append(f'<rect x="{x}" y="{legend_y-9}" width="14" height="14" rx="3" fill="{colors[rep]}"/>')
        svg.append(f'<text x="{x+22}" y="{legend_y+2}" font-size="11" fill="#27313a">{html.escape(LABELS[rep])}</text>')
    svg.append(f'<text x="950" y="{legend_y+2}" font-size="10" fill="#27313a">circle: red · diamond: blue · star: Nash</text>')
    svg.append(f'<text x="{width/2}" y="{height-5}" text-anchor="middle" font-size="10" fill="#525b64">Seed-level mean actions at the reset state; each mean estimated from 8,192 samples (not individual action samples).</text>')
    svg.append('</svg>')
    return "\n".join(svg)


def _robustness_svg(rows, width=1120, height=355):
    summaries = _robustness_summary(rows)
    low = min(0.0, *(item["low"] for item in summaries))
    high = max(0.0, *(item["high"] for item in summaries))
    pad = .12 * (high-low)
    low, high = low-pad, high+pad
    left, right = 480, 1065
    def x_map(value):
        return left + (value-low)/(high-low)*(right-left)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
           '<title id="title">Polar minus Cartesian Beta robustness by house setup</title>',
           '<desc id="desc">Four paired REINFORCE comparisons, ten seeds each. Horizontal lines show exploratory 95 percent paired Student-t intervals. Negative values favor polar actions.</desc>',
           '<rect width="100%" height="100%" fill="white"/>',
           '<text x="560" y="27" font-size="17" font-weight="700">Cross-house robustness: polar minus Cartesian Beta</text>']
    for tick in np.linspace(low, high, 5):
        x = x_map(tick)
        svg.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="50" y2="295" stroke="#e2e6e9"/>')
        svg.append(f'<text x="{x:.1f}" y="318" text-anchor="middle" font-size="11" fill="#525b64">{tick:.3f}</text>')
    zero = x_map(0)
    svg.append(f'<line x1="{zero:.1f}" x2="{zero:.1f}" y1="48" y2="296" stroke="#333" stroke-width="1.5" stroke-dasharray="5 4"/>')
    for i, item in enumerate(summaries):
        y = 76 + i*64
        svg.append(f'<text x="28" y="{y+4}" font-size="13" fill="#27313a">{html.escape(item["label"])}</text>')
        x1, x2, xm = x_map(item["low"]), x_map(item["high"]), x_map(item["delta"])
        svg.append(f'<line x1="{x1:.1f}" x2="{x2:.1f}" y1="{y}" y2="{y}" stroke="#168477" stroke-width="3"/>')
        svg.append(f'<line x1="{x1:.1f}" x2="{x1:.1f}" y1="{y-6}" y2="{y+6}" stroke="#168477" stroke-width="1.5"/>')
        svg.append(f'<line x1="{x2:.1f}" x2="{x2:.1f}" y1="{y-6}" y2="{y+6}" stroke="#168477" stroke-width="1.5"/>')
        svg.append(f'<circle cx="{xm:.1f}" cy="{y}" r="5" fill="#168477" stroke="white" stroke-width="1.2"/>')
    svg.append('<text x="770" y="344" text-anchor="middle" font-size="11" fill="#525b64">Negative favors polar; intervals are exploratory and unadjusted across four setups.</text>')
    svg.append('</svg>')
    return "\n".join(svg)


def _pipeline_drawing():
    from reportlab.graphics.shapes import Drawing, Line, Rect, String
    from reportlab.lib import colors

    d = Drawing(470, 120)
    boxes = [(5, 42, 80, 38, "dog state"), (105, 42, 100, 38, "two PG actors"),
             (225, 42, 85, 38, "(theta, u)"), (330, 42, 120, 38, "map to square")]
    for x, y, w, h, label in boxes:
        d.add(Rect(x, y, w, h, rx=6, ry=6, fillColor=colors.HexColor("#f3f6f8"), strokeColor=colors.HexColor("#82909b")))
        d.add(String(x+w/2, y+14, label, fontSize=9, textAnchor="middle", fillColor=colors.HexColor("#24313a")))
    for x1, x2 in ((85, 105), (205, 225), (310, 330)):
        d.add(Line(x1, 61, x2, 61, strokeColor=colors.HexColor("#495966"), strokeWidth=1.2, endArrowType="filled"))
    d.add(String(60, 23, "s = dog (x, y)", fontSize=8, textAnchor="middle"))
    d.add(String(265, 23, "Von Mises angle + Beta ray fraction", fontSize=8, textAnchor="middle"))
    d.add(String(389, 23, "feasible absolute picks", fontSize=8, textAnchor="middle"))
    d.add(String(235, 106, "u in [0, 1]; r = h*u / max(|cos(theta)|, |sin(theta)|)", fontSize=9, textAnchor="middle", fillColor=colors.HexColor("#168477")))
    return d


def _learning_drawing(rows):
    from reportlab.graphics.shapes import Drawing, Line, String, Circle
    from reportlab.lib import colors
    grouped, _, _ = _summary(rows)
    d = Drawing(470, 205)
    left, top, bottom, panel_w, gap = 35, 30, 164, 125, 20
    y_min = 1e-3
    max_value = max(r["curve"][str(e)]["expected_exploitability"] for r in rows for e in CHECKPOINTS)
    y_max = max(1.0, 10 ** np.ceil(np.log10(max_value)))
    log_min, log_max = np.log10(y_min), np.log10(y_max)
    def y_map(v):
        return bottom - (np.log10(np.clip(v, y_min, y_max)) - log_min) / (log_max-log_min) * (bottom-top)
    for ai, alg in enumerate(ALGORITHMS):
        x0 = left + ai * (panel_w + gap)
        d.add(String(x0 + panel_w/2, 190, ALG_LABELS[alg], fontSize=9, textAnchor="middle"))
        for tick in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
            if tick > y_max:
                continue
            y = y_map(tick)
            d.add(Line(x0+20, y, x0+panel_w, y, strokeColor=colors.HexColor("#e0e5e9"), strokeWidth=0.5))
        threshold_y = y_map(THRESHOLD)
        d.add(Line(x0+20, threshold_y, x0+panel_w, threshold_y,
                   strokeColor=colors.HexColor("#8a5b9e"), strokeWidth=.8, strokeDashArray=[3, 2]))
        for rep in REPRESENTATIONS:
            color = colors.HexColor(COLORS[rep])
            group = grouped[(rep, alg)]
            pts = []
            for ei, episode in enumerate(CHECKPOINTS):
                vals = np.array([r["curve"][str(episode)]["expected_exploitability"] for r in group])
                median = np.median(vals)
                x = x0 + 23 + ei * (panel_w - 30) / (len(CHECKPOINTS)-1)
                y = y_map(median)
                q1, q3 = np.quantile(vals, [0.25, 0.75])
                d.add(Line(x, y_map(q1), x, y_map(q3), strokeColor=color, strokeWidth=1.3))
                pts.append((x, y))
            for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
                d.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=1.7))
            for x, y in pts:
                d.add(Circle(x, y, 2, fillColor=color, strokeColor=color))
        d.add(Line(x0+20, top, x0+20, bottom, strokeColor=colors.HexColor("#69727b"), strokeWidth=.7))
        d.add(Line(x0+20, top, x0+panel_w, top, strokeColor=colors.HexColor("#69727b"), strokeWidth=.7))
    d.add(String(235, 8, "episodes (median line, IQR whiskers)", fontSize=8, textAnchor="middle"))
    d.add(String(2, 177, "Expected gain (log scale)", fontSize=7))
    d.add(String(345, 8, "dashed: 0.004", fontSize=7, fillColor=colors.HexColor("#8a5b9e")))
    return d


def _action_means_drawing(rows):
    from reportlab.graphics.shapes import Circle, Drawing, Line, Polygon, String
    from reportlab.lib import colors
    from doggame.env import DogGameEnv
    from doggame.experiments import DEFAULT_CONFIGS
    from doggame.nash import solve_stage_nash

    env = DogGameEnv(**DEFAULT_CONFIGS[0], discount=0.9)
    nash_red, nash_blue, _ = solve_stage_nash(env.house_red, env.house_blue, env.w, env.domain)
    grouped = _group(rows)
    d = Drawing(470, 235)
    square, y0 = 125, 45
    for ai, alg in enumerate(ALGORITHMS):
        x0 = 8 + ai * 154
        d.add(String(x0+square/2, 219, ALG_LABELS[alg], fontSize=8.5, textAnchor="middle"))
        for tick in (0, .25, .5, .75, 1):
            x, y = x0+tick*square, y0+(1-tick)*square
            d.add(Line(x, y0, x, y0+square, strokeColor=colors.HexColor("#e2e6e9"), strokeWidth=.45))
            d.add(Line(x0, y, x0+square, y, strokeColor=colors.HexColor("#e2e6e9"), strokeWidth=.45))
        for rep in REPRESENTATIONS:
            color = colors.HexColor(COLORS[rep])
            for row in grouped[(rep, alg)]:
                final = row["curve"][str(EPISODES)]
                for key, is_red in (("red_mean", True), ("blue_mean", False)):
                    mx, my = final[key]
                    cx, cy = x0+mx*square, y0+(1-my)*square
                    if is_red:
                        d.add(Circle(cx, cy, 2.5, fillColor=color, strokeColor=color))
                    else:
                        d.add(Polygon([cx,cy+3,cx+3,cy,cx,cy-3,cx-3,cy], fillColor=color, strokeColor=color))
        for action in (nash_red, nash_blue):
            cx, cy = x0+action[0]*square, y0+(1-action[1])*square
            pts=[]
            for i in range(10):
                a=-np.pi/2+i*np.pi/5
                r=4 if i%2==0 else 1.8
                pts.extend([cx+r*np.cos(a),cy+r*np.sin(a)])
            d.add(Polygon(pts, fillColor=colors.HexColor("#252525"), strokeColor=colors.white, strokeWidth=.5))
    d.add(String(235, 27, "Mean actions: circle red, diamond blue; black stars: analytical Nash", fontSize=7.5, textAnchor="middle"))
    d.add(String(235, 10, "Ten seeds per policy; each mean estimated from 8,192 action samples at reset state", fontSize=7, textAnchor="middle"))
    return d


def _robustness_drawing(rows):
    from reportlab.graphics.shapes import Circle, Drawing, Line, String
    from reportlab.lib import colors
    from doggame.experiments import DEFAULT_CONFIGS as DEFAULT_CONFIGS_FOR_REPORT

    summaries = _robustness_summary(rows)
    low = min(0.0, *(item["low"] for item in summaries))
    high = max(0.0, *(item["high"] for item in summaries))
    pad = .12 * (high-low)
    low, high = low-pad, high+pad
    d = Drawing(470, 215)
    left, right = 165, 455
    def x_map(value):
        return left + (value-low)/(high-low)*(right-left)
    for tick in np.linspace(low, high, 5):
        x = x_map(tick)
        d.add(Line(x, 38, x, 180, strokeColor=colors.HexColor("#e2e6e9"), strokeWidth=.6))
        d.add(String(x, 23, f"{tick:.2f}", fontSize=6.5, textAnchor="middle", fillColor=colors.HexColor("#525b64")))
    zero = x_map(0)
    d.add(Line(zero, 35, zero, 181, strokeColor=colors.HexColor("#333333"), strokeWidth=1, strokeDashArray=[3,2]))
    for i, item in enumerate(summaries):
        y = 164-i*35
        d.add(String(2, y-2, f"Cfg {item['config']} (w={DEFAULT_CONFIGS_FOR_REPORT[item['config']]['w']})", fontSize=7.2))
        x1, x2, xm = x_map(item["low"]), x_map(item["high"]), x_map(item["delta"])
        d.add(Line(x1, y, x2, y, strokeColor=colors.HexColor("#168477"), strokeWidth=2))
        d.add(Line(x1, y-3, x1, y+3, strokeColor=colors.HexColor("#168477"), strokeWidth=.8))
        d.add(Line(x2, y-3, x2, y+3, strokeColor=colors.HexColor("#168477"), strokeWidth=.8))
        d.add(Circle(xm, y, 2.8, fillColor=colors.HexColor("#168477"), strokeColor=colors.white, strokeWidth=.7))
    d.add(String(235, 7, "Polar - Cartesian Beta expected-gain difference; negative favors polar", fontSize=7, textAnchor="middle"))
    return d


def build_markdown(rows, path=MARKDOWN, robustness_rows=None):
    _, stats, paired = _summary(rows)
    lines = [
        "# Dog Game: concise policy-gradient update",
        "",
        "**Prepared 15 October 2026 | Research update for discussion with a professor or primary researcher**",
        "",
        "## What was implemented",
        "",
        "The original dog game now supports an absolute polar action policy: each player samples an angle and a normalized ray fraction from the center of the square, which is mapped exactly to a feasible point. The earlier angle-radius policy remains the separate walking game. The DQN/Q-value sketches in the handwritten notes were excluded as requested.",
        "",
        "## Results worth discussing",
        "",
        "The primary endpoint is the maximum one-shot unilateral expected-payoff gain at episode 1,500, calculated exactly from Monte Carlo-estimated action means and variances. It is evaluated at the reset state, not over the full state distribution. Paired-seed polar contrasts against both Cartesian controls are shown below for every algorithm; negative differences favor polar actions. These ten-seed intervals are exploratory, not confirmatory evidence or a novelty claim.",
        "",
        "On the primary setup, the polar policy had higher expected gain than Cartesian Beta in all 30 algorithm-by-seed pairs. The four-house follow-up is mixed: two intervals favor Cartesian Beta, while two include zero; none show a clear polar advantage. This does not support a general improvement from the polar policy package under the tested settings.",
        "",
        "{{figure:polar_learning}}",
        "",
        "| Algorithm | Clipped Gaussian | Cartesian Beta | Polar Beta x von Mises |",
        "|---|---:|---:|---:|",
    ]
    for alg in ALGORITHMS:
        cells = []
        for rep in REPRESENTATIONS:
            s = stats[(rep, alg)]
            cells.append(f"{s['mean']:.4f} ({s['sd']:.4f}); {s['converged']}/10 < 0.004")
        lines.append(f"| {ALG_LABELS[alg]} | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "| Algorithm | Polar − Cartesian Beta | Polar − clipped Gaussian |",
        "|---|---:|---:|",
    ])
    for alg in ALGORITHMS:
        beta = paired[(alg, "cartesian_beta")]
        gaussian = paired[(alg, "cartesian_gaussian_clipped")]
        lines.append(f"| {ALG_LABELS[alg]} | {beta[0]:.4f} [{beta[1]:.4f}, {beta[2]:.4f}] | {gaussian[0]:.4f} [{gaussian[1]:.4f}, {gaussian[2]:.4f}] |")
    lines.extend([
        "",
        "{{figure:polar_actions}}",
        "",
        "| Algorithm | Representation | Mean action distance to Nash | Mean per-player variance trace |",
        "|---|---|---:|---:|",
    ])
    for alg in ALGORITHMS:
        for rep in REPRESENTATIONS:
            s = stats[(rep, alg)]
            lines.append(f"| {ALG_LABELS[alg]} | {LABELS[rep]} | {s['distance']:.4f} | {s['variance']:.5f} |")
    lines.extend([
        "",
        "The analytical equilibrium in this setup is a deterministic corner pair. Every tested actor is stochastic at finite parameters, so action variance can leave positive expected gain even when the mean actions are close to Nash; mean distance alone is not an equilibrium test. The variance column makes that distinction visible.",
        "",
        "These values are distribution-aware one-shot unilateral gains: action means and covariance traces are estimated from 8,192 samples, and the exact best response to the opponent's estimated moments is projected onto the square. The threshold 0.004 was originally chosen for deterministic action pairs, so its counts here are descriptive only.",
        "",
        "## Policy sketch and assumptions",
        "",
        "{{figure:polar_pipeline}}",
        "",
        "The state is the current dog location; each house is fixed per training run. The user confirmed that polar actions are absolute points from the square center. Radius is a fraction of the direction-dependent center-to-edge distance. The policy uses a von Mises angle and Beta radial fraction, both with shape/concentration at least 1. See the separate [assumptions register](ASSUMPTIONS.md) and the frozen [study protocol](docs/POLAR_ACTION_PROTOCOL.md).",
        "",
        "## Limits and relation to prior work",
        "",
        "The primary comparison is one house setup with ten seeds. It compares policy packages, not pure coordinate geometry: the clipped-Gaussian control retains its clipping/log-density behavior, while both Beta policies have exact bounded support. The intervals are exploratory and do not establish generalization or a new algorithm. Chou et al. analyze bounded Beta policies ([ICML/PMLR 2017](https://proceedings.mlr.press/v70/chou17a.html)); gradient dynamics in continuous games need not converge universally ([UAI/PMLR 2020](https://proceedings.mlr.press/v115/chasnov20a.html)). The dog game also overlaps the static structure of the [Battling Influencers Game](https://arxiv.org/abs/2502.01127). Reddit discussions were treated as informal implementation anecdotes, not evidence for research claims.",
        "",
        "No DQN, action-value network, chase game, or full dynamic equilibrium result is included in this update.",
    ])
    if robustness_rows is not None:
        robustness = _robustness_summary(robustness_rows)
        lines.extend([
            "",
            "## Cross-house robustness follow-up",
            "",
            "This exploratory follow-up was specified after the primary study began and is kept separate from its pre-run protocol. It compares REINFORCE with polar and Cartesian Beta policies on the four other predefined house setups, ten paired seeds each. Negative differences favor polar. Intervals are unadjusted and reflect seed variation, not Monte Carlo scoring error. The follow-up protocol is in [docs/POLAR_ACTION_ROBUSTNESS_PROTOCOL.md](docs/POLAR_ACTION_ROBUSTNESS_PROTOCOL.md).",
            "",
            "{{figure:polar_robustness}}",
            "",
            "| House configuration | Cartesian Beta mean (SD) | Polar mean (SD) | Polar − Cartesian Beta [95% t interval] | Polar lower in paired seeds |",
            "|---|---:|---:|---:|---:|",
        ])
        for item in robustness:
            lines.append(
                f"| {item['label']} | {item['cartesian_mean']:.4f} ({item['cartesian_sd']:.4f}) "
                f"| {item['polar_mean']:.4f} ({item['polar_sd']:.4f}) "
                f"| {item['delta']:.4f} [{item['low']:.4f}, {item['high']:.4f}] "
                f"| {item['polar_better']}/10 |"
            )
    Path(path).write_text("\n".join(lines) + "\n")


def build_pdf(rows, robustness_rows, source=MARKDOWN, output=PDF):
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.platypus import KeepTogether, SimpleDocTemplate, Spacer
    from doggame.pdf_kit import kit
    from doggame.research_report_pdf import column_widths, inline, parse

    k = kit()
    blocks = parse(Path(source).read_text())
    story = []
    for kind, payload in blocks:
        if kind == "title":
            story += [k.h1(inline(payload)), Spacer(1, 4)]
        elif kind == "h2":
            story.append(k.h2(inline(payload)))
        elif kind == "para":
            story.append(k.text(inline(payload)))
        elif kind == "table":
            story += [k.table([[inline(c) for c in row] for row in payload], column_widths(payload, 6.85 * inch)), Spacer(1, 5)]
        elif kind == "figure":
            if payload == "polar_learning":
                drawing = _learning_drawing(rows)
            elif payload == "polar_actions":
                drawing = _action_means_drawing(rows)
            elif payload == "polar_robustness":
                drawing = _robustness_drawing(robustness_rows)
            else:
                drawing = _pipeline_drawing()
            story.append(KeepTogether([drawing, Spacer(1, 5)]))
    SimpleDocTemplate(str(output), pagesize=letter, leftMargin=.72*inch, rightMargin=.72*inch,
                      topMargin=.55*inch, bottomMargin=.55*inch,
                      title="Dog Game: concise policy-gradient update", author="").build(story)


def build_all(path=OUTPUT):
    rows = load_complete(path)
    robustness_rows = load_robustness_complete()
    SITE_FIGURE.write_text(_chart_svg(rows))
    ACTION_FIGURE.write_text(_action_means_svg(rows))
    ROBUSTNESS_FIGURE.write_text(_robustness_svg(robustness_rows))
    build_markdown(rows, robustness_rows=robustness_rows)
    build_pdf(rows, robustness_rows)
    print(f"wrote {MARKDOWN}, {PDF}, {SITE_FIGURE}, {ACTION_FIGURE}, and {ROBUSTNESS_FIGURE}")


if __name__ == "__main__":
    build_all()
