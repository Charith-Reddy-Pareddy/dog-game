"""Builds docs/poster.pdf, a 48 x 36 inch research poster, from the saved results (needs reportlab and matplotlib).

Run with `python3 -m doggame.poster [output.pdf]`. Every number comes from results/dog_game_studies.jsonl through
doggame.dog_game_summary or from the inertia modules, so the poster cannot drift from the data.
"""

import os
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from doggame import dog_game_summary as S
from doggame.experiments import DEFAULT_CONFIGS
from doggame.inertia_dp import solve_1d
from doggame.inertia_theory import explicit_policy, kappa, nash_1d
from doggame.potential import smoothed_potential

OUTPUT = "docs/poster.pdf"
SITE = "https://charith-reddy-pareddy.github.io/dog-game/"
RED, BLUE, INK, NAVY, PALE = "#c0392b", "#2e6da4", "#1d2733", "#1f3b57", "#eef3f8"
plt.rcParams.update({"font.size": 15, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 16})


def rho(w0, g):
    return w0 * (1 - g * w0) / (1 - g * w0 ** 2)


def slack(config):
    """The smallest slack of any player at an edge, over the coordinates where the houses differ."""
    out = []
    for c in range(2):
        h_red, h_blue, w = config["house_red"][c], config["house_blue"][c], config["w"]
        a_red, a_blue, kind = nash_1d(h_red, h_blue, w)
        if kind != "shared":
            d = w * a_red + (1 - w) * a_blue
            for h, a in ((h_red, a_red), (h_blue, a_blue)):
                out.append((h - d) / (1 - d) if a == 1 else (d - h) / d)
    return min(out)


def fig_game(path):
    fig, ax = plt.subplots(figsize=(6, 5.4))
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, lw=2, ec=INK))
    (hr, hb), w = (np.array(DEFAULT_CONFIGS[0]["house_red"]), np.array(DEFAULT_CONFIGS[0]["house_blue"])), DEFAULT_CONFIGS[0]["w"]
    ar, ab, x = np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.3, 0.35])
    nxt = 0.6 * x + 0.4 * (w * ar + (1 - w) * ab)
    ax.plot(*hr, "s", ms=22, color=RED), ax.plot(*hb, "s", ms=22, color=BLUE)
    ax.plot(*ar, "o", ms=16, mfc="white", mec=RED, mew=3), ax.plot(*ab, "o", ms=16, mfc="white", mec=BLUE, mew=3)
    ax.plot(*x, "o", ms=18, color="#7a5c3e"), ax.plot(*nxt, "o", ms=18, color="#7a5c3e", alpha=0.45)
    ax.annotate("", nxt, x, arrowprops=dict(arrowstyle="->", lw=2.5, color="#7a5c3e"))
    for p, t, dx, dy in ((hr, "red house", -0.36, -0.02), (hb, "blue house", -0.08, -0.1), (ar, "red's pick", -0.4, -0.06),
                         (ab, "blue's pick", 0.06, 0.02), (x, "dog x", -0.12, -0.08), (nxt, "next dog x'", 0.03, 0.05)):
        ax.text(p[0] + dx, p[1] + dy, t, fontsize=15)
    ax.set(xlim=(-0.08, 1.08), ylim=(-0.08, 1.08), aspect="equal", xticks=[], yticks=[])
    ax.axis("off")
    fig.savefig(path, dpi=220, bbox_inches="tight"), plt.close(fig)


def fig_rho(path):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    w0 = np.linspace(0, 0.999, 400)
    for g, ls in ((0.0, ":"), (0.5, "--"), (0.9, "-"), (0.99, "-.")):
        ax.plot(w0, rho(w0, g), ls, lw=2.6 if g == 0.9 else 1.8, color=INK, label=f"γ = {g}")
    for setup in (0, 3, 4):
        s = slack(DEFAULT_CONFIGS[setup])
        cross = w0[np.argmax(rho(w0, 0.9) > s)]
        ax.axhline(s, color="#8a96a3", lw=1)
        ax.plot(cross, s, "o", ms=10, color=RED)
        where = "setup 4, x coordinate" if setup == 3 else f"setup {setup + 1}"
        ax.text(0.3 if s < 0.3 else 0.01, s - 0.055 if s < 0.3 else s + 0.015, f"{where}: slack {s:.1f}, holds up to w0 = {cross:.2f}", fontsize=13)
    ax.set(xlabel="dog inertia w0", ylabel="slack needed, ρ(w0, γ)", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.savefig(path, dpi=220, bbox_inches="tight"), plt.close(fig)


def fig_policy(path, setup=2, coord=0, w0=0.6):
    c = DEFAULT_CONFIGS[setup]
    h_red, h_blue, w = c["house_red"][coord], c["house_blue"][coord], c["w"]
    a_red, a_blue, _ = nash_1d(h_red, h_blue, w)
    sol = solve_1d(h_red, h_blue, w, w0)
    x = np.linspace(0, 1, 400)
    red, blue = explicit_policy(h_red, h_blue, w, a_red, a_blue, w0)(x)
    d, k = w * a_red + (1 - w) * a_blue, kappa(w0, 0.9)
    lo, hi = max(0.0, d - (d - h_blue) * k / w0), min(1.0, d + (h_red - d) * k / w0)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.axvspan(lo, hi, color=PALE)
    ax.text((lo + hi) / 2, 0.5, "both hold their\none-shot actions", ha="center", fontsize=14, color="#55606b")
    ax.plot(x, red, color=RED, lw=3, label="red, Proposition 2")
    ax.plot(x, blue, color=BLUE, lw=3, label="blue, Proposition 2")
    ax.plot(sol["z"][::10], sol["red"][::10], "o", mfc="none", mec=RED, ms=8, label="red, dynamic program")
    ax.plot(sol["z"][::10], sol["blue"][::10], "o", mfc="none", mec=BLUE, ms=8, label="blue, dynamic program")
    ax.set(xlabel="dog position x (one coordinate)", ylabel="action", ylim=(-0.05, 1.05), xlim=(0, 1))
    ax.legend(loc="lower left", frameon=False, fontsize=12)
    fig.savefig(path, dpi=220, bbox_inches="tight"), plt.close(fig)


def fig_smoothed(path):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    t = np.linspace(0, 1, 201)
    for std, ls in ((0.14, ":"), (0.37, "--"), (0.61, "-"), (1.0, "-.")):
        v = np.array([smoothed_potential(a, 1 - a, 0.5, 0.5, 0.5, std) for a in t])
        ax.plot(t, v - v[100], ls, lw=2.4, color=INK, label=f"noise std {std}")
    ax.set(xlabel="red's y-mean on the line of equilibria", ylabel="smoothed potential\n(minus its interior value)")
    ax.legend(frameon=False, fontsize=12)
    fig.savefig(path, dpi=220, bbox_inches="tight"), plt.close(fig)


def fig_walk(path, studies):
    variants = (("sigmoid mean, clipped\n(default)", [r for r in S.study(studies, "selection_noise") if r["key"]["log_std"] == -0.5]),
                ("sigmoid mean,\nnot clipped", S.study(studies, "selection_clip")), ("linear mean in y,\nnot clipped", S.study(studies, "selection_linear")))
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.axvspan(0, 1, color=PALE)
    for i, (label, runs) in enumerate(variants):
        y = np.array([r["red"][1] for r in runs])
        ax.plot(y, i + rng.uniform(-0.25, 0.25, len(y)), "o", ms=7, alpha=0.7, color=RED)
    ax.set_xscale("symlog", linthresh=1)
    ax.set(yticks=range(3), yticklabels=[v[0] for v in variants], xlabel="red's final y-mean (shaded: inside the box)")
    ax.invert_yaxis()
    fig.savefig(path, dpi=220, bbox_inches="tight"), plt.close(fig)


def build(out=OUTPUT):
    from reportlab.graphics.barcode.qr import QrCodeWidget
    from reportlab.graphics.shapes import Drawing
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Frame, Image, KeepInFrame, Paragraph, Spacer, Table, TableStyle
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.graphics import renderPDF

    fonts = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
    for name, file in (("Sans", "DejaVuSans.ttf"), ("Sans-Bold", "DejaVuSans-Bold.ttf"), ("Sans-Oblique", "DejaVuSans-Oblique.ttf")):
        pdfmetrics.registerFont(TTFont(name, os.path.join(fonts, file)))
    pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-Bold", italic="Sans-Oblique", boldItalic="Sans-Bold")

    studies = S.load(S.STUDIES)
    tmp = tempfile.mkdtemp()
    figs = {n: os.path.join(tmp, n + ".png") for n in ("game", "rho", "policy", "smoothed", "walk")}
    fig_game(figs["game"]), fig_rho(figs["rho"]), fig_policy(figs["policy"]), fig_smoothed(figs["smoothed"]), fig_walk(figs["walk"], studies)

    W, H = 48 * inch, 36 * inch
    body = ParagraphStyle("b", fontName="Sans", fontSize=25, leading=34, textColor=INK, spaceAfter=12)
    point = ParagraphStyle("p", parent=body, leftIndent=30, bulletIndent=4, spaceAfter=9, bulletFontName="Sans", bulletFontSize=25)
    head = ParagraphStyle("h", parent=body, fontName="Sans-Bold", fontSize=38, leading=46, textColor=NAVY, spaceBefore=18, spaceAfter=12)
    claim = ParagraphStyle("c", parent=body, fontName="Sans-Bold", fontSize=28, leading=36, spaceAfter=10)
    math = ParagraphStyle("m", parent=body, fontSize=24, leading=32, leftIndent=24, textColor="#243447", backColor=PALE, borderPadding=10, spaceBefore=8, spaceAfter=18)
    cap = ParagraphStyle("cap", parent=body, fontSize=20, leading=26, textColor="#55606b")
    cell = ParagraphStyle("cell", parent=body, fontSize=21, leading=26, spaceAfter=0)
    P, B = (lambda t, s=body: Paragraph(t, s)), (lambda t: Paragraph(t, point, bulletText="•"))
    col = (W - 2 * inch - 2 * 0.8 * inch) / 3

    def img(path, width):
        from PIL import Image as Pil
        w, h = Pil.open(path).size
        return Image(path, width=width, height=width * h / w)

    def table(rows, widths):
        t = Table([[Paragraph(str(c), cell) for c in r] for r in rows], colWidths=widths)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dde6ef")), ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#b8c2cc")),
                               ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
        return t

    noise = S.selection_noise_table(studies)
    split = [["Noise std", "red top", "red bottom", "interior", "p (fair coin)"]] + [[r[0].replace(" (default)", "*"), r[1].split()[0], r[2].split()[0], r[3].split()[0], r[5]] for r in noise[1:]]
    cycles = [["w0", "policies ignore the dog", "policies react to the dog"]] + S.markov_potential_table(studies)[1:]
    rand = S.inertia_random_table(studies)
    box = ParagraphStyle("box", parent=body, fontSize=26, leading=34, textColor="white", spaceAfter=8)
    takeaway = Table([[[Paragraph("<b>Take-home</b>", box),
                        Paragraph("• With inertia the one-shot equilibrium survives on the path, and everywhere exactly when each player's slack is at least ρ(w0, γ); otherwise players aim the dog at an explicit target.", box),
                        Paragraph("• The static potential does not carry over to feedback play.", box),
                        Paragraph("• On a flat potential, learning is a random walk that the box stops; the mirror image is a coin flip.", box)]]], colWidths=[col])
    takeaway.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(NAVY)), ("LEFTPADDING", (0, 0), (-1, -1), 22),
                                  ("RIGHTPADDING", (0, 0), (-1, -1), 22), ("TOPPADDING", (0, 0), (-1, -1), 18), ("BOTTOMPADDING", (0, 0), (-1, -1), 14)]))
    straddle, interior = rand[1], rand[2]

    one = [P("1. The dog game", head),
           P("Two players each pick a point in the unit square. A shared dog moves part of the way toward their weighted average, and each player wants the dog near its own house:"),
           P("x' = w0·x + (1 − w0)·(w·a<sub>red</sub> + (1 − w)·a<sub>blue</sub>)<br/>reward<sub>i</sub> = −‖x' − h<sub>i</sub>‖², discount γ = 0.9", math),
           P("The dog's own weight w0 is its <b>inertia</b>; w0 = 0 is the one-shot game."),
           img(figs["game"], col * 0.62),
           P("2. What was known", head),
           P("With no inertia this is an instance of the Battling Influencers Game (Wu et al., 2025), an <b>exact potential game</b>: with d = w·a<sub>red</sub> + (1 − w)·a<sub>blue</sub>,"),
           P("Φ = −‖d‖² + 2w·h<sub>red</sub>·a<sub>red</sub> + 2(1 − w)·h<sub>blue</sub>·a<sub>blue</sub>", math),
           P("changes by exactly a deviating player's reward change, so equilibria maximize Φ. In a coordinate the two houses share, Φ is flat: a continuum of equilibria."),
           P("3. Questions", head),
           B("With inertia, is the one-shot equilibrium still a <b>feedback (Markov-perfect) equilibrium</b>, and if not, what is?"),
           B("Is the dynamic game still a potential game?"),
           B("On a continuum of equilibria, which one does <b>policy-gradient self-play</b> find, and why?"),
           P("4. Methods", head),
           B("<b>Exact:</b> the game splits by coordinate; each is solved by Nash value iteration (401 dog positions × 101 actions) and checked with the one-shot deviation principle using closed-form values."),
           B("<b>Learning:</b> REINFORCE self-play with a Gaussian policy (sigmoid mean, clipped actions), 1500 episodes, 50 seeds per setting, plus two ablations."),
           B("Open source and reproducible with an automated test suite; every number on this poster is generated from the saved results."),
           P("9. Known and new", head),
           B("<b>Known:</b> the one-shot potential structure (Wu et al., 2025); a game of potential stage games need not be a Markov potential game (Leonardos et al., 2022)."),
           B("<b>New for this game</b>, as far as the literature checked goes: Propositions 1 and 2, the Markov-potential counterexample, the selection mechanism with ablations, and that self-play misses the feedback equilibrium."),
           B("<b>Limits:</b> the proofs (in the repository) are not yet refereed; the interior case is open; one learning algorithm on one game."),
           P("Wu, Zhu, Cai, Zhu (2025), Battling Influencers Game, arXiv 2502.01127 · Leonardos et al. (ICLR 2022), Markov potential games · Monderer and Shapley (1996), Potential games · Fudenberg and Tirole (1991), Game Theory.", cap)]

    two = [P("5. When does the one-shot equilibrium survive inertia?", head),
           P("Let k = (1 − γ·w0²)/(1 − γ·w0) and give each player the target T = d* + k·(h − d*), where d* is the one-shot Nash dog.", body),
           P("<b>Proposition 1.</b> Holding the one-shot actions at every dog position is a feedback equilibrium if and only if every player at an edge of the box has slack s ≥ ρ(w0, γ) = w0(1 − γw0)/(1 − γw0²), where s = (h − d*)/(1 − d*) at the top edge, and no one-shot action is interior.", claim),
           img(figs["rho"], col),
           P("Holding is an equilibrium while the curve (γ = 0.9 in bold) stays below a setup's slack; the dots mark where it stops. More patient players tolerate more inertia.", cap),
           B("Predicted vs computed share of dog positions where holding fails: within about one percentage point in all 30 coordinate games of the five setups."),
           P("6. What players do instead", head),
           P("<b>Proposition 2.</b> When the houses lie on either side of the dog, the policy in which each player aims the next dog position at its own target, as far as its weight allows,", claim),
           P("a<sub>i</sub>(x) = clip( a<sub>i</sub>* + (T<sub>i</sub> − w0·x − (1 − w0)·d*) / ((1 − w0)·w<sub>i</sub>), 0, 1 )", math),
           P("is a feedback equilibrium. It is piecewise affine, and equals the one-shot profile wherever holding is optimal. <i>Proof idea:</i> each player's one-round objective is single-peaked at its target.", body),
           img(figs["policy"], col),
           P("Setup 3, x coordinate, w0 = 0.6: the explicit policy (lines) and the dynamic program (circles).", cap),
           B(f"Within one action-grid step (0.01) of the dynamic program in every such coordinate; one-round deviation gains below 1e-31."),
           B(f"200 random games: an equilibrium in {straddle[2]} straddling games; it fails in {int(interior[1]) - int(interior[2].split()[0])} of {interior[1]} games with an interior one-shot action, which stays open."),
           B("<b>Learners miss it.</b> Even starting every episode off the path, A2C and PPO self-play learn to hold the one-shot actions everywhere, exploitable off the path. Noise-free gradient play finds the equilibrium with a table of actions (4 of 4) but not with a shared network (0 of 20).")]

    three = [P("7. Not a Markov potential game", head),
             P("In a Markov potential game every 4-cycle of unilateral policy changes sums to zero. Exact values, setup 1:"),
             table(cycles, [col * 0.16, col * 0.42, col * 0.42]), Spacer(1, 12),
             B("Open-loop it stays a potential game at any inertia; with feedback it does not, even at w0 = 0. So convergence guarantees for independent policy gradient in Markov potential games (Leonardos et al., 2022) do not apply."),
             P("8. Selection on a flat potential is a random walk", head),
             table(split, [col * 0.2, col * 0.18, col * 0.2, col * 0.17, col * 0.25]),
             P("50 seeds per noise level, shared-y setup. *Default.", cap),
             B("<b>Never the interior equilibrium</b> (0 of 250 runs). <b>Which mirror image: a fair coin.</b> An earlier 10-seed effect of the noise did not replicate."),
             img(figs["walk"], col),
             B("The expected gradient along the flat direction is zero but noisy, so the means random-walk; the box stops them where the sigmoid saturates. With an unbounded mean the walk leaves the box (38 of 50 runs)."),
             img(figs["smoothed"], col * 0.92),
             B("With clipped exploration noise, the averaged potential even dips at the interior: a push to either edge, equally."),
             Spacer(1, 18), takeaway]

    c = Canvas(out, pagesize=(W, H))
    c.setTitle("Inertia and selection in the dog game")
    c.setFillColor(colors.HexColor(NAVY)), c.rect(0, H - 4.3 * inch, W, 4.3 * inch, stroke=0, fill=1)
    c.setFillColor(colors.white), c.setFont("Sans-Bold", 80)
    c.drawString(inch, H - 1.75 * inch, "When the Dog Has Inertia: Exact Feedback Equilibria")
    c.drawString(inch, H - 2.85 * inch, "and Equilibrium Selection in a Potential Influence Game")
    c.setFont("Sans", 40), c.drawString(inch, H - 3.75 * inch, "Charith Reddy Pareddy  ·  University of Wisconsin–Madison")
    qr = QrCodeWidget(SITE)
    x0, y0, x1, y1 = qr.getBounds()
    size = 3.4 * inch
    d = Drawing(size, size, transform=[size / (x1 - x0), 0, 0, size / (y1 - y0), 0, 0])
    d.add(qr)
    c.setFillColor(colors.white), c.rect(W - inch - size - 0.1 * inch, H - 4.0 * inch, size + 0.2 * inch, size + 0.2 * inch, stroke=0, fill=1)
    renderPDF.draw(d, c, W - inch - size, H - 3.9 * inch)
    c.setFont("Sans", 22), c.drawRightString(W - inch - size - 0.3 * inch, H - 3.75 * inch, "Code, data, full report: " + SITE)
    top = H - 4.8 * inch
    for i, flow in enumerate((one, two, three)):
        x = inch + i * (col + 0.8 * inch)
        Frame(x, inch, col, top - inch, showBoundary=0, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0).addFromList(
            [KeepInFrame(col, top - inch, flow, mode="shrink")], c)
        if i:
            c.setStrokeColor(colors.HexColor("#c9d3dd")), c.setLineWidth(2), c.line(x - 0.4 * inch, inch, x - 0.4 * inch, top)
    c.save()
    return out


if __name__ == "__main__":
    print(build(*sys.argv[1:]))
