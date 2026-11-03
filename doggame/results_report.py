"""The results report: results at a glance, recent updates, and the seven main graphs and trajectories. The numbers
come from the tables of doggame.dog_game_summary and the figures from doggame.summary_figures (fig8, fig10, fig3,
fig4 and fig6 of the summary, and fig11 and fig12 of the dynamic game); see docs/CODE_GUIDE.md for where each one is computed.

Run with `python3 -m doggame.results_report [output.pdf]`; it draws the figures first.
"""

import sys
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

from doggame import dog_game_summary as S
from doggame import summary_figures

OUTPUT = "build/report/Dog_Game_Results.pdf"
ink, ink2, rule, tint = HexColor("#0b0b0b"), HexColor("#52514e"), HexColor("#c9c8c3"), HexColor("#f3f2ee")
title = ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=17, leading=21, textColor=ink)
sub = ParagraphStyle("s", fontName="Helvetica", fontSize=9.5, leading=13, textColor=ink2)
h = ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=11.5, leading=14, textColor=ink, spaceBefore=9, spaceAfter=4, keepWithNext=1)
body = ParagraphStyle("b", fontName="Helvetica", fontSize=9.1, leading=12.3, textColor=ink, spaceAfter=3)
cap = ParagraphStyle("c", fontName="Helvetica", fontSize=8.3, leading=11, textColor=ink2, spaceAfter=9)
cell = ParagraphStyle("cell", fontName="Helvetica", fontSize=8.2, leading=10.4, textColor=ink)
head = ParagraphStyle("head", parent=cell, fontName="Helvetica-Bold")
thm = ParagraphStyle("thm", parent=body, leftIndent=8, rightIndent=8, spaceBefore=2, spaceAfter=5)
FULL = 7.0 * inch


def fig(folder, name, caption, width=FULL):
    w, hgt = PILImage.open(folder / name).size
    return KeepTogether([Image(str(folder / name), width=width, height=width * hgt / w), Spacer(1, 3), Paragraph(caption, cap)])




def bullets(items):
    return ListFlowable([ListItem(Paragraph(p, body), leftIndent=12, value="circle") for p in items], bulletType="bullet",
                        start="•", leftIndent=12, bulletFontSize=8)


def table(rows, widths, shade_first=True):
    data = [[Paragraph(str(c), head if i == 0 and shade_first else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1)
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, 0), 0.8, ink2),
             ("LINEBELOW", (0, 1), (-1, -1), 0.3, rule), ("TOPPADDING", (0, 0), (-1, -1), 2.5),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5), ("LEFTPADDING", (0, 0), (-1, -1), 4)]
    if shade_first:
        style.append(("BACKGROUND", (0, 0), (-1, 0), tint))
    t.setStyle(TableStyle(style))
    return t


def box(paragraphs):
    t = Table([[[Paragraph(p, thm) for p in paragraphs]]], colWidths=[FULL])
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, rule), ("BACKGROUND", (0, 0), (-1, -1), tint),
                           ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    return t


def receiver_rows(rows):
    """Per number of influencers, without inertia and at inertia 0.9: how often the median receiver loses less, the
    price of anarchy of both plain receivers, and what compensation saves over plain averaging."""
    wide = {(r[0], r[1]): r for r in S.mechanisms_wide_table(rows)[1:]}
    comp = {(r[0], r[1]): r for r in S.mechanisms_compensated_table(rows)[1:]}
    out = [["Influencers", "Inertia", "Median receiver loses less", "Price of anarchy: averaging / median",
            "Compensated averaging minus plain averaging: mean (s.e.)"]]
    for n in ("3", "5", "7"):
        for w0 in ("0.0", "0.9"):
            out.append([n, w0, wide[n, w0][3], wide[n, w0][2], comp[n, w0][3].split(",")[0]])
    return out


def nash_q_rows(rows):
    out = [["Move encoding", "Seeds", "Exploitability from the houses: median (range)", "From random starts",
            "Seeds below 0.1 from random starts"]]
    return out + [[r[0].capitalize(), *r[1:]] for r in S.continuous_q_table(rows)[1:]]


glance = [
    ["Result", "Status", "Figure"],
    ["Inertia reverses the ranking of receivers: averaging is better without inertia, the median receiver in 81 to 88% of "
     "random games at inertia 0.9", "Numerical, 100 games per point, 3/5/7 influencers", "3"],
    ["With any inertia the median receiver is no longer strategy-proof; the pivotal influencer exaggerates", "Proved", "1"],
    ["Averaging behaves exactly like a generalized median with phantoms at k/N, at every inertia", "Proved", "1"],
    ["A slow plain receiver should move toward one order statistic of the targets (median if uniform, higher if skewed)",
     "Proved to first order; skew check 46 of 60 games", "4"],
    ["<b>Compensating for inertia restores truthful reporting</b> for every strategy-proof rule and every reachable set that "
     "lets the receiver stay put and moves up with its position; neither condition can be dropped", "Proved (Theorem 1), two counterexamples", "1, 5"],
    ["<b>Exact welfare</b> of every compensated receiver; the planner's path is the compensated path toward the mean target",
     "Proved (Theorem 2)", ""],
    ["<b>For uniform targets the best compensated receiver does not depend on inertia</b>: averaging. With clustered targets inertia can switch between static local optima", "Proved (Theorem 3); grids of 500 to 1,000 games",
     "4"],
    ["Closed-form feedback equilibrium of the two-player dog game with inertia, and when holding the one-shot actions is one",
     "Proved (Props. 1-2); 77 of 77 random games", ""],
    ["<b>With a pivot the averaging receiver lands exactly on BIG's one-shot outcome</b> in finitely many rounds (closed form); "
     "without one it only approaches it", "Proved when edge players are far enough from the pivot (Prop. 18); otherwise numerical", "6, 7"],
    ["A moderate inertia (0.45 to 0.75) lowers the total loss when the new information is noisy", "Numerical (planner part proved)", ""],
    ["Self-play learns the one-shot equilibrium; networks miss the feedback equilibrium, and fitting per-state best responses fixes it",
     "Numerical, 3 games (5 seeds)", ""],
    ["Continuous-move Nash-Q reaches the Nash corners with Cartesian or square moves, never with polar moves", "Numerical, 5 seeds", "2"],
]

updates = [
    "<b>The dynamic game, plotted.</b> Vector fields of where the receiver moves next from every position, equilibrium paths "
    "with every report, and the influencers' response fields, for the averaging and the median receiver (Figs. 6 and 7).",
    "<b>Proposition 18.</b> When BIG's one-shot equilibrium has a pivot, the averaging receiver lands exactly on the one-shot "
    "outcome after finitely many rounds, with a closed-form count; this covers part of a case listed as open before.",
    "<b>Compensated receivers, three theorems.</b> Truthfulness now holds for any reachable set with two conditions (the "
    "receiver can stay where it is; the reachable range never moves down when the position moves up), with a counterexample "
    "for each (gains from lying of 0.085 and 1.24). New: an exact welfare formula, and a proof that the best fixed points do not "
    "depend on inertia.",
    "<b>Random shocks.</b> With shocks on the receiver's position, truthful reporting is no longer exact, but the gain from "
    "lying is small (about 0.0015 at inertia 0.8).",
    "<b>Continuous-move Nash-Q.</b> The network Q(s, a<sub>1</sub>, a<sub>2</sub>) with Cartesian (v<sub>x</sub>, "
    "v<sub>y</sub>) or polar (r, &theta;) moves, best responses by grid, gradient ascent, finite differences or a fitted "
    "quadratic, and simultaneous best-response dynamics: results and trajectories now in the summary.",
    "<b>Explainable code.</b> Modules that no current result needs were moved to <i>archive/</i> (the tag <i>pre-archive</i> "
    "keeps the old tree). <i>docs/CODE_GUIDE.md</i> maps each result to its proof, study, table and test. Two CI defects "
    "were fixed (a missing pypdf dependency; NumPy 2's removal of <i>trapz</i>). All tests pass locally and in CI.",
]

interesting = [
    "<b>Compensation separates incentives from dynamics.</b> Theorems 1 to 3 together say that a receiver which moves as close to "
    "the strategy-proof outcome as it can (rather than a fixed fraction of the way) keeps everyone truthful, has a closed-form "
    "welfare, and for evenly spread targets is designed exactly as if there were no inertia. Without compensation all three "
    "fail: the best slow receiver "
    "changes from averaging to the median receiver. The one-round step is a known static fact (a generalized median clipped to "
    "an interval is strategy-proof); the dynamic versions are new as far as our literature check went.",
    "<b>Speed against location.</b> The reversal is not about where receivers end up, which inertia does not change, but about how "
    "fast they get there: under averaging the influencers on the far side keep pulling, a tug of war; the median receiver moves "
    "at full speed once a majority pushes one way. A sticky receiver magnifies that difference.",
    "<b>A pivot makes the slow receiver arrive.</b> Under BIG's averaging, a coordinate whose one-shot equilibrium has a pivot "
    "reaches the one-shot outcome exactly, in a number of rounds that is logarithmic in the distance; a coordinate without one "
    "only approaches it. The proof is short and covers the case where every edge player is far enough from the pivot; it has "
    "not yet been checked carefully against the literature.",
    "<b>What a learner can represent decides what it learns.</b> In the dog game, whose feedback equilibrium jumps, a table "
    "learns it and a network does not; in a sticky-price duopoly, whose equilibrium is linear, it is the other way round. "
    "Fitting each state's best response removes the failure for networks in all three games tested.",
]

limits = [
    "Everything is one-dimensional per coordinate and, apart from the shock checks, deterministic. The proofs are ours and "
    "have not been refereed.",
    "The slow-receiver result is first order; its remainder is not shown to be uniform in the phantoms. The skew check is "
    "smaller than predicted (about a third of the first-order gain).",
    "The feedback equilibrium with an interior one-shot action is proved only when every edge player is far enough from the "
    "pivot (Prop. 18); otherwise the target fixed point is an equilibrium in 122 of 123 random two-player games, and in 10 of "
    "12 checked games with three or five players, numerically. Uniqueness is not shown.",
    "The learning results use 5 seeds per setting (1 for tables); the Nash-Q results use 5 seeds and one step length.",
]


def build(out=OUTPUT):
    out = Path(out)
    folder = summary_figures.build(out.parent)
    summary_figures.fig11(folder / "fig11.png")
    summary_figures.fig12(folder / "fig12.png")
    rows = S.load(S.STUDIES)
    story = [
        Paragraph("The dog game with a persistent receiver: results report", title), Spacer(1, 4),
        Paragraph("Charith Reddy Pareddy &middot; November 3, 2026 &middot; draft", sub), Spacer(1, 8),
        Paragraph("The game in one paragraph", h),
        Paragraph("The dog game is the Battling Influencers Game (BIG; Wu, Zhu, Cai and Zhu, 2025): N influencers with targets "
                  "t<sub>i</sub> in the unit square report actions, a receiver (the dog) aggregates them, and each influencer loses "
                  "the squared distance between the receiver and its target, discounted over rounds. The losses add up across "
                  "coordinates, so each coordinate is a one-dimensional game. Our change: the receiver keeps part of its position, "
                  "x<sub>t+1</sub> = &lambda; x<sub>t</sub> + (1 - &lambda;) &times; aggregate (inertia &lambda;, written w<sub>0</sub> in "
                  "the figures). We compare BIG's averaging receiver with median receivers, introduce a <i>compensated</i> receiver "
                  "that moves as close to the aggregate as its inertia allows, and study what learning algorithms find.", body),
        Paragraph("Results at a glance", h),
        table(glance, [3.95 * inch, 2.35 * inch, 0.7 * inch]),
        Paragraph("Updates since October 31", h),
        bullets(updates),
        Paragraph("Trajectories", h),
        Paragraph("Figures 1 and 2 show what the influencers and the receiver actually do, round by round: equilibrium play under "
                  "different receivers, and the paths learned by continuous-move Nash-Q.", body),
        fig(folder, "fig8.png", "<b>Fig. 1.</b> Three influencers, same targets and start, inertia 0.9, each coordinate aggregated "
            "separately. Top: BIG's averaging receiver and the median receiver at their equilibria; influencers exaggerate to the edges "
            "(circles); the median receiver is within 0.01 of its end point after 11 rounds, averaging after 27. Bottom: the compensated "
            "averaging receiver with truthful reports (on the targets), after 10 rounds, and the planner, after 7.", width=4.6 * inch),
        fig(folder, "fig10.png", "<b>Fig. 2.</b> Nash-Q with continuous moves (seed 0): greedy equilibrium paths from the houses (solid) and "
            "with each player starting in the other's corner (faint). With Cartesian and square moves both players reach their one-shot "
            "Nash corners, sliding along an edge where a straight line would leave the square; with polar moves the second player barely "
            "moves and the first stops short of its corner."),
        table(nash_q_rows(rows), [1.1 * inch, 0.55 * inch, 2.1 * inch, 1.5 * inch, 1.75 * inch]),
        Spacer(1, 4),
        Paragraph("A batched grid is the exact and fastest best response (about 0.3 ms for 1,257 to 1,722 moves); gradient ascent "
                  "leaves 10 to 14% of the Q spread on Cartesian and square moves and 27 to 32% on polar moves, which is why polar "
                  "learning fails. Finite differences agree with autograd, so the analytic gradient is not needed.", body),

        Paragraph("1. Inertia reverses the ranking of receivers", h),
        Paragraph("Without inertia BIG's averaging receiver has the lower social loss in most games, as known. With a sticky receiver "
                  "the median receiver wins: it reaches its end point much faster (Fig. 1), and speed matters more than location "
                  "when the receiver is slow. The median receiver also stops being strategy-proof once &lambda; &gt; 0: it moves only "
                  "part of the way, so the pivotal influencer exaggerates to make up for it.", body),
        fig(folder, "fig3.png", "<b>Fig. 3.</b> Share of 100 random games (targets uniform on [0, 1]) in which the median receiver has the "
            "lower total discounted loss, by inertia. Dashed line: 50%.", width=6.0 * inch),
        table(receiver_rows(rows), [0.85 * inch, 0.6 * inch, 1.4 * inch, 1.75 * inch, 2.4 * inch]),
        Spacer(1, 3),
        Paragraph("Price of anarchy: the equilibrium loss over the planner's, mean over games. Last column: the compensated receiver "
                  "with truthful reports against BIG's receiver at its equilibrium, same games (negative = compensation loses less).", cap),

        Paragraph("2. Which receiver is best?", h),
        Paragraph("Every strategy-proof receiver is a median of the reports and N - 1 fixed points (phantoms). Averaging behaves exactly "
                  "like phantoms at 1/N, ..., (N - 1)/N, so one family covers both receivers, and the phantoms decide where the receiver "
                  "settles and how fast. For a slow plain receiver we prove (to first order) that the best one moves toward the order "
                  "statistic of the targets closest on average to the mean target: the median for uniform targets, a higher one "
                  "for strongly skewed targets. With Beta(0.1, 1) targets, seven influencers and inertia 0.99, the 5th-smallest "
                  "receiver beats the median receiver in 46 of 60 games, by 0.008 on average (s.e. 0.0025), about a third of the "
                  "first-order prediction.", body),
        fig(folder, "fig4.png", "<b>Fig. 4.</b> Extra loss of phantoms (p, 1 - p) over the best at each inertia, three influencers. Left: plain "
            "receivers at their equilibria (100 games each); the best p moves from averaging (near 1/3) to the median receiver (p = 0). "
            "Right: compensated receivers with truthful reports (1,000 games each); the best p stays next to averaging at every inertia."),

        Paragraph("3. Compensated receivers", h),
        Paragraph("A compensated receiver with rule m and reachable set R(x) = [a(x), b(x)] moves to clip(m(r), a(x), b(x)), the reachable "
                  "position closest to the rule's outcome. Proportional inertia is R(x) = [&lambda;x, &lambda;x + 1 - &lambda;]; a speed limit "
                  "and different limits up and down are other examples. Two conditions: <b>(R1)</b> a(x) &le; x &le; b(x), the receiver "
                  "can stay put; <b>(R2)</b> a and b are nondecreasing.", body),
        box(["<b>Theorem 1 (truthful).</b> Under (R1) and (R2), for any generalized median voter scheme m (any strategy-proof rule, "
             "including a dictator), any number of influencers, discount and loss increasing in distance, truthful reporting is a "
             "Markov perfect equilibrium. The truthful path after k rounds is clip(m(t), A<sub>k</sub>, B<sub>k</sub>) with "
             "A<sub>k+1</sub> = a(A<sub>k</sub>), B<sub>k+1</sub> = b(B<sub>k</sub>). Without (R1) lying gains 0.085 in an example, "
             "without (R2) 1.24. Compensation cannot rescue a manipulable rule such as the mean of the reports.",
             "<b>Theorem 2 (welfare).</b> With convex loss minimized at s*, the planner's path is the compensated path toward s*, and the "
             "compensated path toward s loses L(s) = L(s*) + &int; &Lambda;'(u) W(u) du (from s* to s), where W(u) is the discounted "
             "expected time during which u is reachable.",
             "<b>Theorem 3 (design).</b> For i.i.d. targets and quadratic loss, &part;L/&part;&alpha;<sub>k</sub> = 2N W(&alpha;<sub>k</sub>) "
             "E[(&alpha;<sub>k</sub> - mean target) 1{exactly N - k targets below &alpha;<sub>k</sub>}]: each phantom's derivative has "
             "the sign it has without inertia. For uniform targets the unique optimum is &alpha;<sub>k</sub> = k/N, BIG's averaging "
             "receiver, at every inertia, discount and start distribution."]),
        Spacer(1, 6),
        fig(folder, "fig6.png", "<b>Fig. 5.</b> Mean extra loss over the planner in 100 random games per point: BIG's averaging receiver and "
            "the median receiver at their equilibria, and the compensated averaging receiver with truthful reports. Compensation loses "
            "less than both plain receivers at every positive inertia, for 3, 5 and 7 influencers."),
        Paragraph("Why, for uniform targets, the design does not depend on inertia: by Theorem 2 the loss is single-dipped in the end point, with its "
                  "minimum at the mean target, whatever the inertia. Inertia only changes the weight W(u), which is lowest near the "
                  "edges because they are reached last. So of two compensated receivers whose end points lie on the same side of the "
                  "mean target, the closer one loses less at every inertia.", body),
        Paragraph("<b>Random shocks (numerical).</b> If the receiver's position is also hit by shocks (sd 0.15, clipped to [0, 1]), the "
                  "median influencer gains a little by shading its report: about 0.0015 at inertia 0.8 against 0.0001 without inertia "
                  "(targets 0.15, 0.30, 0.80); with targets 0.2, 0.5, 0.8 no gain was found.", body),

        Paragraph("4. The feedback equilibrium of the dog game with inertia", h),
        Paragraph("Under BIG's averaging with two players, the one-shot answer (each player at the edge or corner nearest its target) is not "
                  "always an equilibrium of the dynamic game. Let d* be the one-shot dog position and k = (1 - g&lambda;<sup>2</sup>)/(1 - g&lambda;). "
                  "<b>Proposition 1:</b> holding the one-shot actions at every dog position is Markov-perfect iff every edge player's slack "
                  "s = (t - d*)/(1 - d*) is at least &rho;(&lambda;, g) = &lambda;(1 - g&lambda;)/(1 - g&lambda;<super>2</super>). "
                  "<b>Proposition 2:</b> when the targets straddle d*, the policy in which each player aims the next dog position at "
                  "d* + k(t - d*), as far as its weight allows, is a feedback equilibrium at every inertia and discount. It matches the "
                  "dynamic program to within one grid step and is an equilibrium in 77 of 77 random straddling games. In continuous "
                  "time the threshold becomes (1 + r)/(2 + r). In house setup 3 (x-coordinate, targets 0.7 and 0.6) the policy's "
                  "largest one-round deviation gain over 21 dog positions is 0, while holding the one-shot actions gives 0.076 at "
                  "&lambda; = 0.6 and 0.121 at 0.9.", body),
        Paragraph("<b>Proposition 18 (a pivot gives exact arrival).</b> With three or more influencers the one-shot equilibrium often "
                  "has a pivot: one influencer whose one-shot action is strictly inside the box, so that d* is its target. Let U and L be "
                  "the dog with the pivot at 1 and at 0 and everyone else at the one-shot actions. If every edge influencer holds its "
                  "one-shot action and the pivot moves the dog as close to its target as it can, the dog lands exactly on d* after "
                  "ceil(log((U - x)/(U - d*)) / log(1/&lambda;)) rounds from x &lt; d* (the mirror formula above d*) and stays "
                  "there; without a pivot it only approaches d*. This policy is a feedback equilibrium at every inertia and discount "
                  "when every edge influencer is far enough from the pivot (slack (t - L)/(1 - L) &ge; &lambda; at the top, "
                  "(U - t)/U &ge; &lambda; at the bottom). Otherwise the equilibrium in target form still lands exactly in random games "
                  "with three and five influencers (numerical), but in 2 of 12 games with five influencers at inertia 0.9 it is not "
                  "an equilibrium (largest gain 0.005).", body),
        fig(folder, "fig11.png", "<b>Fig. 6.</b> Averaging and median receivers in the dynamic game, inertia 0.8, targets and start as in "
            "Fig. 1. Left: the receiver's one-round move from each position (scaled per panel, darker = faster) and paths from the four "
            "corners. Right: one equilibrium path with every report. The averaging receiver lands exactly on y = 0.55 in round 5, "
            "because the y-coordinate has a pivot (influencer 3), but only approaches x = 1/3; the median receiver lands exactly in "
            "both coordinates (rounds 6 and 3).", width=4.6 * inch),
        fig(folder, "fig12.png", "<b>Fig. 7.</b> Response vector fields of the same game: at each receiver position, an arrow toward the "
            "report each influencer makes there in equilibrium (fixed length, darker = farther), and the reports themselves (circles). "
            "Under averaging the influencers report at the edges, with two or three distinct reports each over the whole square; under "
            "the median the reports move with the receiver's position (8, 5 and 5 distinct reports)."),

        Paragraph("5. Cooling off: some inertia can help", h),
        Paragraph("Without noise, inertia only slows the outcome. If each round's new aggregate is noisy, inertia filters it: for the "
                  "planner and both rules the loss is U-shaped in inertia, lowest at about 0.45 to 0.75 for shocks of 0.1 and 0.2, and "
                  "the best inertia rises with the shock. If instead the shock hits the outcome, the planner wants no inertia (proved "
                  "for the planner; the rest is numerical, 20 draws per point).", body),

        Paragraph("6. What learning finds", h),
        bullets([
            "<b>Policy gradient finds the one-shot equilibrium.</b> 74 of 75 REINFORCE runs and 25 of 25 PPO runs reached "
            "exploitability below 0.004 within 1,500 episodes; in the walking game every seed walks to the Nash corners.",
            "<b>With a persistent dog it learns the static answer, not the dynamic one.</b> Starting every episode from a random dog "
            "position, A2C and PPO hold the one-shot actions everywhere in 19 of 20 runs, exploitable off the equilibrium path. Exact "
            "(noise-free) gradient play shows why: a table of actions reaches Proposition 2's equilibrium every time, a shared network "
            "never (0 of 20). The same split appears in soccer.",
            "<b>Fitting per-state best responses fixes the network.</b> Every 25 steps each player's exact best action is computed at "
            "every state and the policy is fitted to it. This brings the largest one-round gain from 0.048 and 0.104 down to at most "
            "0.001 in the dog game (5 of 5 seeds), keeps the duopoly right, and in soccer (with value-iteration targets) cuts the mean "
            "duality gap from 0.255 to 0.019. For tables it does not help.",
        ]),

        Paragraph("7. The most interesting results, and how new they are", h),
        bullets(interesting),
        Paragraph("Also known and confirmed here: with many houses the one-shot dog converges to the solution of d = 1 - F(d), not to "
                  "the median house (Renault and Trannoy, 2005); without inertia averaging is the welfare-best strategy-proof receiver "
                  "for uniform targets (Jennings et al., 2021).", body),
        Paragraph("Limits and open problems", h),
        bullets(limits),
        Paragraph("Reproducibility", h),
        Paragraph("All numbers come from saved study results through <i>doggame/dog_game_summary.py</i>; <i>docs/CODE_GUIDE.md</i> maps each "
                  "result to its proof, study, table and test. <i>python3 -m doggame.results_report</i> rebuilds this report, "
                  "and <i>python3 -m doggame.summary_pdf</i> the summary; the tests "
                  "check the theorems numerically.", body),
    ]
    SimpleDocTemplate(str(out), pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch, topMargin=0.65 * inch,
                      bottomMargin=0.65 * inch, title="The dog game with a persistent receiver: results report",
                      author="Charith Reddy Pareddy").build(story, onFirstPage=number_pages, onLaterPages=number_pages)
    return out


def number_pages(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(ink2)
    canvas.drawRightString(letter[0] - 0.75 * inch, 0.4 * inch, str(doc.page))
    canvas.restoreState()


if __name__ == "__main__":
    print(build(*sys.argv[1:]))
