"""The key-results summary PDF (six pages): the main results in plain words, with nine figures of
doggame.summary_figures. The numbers in the text come from the tables of doggame.dog_game_summary and the
propositions in docs/INERTIA_PROOFS.md; see docs/CODE_GUIDE.md for where each one is computed.

Run with `python3 -m doggame.summary_pdf [output.pdf]`; it draws the figures first.
"""

import sys
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Image, KeepTogether, ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from doggame import summary_figures

OUTPUT = "build/summary/Dog_Game.pdf"
ink, ink2 = HexColor("#0b0b0b"), HexColor("#52514e")
title = ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=16, leading=20, textColor=ink)
sub = ParagraphStyle("s", fontName="Helvetica", fontSize=9.5, leading=13, textColor=ink2)
h = ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=ink, spaceBefore=8, spaceAfter=4)
body = ParagraphStyle("b", fontName="Helvetica", fontSize=9.3, leading=12.6, textColor=ink)
cap = ParagraphStyle("c", fontName="Helvetica", fontSize=8.5, leading=11.5, textColor=ink2, spaceAfter=10)


def fig(folder, name, height, caption, width=7.0 * inch):
    img = Image(str(folder / name), width=width, height=height)
    return KeepTogether([img, Spacer(1, 3), Paragraph(caption, cap)])


def side(folder, name, width, caption):
    img = Image(str(folder / name), width=width, height=width * 1100 / 1011)
    t = Table([[img, Paragraph(caption, cap)]], colWidths=[width + 0.2 * inch, 7.0 * inch - width - 0.2 * inch])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    return t


points = [
    "<b>Inertia changes which receiver is better.</b> Without inertia the median receiver does better in only about a fifth "
    "of random games (in many others the two tie). With high inertia it does better in 81 to 88% of games, for 3, 5 and 7 "
    "influencers (Fig. 3). The reason is speed: the median receiver moves to where it ends up as fast as possible, while "
    "under averaging the influencers on the far side keep pulling against the move (Fig. 2).",
    "<b>The median receiver is no longer strategy-proof.</b> Reporting your true target is an equilibrium without inertia, "
    "but not for any positive inertia: the receiver moves only part of the way, so the pivotal influencer exaggerates to make up for it.",
    "<b>Exaggeration persists over time.</b> Under the averaging receiver each influencer pushes to an edge instead of playing "
    "its target (Fig. 1), as in BIG's static result.",
    "<b>One family covers both receivers.</b> The averaging receiver behaves exactly like the median of the actions together "
    "with fixed points at 1/N, ..., (N-1)/N; the median receiver puts those points at the edges. The fixed points decide both "
    "where the receiver settles and how fast it gets there.",
    "<b>The best receiver depends on inertia.</b> Among all strategy-proof receivers (medians with fixed points), averaging "
    "is best without inertia, a known result. For a slowly adjusting receiver we prove that the best one moves toward a "
    "single order statistic of the targets: the one closest on average to the mean target. For uniformly spread targets "
    "that is the median receiver (checked at inertia 0.99). In between, the best fixed points move steadily toward the edges, the best receiver "
    "was symmetric in every case checked, and it saves up to 6% over averaging (Fig. 4).",
    "<b>Some inertia can help.</b> When the incoming information is noisy, a moderate inertia (about 0.45 to 0.75) gives the "
    "lowest total loss, because it smooths out the noise.",
    "<b>Skewed targets call for a different receiver.</b> When most targets sit near one edge, the theorem picks an order "
    "statistic above the median (Fig. 5, left). In the full dynamic game (7 influencers, inertia 0.99) that receiver beats "
    "the median receiver in 46 of 60 random games, by 0.008 on average (standard error 0.0025), about a third of the size "
    "the theorem's first-order term predicts. With a milder skew and 5 influencers the predicted gain is too small to detect "
    "in 40 games.",
    "<b>Compensating for inertia restores truthful reporting.</b> If the receiver moves as close to the median report (with "
    "its fixed points) as its inertia allows, instead of only part of the way, then reporting your true target is an "
    "equilibrium at every inertia and for any fixed points (our proof; no profitable deviation in any game checked). The "
    "same holds for every strategy-proof rule, including ones that treat influencers differently, such as a dictator; "
    "compensation cannot rescue a manipulable rule such as plain averaging of the reports. Truthful reporting also stays an "
    "equilibrium for any receiver that can always stay where it is and whose reachable range never moves down when its "
    "position moves up, such as a speed limit or different limits for moving up and down; we show by example that neither "
    "condition can be dropped. With truthful reports the compensated averaging receiver loses less on average than both "
    "plain receivers at every positive inertia, for 3, 5 and 7 influencers (Fig. 6); in the example of Fig. 2 it also "
    "settles sooner than either.",
    "<b>With compensation and evenly spread targets the best receiver no longer depends on inertia.</b> We prove this "
    "exactly, at every inertia: a "
    "compensated receiver's loss is the planner's loss plus a weighted version of the no-inertia criterion, where the weight "
    "of a point is the discounted time during which the receiver can still reach it (Fig. 7). So for evenly spread targets "
    "the best fixed points are those without inertia, exactly BIG's averaging receiver, where without compensation a slow "
    "receiver should be the median receiver. On fine grids of fixed points (1,000 random "
    "games for 3 influencers, 500 for 5 and 7) the best compensated receiver is the same at every inertia from 0 to 0.995 "
    "and within sampling error of averaging, while the median receiver stays clearly worse (Fig. 4, right).",
    "<b>Random shocks break exact truthfulness, slightly.</b> The results above assume the receiver moves exactly as "
    "told. If its position is also hit by random shocks (standard deviation 0.15, clipped to [0, 1]), the median influencer "
    "gains a little by shading its report: 0.0024 at inertia 0.8 against 0.0003 without inertia, for targets 0.15, 0.30 and "
    "0.80. This is a numerical check, not a theorem; with targets 0.2, 0.5 and 0.8 we found no gain.",
]
pg = [
    "<b>PG self-play finds the one-shot equilibrium.</b> 74 of 75 REINFORCE runs and 25 of 25 PPO runs reached exploitability below "
    "0.004 within 1,500 episodes; in the walking game every REINFORCE seed walks to the one-shot Nash corners (Fig. 8).",
    "<b>With a persistent dog, PG learns the static answer, not the dynamic one.</b> Even starting every episode from a random dog "
    "position, A2C and PPO learn to hold the one-shot actions everywhere (19 of 20 runs), which can be exploited away from the "
    "equilibrium path. Fitting the policy network to exact per-state targets fixes this in three games, 5 of 5 seeds each.",
]
nash_q = [
    "<b>Cartesian moves work, polar moves do not.</b> One network reads the state and both players' moves and returns both "
    "Q-values, Q(s, a<sub>1</sub>, a<sub>2</sub>); its learning target uses the equilibrium of each stage game, found by both "
    "players climbing their own Q at once. With moves written as (v<sub>x</sub>, v<sub>y</sub>) in a disk, or as a square, all 5 "
    "seeds bring both players to the one-shot Nash corners, from the houses and from 8 random starts; with polar moves "
    "(r, &theta;) none of the 5 does (Fig. 9).",
    "<b>Best response: a batched grid is exact and fastest.</b> At 50 random states a dense grid (1,257 to 1,722 evaluations in "
    "one batch) takes about 0.3 ms on a CPU. Gradient ascent stops at local peaks and leaves 10 to 14% of the Q spread on "
    "Cartesian and square moves; finite differences and autograd agree, so the analytic gradient is not needed. A quadratic "
    "fitted through five points needs 15 evaluations and does about as well as gradient ascent. On polar moves every local "
    "method leaves 27 to 32%, which is why polar learning fails.",
    "<b>Both players at once.</b> Simultaneous gradient steps of 0.1 to 1 (in units of the move set's radius) rarely settle "
    "exactly within 500 iterations, but where they hover leaves either player at most about 0.001 to gain on Cartesian and "
    "square moves; on polar moves the gap stays at 0.57 to 0.76.",
]


def build(out=OUTPUT):
    out = Path(out)
    folder = summary_figures.build(out.parent)
    story = [
        Paragraph("Dog game with a persistent receiver: key results", title), Spacer(1, 4),
        Paragraph("Charith Reddy Pareddy &middot; November 1, 2026 &middot; draft, please do not circulate", sub),
        Spacer(1, 10),
        Paragraph("Setting", h),
        Paragraph("The dog game is the Battling Influencers Game (Wu, Zhu, Cai and Zhu, 2025) in the unit square, solved coordinate by "
                  "coordinate (the losses add up across coordinates). Here the receiver "
                  "(the dog) keeps part of its position each round: x<sub>t+1</sub> = w<sub>0</sub> x<sub>t</sub> + (1 - w<sub>0</sub>) "
                  "&times; (aggregate of the influencers' actions), and each influencer loses (x - t<sub>i</sub>)<super>2</super> per "
                  "round, discounted. We compare BIG's averaging receiver with a median receiver.", body),
        Paragraph("Key results", h),
        ListFlowable([ListItem(Paragraph(p, body), leftIndent=12, value="circle") for p in points], bulletType="bullet", start="•", leftIndent=12, bulletFontSize=8),
        Spacer(1, 8),
        side(folder, "fig1.png", 2.4 * inch, "<b>Fig. 1.</b> Two players in BIG with a persistent dog (inertia 0.8), the dog starting near the "
            "corner (0.05, 0.05). The players' actions (circles) jump to the edges and corners of the square instead of their targets "
            "(stars), while the dog drifts round by round to the middle. Grey cross: the mean target, where a planner would put the dog."),
        fig(folder, "fig8.png", 4.7 * inch * 1826 / 1628, "<b>Fig. 2.</b> Three influencers with the same targets and start, inertia 0.9; "
            "each coordinate is aggregated separately. Top: BIG's averaging receiver and the median receiver at their equilibria; the "
            "influencers exaggerate to the edges (circles), and the median receiver is within 0.01 of where it ends after 11 rounds, "
            "averaging after 27. Bottom: the compensated averaging receiver with truthful reports (on the targets), after 10 rounds, "
            "and the planner, after 7.", width=4.7 * inch),
        fig(folder, "fig3.png", 6.2 * inch * 572 / 1584, "<b>Fig. 3.</b> Share of 100 random games (targets uniform on [0, 1]) in which the "
            "median receiver has the lower total discounted loss, by inertia. Dashed line: 50%.", width=6.2 * inch),
        fig(folder, "fig4.png", 6.6 * inch * 638 / 1672, "<b>Fig. 4.</b> Choosing the receiver, three influencers: extra loss of the "
            "receiver with fixed points (p, 1 - p) over the best one at each inertia. Left: plain receivers at their equilibria (100 "
            "random games each); the best position moves from averaging (p near 1/3) without inertia to the median receiver (p = 0) "
            "at high inertia. Right: receivers that compensate for inertia, with truthful reports (1,000 random games each); the best "
            "position stays next to averaging at every inertia.", width=6.6 * inch),
        fig(folder, "fig5.png", 7.0 * inch * 638 / 1672, "<b>Fig. 5.</b> Slow receivers and skewed targets. Left: the theorem's criterion "
            "for 7 influencers, the mean squared distance between the j-th smallest target and the mean target; the best slow "
            "receiver minimizes it (the median for uniform targets, the 5th smallest for Beta(0.1, 1) targets). Right: in the dynamic "
            "game with Beta(0.1, 1) targets, the loss of the 5th-smallest receiver minus the median receiver's, game by game."),
        fig(folder, "fig6.png", 7.0 * inch * 594 / 1672, "<b>Fig. 6.</b> Receivers that compensate for inertia: mean extra loss over the "
            "planner in 100 random games per point (targets uniform on [0, 1]), for BIG's averaging receiver and the median receiver "
            "at their equilibria, and for the compensated averaging receiver with truthful reports. The compensated median receiver "
            "(not shown) follows the same path as the median receiver's equilibrium."),
        fig(folder, "fig9.png", 7.0 * inch * 638 / 1672, "<b>Fig. 7.</b> Why compensation removes the dependence on inertia "
            "(discount 0.9). Left: the weight of each point u, the discounted share of rounds in which a compensated receiver can still "
            "reach it; inertia lowers it most near the edges. Right: extra loss over the planner of a compensated receiver by where it "
            "ends, for targets 0.15, 0.30 and 0.90; the averaging receiver ends closer to the mean target than the median receiver, "
            "so it loses less at every inertia."),
        Paragraph("Policy gradient (PG)", h),
        ListFlowable([ListItem(Paragraph(p, body), leftIndent=12, value="circle") for p in pg], bulletType="bullet", start="\u2022", leftIndent=12, bulletFontSize=8),
        Spacer(1, 8),
        fig(folder, "fig7.png", 7.0 * inch * 946 / 1628, "<b>Fig. 8.</b> REINFORCE self-play in the walking game: each player starts at its target and moves "
            "at most 0.2 per step. Before training the players wander; after 300 training episodes both walk straight to the one-shot "
            "Nash corners, exaggerating as BIG predicts, and the dog ends in the middle. All 5 seeds end at the same corners."),
        Paragraph("Nash-Q with continuous moves", h),
        ListFlowable([ListItem(Paragraph(p, body), leftIndent=12, value="circle") for p in nash_q], bulletType="bullet", start="\u2022", leftIndent=12, bulletFontSize=8),
        Spacer(1, 8),
        fig(folder, "fig10.png", 7.0 * inch * 704 / 1672, "<b>Fig. 9.</b> Nash-Q with continuous moves in the walking game: greedy "
            "equilibrium paths of the seed-0 networks. With Cartesian and square moves both players head for their one-shot Nash "
            "corners, from their houses (solid) and from each other's corner (faint), sliding along the edge where a straight line "
            "would leave the square; with polar moves the second player barely moves and the first stops short of its corner."),
    ]

    SimpleDocTemplate(str(out), pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch, topMargin=0.7 * inch, bottomMargin=0.7 * inch,
                      title="Dog game with a persistent receiver: key results", author="Charith Reddy Pareddy").build(story)
    return out


if __name__ == "__main__":
    print(build(*sys.argv[1:]))
