"""The key-results summary PDF (four pages): the main results in plain words, with the seven figures of
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
    "compensation cannot rescue a manipulable rule such as plain averaging of the reports. With truthful reports the "
    "compensated averaging receiver loses less on average than both plain receivers at every positive inertia, for 3, 5 "
    "and 7 influencers (Fig. 6).",
    "<b>With compensation the best receiver no longer depends on inertia.</b> We prove that a slowly adjusting compensated "
    "receiver should use the same fixed points as a receiver without inertia: for evenly spread targets BIG's averaging "
    "receiver, where without compensation it would be the median receiver. On fine grids of fixed points (1,000 random "
    "games for 3 influencers, 500 for 5 and 7) the best compensated receiver is the same at every inertia from 0 to 0.995 "
    "and within sampling error of averaging, while the median receiver stays clearly worse (Fig. 4, right).",
]
pg = [
    "<b>PG self-play finds the one-shot equilibrium.</b> 74 of 75 REINFORCE runs and 25 of 25 PPO runs reached exploitability below "
    "0.004 within 1,500 episodes; in the walking game every REINFORCE seed walks to the one-shot Nash corners (Fig. 7).",
    "<b>With a persistent dog, PG learns the static answer, not the dynamic one.</b> Even starting every episode from a random dog "
    "position, A2C and PPO learn to hold the one-shot actions everywhere (19 of 20 runs), which can be exploited away from the "
    "equilibrium path. Fitting the policy network to exact per-state targets fixes this in three games, 5 of 5 seeds each.",
]


def build(out=OUTPUT):
    out = Path(out)
    folder = summary_figures.build(out.parent)
    story = [
        Paragraph("Dog game with a persistent receiver: key results", title), Spacer(1, 4),
        Paragraph("Charith Reddy Pareddy &middot; October 31, 2026 &middot; draft, please do not circulate", sub),
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
        fig(folder, "fig2.png", 7.0 * inch * 946 / 1628, "<b>Fig. 2.</b> Three influencers with the same targets and start under BIG's averaging "
            "receiver (left) and the median receiver (right), inertia 0.9; each coordinate is aggregated separately. The median receiver "
            "is within 0.01 of where it ends after 11 rounds, averaging after 27."),
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
        Paragraph("Policy gradient (PG)", h),
        ListFlowable([ListItem(Paragraph(p, body), leftIndent=12, value="circle") for p in pg], bulletType="bullet", start="\u2022", leftIndent=12, bulletFontSize=8),
        Spacer(1, 8),
        fig(folder, "fig7.png", 7.0 * inch * 946 / 1628, "<b>Fig. 7.</b> REINFORCE self-play in the walking game: each player starts at its target and moves "
            "at most 0.2 per step. Before training the players wander; after 300 training episodes both walk straight to the one-shot "
            "Nash corners, exaggerating as BIG predicts, and the dog ends in the middle. All 5 seeds end at the same corners."),
    ]

    SimpleDocTemplate(str(out), pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch, topMargin=0.7 * inch, bottomMargin=0.7 * inch,
                      title="Dog game with a persistent receiver: key results", author="Charith Reddy Pareddy").build(story)
    return out


if __name__ == "__main__":
    print(build(*sys.argv[1:]))
