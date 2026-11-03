"""Frequently asked questions (three pages, text only): the game, the results, learning, the evidence and the code.
A companion to doggame.results_report, whose styles it uses.

Run with `python3 -m doggame.faq [output.pdf]`.
"""

import sys
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer

from doggame.results_report import body, h, number_pages, sub, title

OUTPUT = "build/faq/Dog_Game_FAQ.pdf"
q = ParagraphStyle("q", parent=body, fontName="Helvetica-Bold", spaceBefore=5, spaceAfter=1, keepWithNext=1)

sections = [
    ("The game", [
        ("What is the dog game?",
         "The Battling Influencers Game (Wu, Zhu, Cai and Zhu, 2025) in the unit square: influencers with targets report "
         "points, a receiver (the dog) aggregates them, and each influencer loses the squared distance from the receiver to its "
         "target, discounted over rounds. We add inertia: the receiver keeps part of its old position each round."),
        ("Why solve one coordinate at a time?",
         "The losses add up over the two coordinates and every receiver aggregates each coordinate separately, so the 2D game "
         "splits into two independent 1D games. A pair of 1D equilibria is an equilibrium of the 2D game."),
        ("Why do the players exaggerate?",
         "Under averaging a report moves the outcome by only the player's weight, so the best response overshoots the target "
         "and stops at the edge of the box. That is BIG's main result, and it persists with inertia."),
        ("What does inertia model?",
         "A receiver that cannot jump: opinion that changes gradually, a price that adjusts part of the way, a robot with a "
         "speed limit. Formally x' = &lambda;x + (1 - &lambda;) &times; aggregate, with &lambda; the inertia."),
        ("What are phantoms?",
         "N - 1 fixed points added to the reports before taking the median. They describe every strategy-proof rule without "
         "inertia. Phantoms at the edges give the median receiver; phantoms at 1/N, ..., (N - 1)/N behave exactly like BIG's "
         "averaging."),
    ]),
    ("The results", [
        ("Is the median receiver better than averaging?",
         "It depends on inertia. Without inertia averaging has the lower social loss in most random games (the median wins in "
         "about a fifth). At inertia 0.9 the median receiver wins in 81 to 88% of games, for 3, 5 and 7 influencers. This is "
         "numerical, from 100 random games per point."),
        ("Why should inertia matter if receivers end up at the same place?",
         "Because the loss is paid along the way. The median receiver moves at full speed once a majority pushes one way, while "
         "under averaging the influencers on the far side keep pulling against the move. In the three-influencer example the "
         "median receiver is settled after 11 rounds, averaging after 27."),
        ("Is truthful reporting an equilibrium?",
         "For the plain median receiver only without inertia; with any inertia the pivotal influencer over-reports to make up "
         "for the partial move (proved). Under averaging never: everyone exaggerates. For a compensated receiver with any "
         "strategy-proof rule it is, at every inertia (Theorem 1)."),
        ("What is a compensated receiver, in plain words?",
         "Instead of moving a fixed fraction of the way to the aggregate, it moves as close to the rule's outcome as it can "
         "reach this round. From far away it moves as fast as allowed; once the outcome is reachable it goes there exactly."),
        ("Is compensation realistic?",
         "It is a design proposal, not observed behaviour. It needs a receiver that knows its own limits and computes the rule's "
         "outcome each round. It covers proportional inertia, a speed limit and different limits up and down; it fails if the "
         "receiver cannot stay put or if its reach can fall when its position rises (the two counterexamples)."),
        ("Does compensation work for every rule?",
         "For every strategy-proof rule, including ones that treat influencers differently such as a dictator. Not for a "
         "manipulable rule: the compensated mean of the reports is not truthful even without inertia."),
        ("Why does the best compensated receiver not depend on inertia?",
         "Theorem 2 writes its loss as the planner's loss plus the static criterion weighted by W(u), the discounted time during "
         "which u is reachable. W is positive and only rescales, so each phantom's derivative keeps the sign it has without "
         "inertia (Theorem 3). For uniform targets the optimum is k/N, BIG's averaging, at every inertia."),
        ("Does compensation actually lose less?",
         "Yes, numerically: with truthful reports the compensated averaging receiver loses less on average than both plain "
         "receivers at their equilibria at every positive inertia, for 3, 5 and 7 influencers (at inertia 0.9: 0.20, 0.33 and "
         "0.52 less than plain averaging)."),
        ("What happens with noise?",
         "With random shocks on the receiver's position, truthful reporting stops being exact, but the gain from lying is small "
         "(0.0024 at inertia 0.8). And if the new information is noisy, a moderate inertia (0.45 to 0.75) gives the lowest total "
         "loss, because it filters the noise."),
        ("What changes with skewed targets?",
         "For a slow plain receiver the best rule moves toward the order statistic of the targets closest on average to the mean "
         "target: the median for uniform targets, a higher one for strongly skewed targets. In the full game the predicted "
         "receiver wins 46 of 60 games, by about a third of the first-order prediction."),
    ]),
    ("Learning", [
        ("Does policy-gradient self-play converge?",
         "In the one-shot dog game, yes: 74 of 75 REINFORCE runs and 25 of 25 PPO runs reached exploitability below 0.004 "
         "within 1,500 episodes. The game is favourable (best responses are affine and the box makes the recursion a "
         "contraction), so this is not a general result."),
        ("Does it learn the dynamic equilibrium with inertia?",
         "No. Even starting from random dog positions, A2C and PPO learn to hold the one-shot actions everywhere (19 of 20 "
         "runs), which can be exploited off the equilibrium path. Exact gradient play finds the feedback equilibrium with a "
         "table of actions every time and with a shared network never."),
        ("How is that fixed?",
         "By best-response regression: every 25 steps compute each player's exact best one-round action at every state and "
         "fit the network to it. In the dog game the largest gain falls from 0.048 and 0.104 to at most 0.001 (5 of 5 seeds); "
         "in soccer, with value-iteration targets, the mean duality gap falls from 0.255 to 0.019."),
        ("Why do polar moves fail for Nash-Q?",
         "With polar moves (r, &theta;) every local best-response method leaves 27 to 32% of the Q spread, so the stage "
         "equilibria in the learning target are wrong. With Cartesian or square moves all 5 seeds reach the Nash corners; with "
         "polar moves none does."),
        ("Which best-response method should be used?",
         "A dense grid evaluated in one batch: it is exact and the fastest (about 0.3 ms for 1,257 to 1,722 moves). Gradient "
         "ascent stops at local peaks; finite differences agree with autograd, so an analytic gradient is not needed."),
    ]),
    ("The evidence", [
        ("What is proved and what is numerical?",
         "Proved: the median receiver's loss of strategy-proofness, the phantom equivalence, Theorems 1 to 3, Propositions 1 and "
         "2 for the feedback equilibrium, and (to first order) the slow-receiver result. Numerical: the reversal, the gains "
         "from compensation, the noise and skew checks, and all learning results. Every theorem also has a numerical test."),
        ("Are the results new?",
         "As far as our literature check went, Theorems 1 to 3 for receivers with inertia are new; their one-round step is a "
         "known static fact (a generalized median clipped to an interval is strategy-proof). Known and confirmed: averaging is "
         "the best strategy-proof receiver without inertia for uniform targets, and the many-player one-shot outcome is the "
         "solution of d = 1 - F(d)."),
        ("How reliable are the numbers?",
         "Plain-receiver equilibria come from an 801-point grid, accurate to about 0.015 in loss (smaller differences count "
         "as ties). Deviation checks use 21 positions and 201 reports. Random-game results use 100 to 1,000 draws with paired "
         "standard errors; learning results use 5 seeds (1 for tables). A 10-seed result that did not replicate at 50 seeds "
         "was corrected."),
        ("What is not shown?",
         "Everything is one-dimensional per coordinate and deterministic apart from the shock checks. Uniqueness of the "
         "feedback equilibrium, the feedback equilibrium with an interior one-shot action, and a uniform remainder for the "
         "slow-receiver result are open. The proofs have not been refereed."),
    ]),
    ("The code", [
        ("Where is each result computed?",
         "docs/CODE_GUIDE.md maps every result to its proof, study, table and test. The receiver math is all in "
         "doggame/mechanisms.py; the feedback equilibrium in doggame/inertia_theory.py."),
        ("How do I reproduce a number?",
         "Find its table in doggame/dog_game_summary.py, the study that feeds it in doggame/dog_game_studies.py, and the "
         "function the study calls. <i>python3 -m doggame.dog_game_summary</i> prints every table from the saved runs."),
        ("What was archived and why?",
         "Finished threads that no current result needs, such as the Nash-Q studies, the polar-action and walking-equilibrium "
         "studies, the soccer extras and old report builders, moved to archive/. They are frozen at the tag pre-archive, where "
         "they still run, so the active code stays small enough to explain."),
        ("How are the documents built?",
         "<i>python3 -m doggame.results_report</i> builds the results report, <i>python3 -m doggame.methods_guide</i> the "
         "methods guide, <i>python3 -m doggame.faq</i> these questions, <i>python3 -m doggame.summary_pdf</i> the summary, "
         "and <i>tectonic paper/paper.tex</i> the paper. The test suite runs locally and in CI on every push."),
    ]),
]


def build(out=OUTPUT):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    story = [Paragraph("The dog game with a persistent receiver: frequently asked questions", title), Spacer(1, 4),
             Paragraph("Charith Reddy Pareddy &middot; November 2, 2026 &middot; companion to the results report", sub), Spacer(1, 6)]
    for name, items in sections:
        story.append(Paragraph(name, h))
        for question, answer in items:
            story.append(KeepTogether([Paragraph(question, q), Paragraph(answer, body)]))
    SimpleDocTemplate(str(out), pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch, topMargin=0.65 * inch,
                      bottomMargin=0.65 * inch, title="The dog game: frequently asked questions",
                      author="Charith Reddy Pareddy").build(story, onFirstPage=number_pages, onLaterPages=number_pages)
    return out


if __name__ == "__main__":
    print(build(*sys.argv[1:]))
