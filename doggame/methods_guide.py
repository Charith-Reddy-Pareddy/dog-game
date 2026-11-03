"""The methods guide (five pages, no figures): definitions of every term, what each module implements, how each number
is computed, why each result holds, and a worked example. A companion to doggame.results_report, whose styles it uses.

Run with `python3 -m doggame.methods_guide [output.pdf]`.
"""

import sys
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from doggame.results_report import body, box, bullets, h, number_pages, sub, table, title

OUTPUT = "build/guide/Dog_Game_Methods_Guide.pdf"
W = [1.55 * inch, 5.45 * inch]

game_terms = [
    ["Term", "Meaning"],
    ["Influencer (player)", "One of N agents. Each has a target and reports a point every round. In the two-player dog game they are red and blue."],
    ["Target t<sub>i</sub> (house h<sub>i</sub>)", "Where influencer i wants the receiver to be. Called the house in the dog game."],
    ["Receiver (dog)", "The outcome x that the reports move. One coordinate of the dog's position; the two coordinates are solved separately because the losses add up."],
    ["Report r<sub>i</sub> (action)", "The point in [0, 1] (per coordinate) that influencer i sends each round. It need not equal its target."],
    ["Loss", "Each round influencer i loses (x - t<sub>i</sub>)<super>2</super>, the squared distance from the receiver to its target, discounted over rounds."],
    ["Discount &delta; (g)", "How much a round counts relative to the one before: 0.9 throughout unless stated."],
    ["Inertia &lambda; (w<sub>0</sub>)", "The share of its old position the receiver keeps: x' = &lambda;x + (1 - &lambda;) &times; aggregate. &lambda; = 0 is BIG's original game."],
    ["Weight w", "In the two-player dog game, red's share of the aggregate: aggregate = w a<sub>red</sub> + (1 - w) a<sub>blue</sub>."],
    ["One-shot (stage) game", "A single round without inertia. Its Nash equilibrium puts each player at the edge or corner nearest its target."],
    ["Exaggeration", "Reporting further out than one's target, to pull the aggregate toward it. The BIG result: under averaging every player exaggerates."],
    ["Pivot", "With many players, the at most one player whose one-shot report is strictly inside the box; its target is the one-shot outcome."],
    ["Social loss &Lambda;", "The sum of all influencers' losses, discounted and averaged over 101 evenly spread starting positions."],
    ["Planner", "The benchmark: moves the receiver as close to the mean target as it can reach every round. Without shocks this is the best possible path."],
    ["Price of anarchy", "An equilibrium's social loss divided by the planner's."],
]

receiver_terms = [
    ["Term", "Meaning"],
    ["Averaging receiver (BIG)", "Aggregate = mean of the reports."],
    ["Median receiver", "Aggregate = median of the reports."],
    ["Phantoms &alpha;", "N - 1 fixed points added to the reports before taking the median. Half at 0 and half at 1 (N odd) give the median; phantoms k/N reproduce averaging."],
    ["Generalized median", "median(r<sub>1</sub>, ..., r<sub>N</sub>, &alpha;<sub>1</sub>, ..., &alpha;<sub>N-1</sub>). Without inertia these are exactly the strategy-proof rules (Moulin, 1980)."],
    ["Order-statistic receiver", "A generalized median that always picks the j-th smallest report (phantoms: N - j at 0, j - 1 at 1)."],
    ["Plain receiver", "Moves a fixed fraction (1 - &lambda;) of the way to the aggregate."],
    ["Reachable set R(x)", "[a(x), b(x)], the positions the receiver can reach from x in one round. Proportional inertia: [&lambda;x, &lambda;x + 1 - &lambda;]."],
    ["Compensated receiver", "Moves to clip(m(r), a(x), b(x)): the reachable position closest to the rule's outcome m(r), instead of a fixed fraction of the way."],
    ["(R1), (R2)", "(R1) a(x) &le; x &le; b(x): the receiver can stay where it is. (R2) a and b are nondecreasing: its reach never moves down when its position moves up."],
    ["Reachability weight W(u)", "The discounted expected number of rounds in which the position u can be reached. It weights the static criterion in Theorem 2."],
]

eq_terms = [
    ["Term", "Meaning"],
    ["Strategy-proof", "Reporting your target is best whatever the others report (in one round)."],
    ["Uncompromising", "An influencer above the outcome cannot move it up by any report, and one below cannot move it down. The only property of the rule the proofs use."],
    ["Truthful reporting", "Every influencer reports its target at every position."],
    ["Markov (feedback) strategy", "A report that depends only on the current position x."],
    ["Markov perfect equilibrium", "A profile of Markov strategies from which no influencer gains by deviating at any position, on or off the equilibrium path."],
    ["One-shot deviation principle", "With discounted, bounded losses, a profile is Markov perfect iff no player gains by changing its report for one round and then returning to it. Every check below uses it."],
    ["Deviation gain", "The largest such one-round gain over a grid of positions and reports. 0 means an equilibrium (up to the grid)."],
    ["Target form", "Each influencer aims the next position at a fixed target as far as its report allows: the shape every equilibrium we compute has."],
    ["Holding", "Playing the one-shot Nash actions at every position, ignoring the receiver's state."],
    ["d*, k, slack s, &rho;", "d*: the one-shot outcome. k = (1 - g&lambda;<super>2</super>)/(1 - g&lambda;). Slack s = (t - d*)/(1 - d*) for a player at the top edge. &rho;(&lambda;, g) = &lambda;(1 - g&lambda;)/(1 - g&lambda;<super>2</super>), the slack needed for holding."],
    ["Straddle / interior / shared", "Kinds of one-shot equilibrium in a coordinate: the targets lie on either side of d* (both at edges), one action is inside the box, or the targets coincide."],
]

learn_terms = [
    ["Term", "Meaning"],
    ["Self-play", "One policy trained for both players at once against itself."],
    ["REINFORCE, A2C, PPO", "Policy-gradient methods: the plain likelihood-ratio gradient; with a learned value function as baseline; and with a clipped probability ratio and several epochs per episode."],
    ["Exploitability", "The most a player gains by deviating from the learned actions in the one-shot game (grid of 101 &times; 101 points). 0 at a Nash equilibrium."],
    ["Feedback exploitability", "The same with inertia: the largest one-round gain over an 11 &times; 11 grid of dog positions, continuation simulated exactly. 0 at any feedback equilibrium."],
    ["Walking game", "Players stand in the square and walk at most 0.2 per step; the dog follows a blend of their positions."],
    ["Nash-Q (continuous)", "One network Q(s, a<sub>1</sub>, a<sub>2</sub>) returns both players' Q-values; its target uses the equilibrium of each stage game."],
    ["Cartesian / polar / square moves", "A move written as (v<sub>x</sub>, v<sub>y</sub>) in a disk, as (r, &theta;) in the same disk, or as (v<sub>x</sub>, v<sub>y</sub>) in a square."],
    ["Exact gradient play", "Noise-free learning: each player's payoff is computed exactly by following the deterministic dog, and both climb their own gradient (Adam) at once."],
    ["Table / network policy", "A table has one action per grid position (interpolated); a network shares parameters across positions."],
    ["Best-response regression", "Every 25 steps compute each player's exact best one-round action at every state and fit the policy to it (soccer: the stage equilibrium)."],
    ["Duality gap", "In zero-sum soccer, how far a state's values are from the stage-game equilibrium; 0 only at a Markov perfect equilibrium."],
    ["Soccer, sticky-price duopoly", "The two other games of the learning results: a 7 &times; 5 zero-sum grid game, and a duopoly whose price moves part of the way to inverse demand."],
]

modules = [
    ["Module", "What it computes"],
    ["mechanisms.py", "All the receiver math: next outcome for every receiver, equilibrium targets (<i>target_equilibrium</i>), social loss, deviation gains, the planner, the compensated receiver and its losses, the reachability weight, the slow-receiver criteria."],
    ["many_houses.py", "BIG's one-shot equilibrium with N players (<i>one_shot</i>) and the averaging receiver's next outcome (<i>next_dog</i>)."],
    ["inertia_theory.py", "Propositions 1 and 2: the hold condition, the explicit feedback policy, deviation gains, the target fixed point, the continuous-time limit."],
    ["inertia_mpe.py, inertia_dp.py", "Independent checks of inertia_theory: exact one-round gains under holding, and the feedback equilibrium by dynamic programming."],
    ["nash.py, exact_nash.py, discretize.py, verify.py", "One-shot equilibria (best-response iteration, Lemke-Howson, linear programming), the action grid, and exploitability."],
    ["env.py, agents.py, networks.py", "The game as an environment, the Gaussian policies, the network architectures (separate, shared, partially shared)."],
    ["train.py, train_ac.py, experiments.py", "REINFORCE, A2C and PPO self-play, and the convergence study."],
    ["feedback_learning.py, exact_play.py", "Learning from random dog positions, exact gradient play, and best-response regression."],
    ["soccer*.py, sticky_duopoly.py, duopoly_learning.py", "The two other games and their exact learning runs."],
    ["polar*.py, dog_game_study.py, dog_game_trajectories.py", "The walking game, its policies, runs and saved paths."],
    ["dog_game_studies.py", "Runs each study and appends one JSON row per run to results/dog_game_studies.jsonl."],
    ["dog_game_summary.py", "Turns the saved rows into every table quoted in the documents."],
    ["summary_figures.py, summary_pdf.py, results_report.py, methods_guide.py", "The figures, the key-results summary, the results report and this guide."],
    ["archive/", "Finished threads no current result needs (Nash-Q on a grid and with continuous moves, polar-action studies, soccer extras, report builders), frozen at the tag pre-archive where they still run."],
]

methods = [
    "<b>Equilibria of the plain receivers</b> (<i>target_equilibrium</i>). Start with targets equal to the houses. For each "
    "influencer, evaluate on a grid of 801 next positions this round's loss plus the discounted loss of the path that follows, "
    "take the best grid point as its new target, average old and new targets (damping 0.5), and repeat until nothing changes. "
    "Accuracy is about 0.015 in loss, the tolerance used for ties.",
    "<b>Losses</b> (<i>social_loss</i>). Follow the receiver for 200 rounds from 101 evenly spread starts and add every "
    "influencer's discounted squared distance (0.9<super>200</super> is negligible). With shocks, a Bellman equation on a "
    "grid with Gauss-Hermite quadrature replaces the simulation.",
    "<b>Equilibrium checks</b> (<i>deviation_gain</i>). At 21 positions, for each influencer, try every report on a 201-point "
    "grid for one round, then return to the profile, and compare. The largest improvement is reported; 0 means equilibrium on "
    "that grid.",
    "<b>Random games.</b> Targets drawn uniformly on [0, 1] (or from a Beta distribution for skew), 100 draws per point unless "
    "stated (500 to 1,000 for the design grids); differences are paired by draw and reported with standard errors.",
    "<b>Theorems checked numerically.</b> Each theorem has a test that computes both sides: zero deviation gain for compensated "
    "receivers, the welfare integral against simulation, the phantom derivative against finite differences.",
    "<b>Feedback equilibrium of the dog game.</b> Proposition 2's policy in closed form; checked against dynamic programming "
    "(value iteration on a grid) and by deviation gains over 200 random one-coordinate games.",
    "<b>Learning.</b> REINFORCE, A2C and PPO with Adam (defaults: learning rate 0.01, 20 steps per episode; PPO clip 0.2, 4 epochs), "
    "scored by exploitability. Exact gradient play: 41 starting positions, Adam, 3,000 steps. Best-response regression refits "
    "every 25 steps. Continuous Nash-Q: best responses by a dense batched grid, gradient ascent (autograd or finite "
    "differences) or a quadratic fitted to five points; stage equilibria by simultaneous gradient steps.",
    "<b>From code to document.</b> A study in dog_game_studies.py saves one JSON row per run; dog_game_summary.py turns the "
    "rows into tables; the documents quote those tables. To trace a number: table, then study, then function.",
]

logic = [
    "<b>Why players exaggerate (BIG).</b> Under averaging a player's report moves the outcome by only its weight, so the "
    "best response to any profile overshoots the target; the box stops it at an edge. The one-shot game is a potential game, "
    "and its equilibrium is a weighted generalized median with at most one pivot inside.",
    "<b>Why inertia breaks strategy-proofness.</b> With truthful reports a plain median receiver moves only part of the way to "
    "the median target. The median influencer can then put the receiver exactly on its target by over-reporting, which helps "
    "it this round and every later round.",
    "<b>Why averaging is a generalized median.</b> In target form, the averaging receiver's equilibrium targets coincide with "
    "those of the median with phantoms k/N, so both have the same paths and losses at every inertia.",
    "<b>Why the ranking reverses.</b> Inertia does not change where a receiver ends up, only how fast. Under averaging the "
    "influencers on the far side keep pulling against the move; the median receiver moves at full speed once a majority "
    "pushes one way. A slow receiver makes speed matter more, so the median receiver wins more often.",
    "<b>Why compensation is truthful (Theorem 1).</b> The truthful transition G(x) = clip(s, a(x), b(x)) is nondecreasing by "
    "(R2) and moves x toward s without crossing it by (R1). A deviation by an influencer above s can only push today's position "
    "down (the rule is uncompromising), and because G preserves order, every later position is weakly lower too, so every "
    "round's loss weakly rises. By the one-shot deviation principle, no deviation pays.",
    "<b>Why welfare has a closed form (Theorem 2).</b> The set reachable after k rounds is [A<sub>k</sub>, B<sub>k</sub>] "
    "whatever path is taken, and the planner's best position in it is the one closest to the mean target, so the planner is "
    "a compensated receiver. Differentiating the loss in the end point s gives &Lambda;'(s) W(s).",
    "<b>Why the design ignores inertia (Theorem 3).</b> Moving one phantom moves the end point only when that phantom is the "
    "median, an event that depends on that phantom alone. So the gradient separates by phantom, and W only rescales each "
    "term without changing its sign: the optimum is the static one, k/N for uniform targets.",
    "<b>Why the feedback equilibrium has a closed form (Propositions 1-2).</b> Under a stationary profile a player's one-round "
    "objective depends only on the next dog position. When it is single-peaked, the best reply is to aim at the same target "
    "everywhere, as far as the weight allows; under holding the peak is d* + k(t - d*).",
    "<b>Why networks miss the feedback equilibrium.</b> A plausible mechanism, not tested: a shared network carries "
    "'push to the edge' from the many positions where it is right to the few where easing off is right. Fitting per-state "
    "best responses weights every state equally and removes the failure.",
]

example = [
    ["Question (targets 0.1, 0.2, 0.9; inertia 0.6)", "Answer"],
    ["BIG without inertia: reports and outcome", "(0, 0, 1), outcome 1/3: every influencer exaggerates"],
    ["Plain median, truthful, from x = 0", "0.6 &middot; 0 + 0.4 &middot; 0.2 = 0.08: only 40% of the way"],
    ["Plain median in equilibrium, from x = 0", "the median influencer reports 0.5; outcome 0.2"],
    ["Compensated median, truthful, from 0 and from 0.9", "clip(0.2, 0, 0.4) = 0.2 and clip(0.2, 0.54, 0.94) = 0.54"],
    ["Planner from 0", "clip(0.4, 0, 0.4) = 0.4 (mean target 0.4)"],
    ["Social loss: planner / compensated averaging / plain averaging / plain median", "3.8138 / 3.9413 / 4.0264 / 4.9182"],
    ["Deviation gain with truthful reports: plain median / compensated median / compensated mean", "0.0544 / 0 / 0.0322"],
]


def build(out=OUTPUT):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    story = [
        Paragraph("The dog game with a persistent receiver: definitions, implementation, methods and logic", title), Spacer(1, 4),
        Paragraph("Charith Reddy Pareddy &middot; November 2, 2026 &middot; companion to the results report", sub), Spacer(1, 8),
        Paragraph("This guide explains every term the results use, what the code actually implements, how each number is "
                  "computed, and the reasoning behind each result, so that every piece can be explained and checked without "
                  "reading the code first.", body),
        Paragraph("1. The model", h),
        box(["One coordinate: position x in [0, 1]. Every round each influencer reports r<sub>i</sub> in [0, 1]; the receiver "
             "moves to x'; influencer i loses (x' - t<sub>i</sub>)<super>2</super>, discounted by &delta; = 0.9.",
             "Plain receiver: x' = &lambda;x + (1 - &lambda;) &times; aggregate(r), where the aggregate is the mean, the median, "
             "or a generalized median median(r, &alpha;).",
             "Compensated receiver: x' = clip(median(r, &alpha;), a(x), b(x)); for proportional inertia [a(x), b(x)] = "
             "[&lambda;x, &lambda;x + 1 - &lambda;], exactly the positions a plain receiver can reach.",
             "Planner: x' = clip(mean target, a(x), b(x))."]),
        Paragraph("2. Definitions", h),
        Paragraph("<b>The game</b>", body), table(game_terms, W),
        Spacer(1, 6), Paragraph("<b>Receivers</b>", body), table(receiver_terms, W),
        Spacer(1, 6), Paragraph("<b>Equilibrium</b>", body), table(eq_terms, W),
        Spacer(1, 6), Paragraph("<b>Learning</b>", body), table(learn_terms, W),
        Paragraph("3. What is implemented", h),
        Paragraph("The repository has two sets of code: the receiver results need only <i>mechanisms.py</i>, <i>many_houses.py</i> "
                  "and the study and table functions; the learning results use the environment, policies and training loops. Every "
                  "module has a test, and docs/CODE_GUIDE.md maps each result to its proof, study, table and test.", body),
        table(modules, [2.15 * inch, 4.85 * inch]),
        Paragraph("4. Methods: how each number is computed", h),
        bullets(methods),
        Paragraph("5. Logic: why each result holds", h),
        bullets(logic),
        Paragraph("6. A worked example to check by hand", h),
        table(example, [3.6 * inch, 3.4 * inch]),
        Spacer(1, 4),
        Paragraph("Losses are averaged over 101 evenly spread starts. The plain median's equilibrium and the compensated "
                  "median with truthful reports follow the same path, so they have the same loss.", body),
        Paragraph("7. What the methods cannot show", h),
        bullets(["Grids: equilibria to about 0.015 in loss, deviation checks on 21 positions and 201 reports; \"gain 0\" means no "
                 "gain on that grid.",
                 "The slow-receiver results are first order; their numerical checks hold to about 4 to 15% at inertia 0.99 to "
                 "0.995, less well at 0.9.",
                 "\"Numerical\" results rest on the saved draws and seeds, not on a proof. The proofs have not been refereed."]),
    ]
    SimpleDocTemplate(str(out), pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch, topMargin=0.65 * inch,
                      bottomMargin=0.65 * inch, title="The dog game: definitions, implementation, methods and logic",
                      author="Charith Reddy Pareddy").build(story, onFirstPage=number_pages, onLaterPages=number_pages)
    return out


if __name__ == "__main__":
    print(build(*sys.argv[1:]))
