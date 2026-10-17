"""Writes docs/WORKING_PAPER.md (and, with reportlab, docs/WORKING_PAPER.pdf): a short working paper on the dog game
with inertia. The tables and the counts quoted in the text come from results/dog_game_studies.jsonl.

Run with `python3 -m doggame.working_paper`.
"""

import sys

from doggame import dog_game_summary as S

OUTPUT = "docs/WORKING_PAPER.md"


def counts(studies):
    random = {r[0]: r for r in S.inertia_random_table(studies)[1:]}
    straddle, interior = random["straddle"], random["interior"]
    return dict(straddle=straddle[1], interior=interior[1], interior_ok=interior[2].split()[0], holds=straddle[4].split()[0])


def text(studies):
    c, md = counts(studies), S.markdown
    return f"""# When the Dog Has Inertia: Feedback Equilibria, Lost Potential and Equilibrium Selection in an Influence Game

Charith Reddy Pareddy, University of Wisconsin-Madison. Working paper draft, not peer reviewed. Code, data and
tests: github.com/Charith-Reddy-Pareddy/dog-game.

## Abstract

Two players repeatedly pull a shared point, the dog, toward their own houses. Without inertia the game is an instance
of the Battling Influencers Game, an exact potential game (Wu et al., 2025). We study what survives when the dog keeps
part of its position each round. We prove that holding the one-shot equilibrium actions is a Markov-perfect
equilibrium if and only if each player's slack beyond the dog is at least w0(1 - g w0)/(1 - g w0^2), where w0 is the
inertia and g the discount, and we give an explicit piecewise-affine feedback equilibrium whenever the houses lie on
either side of the dog; numerically, a fixed point for the players' targets gives a feedback equilibrium in 199 of 200
random games. The condition extends to many houses, where the admissible inertia shrinks like 1/N. We show
that the dynamic game is not a Markov potential game once policies react to the dog. For learning, we find that
policy-gradient self-play from random starts learns to hold the one-shot actions, which is exploitable off the
equilibrium path, and that on the potential's flat direction learning is a random walk stopped by the action box,
so which of two mirror equilibria wins is a fair coin. Exact gradient play finds the feedback equilibrium with a
table of actions but not with a shared network.

## 1. Introduction

Influence games, in which agents with fixed targets push a shared quantity, are a clean testbed for multi-agent
learning: the static game has a potential, its equilibria are computable, and its learning dynamics can be checked
exactly. The Battling Influencers Game proves the potential structure and characterizes pure equilibria of the
one-shot game (Wu et al., 2025). Real influence is persistent, though: opinions, prices or a dog on a leash keep part
of their state. This paper asks what the static results imply once the shared point has inertia, and what
policy-gradient learners find. Our contributions, each verified computationally in the accompanying repository:

- A necessary and sufficient condition for the one-shot equilibrium to remain Markov-perfect under inertia
  (Proposition 1), and an explicit piecewise-affine feedback equilibrium when it does not (Proposition 2), both with
  proofs and checked against an exact dynamic program.
- The extension of the condition to many houses, where the static outcome is the generalized median of the average
  voting rule (Renault and Trannoy, 2005), and the finding that the admissible inertia shrinks like 1/N.
- A concrete check that the dynamic game is not a Markov potential game, although every state game is a potential game.
- Equilibria in target form beyond Proposition 2, found by a two-dimensional fixed point, and the continuous-time
  limit, where the equilibrium is bang-bang and a noisy dog moves its switching points.
- Two learning findings: self-play misses the feedback equilibrium, and selection along a flat potential is a random
  walk stopped by the action box. The second corrects an earlier small-sample result of ours.

The static potential structure and the general fact that stage-wise potential games need not be Markov potential
games are known; we claim novelty only for the results above, for this game, as far as the literature we checked goes.

## 2. Related work

Potential games go back to Monderer and Shapley (1996); noisy learning in them selects potential maximizers (Blume,
1993). The Battling Influencers Game (Wu et al., 2025) is the static game studied here. Markov potential games
(Leonardos et al., ICLR 2022) give convergence of independent policy gradient, and a Markov game whose stage games are
potential games need not be one; a continuous-state counterexample to sufficient conditions appears in arXiv
2405.08206. Potential differential games reduce to one optimal-control problem in open loop (Kavuncu, Yaraneri and
Mehr, RSS 2021). Feedback equilibria of constrained linear-quadratic games are piecewise affine over polyhedral
regions (arXiv 2001.01826; Benenati and Belgioioso, arXiv 2512.07749); Proposition 2 is an explicit instance with a
proof specific to this game. The average voting rule's equilibrium outcome is a generalized median (Renault and
Trannoy, 2005). Policy gradient provably finds Nash equilibria of unconstrained linear-quadratic games (Hambly, Xu and
Yang, JMLR 2023); equilibrium selection by multi-agent policy gradient is studied in finite games (ICML 2026,
arXiv 2605.18078).

## 3. Model

Two players choose a_red and a_blue in the unit square. The dog moves to x' = w0 x + (1 - w0)(w a_red + (1 - w)
a_blue), with inertia w0 in [0, 1) and red's weight w in (0, 1), and player i receives -||x' - h_i||^2, discounted by
g in [0, 1). Rewards, dynamics and the action box all separate by coordinate, so the game splits into two independent
one-dimensional games, and a profile is a feedback (Markov-perfect) equilibrium iff it is one in each coordinate. With
w0 = 0 this is the one-shot game, with potential Phi = -||d||^2 + 2 w h_red . a_red + 2 (1 - w) h_blue . a_blue, where
d = w a_red + (1 - w) a_blue. Let (a_red*, a_blue*) be the one-shot equilibrium, d* its dog, k = (1 - g w0^2)/(1 - g w0)
and T_i = d* + k (h_i - d*) player i's target.

## 4. Feedback equilibria with inertia

**Proposition 1.** Holding the one-shot actions at every dog position is a feedback equilibrium iff every player at an
edge of the box has slack s >= rho(w0, g) = w0 (1 - g w0)/(1 - g w0^2), with s = (h - d*)/(1 - d*) at the top edge and
(d* - h)/d* at the bottom, and, when w0 > 0, no one-shot action is strictly inside the box.

**Proposition 2.** If the houses lie on either side of the dog (h_blue <= w <= h_red after relabeling), the policy in
which each player aims the next dog position at its target T_i, as far as its own weight allows, with the other at its
one-shot action, is a feedback equilibrium for every w0 in [0, 1) and g in [0, 1). It is piecewise affine in the dog's
position and equals the one-shot profile wherever holding is optimal.

Both proofs use the one-shot deviation principle. Under holding, a player's one-round objective is a concave
quadratic in the next dog position peaked at T_i, which gives Proposition 1. For Proposition 2, a contraction
argument shows the one-round objective under the policy is single-peaked at T_i on all of [0, 1], so the clipped
target is a best reply. Full proofs are in docs/INERTIA_PROOFS.md.

**Checks.** An exact dynamic program (Nash value iteration, 401 positions by 101 actions per coordinate) agrees with
both propositions to one action-grid step in every coordinate where the houses differ, in five house setups at w0 =
0.3, 0.6 and 0.9 (where the houses coincide, the game has a continuum of equilibria and the program picks another). On 200
random one-coordinate games, with the deviation gain computed by simulating the dog rather than on a grid:

{md(S.inertia_random_table(studies))}

Proposition 2 holds in all {c['straddle']} straddling games, and holding alone in {c['holds']} of them, as
Proposition 1 predicts. With an interior one-shot action the policy is an equilibrium in only {c['interior_ok']} of
{c['interior']} games: the interior player pushes at an edge off its house, the dog returns faster than under holding,
and the other player's best target moves. Because a player's one-round objective depends only on the next dog
position, each player can still aim at one target everywhere when that objective is single-peaked; with the targets
from a two-dimensional fixed point:

{md(S.target_random_table(studies))}

The fixed point is a feedback equilibrium in 199 of 200 random games; a proof, and uniqueness, are open.

**Many houses.** With N players the one-shot dog is the generalized median of the average voting rule, at most one
player (the pivot) is inside the box, and Proposition 1 holds verbatim. Drawing N houses from Beta(2, 5) with equal
weights (200 draws per N, discount 0.9):

{md(S.many_houses_table(studies))}

A pivot inside the box (about two draws in three) rules holding out at any inertia; otherwise the largest admissible
inertia scales like 0.35/N. With many houses the static equilibrium is almost never a feedback equilibrium. The
target form carries over: if every player aims at a target, each round is the one-shot game with the targets as houses,
and fixed-point targets give a feedback equilibrium whenever every player's one-round objective is single-peaked:

{md(S.many_targets_table(studies))}

## 5. The dynamic game is not a Markov potential game

In a Markov potential game, every 4-cycle of unilateral policy changes sums to zero. Exact values from the starting
dog position of one setup:

{md(S.markov_potential_table(studies))}

With policies that ignore the dog, the dynamic game remains a potential game at any inertia (open loop); with
policies that react to it, it is not, even at w0 = 0. The convergence theory of independent policy gradient in Markov
potential games therefore does not cover the feedback play studied below.

## 6. Learning

**Self-play misses the feedback equilibrium.** We train self-play from uniformly random dog positions and score the
learned mean policy by its feedback exploitability: the largest and mean gain, over dog positions and players, from a
one-round deviation followed by the learned policy, with the dog simulated exactly. Every feedback equilibrium scores
0, so the score does not assume uniqueness.

{md(S.feedback_learning_table(studies))}

A2C and PPO learn to hold the one-shot actions at every position, which is unexploitable on the equilibrium path and
exploitable off it, and this survives five times longer training, a non-saturating Beta policy and a setting with 3.5
times the off-path incentive. REINFORCE does not do better. To separate the learning dynamics from estimation noise,
we also run exact gradient play on one coordinate: deterministic policies, each player ascending the exact gradient of
its own payoff averaged over starting positions.

{md(S.exact_play_table(studies))}

A table of actions reaches the feedback equilibrium every time and a shared network never does, so function
approximation, not sampling noise, is what stops the learners; a plausible but untested mechanism is that the network
carries "push to the edge" from the many positions where it is right to the few where easing off is right. Low
on-path exploitability therefore does not certify a feedback equilibrium.

**Selection on a flat potential.** When the houses share a coordinate, the potential is flat along a continuum of
equilibria. REINFORCE self-play, 50 seeds per initial exploration noise:

{md(S.selection_noise_table(studies))}

No run ends at the interior equilibrium, and which mirror image wins is consistent with a fair coin; an earlier
10-seed analysis of ours had suggested that the noise level flips it, which did not replicate. Two ablations show why:

{md(S.selection_clip_table(studies))}

The expected gradient along the flat direction is zero but noisy, so the policy means random-walk; with an unbounded
mean the walk leaves the box, and the default sigmoid policy stops it at the edges. Clipped exploration noise also
makes the averaged potential dip at the interior, pushing either way equally.

## 7. Continuous time and noise

As the time step dt shrinks (w0 = 1 - dt, discount exp(-r dt)), k tends to (r + 2)/(r + 1) and rho to (r + 1)/(r + 2):
holding the one-shot actions needs slack at least (1 + r)/(2 + r), and Proposition 2's equilibrium becomes bang-bang,
each player pushing its edge toward its target and switching there. With a Gaussian shock on the dog the
equilibrium stays bang-bang, but its switching points move with the noise, also without clipping, which a
linear-quadratic benchmark with an action cost would not show (its policies are certainty-equivalent):

{md(S.noisy_inertia_table(studies))}

The shift grows smoothly from zero noise and is the same with half the time step and with an unclipped dog; why noise
moves the switching points this way is open, and one clipped run at sigma = 0.4 did not converge.

## 8. Discussion and limits

The static potential structure does not carry over to feedback play, but the static equilibrium survives on the
equilibrium path, and Proposition 1 says exactly where it fails off the path. The feedback equilibrium is explicit when
the houses straddle the dog and has the same target form numerically when they do not; learners nevertheless converge
to the static profile. Limits: the proofs are ours and not refereed; Proposition 2 excludes interior one-shot actions,
where the target fixed point is only numerical; learning results use one game, a few settings and five
seeds per setting; the continuous-time statements are limits of discrete-time equilibria, and continuous-time
q-learning is not studied.

## References

Blume (1993), The statistical mechanics of strategic interaction, Games and Economic Behavior. Benenati and
Belgioioso (2025), arXiv 2512.07749. Fudenberg and Tirole (1991), Game Theory. Hambly, Xu and Yang (2023), Policy
gradient methods find the Nash equilibrium in N-player general-sum linear-quadratic games, JMLR. Kavuncu, Yaraneri
and Mehr (2021), Potential iLQR, RSS. Leonardos, Overman, Panageas and Piliouras (2022), Global convergence of
multi-agent policy gradient in Markov potential games, ICLR. Monderer and Shapley (1996), Potential games, Games and
Economic Behavior. Renault and Trannoy (2005), Protecting minorities through the average voting rule, Journal of
Public Economic Theory 7, 169-199. Wu, Zhu, Cai and Zhu (2025), Battling Influencers Game, arXiv 2502.01127.
arXiv 2001.01826 (feedback Nash equilibria of constrained dynamic games). arXiv 2405.08206 (Beyond Theorems: a
counterexample to potential Markov game criteria). arXiv 2605.18078 (equilibrium selection in multi-agent policy
gradients via opponent-aware basin entry, ICML 2026).
"""


def build(out=OUTPUT):
    studies = S.load(S.STUDIES)
    with open(out, "w") as f:
        f.write(text(studies))
    return out


if __name__ == "__main__":
    print(build(*sys.argv[1:]))
