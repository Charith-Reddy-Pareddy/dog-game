# When the Dog Has Inertia: Feedback Equilibria, Lost Potential and Equilibrium Selection in an Influence Game

Charith Reddy Pareddy, University of Wisconsin-Madison. Working paper draft, not peer reviewed. Code, data and
tests: github.com/Charith-Reddy-Pareddy/dog-game.

## Abstract

Two players repeatedly pull a shared point, the dog, toward their own houses. Without inertia the game is an instance
of the Battling Influencers Game, an exact potential game (Wu et al., 2025). We study what survives when the dog keeps
part of its position each round. We prove that holding the one-shot equilibrium actions is a Markov-perfect
equilibrium if and only if each player's slack beyond the dog is at least w0(1 - g w0)/(1 - g w0^2), where w0 is the
inertia and g the discount, and we give an explicit piecewise-affine feedback equilibrium whenever the houses lie on
either side of the dog. The condition extends to many houses, where the admissible inertia shrinks like 1/N. We show
that the dynamic game is not a Markov potential game once policies react to the dog. For learning, we find that
policy-gradient self-play from random starts learns to hold the one-shot actions, which is exploitable off the
equilibrium path, and that on the potential's flat direction learning is a random walk stopped by the action box,
so which of two mirror equilibria wins is a fair coin.

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
both propositions to one action-grid step in every coordinate of five house setups at w0 = 0.3, 0.6 and 0.9. On 200
random one-coordinate games, with the deviation gain computed by simulating the dog rather than on a grid:

| One-shot equilibrium | Games | Explicit policy is an equilibrium (gain < 1e-9) | Largest one-round gain | Holding is Markov-perfect |
|---|---|---|---|---|
| straddle | 77 | 77 of 77 | 0.0e+00 | 30 of 77 |
| interior | 123 | 70 of 123 | 2.3e-02 | 0 of 123 |

Proposition 2 holds in all 77 straddling games, and holding alone in 30 of them, as
Proposition 1 predicts. With an interior one-shot action the policy is an equilibrium in only 70 of
123 games: the interior player pushes at an edge off its house, the dog returns faster than under holding,
and the other player's best target moves. That case, and uniqueness, are open.

**Many houses.** With N players the one-shot dog is the generalized median of the average voting rule, at most one
player (the pivot) is inside the box, and Proposition 1 holds verbatim. Drawing N houses from Beta(2, 5) with equal
weights (200 draws per N, discount 0.9):

| Houses N | One-shot dog: mean (sd); limit 0.339 | No pivot inside (limit 0.34) | Largest inertia: mean | N times it, median when no pivot |
|---|---|---|---|---|
| 2 | 0.368 (0.115) | 0.20 | 0.0399 | 0.27 |
| 4 | 0.340 (0.090) | 0.28 | 0.0357 | 0.43 |
| 8 | 0.340 (0.055) | 0.34 | 0.0206 | 0.27 |
| 16 | 0.337 (0.038) | 0.29 | 0.0111 | 0.45 |
| 32 | 0.340 (0.028) | 0.35 | 0.0058 | 0.33 |
| 64 | 0.341 (0.019) | 0.32 | 0.0024 | 0.32 |
| 128 | 0.339 (0.013) | 0.34 | 0.0012 | 0.34 |
| 256 | 0.340 (0.010) | 0.34 | 0.0007 | 0.33 |
| 512 | 0.340 (0.007) | 0.28 | 0.0003 | 0.43 |
| 1024 | 0.340 (0.005) | 0.32 | 0.0002 | 0.35 |

A pivot inside the box (about two draws in three) rules holding out at any inertia; otherwise the largest admissible
inertia scales like 0.35/N. With many houses the static equilibrium is almost never a feedback equilibrium.

## 5. The dynamic game is not a Markov potential game

In a Markov potential game, every 4-cycle of unilateral policy changes sums to zero. Exact values from the starting
dog position of one setup:

| Inertia w0 | Policies that ignore the dog | Policies that react to the dog |
|---|---|---|
| 0.0 | +6.7e-15 | +7.90 |
| 0.3 | +3.1e-15 | +7.12 |
| 0.6 | +1.8e-15 | +5.63 |
| 0.9 | +5.6e-15 | +1.85 |

With policies that ignore the dog, the dynamic game remains a potential game at any inertia (open loop); with
policies that react to it, it is not, even at w0 = 0. The convergence theory of independent policy gradient in Markov
potential games therefore does not cover the feedback play studied below.

## 6. Learning

**Self-play misses the feedback equilibrium.** We train self-play from uniformly random dog positions and score the
learned mean policy by its feedback exploitability: the largest and mean gain, over dog positions and players, from a
one-round deviation followed by the learned policy, with the dog simulated exactly. Every feedback equilibrium scores
0, so the score does not assume uniqueness.

| Setting | Policy | Runs | Largest one-round gain: median (range) | Mean one-round gain: median |
|---|---|---|---|---|
| setup 3, w0 = 0.6 | Proposition 2 | 1 | 0.0000 | 0.0000 |
| setup 3, w0 = 0.6 | hold the one-shot actions | 1 | 0.0756 | 0.0114 |
| setup 3, w0 = 0.6 | REINFORCE | 5 | 0.1537 (0.1156 to 0.6714) | 0.0275 |
| setup 3, w0 = 0.6 | A2C | 5 | 0.0756 (0.0755 to 0.0759) | 0.0115 |
| setup 3, w0 = 0.6 | PPO | 5 | 0.0756 (0.0756 to 0.4599) | 0.0115 |
| setup 3, w0 = 0.6 | REINFORCE, Beta policy | 5 | 0.0900 (0.0855 to 0.1287) | 0.0247 |
| setup 3, w0 = 0.6 | A2C, Beta policy | 5 | 0.0854 (0.0813 to 0.0891) | 0.0222 |
| setup 3, w0 = 0.6 | PPO, Beta policy | 5 | 0.0991 (0.0700 to 0.4077) | 0.0323 |
| setup 3, w0 = 0.6 | A2C, 15000 episodes | 3 | 0.0756 (0.0756 to 0.0756) | 0.0114 |
| setup 3, w0 = 0.6 | A2C, Beta policy, 15000 episodes | 3 | 0.0791 (0.0776 to 0.0798) | 0.0150 |
| setup 5, w0 = 0.9 | Proposition 2 | 1 | 0.0000 | 0.0000 |
| setup 5, w0 = 0.9 | hold the one-shot actions | 1 | 0.2084 | 0.0397 |
| setup 5, w0 = 0.9 | REINFORCE | 5 | 0.2014 (0.1506 to 0.3832) | 0.0337 |
| setup 5, w0 = 0.9 | A2C | 5 | 0.2084 (0.2084 to 0.2084) | 0.0397 |
| setup 5, w0 = 0.9 | PPO | 5 | 0.2084 (0.2084 to 0.2084) | 0.0397 |

A2C and PPO learn to hold the one-shot actions at every position, which is unexploitable on the equilibrium path and
exploitable off it, and this survives five times longer training, a non-saturating Beta policy and a setting with 3.5
times the off-path incentive. REINFORCE does not do better. Low on-path exploitability therefore does not certify a
feedback equilibrium. Why learners miss it is open; small off-path gains relative to gradient noise is one untested
explanation.

**Selection on a flat potential.** When the houses share a coordinate, the potential is flat along a continuum of
equilibria. REINFORCE self-play, 50 seeds per initial exploration noise:

| Initial noise (std) | red top, blue bottom | red bottom, blue top | both at y = 0.5 | other | p (mirror images equally likely) |
|---|---|---|---|---|---|
| 0.14 | 19 of 50 | 22 of 50 | 0 of 50 | 9 of 50 | 0.76 |
| 0.37 | 31 of 50 | 19 of 50 | 0 of 50 | 0 of 50 | 0.12 |
| 0.61 (default) | 31 of 50 | 19 of 50 | 0 of 50 | 0 of 50 | 0.12 |
| 1.00 | 30 of 50 | 20 of 50 | 0 of 50 | 0 of 50 | 0.20 |

No run ends at the interior equilibrium, and which mirror image wins is consistent with a fair coin; an earlier
10-seed analysis of ours had suggested that the noise level flips it, which did not replicate. Two ablations show why:

| Policy mean in y, actions and dog | red top, blue bottom | red bottom, blue top | both at y = 0.5 | other | Red's final y: quartiles | Red's y outside [0, 1] |
|---|---|---|---|---|---|---|
| sigmoid, clipped (default) | 31 of 50 | 19 of 50 | 0 of 50 | 0 of 50 | 0.00 / 0.99 / 1.00 | 0 of 50 |
| sigmoid, not clipped | 20 of 50 | 21 of 50 | 0 of 50 | 9 of 50 | 0.03 / 0.64 / 0.97 | 0 of 50 |
| linear, not clipped | 17 of 50 | 20 of 50 | 0 of 50 | 13 of 50 | -2.26 / 0.33 / 1.29 | 38 of 50 |

The expected gradient along the flat direction is zero but noisy, so the policy means random-walk; with an unbounded
mean the walk leaves the box, and the default sigmoid policy stops it at the edges. Clipped exploration noise also
makes the averaged potential dip at the interior, pushing either way equally.

## 7. Discussion and limits

The static potential structure does not carry over to feedback play, but the static equilibrium does survive on the
equilibrium path, and exactly characterizes where it fails off the path. The feedback equilibrium is explicit when the
houses straddle the dog; learners nevertheless converge to the static profile. Limits: the proofs are ours and not
refereed; Proposition 2 excludes interior one-shot actions; learning results use one game, a few settings and five
seeds per setting; and the noisy continuous-time version of the game, which needs an action cost for a Riccati
benchmark, is not studied.

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
