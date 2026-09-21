# Research Report: Solving the Dog Game

**Live demo:** https://charith-reddy-pareddy.github.io/dog-game/
**Code:** this repository (`doggame/`), reproducible with `pytest` and
`python3 -m doggame.experiments`.

## 1. Background and motivation

The broader research program is about solving multi-agent Markov
games. Earlier work built an exact minimax-value-iteration solver for
a grid soccer game and used it to study when equilibria involve mixed
(randomized) strategies, and to look for game states where the optimal
policy is non-intuitive. That work is planning, not learning: rewards
and transitions are known exactly, so the problem is finding an
equilibrium, not estimating one from samples.

The dog game (also called the "Battling Influencer Game" or
"AI-dog/BIG") was introduced as a smaller, more tractable
general-sum companion to the soccer game, specifically so two
solution methods discussed for the soccer game — DQN-style Nash-Q
learning and policy-gradient self-play — could be tried somewhere with
a known, provably unique equilibrium before being applied to soccer,
where equilibria are not unique and ground truth is harder to pin
down.

## 2. The game

Two players, red and blue, each own a house at a fixed point in a
bounded, convex domain (a unit square). Each round, both players pick
a point in the domain; a shared "dog" moves to the `w / (1 - w)`
weighted average of the two picks. Each player is rewarded by the
negative squared distance from the dog to their own house. This is a
**general-sum** game (not zero-sum, unlike soccer), and it is
structurally identical to two influencers each staking out an opinion
to pull a shared audience toward them: because the domain is bounded,
a player who wants the dog exactly on their house must *overshoot* —
pick further out than their own house — to counteract the other
player's pull. Implementation: [`doggame/env.py`](doggame/env.py).

## 3. Analytical solution

Solving each player's unconstrained best response and substituting one
into the other shows there is no interior fixed point unless the two
houses share a coordinate — the unconstrained recursion just pushes
the action further in one direction forever. The bounded domain turns
this into a genuine equilibrium: both players get pushed to the
boundary. [`doggame/nash.py`](doggame/nash.py) finds this fixed point
by iterated best response and is used throughout the rest of this repo
as ground truth. `tests/test_nash.py` checks it against a hand-derived
symmetric case, an identical-houses degenerate case, and a
brute-force "no unilateral deviation helps" grid search.

## 4. Two solvers, as assigned in the planning meeting

### 4.1 Policy gradient (self-play REINFORCE)

[`doggame/agents.py`](doggame/agents.py) and
[`doggame/train.py`](doggame/train.py) implement a Gaussian
policy — `TwoPlayerPolicy` — trained with REINFORCE via `torch.optim`
(no hand-derived gradient code, per the meeting's explicit
instruction once a training package is in play). Three network
architectures are implemented behind one interface
([`doggame/networks.py`](doggame/networks.py)):

- **separate** — two fully independent networks.
- **shared** — one trunk, two output heads.
- **partial** — a shared trunk feeding two separate per-player
  branches (equivalent to one network with the cross-player weights
  frozen at zero, but simpler to express directly).

### 4.2 DQN / Nash-Q (discretized actions)

[`doggame/discretize.py`](doggame/discretize.py) turns the continuous
square into an N x N action grid and builds both players' payoff
matrices by calling the *same* transition/reward functions the
continuous environment uses, so the discretized game and the
continuous game agree on the rules by construction, not by luck.

[`doggame/fictitious_play.py`](doggame/fictitious_play.py) solves that
discretized stage game by iterated best response against the
opponent's running-average strategy — the "best-response dynamics"
approach the meeting raised as the practical alternative to an LP
solver, at the cost of being only approximate.

[`doggame/dqn.py`](doggame/dqn.py) then iterates the Nash-Q Bellman
backup `Q = R + discount * NashV(Q)` to a fixed point. Because the dog
game's transition never depends on the incoming state, every state
faces an identical one-shot game, so this backup collapses to
iterating on a single pair of payoff matrices rather than looping over
a state space — a property specific to this game's transition, not a
general shortcut (documented in the module's docstring, verified in
`tests/test_dqn.py` against the closed-form fixed point this collapse
implies: discounted value equals one-shot Nash value divided by
`1 - discount`).

## 5. Verification methodology

The meeting specified the actual check to use: for a mixed strategy,
compute the value of every action played with positive probability —
they should all be equal — and confirm no unplayed action would do
better. [`doggame/verify.py`](doggame/verify.py) implements two
versions of this:

- `exploitability` — for a continuous action pair, grid-search for the
  largest one-sided improvement either player could get by deviating.
  Used both offline (`doggame/experiments.py`) and live, in-browser,
  on the [demo page](https://charith-reddy-pareddy.github.io/dog-game/).
- `mixed_strategy_indifference_gap` — the discrete-strategy version of
  the same check, with a probability floor (`tol`) tuned to fictitious
  play's known noise floor (its running average never fully forgets
  its random initial pick, leaving a permanent ~1/iterations residual
  that is not a real equilibrium violation).

**A defect found and fixed while testing this module:** the floor-based
"which actions are played" filter, `strategy_row > tol`, can select
*zero* actions on a fine enough action grid — once the grid has more
than `1/tol` actions, a legitimately uniform mixed strategy spreads
every action's probability below the floor. That left
`best_response_gap` computed against an empty array, silently
returning `+inf` instead of a meaningful number (verified directly:
a 1024-action uniform strategy already triggers it). The fix falls
back to the action(s) actually holding the strategy's maximum
probability when none clear the floor, so the check stays finite and
usable as action-grid resolution grows — which is exactly the
direction question 6 (discretization resolution) pushes this code.
Covered by a regression test in `tests/test_verify.py`.

## 6. Findings

Findings 1 and 2 below are also presented, with live figures, on the
[GitHub Pages demo](https://charith-reddy-pareddy.github.io/dog-game/#findings);
they are reproduced verbatim from real runs of `doggame.experiments`
(15 runs: 5 house/weight configurations x 3 architectures, 1500
self-play episodes each), not hand-picked or simulated for
presentation.

### Finding 1 — "can policy gradient converge?" needs a number attached to it

14 of 15 (configuration, architecture) runs reached exploitability
below 0.004 within 1500 episodes. The 15th — the `separate`
architecture on `house_red=(0.8, 0.8)`, `house_blue=(0.2, 0.3)`,
`w=0.3` — was still at 0.183, clearly unconverged. Training that exact
run 3000 episodes further brought it to 0.004: it was not stuck in a
bad equilibrium, just slower, because one action dimension took longer
to get pushed out to the domain boundary than the others. This answers
question 2 honestly rather than optimistically: self-play policy
gradient does converge here, but "1500 episodes" is meaningless
without naming the configuration and architecture it was measured on.

### Finding 2 — multiple equilibria are real, not a training bug

When both houses share a y-coordinate (`house_red=(0.95, 0.5)`,
`house_blue=(0.05, 0.5)`), the y-axis of the stage game becomes
degenerate: any pair of y-actions that averages back to 0.5 is equally
optimal. `doggame.nash`'s iterated best response lands on the "obvious"
solution (both players at y = 0.5 exactly); policy-gradient training
instead consistently found a boundary-cancellation solution (red near
y = 1, blue near y = 0, `0.5*1 + 0.5*0 = 0.5`) in every architecture.
Checking both with `exploitability` confirms they are both genuinely
unexploitable (< 0.0002) — two different, equally valid equilibria,
directly matching the degenerate case raised in question 8, not a
training failure. This is why `doggame.verify` checks optimality
directly rather than distance to one precomputed candidate.

### Finding 3 — network architecture did not change *whether* training converged

Across the same 15 runs, `separate`, `shared`, and `partial`
architectures converged to equivalently low exploitability on 14 of
15 configurations; the one slow case above was slow specifically for
`separate`, while `shared` and `partial` converged on it within the
same 1500 episodes. That is a single data point, not a general
architecture ranking (question 7 remains open on *why*), but it is
evidence that parameter sharing does not obviously hurt convergence in
this game, and may help on harder configurations.

## 7. Open questions

The following questions from `RESEARCH_QUESTIONS.md` are not yet
answered by this repository:

- **Pre-training** (question 3): not yet tested — no experiment here
  initializes a policy from a related configuration before self-play.
- **Equilibrium selection consistency** (question 4): Finding 2 shows
  policy gradient can consistently prefer a boundary-cancellation
  equilibrium over the analytical solver's, but this has only been
  checked on one degenerate configuration; whether the *same*
  non-obvious equilibrium is selected across seeds, or only across
  architectures, is untested.
- **Stage-game solution quality vs. outer convergence** (question 5b):
  `doggame.dqn` uses fictitious play with a fixed iteration count; how
  the outer Bellman iteration's fixed point degrades as the inner
  fictitious-play solution gets coarser has not been measured.
- **Resolution vs. solution quality trade-off** (question 6):
  `tests/test_dqn.py` checks two resolutions (5 and 9); no systematic
  sweep exists yet, though the fix in Section 5 specifically removes a
  correctness blocker for testing much finer grids.
- **Clean vs. noisy mixed probabilities** (question 10): the
  fictitious-play noise floor is documented and accounted for in
  `mixed_strategy_indifference_gap`'s `tol`, but no rounding/cleanup
  pass exists to recover exact fractions (50/50, 1/3-2/3) the way the
  meeting's soccer-game discussion called for.
- **Transfer to the soccer game** (question 11): entirely open. The
  dog game's state-independent transition is explicitly called out in
  `doggame/dqn.py` as a simplification that a soccer game (state
  depends on ball possession) would not have, so the Nash-Q backup
  collapse this repo relies on would not directly apply there.
