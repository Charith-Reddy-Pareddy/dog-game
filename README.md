# Dog Game

**Live demo:** https://charith-reddy-pareddy.github.io/dog-game/
**Research questions:** [RESEARCH_QUESTIONS.md](RESEARCH_QUESTIONS.md)
**Assumptions:** [ASSUMPTIONS.md](ASSUMPTIONS.md)
**Dog-game results:** [PDF](dog_game_results.pdf) (convergence over seeds, and the angle-radius version)
**Research report:** [RESEARCH_REPORT.md](RESEARCH_REPORT.md) ([PDF version](research_report.pdf), built from it, with direct answers to each research question)

A two-player, general-sum game: a red house and a blue house each want a
shared "dog" to end up as close as possible to their own doorstep. Each
player picks a point in a bounded domain every round; the dog's next
position is a weighted average of the two points. Because the domain is
bounded, players are pushed to exaggerate their pick toward the edge of
the domain to counteract the other player's pull -- the same dynamic
that shows up in the "battling influencers" framing of the game, where
two influencers stake out opinions further from their true position to
drag a shared audience opinion toward them.

This repo implements the continuous-state, continuous-action version of
the game, following both subgroups' assignments from the planning
meeting: a policy-gradient solver and a DQN-style solver, three
multi-agent network architectures, an analytical Nash equilibrium
solver to check both against, and a browser visualization.

## Layout

- `doggame/env.py` -- `DogGameEnv`: the repeated game. Transition and
  reward functions are injected rather than hardcoded, so the same
  loop can drive a different game later without touching this class.
  `make_inertial_transition(w0)` builds the more general three-weight
  transition (`w0 * dog + w1 * pick_red + w2 * pick_blue`) for a dog
  with some momentum instead of fully re-targeting every round.
- `doggame/nash.py` -- closed-form-style continuous stage-game Nash
  solver via iterated best response. Because the unconstrained best
  response has no fixed point unless both houses share a coordinate,
  the bounded domain is what forces convergence, typically at the
  boundary.
- `doggame/networks.py` -- three multi-agent policy architectures
  behind one interface: `separate` (two independent networks),
  `shared` (one trunk, two output heads), and `partial` (a shared
  trunk with separate per-player branches).
- `doggame/agents.py` -- `TwoPlayerPolicy`: wraps one of the above
  architectures with a learned per-player log-std and samples/scores
  actions with `torch.distributions.Normal`. Training goes through
  `torch.optim`, not a hand-derived gradient.
- `doggame/train.py` -- REINFORCE self-play loop (the policy-gradient
  subgroup's task), with a `__main__` entry point that compares the
  learned policy to the analytical Nash equilibrium.
- `doggame/discretize.py` -- turns the continuous domain into an N x N
  action grid and builds the resulting stage game's payoff matrices,
  reusing the same transition/reward functions as `env.py`.
- `doggame/fictitious_play.py` -- approximates a general-sum bimatrix
  game's Nash equilibrium via iterated best response against the
  opponent's running-average strategy -- the "best response dynamics"
  approach discussed as the practical alternative to an LP solver.
- `doggame/exact_nash.py` -- exact (not approximate) bimatrix Nash
  solvers: `lemke_howson_nash` (NashPy, general-sum) and
  `zero_sum_lp_nash` (an LP via scipy, for the zero-sum case a future
  soccer-game stage game would actually need -- LP duality doesn't
  apply to general-sum games, so this isn't a substitute for the
  Lemke-Howson path on the dog game itself).
- `doggame/dqn.py` -- the DQN subgroup's task: solves the discretized
  stage game with fictitious play by default, or exactly via
  `solver="exact"`, then iterates the Nash-Q Bellman backup
  `Q = R + discount * NashV(Q)` to a fixed point. `resync_every`
  freezes the stage-game strategy for several outer iterations between
  re-solves instead of re-solving every time. See the module docstring
  for why the backup doesn't need to loop over states in this
  particular game.
- `doggame/verify.py` -- the check described in the meeting for
  whether a solved or learned policy is actually a Nash equilibrium:
  `exploitability` (grid-search for a unilateral improvement, for
  continuous actions) and `mixed_strategy_indifference_gap` (the
  "actions played with positive probability should all have equal
  value" check, for a discrete mixed strategy).
- `doggame/polar.py`, `doggame/polar_agent.py` -- a variant where the
  players walk: each picks an angle and a step length (up to `max_step`)
  and moves from where it stands. The policy uses a von Mises for the
  angle and a Beta for the step, trained with REINFORCE.
- `doggame/polar_dqn.py` -- a deep Nash-Q for that game: 10 compass
  directions, one network for both players' Q-values, a target network,
  a replay buffer, and an exact Nash solve at every target.
- `doggame/continuous_br.py` -- best response over a continuous action
  by bisection, finite-difference gradient, or a quadratic fit.
- `doggame/soccer.py`, `doggame/soccer_solver.py` -- a deterministic grid
  soccer game (zero-sum, with ball possession) and its exact solution by
  minimax value iteration.
- `doggame/soccer_pg.py` -- REINFORCE, A2C and PPO for soccer, trained by
  self-play or by fictitious play (each player learns a best response to
  the opponent's earlier policies). It never solves a game.
- `doggame/soccer_eval.py`, `doggame/soccer_experiments.py` -- count wins,
  losses and ties against random, exact-equilibrium and exact
  best-response opponents; run the comparison with
  `python3 -m doggame.soccer_experiments`.
- `doggame/soccer_averaging.py`, `doggame/soccer_report.py` -- score the saved
  fictitious-play snapshots under different ways of averaging them, and build
  a PDF of those results from the files in `results/` (the PDF is not kept in
  the repo).
- `doggame/dog_game_study.py`, `doggame/dog_game_trajectories.py`,
  `doggame/dog_game_report.py` -- the multi-seed dog-game runs (convergence, and
  the angle-radius walk), the recorded paths of the players and the dog, and the
  PDF built from them.
- `doggame/train_ac.py` -- A2C and PPO self-play for the dog game, as alternatives
  to REINFORCE in `train.py`.
- `doggame/dog_game_studies.py`, `doggame/dog_game_summary.py` -- seven experiments
  that answer open research questions (algorithms, equilibrium selection, inertia,
  pre-training, stage-solver precision, grid resolution, fictitious-play noise), and
  the tables built from their saved results.
- `doggame/research_report_pdf.py` -- builds `research_report.pdf` from
  `RESEARCH_REPORT.md`.
- `doggame/experiments.py` -- runs policy-gradient self-play across
  several house/weight configurations and reports exploitability for
  each, to actually answer "can this converge?" instead of trusting
  one demo run. See Findings below.
- `visualize/index.html` -- a self-contained, no-build browser page.
  Drag the houses, tune `w`, raise `w0` to give the dog inertia, set a
  max step so players walk instead of jump, and step through
  best-response play.
- `docs/index.html` -- the GitHub Pages version of the same live
  simulator, plus the real training results from `doggame.experiments`
  and the write-up in Findings below. Deployed at
  https://charith-reddy-pareddy.github.io/dog-game/.

## Usage

```bash
pip install -r requirements.txt
python3 -m pytest            # run the test suite
python3 -m doggame.train     # policy-gradient self-play vs. the analytical Nash
```

To compare all three network architectures, or to run the DQN solver:

```python
from doggame.env import DogGameEnv
from doggame.agents import TwoPlayerPolicy
from doggame.train import train_self_play

env = DogGameEnv(house_red=(0.9, 0.2), house_blue=(0.1, 0.8), w=0.5)
policy = TwoPlayerPolicy(env.domain, architecture="shared")  # or "separate" / "partial"
train_self_play(env, policy, episodes=1500)
```

```python
from doggame.discretize import action_grid, payoff_matrices
from doggame.dqn import solve_discretized_nash_q
from doggame.env import Square, weighted_average_transition, negative_squared_distance_reward

domain = Square(0.0, 1.0)
grid = action_grid(domain, resolution=9)
payoff_red, payoff_blue = payoff_matrices(
    grid, (0.9, 0.2), (0.1, 0.8), w=0.5, domain=domain,
    transition_fn=weighted_average_transition, reward_fn=negative_squared_distance_reward,
)
q_red, q_blue, strategy_red, strategy_blue = solve_discretized_nash_q(payoff_red, payoff_blue, discount=0.9)

# exact instead of approximate, resolving the stage game every 5th
# outer iteration instead of every iteration:
q_red, q_blue, strategy_red, strategy_blue = solve_discretized_nash_q(
    payoff_red, payoff_blue, discount=0.9, solver="exact", resync_every=5,
)
```

```python
from doggame.env import DogGameEnv, make_inertial_transition

# the dog only closes 30% of the gap to each round's target, instead
# of fully re-targeting every round
env = DogGameEnv(
    house_red=(0.9, 0.2), house_blue=(0.1, 0.8), w=0.5,
    transition_fn=make_inertial_transition(w0=0.7),
)
```

Run the convergence study (`python3 -m doggame.experiments`) to see
exploitability across configurations and architectures.

Open `visualize/index.html` directly in a browser (no server needed).

## The game

Two houses sit at fixed points `house_red` and `house_blue` in a
bounded, convex domain (a unit square by default). Each round, both
players pick a point `a_red`, `a_blue` in that domain, and the dog
moves to:

```
dog = w * a_red + (1 - w) * a_blue
```

Each player is rewarded by `-||dog - house||^2`. Because the domain is
bounded, a player who wants the dog to land exactly on their house has
to overshoot their own pick to counteract the other player's pull --
the same logic behind two influencers each staking out an exaggerated
opinion to drag a shared audience toward their true position.

## Findings

The full results, with the number of runs behind each claim, are in the
[research report](RESEARCH_REPORT.md) ([PDF](research_report.pdf)), and the
questions they answer are in [RESEARCH_QUESTIONS.md](RESEARCH_QUESTIONS.md).
In short, from saved runs in `results/`:

- **Policy-gradient self-play converges on the dog game.** 74 of 75 REINFORCE
  runs (5 house setups x 3 network layouts x 5 seeds) and 25 of 25 PPO runs reach
  exploitability below 0.004 within 1500 episodes. PPO needs far fewer episodes
  (19 of 25 runs after 100, against none for REINFORCE and A2C), though it takes
  four gradient steps per episode.
- **The one miss is not a bad equilibrium.** It is the `separate` layout on one
  setup, one seed, at 0.18; the other four seeds of that setup converge.
- **Shared coordinates give many equilibria.** When both houses share a
  y-coordinate, the best-response solver returns the interior point, but none of
  30 trained policies did: all ended at the edges, one player on top and one at
  the bottom (23 with red on top, 7 the other way). They are all genuinely
  unexploitable, which is why `doggame.verify` checks optimality directly.
- **Dog inertia slows training**, sharply on the harder setup (4, 5, 1 and 0 of 5
  runs converge at `w0` of 0, 0.3, 0.6 and 0.9).
- **Pre-training helps or hurts** depending on whether the source setup has the
  same equilibrium corners as the target.
- **Nash-Q:** the stage-game error falls about as 1 over the fictitious-play
  iterations, the grid is exact from 5 x 5, and the exact equilibrium is always one
  action pair, so fictitious play's apparent mixing is noise.

Reproduce with `python3 -m doggame.experiments`,
`python3 -m doggame.dog_game_study` and `python3 -m doggame.dog_game_studies`; print
every table with `python3 -m doggame.dog_game_summary`.

## Status

Work in progress, built incrementally. See commit history for the
build-out order.
