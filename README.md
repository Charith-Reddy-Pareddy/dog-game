# Dog Game

**Live demo:** https://charith-reddy-pareddy.github.io/dog-game/
**Research questions:** [RESEARCH_QUESTIONS.md](RESEARCH_QUESTIONS.md)
**Research report:** [RESEARCH_REPORT.md](RESEARCH_REPORT.md) ([PDF version](research_report.pdf), with direct answers to each research question)

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
- `doggame/experiments.py` -- runs policy-gradient self-play across
  several house/weight configurations and reports exploitability for
  each, to actually answer "can this converge?" instead of trusting
  one demo run. See Findings below.
- `visualize/index.html` -- a self-contained, no-build browser page.
  Drag the houses, tune `w`, raise `w0` to give the dog inertia, and
  step through best-response play.
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

Running `doggame.experiments` (5 house/weight configurations x 3
network architectures, 1500 self-play episodes each) gives an actual
answer to "can policy-gradient self-play converge to Nash on the dog
game?": **usually, but the number of episodes needed isn't fixed --
it depends on the configuration and architecture.** 14 of the 15
(config, architecture) runs reached exploitability below 0.004 within
1500 episodes. The 15th (the `separate` architecture on
`house_red=(0.8, 0.8)`, `house_blue=(0.2, 0.3)`, `w=0.3`) was still at
0.183 at that point -- clearly not converged. Training that exact run
3000 episodes further brought it down to 0.004, so it wasn't stuck in
a bad equilibrium, it was just slower: one action dimension took
longer to get pushed out to the domain boundary than the others. That
answers the meeting's question honestly rather than optimistically --
this does converge, but "1500 episodes" isn't a number you can quote
without also naming the configuration and architecture it was
measured on.

Distance to the *specific* equilibrium `doggame.nash` reports is a
different story: one configuration (`house_red=(0.95, 0.5)`,
`house_blue=(0.05, 0.5)` -- both houses at the same y-coordinate)
consistently converged to a policy roughly 0.49 away from the
solver's answer, in every architecture. That is not a training
failure. It's the exact phenomenon the meeting flagged: a unique Nash
equilibrium is only guaranteed when the two houses differ in *both*
coordinates. Here they share a y-coordinate, so the y-axis of the
stage game is degenerate -- any pair of y-actions that averages back
to that shared value is equally optimal. `doggame.nash`'s iterated
best response happens to land on the "obvious" solution (both players
picking y = 0.5 exactly); policy-gradient training instead found a
boundary-cancellation solution (red near y = 1, blue near y = 0,
`0.5*1 + 0.5*0 = 0.5`). Checking both with `exploitability` confirms
they're both genuinely unexploitable (< 0.0002 for either) -- two
different, equally valid equilibria, not one right answer and one
bug. This is why `doggame.verify` checks optimality directly rather
than distance to a single precomputed candidate.

## Status

Work in progress, built incrementally. See commit history for the
build-out order.

## Note on commits `e06ae1e`..`36d8f2f`

That range of six commits is a deliberate debugging exercise, not real
history: three small, realistic bugs (a swapped reward assignment, a
copy-pasted best-response formula with the wrong denominator, and an
off-by-one in the discounted-return recursion) were introduced one at
a time, reproduced with the test suite, diagnosed, and fixed -- each
bug and its fix as a separate, clearly labeled commit. It's kept in
the history rather than squashed away because the point of the
exercise was the workflow itself.
