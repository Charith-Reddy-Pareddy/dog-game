# Dog Game

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
- `doggame/dqn.py` -- the DQN subgroup's task: solves the discretized
  stage game with fictitious play, then iterates the Nash-Q Bellman
  backup `Q = R + discount * NashV(Q)` to a fixed point. See the
  module docstring for why that backup doesn't need to loop over
  states in this particular game.
- `visualize/index.html` -- a self-contained, no-build browser page.
  Drag the houses, tune `w`, and step through best-response play.

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
```

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
