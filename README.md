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
the game: an environment, an analytical stage-game Nash equilibrium
solver, a simple policy-gradient learner, and a browser visualization.

## Layout

- `doggame/env.py` -- `DogGameEnv`: the repeated game. Transition and
  reward functions are injected rather than hardcoded, so the same
  loop can drive a different game later without touching this class.
- `doggame/nash.py` -- closed-form-style stage-game Nash solver via
  iterated best response. Because the unconstrained best response has
  no fixed point unless both houses share a coordinate, the bounded
  domain is what forces convergence, typically at the boundary.
- `doggame/agents.py` -- `LinearGaussianPolicy` and a REINFORCE update
  rule, kept dependency-light (numpy only, no deep learning framework).
- `doggame/train.py` -- self-play loop for two agents, with a `__main__`
  entry point that compares the learned policy to the analytical Nash
  equilibrium.
- `visualize/index.html` -- a self-contained, no-build browser page.
  Drag the houses, tune `w`, and step through best-response play.

## Usage

```bash
pip install -r requirements.txt
python3 -m pytest            # run the test suite
python3 -m doggame.train     # train two agents against each other
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
