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

## Status

Work in progress, built incrementally. See commit history for the
build-out order.
