# Assumptions

Where the instructions left something open, I picked the simplest
reasonable option. This file lists those choices so they can be checked
against what was actually intended.

## The game

- The domain is the unit square, and the houses are fixed points.
- The default transition is `dog = w * red + (1 - w) * blue`. It ignores
  the dog's previous position.
- The reward is the negative squared distance from the dog to a player's
  own house.
- The discount is 0.9 (0.95 in `DogGameEnv` by default). Neither value
  was specified.
- The dog starts at `w * house_red + (1 - w) * house_blue`.
- `make_inertial_transition` reads `w0 * dog` as the dog's own previous
  position and forces `w0 + w1 + w2 = 1`. The Battling Influencers paper
  (arXiv 2502.01127) instead defines `x0` as a fixed background point,
  with weights that need not be normalized. My reading may be wrong.
- The live site updates players one after the other (red responds, then
  blue responds to the new red), not simultaneously. Its default house
  positions and its 31x31 exploitability grid are my choices.

## Policy gradient

- The policy is Gaussian. The mean goes through a sigmoid, and the std is
  a learned value that doesn't depend on the state.
- Actions are clipped to the square before the log-probability is taken,
  which makes the gradient slightly biased.
- The baseline is the mean return within the episode, not a learned value
  function. Only REINFORCE is implemented, not A2C or PPO.
- Episodes are 20 steps long, trained with Adam at lr 0.01. Both players
  update together on one optimizer.
- The state is only the dog's position, and a new policy is trained for
  each house configuration.

## Angle-radius version (`polar.py`, `polar_agent.py`)

- I read (theta, r) as a move from the player's current position, with
  r capped at `max_step`. Players start on their houses. This is a guess
  at what the notes mean.
- The dog is `w0 * dog + w1 * red + w2 * blue`, where red and blue are
  the players' positions. Positions are clipped at the edge of the
  square.
- The angle is sampled from a von Mises and the step fraction from a
  Beta (both parameters at least 1, so each is single-peaked). I chose
  these so the angle wraps cleanly and nothing needs clipping.
- The network sees the dog and both players' positions (6 numbers), with
  one small network per player. Training is REINFORCE with the same
  mean-return baseline as above, 15 steps per episode.
- The reference answer is the one-shot Nash corners from `nash.py`. Walking
  there is only checked on the default house setup; other setups were
  only checked for NaNs.
- The 10-direction deep Nash-Q (`polar_dqn.py`) lets each player walk one
  of 10 evenly spaced compass directions, always a full `max_step`.
  One network outputs both players' 10x10 Q-values. The target is
  `r + discount * Nash value` of the target network's Q at the next
  state, with a replay buffer and a frozen target network. Each player
  explores with a random move at a decaying rate, and otherwise samples
  the Nash strategy of its Q.
- Each stage game is solved exactly, with up to 0.03 s per attempt. A game
  that cycles is retried from another starting label (see below).
- `continuous_br.py` has the three continuous best-response methods from
  the notes (bisection, finite-difference gradient, quadratic fit). They
  assume a single peak and are standalone: the deep Nash-Q uses discrete
  directions and does not call them.
- The simulator's walking mode is not learned. It moves each player toward
  their best-response pick by at most `max_step`.

## Soccer (`soccer*.py`)

- The group's own soccer game wasn't available, so this is my version: a
  deterministic, two-player grid soccer game with 4 moves, ball possession,
  and a three-cell goal on a 7x5 board (a smaller 5x3 board with a one-cell
  goal is also used). Both players move at the same time. The collision
  and tackle rules are in the `soccer.py` docstring, and I wrote them so
  the game is identical for both players. They are not the "A10" rules.
- A score ends the game (+1 / -1). The exact solution is discounted with
  0.9. Games are cut off at 100 steps and count as ties. The network input
  has no step counter, because 0.9 to the 100th power is negligible.
- This version has no states that need a mixed strategy, which keeps the
  comparison simple (fewer mixed-equilibrium states is better). It also means
  it can't show whether PG finds mixed strategies.
- Policy gradient never solves a game. The exact solution is used only
  to judge the learned policies.
- Training settings (batch of 256 games, lr 0.003, entropy bonus 0.001,
  one seed) are my choices. The entropy and lr were picked by trying four
  combinations on one seed, so they are tuned to that run. Results swung a
  lot between settings.
- "Exploitability" is how much an exact best response wins against the
  learned policy. It is harsh: a policy that never loses to the exact
  equilibrium can still score high, because a best response searches out
  every state it plays badly.

## Solvers and checks

- The analytic solver finds one pure Nash equilibrium by iterated best
  response. When equilibria are not unique it returns an arbitrary one.
- Fictitious play runs 1000 iterations from a random start (seed 0).
  Lemke-Howson returns the first valid equilibrium it finds. NashPy
  overflows or cycles on a few games, so each answer is checked and a
  failed attempt is retried from the next starting label.
- The original dog-game "DQN" is tabular Nash-Q on a single state, with
  no neural network. The angle-radius game has a real one (below).
- Exploitability is searched on a 101x101 grid, and "converged" means
  below 0.004 (my threshold).

## Evidence

- Every reported run uses one seed (0). The claim about architecture and
  convergence speed rests on a single data point.
- The "3000 more episodes brought it to 0.004" figure came from an
  earlier note. I have not re-run it.
- The explanations in the research report for why PG picks one
  equilibrium over another are hypotheses, not tested results.

## Settled points

- **Which game for PG:** any one where the solver is correct and the comparison
  is meaningful; fewer mixed-equilibrium states is better.
- **Discounting:** PG can only approximate finite-horizon discounted
  rewards, so use both (discounted, with a 100-step limit).
- **Fictitious play:** it is how PG solves things. Each player learns a
  best response to the opponent's earlier policies. There is no explicit
  game solving in PG, and it is not applied to the exact stage-game matrices.
- **Win rate:** play repeated games and count the wins.
- **Continuous-action methods** (bisection, finite-difference gradient,
  quadratic fit): those are for DQN. PG handles continuous actions directly,
  as the angle-radius policy does.

## Still open

1. Is the angle-radius action (theta, r) a per-step move of the player's
   own position (with r up to some delta), or an absolute polar point?
2. In `w0*dog + w1*x_red + w2*x_blue`, are `x_red` and `x_blue` the
   players' positions? Is `w0*dog` the dog's current position or the
   paper's fixed `x0`? Must the weights sum to 1?
3. Should the soccer comparison use the group's own game (7x5, random move
   order, or the "A10" rules) instead of mine?
4. The angle-radius game has a deep Nash-Q. Should the original dog game
   have one too?
5. How should the angle's wraparound and the board edge be handled?
