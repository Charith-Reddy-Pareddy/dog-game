# Polar-action policy-gradient comparison

## Question and scope

Does representing an absolute dog-game action as an angle and a normalized
radius improve policy-gradient self-play relative to bounded Cartesian
actions? The original game is used: each player chooses an absolute point in
the unit square, both actions are simultaneous, and the dog moves to their
weighted average. This is not the separate walking game. The DQN and Q-value
surface sketches in the handwritten notes are out of scope for this study.

## Protocol fixed before running the comparison

- Primary setup: `DEFAULT_CONFIGS[0]`, houses red `(0.9, 0.2)`, blue
  `(0.1, 0.8)`, `w=0.5`; discount 0.9.
- Algorithms: REINFORCE, A2C and PPO. Architecture: separate player networks.
- Action representations: the existing clipped Cartesian Gaussian; an
  independent bounded Beta per Cartesian coordinate; and a von Mises angle
  with a Beta normalized radius mapped to the square.
- Ten seeds (0-9) for every algorithm and representation; 1500 episodes, 20
  steps, learning rate 0.01. PPO uses four epochs and clip 0.2. No setting is
  tuned on these outcomes.
- Checkpoints: 100, 250, 500, 1000 and 1500 episodes.
- Primary outcome: maximum exact unilateral expected-payoff gain at episode
  1500. Action means and covariance traces are estimated from 8192 sampled
  action pairs at the initial state, using a fixed evaluation seed. Since
  rewards are quadratic and actions enter linearly, the best response to the
  opponent distribution is the projection of the desired action onto the
  square; no action grid is used for this score.
- Secondary outcomes: checkpoint scores, mean-action distance to the
  analytical equilibrium, action covariance and mass at the square boundary.
  The legacy 0.004 threshold is shown descriptively; it was designed for a
  deterministic action-pair score and is not treated as a confirmatory test
  threshold for this distribution-aware score.
- Paired-seed differences are shown with a two-sided 95% Student-t interval.
  With ten seeds and several secondary comparisons, these intervals are
  exploratory; no significance or generalization claim is made from them.

All run rows are appended to `results/polar_action_pg.jsonl`; each completed
row is saved immediately. A rerun skips matching representation/algorithm/
seed/config keys. The analysis script does not delete or synthesize results.

## Interpretation limits

This comparison changes both coordinates and distribution family: the polar
actor uses angle/radius distributions, while the Cartesian bounded control
uses coordinate-wise Betas. The clipped-Gaussian baseline also retains the
existing action clipping and density-evaluation behavior. The study therefore
compares implementable policy parameterizations as packages; it cannot isolate
the causal effect of polar coordinates alone. One house setup and ten seeds
are a controlled pilot, not conference-grade evidence or a novelty claim.
