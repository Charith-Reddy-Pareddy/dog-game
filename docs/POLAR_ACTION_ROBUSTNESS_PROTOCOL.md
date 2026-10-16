# Polar-action robustness follow-up

This is a separate exploratory follow-up to the predeclared one-setup study in
[`POLAR_ACTION_PROTOCOL.md`](POLAR_ACTION_PROTOCOL.md). It was specified after
that primary study had begun; it is not part of that study's pre-run protocol
and is not confirmatory evidence.

## Fixed comparison

- Use all four remaining configurations (`DEFAULT_CONFIGS[1:5]`) from
  `doggame.experiments`; do not select configurations based on the primary
  study's outcomes.
- Compare only the polar Beta/von-Mises actor with the bounded Cartesian Beta
  actor. Both have exact bounded support and use the same score-function
  REINFORCE trainer, so this narrows the follow-up to two policy packages.
- Use seeds 0-9 for every configuration and representation; 1,500 episodes,
  20 steps per episode, learning rate 0.01, separate player networks, and
  discount 0.9. No outcome-based tuning or seed replacement.
- Score at 100, 250, 500, 1,000 and 1,500 episodes using 8,192 sampled action
  pairs at the reset state. Compute the maximum unilateral expected gain from
  sample means and variance traces and the exact box-constrained best response.
- Report per-configuration paired-seed differences at episode 1,500 with
  exploratory two-sided 95% Student-t intervals. These intervals capture
  training-seed variation, not Monte Carlo scoring error. No
  multiple-comparison, generalization, or novelty claim is made.

The purpose is to check whether the primary action-package comparison changes
across the repository's fixed house configurations. Results are stored
separately in `results/polar_action_pg_robustness.jsonl`; completed runs are
appended and never synthesized or overwritten.
