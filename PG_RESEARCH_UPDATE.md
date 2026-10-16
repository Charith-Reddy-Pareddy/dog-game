# Dog Game: concise policy-gradient update

**Prepared 15 October 2026 | Research update for discussion with a professor or primary researcher**

## What was implemented

The original dog game now supports an absolute polar action policy: each player samples an angle and a normalized ray fraction from the center of the square, which is mapped exactly to a feasible point. The earlier angle-radius policy remains the separate walking game. The DQN/Q-value sketches in the handwritten notes were excluded as requested.

## Results worth discussing

The primary endpoint is the maximum one-shot unilateral expected-payoff gain at episode 1,500, calculated exactly from Monte Carlo-estimated action means and variances. It is evaluated at the reset state, not over the full state distribution. Paired-seed polar contrasts against both Cartesian controls are shown below for every algorithm; negative differences favor polar actions. These ten-seed intervals are exploratory, not confirmatory evidence or a novelty claim.

On the primary setup, the polar policy had higher expected gain than Cartesian Beta in all 30 algorithm-by-seed pairs. The four-house follow-up is mixed: two intervals favor Cartesian Beta, while two include zero; none show a clear polar advantage. This does not support a general improvement from the polar policy package under the tested settings.

{{figure:polar_learning}}

| Algorithm | Clipped Gaussian | Cartesian Beta | Polar Beta x von Mises |
|---|---:|---:|---:|
| REINFORCE | 0.0193 (0.0022); 0/10 < 0.004 | 0.0281 (0.0007); 0/10 < 0.004 | 0.0689 (0.0060); 0/10 < 0.004 |
| A2C | 0.0205 (0.0020); 0/10 < 0.004 | 0.0269 (0.0015); 0/10 < 0.004 | 0.0620 (0.0051); 0/10 < 0.004 |
| PPO | 0.0093 (0.0044); 1/10 < 0.004 | 0.0184 (0.0023); 0/10 < 0.004 | 0.1374 (0.1350); 0/10 < 0.004 |

| Algorithm | Polar − Cartesian Beta | Polar − clipped Gaussian |
|---|---:|---:|
| REINFORCE | 0.0407 [0.0367, 0.0447] | 0.0496 [0.0453, 0.0539] |
| A2C | 0.0350 [0.0317, 0.0383] | 0.0415 [0.0378, 0.0451] |
| PPO | 0.1190 [0.0232, 0.2148] | 0.1282 [0.0318, 0.2245] |

{{figure:polar_actions}}

| Algorithm | Representation | Mean action distance to Nash | Mean per-player variance trace |
|---|---|---:|---:|
| REINFORCE | Cartesian Gaussian (clipped) | 0.0695 | 0.00262 |
| REINFORCE | Cartesian Beta | 0.1118 | 0.00289 |
| REINFORCE | Polar Beta x von Mises | 0.3012 | 0.01514 |
| A2C | Cartesian Gaussian (clipped) | 0.0763 | 0.00313 |
| A2C | Cartesian Beta | 0.1046 | 0.00255 |
| A2C | Polar Beta x von Mises | 0.2670 | 0.01329 |
| PPO | Cartesian Gaussian (clipped) | 0.0297 | 0.00062 |
| PPO | Cartesian Beta | 0.0672 | 0.00112 |
| PPO | Polar Beta x von Mises | 0.3899 | 0.00721 |

The analytical equilibrium in this setup is a deterministic corner pair. Every tested actor is stochastic at finite parameters, so action variance can leave positive expected gain even when the mean actions are close to Nash; mean distance alone is not an equilibrium test. The variance column makes that distinction visible.

These values are distribution-aware one-shot unilateral gains: action means and covariance traces are estimated from 8,192 samples, and the exact best response to the opponent's estimated moments is projected onto the square. The threshold 0.004 was originally chosen for deterministic action pairs, so its counts here are descriptive only.

## Policy sketch and assumptions

{{figure:polar_pipeline}}

The state is the current dog location; each house is fixed per training run. The user confirmed that polar actions are absolute points from the square center. Radius is a fraction of the direction-dependent center-to-edge distance. The policy uses a von Mises angle and Beta radial fraction, both with shape/concentration at least 1. See the separate [assumptions register](ASSUMPTIONS.md) and the frozen [study protocol](docs/POLAR_ACTION_PROTOCOL.md).

## Limits and relation to prior work

The primary comparison is one house setup with ten seeds. It compares policy packages, not pure coordinate geometry: the clipped-Gaussian control retains its clipping/log-density behavior, while both Beta policies have exact bounded support. The intervals are exploratory and do not establish generalization or a new algorithm. Chou et al. analyze bounded Beta policies ([ICML/PMLR 2017](https://proceedings.mlr.press/v70/chou17a.html)); gradient dynamics in continuous games need not converge universally ([UAI/PMLR 2020](https://proceedings.mlr.press/v115/chasnov20a.html)). The dog game also overlaps the static structure of the [Battling Influencers Game](https://arxiv.org/abs/2502.01127). Reddit discussions were treated as informal implementation anecdotes, not evidence for research claims.

No DQN, action-value network, chase game, or full dynamic equilibrium result is included in this update.

## Cross-house robustness follow-up

This exploratory follow-up was specified after the primary study began and is kept separate from its pre-run protocol. It compares REINFORCE with polar and Cartesian Beta policies on the four other predefined house setups, ten paired seeds each. Negative differences favor polar. Intervals are unadjusted and reflect seed variation, not Monte Carlo scoring error. The follow-up protocol is in [docs/POLAR_ACTION_ROBUSTNESS_PROTOCOL.md](docs/POLAR_ACTION_ROBUSTNESS_PROTOCOL.md).

{{figure:polar_robustness}}

| House configuration | Cartesian Beta mean (SD) | Polar mean (SD) | Polar − Cartesian Beta [95% t interval] | Polar lower in paired seeds |
|---|---:|---:|---:|---:|
| Red (0.8, 0.8) / blue (0.2, 0.3), w=0.3 | 0.0340 (0.0053) | 0.1218 (0.0859) | 0.0879 [0.0290, 0.1467] | 0/10 |
| Red (0.7, 0.1) / blue (0.6, 0.9), w=0.7 | 0.0238 (0.0026) | 0.0216 (0.0052) | -0.0022 [-0.0065, 0.0020] | 7/10 |
| Red (0.95, 0.5) / blue (0.05, 0.5), w=0.5 | 0.0219 (0.0079) | 0.0184 (0.0024) | -0.0035 [-0.0097, 0.0027] | 6/10 |
| Red (0.6, 0.6) / blue (0.4, 0.4), w=0.5 | 0.0065 (0.0004) | 0.0151 (0.0015) | 0.0086 [0.0074, 0.0097] | 0/10 |
