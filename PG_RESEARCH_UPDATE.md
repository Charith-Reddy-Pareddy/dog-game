# Dog Game: concise research update

**Prepared 15 October 2026 | Internal research note for discussion with a professor or primary researcher**

## What changed since the last public version

This update repairs the published report rendering, corrects one contradictory statement about the PPO follow-up, makes the live simulator's one-shot metric limitation explicit, and integrates the local studies into the reports and question dashboard. It also fixes a faulty negative control in the inertia study. The full assumptions register is available as Markdown and PDF alongside this note.

## New evidence worth discussing

### 1. A restricted dynamic-equilibrium check for dog inertia

For five house configurations, I checked each player against unilateral action sequences while the other player holds its one-shot Nash action. The dog starts at the weighted house midpoint. A coordinate-wise argument gives an exact sufficient certificate: if the house is beyond the player's attainable dog interval and the initial dog is on the same side, the nearest endpoint is the best next state and inertia preserves that side; if the house is attainable and the dog starts there, following earns the maximum possible reward, zero. In all five configurations, every player/coordinate satisfies one of these conditions. Induction proves the fixed-opponent one-shot action is optimal against every unilateral sequence for any horizon and any `w0` in `[0, 1]`.

The five-setup certificate is coded and tested, and the 15 saved rows at `w0 = 0.3, 0.6, 0.9` report zero unilateral gains. The negative control matters: the earlier midpoint opponent was accidentally still a best-response case. With blue fixed at `(0, 0)`, the certificate fails and the convex optimizer finds red a gain of about `0.214`. This does not solve feedback strategies or establish a universal theorem for arbitrary house/action sets.

### 2. More training on the hard walking setup

The 900-episode extension is complete for five seeds per algorithm. Score is the mean distance to the one-shot Nash corners over 10 sampled games at each checkpoint; `under 0.1` uses the project's chosen threshold. Late checkpoints are episodes 450-900 inclusive.

| Method | Mean final distance (SD) | Seeds under 0.1 at episode 900 | Late checkpoints under 0.1 |
|---|---:|---:|---:|
| REINFORCE | 0.563 (0.628) | 2/5 | 29/95 |
| A2C | 0.103 (0.205) | 4/5 | 68/95 |
| PPO, four epochs | 1.249 (0.775) | 1/5 | 21/95 |

Four A2C seeds are near the corners, but one remains at 0.470. PPO has one seed at distance 0 and four above 1.17. This is high seed variability from only five runs, not a reliable ranking or a general learnability claim.

### 3. PPO network width and inertia: exploratory only

On the default walking setup (five seeds, 300 episodes), PPO with 64 hidden units and one epoch ends under the 0.1 threshold in 4/5 seeds (mean distance 0.218; 26/35 later checkpoints). The same width with four epochs reaches it in 1/5 (0.704; 7/35). The 16-unit/four-epoch default reaches it in 2/5 (0.371; 14/35). Width and update reuse appear to interact in this small sample; the experiment does not identify a mechanism. No entropy bonus was tested.

PPO with dog inertia ends under threshold in 4/5 seeds at `w0 = 0.3` (mean 0.200; 24/35 later checkpoints) and 2/5 at `w0 = 0.6` (mean 0.569; 9/35). These descriptive counts do not show that inertia helps; they use five seeds and one house setup.

## Relation to prior work and novelty boundary

The static dog game overlaps the bounded-action, affine-receiver, squared-loss structure of Wu et al.'s Battling Influencers Game (BIG), which already establishes potential-game and pure-equilibrium results. BIG's background influence is fixed; this project's inertia term uses the dog's previous state. Nash-Q and policy-gradient methods also have a substantial literature, including Hu and Wellman's stochastic-game Nash-Q and convergence results for natural policy gradient in specific stochastic linear-quadratic games. Those results do not directly transfer to this deterministic, clipped policy class.

The defensible contribution today is an inspectable educational/research codebase and a set of controlled empirical case studies with explicit caveats. I do **not** claim a new general algorithm, a novel static equilibrium theorem, or publication readiness. This review was not a systematic literature search, and the present evidence is not sufficient for a top-conference submission.

## Assumptions and next research tests

The handwritten group notes are treated as design context, not as instructions that define missing rules. The current walking interpretation (angle and radius as movement from the player's current location) remains an assumption; the chase sketch has no implemented game. The live simulator's displayed 31-by-31 score is an unconstrained one-shot static proxy; it ignores current dog state, inertia and movement constraints and must not be read as dynamic exploitability.

The strongest next tests are: (i) extend or independently verify the fixed-opponent certificate for broader action sets and initial states; (ii) solve or approximate the full feedback equilibrium under inertia; (iii) increase seeds and report uncertainty for the hard-setup and PPO interaction studies; (iv) test entropy only under a declared, non-selected analysis plan; and (v) compare the walking game against a dynamic-game equilibrium rather than the one-shot corner reference. These are proposals, not completed experiments.

## Primary references

- Wu, Y., Zhu, Y., Cai, J.-Y., and Zhu, X. (2025). [The Battling Influencers Game: Nash Equilibria Structure of a Potential Game and Implications to Value Alignment](https://arxiv.org/abs/2502.01127).
- Hu, J. and Wellman, M. P. (2003). [Nash Q-Learning for General-Sum Stochastic Games](https://www.jmlr.org/papers/v4/hu03a.html). *Journal of Machine Learning Research*, 4, 1039-1069.
- Hambly, B., Xu, R., and Yang, H. (2023). [Policy Gradient Methods Find the Nash Equilibrium in N-player General-sum Linear-quadratic Games](https://www.jmlr.org/papers/v24/21-0842.html). *Journal of Machine Learning Research*, 24(139), 1-56.
- Schulman, J. et al. (2017). [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).
