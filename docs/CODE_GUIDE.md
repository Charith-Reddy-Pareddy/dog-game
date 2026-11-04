# What is implemented: a guide to the code

This guide is for anyone who needs to explain or check the results without having written the code. It says
which parts of the repository the current results depend on and which parts belong to other, finished
threads. For every result it names the function that computes it, the study that ran it, and the test that
checks it. Every number in the worked example below can be reproduced with the commands in Section 6.

## 1. What you need, and what you can ignore

After the archive (Section 8) the active code is 32 modules (about 5,600 lines) with 34 test
files. The receiver (mechanism) results need only a small part of that:

| File | What it is | Size |
|---|---|---|
| `doggame/mechanisms.py` | All the math for receivers: next outcome, equilibrium, losses, deviations | 210 lines |
| `doggame/many_houses.py` | Two helpers: `one_shot` (BIG's static equilibrium) and `next_dog` (the averaging receiver) | 2 functions |
| `doggame/dog_game_studies.py` | Only the `study_mechanisms*` functions run the experiments | about 250 lines |
| `doggame/dog_game_summary.py` | Only the `mechanisms*_table` functions turn saved runs into the numbers in the docs | about 250 lines |
| `results/dog_game_studies.jsonl` | The saved runs, one JSON row per run | data |
| `tests/test_mechanisms.py` | 33 tests of the claims below | |
| `doggame/summary_figures.py` | The nine figures of the key-results summary PDF (`fig8`, `fig9` and `fig10` are its Figs. 2, 7 and 9), `fig11` and `fig12` (the dynamic game for averaging and median: the dog's vector field with paths and reports, and each influencer's response field), and `paper()`, which writes four of them as vector PDFs | one function per figure |
| `doggame/summary_pdf.py` | The key-results summary PDF itself: text, figures, captions | |
| `doggame/results_report.py` | The eight-page results report: results at a glance, recent updates, and the seven main graphs and trajectories (`fig8`, `fig10`, `fig3`, `fig4`, `fig6`, and `fig11`, `fig12` of the dynamic game) | |
| `docs/INERTIA_PROOFS.md` | The proofs (Propositions 1 to 5 and 7 to 19) | |
| `docs/Dog_Game_Results_and_Updates.pdf`, `docs/Dog_Game_FAQ.pdf` | Snapshots of the latest results and a FAQ, linked from the site; `docs/figures/` holds their plots (`fig11`, `fig12`, and the distance to the rest point) | |

The policy-gradient results use a second, separate set: `env.py` (the game), `agents.py` and `networks.py`
(the policies), `train.py` and `train_ac.py` (REINFORCE, A2C, PPO), `verify.py` (exploitability),
`experiments.py`, `feedback_learning.py`, `exact_play.py`, and `dog_game_trajectories.py` (the walking-game
paths).

The other active modules support these two sets or check them: `inertia_theory` (Propositions 1 and 2, used by
the learning results) with `inertia_mpe` and `inertia_dp` as independent checks of it; `nash`, `exact_nash` and
`discretize` (one-shot equilibria); `polar`, `polar_agent`, `polar_dqn` and `dog_game_study` (the walking game
and its runs); `soccer`, `soccer_solver`, `soccer_exact_play`, `sticky_duopoly` and `duopoly_learning` (the other
two games of the learning fix).

Worth knowing:
- `inertia_theory` (closed form) and `inertia_mpe` (exact one-round gain at every dog position) both ask whether
  holding the one-shot actions is a feedback equilibrium, and `inertia_dp` computes the feedback equilibrium
  itself by dynamic programming; `tests/test_inertia_agreement.py` checks them against each other.
- Every module has a test.

## 2. The model in one paragraph

There is one outcome x in [0, 1] (the dog's position in one coordinate; the two coordinates are solved
separately because the losses add up). N players have targets h_1, ..., h_N. Each round every player
reports a point r_i in [0, 1], and the receiver moves the outcome. Each player loses (x - h_i)^2 per
round, discounted by g = 0.9. Inertia w0 is how much of its old position the receiver keeps. The receivers
are:

| Receiver | Next outcome | Function |
|---|---|---|
| Averaging (BIG) | w0 x + (1 - w0) mean(r) | `many_houses.next_dog` (via `mechanisms.step(..., "mean")`) |
| Median | w0 x + (1 - w0) median(r) | `mechanisms.next_median` |
| Generalized median, phantoms a | w0 x + (1 - w0) median(r, a) | `mechanisms.next_phantom` |
| Compensated, phantoms a | the point of [w0 x, w0 x + 1 - w0] closest to median(r, a) | `mechanisms.compensated` |
| Planner (benchmark) | the point of [w0 x, w0 x + 1 - w0] closest to the mean target | `mechanisms.planner_step` |

[w0 x, w0 x + 1 - w0] is everywhere the outcome can reach in one round. "Phantoms" are N - 1 fixed points
added to the reports before taking the median. Phantoms (1/3, 2/3) for 3 players reproduce averaging
(Proposition 7); phantoms (0, 1) give the median.

## 3. A worked example to check by hand

Three players with targets (0.1, 0.2, 0.9) and mean target 0.4. Inertia 0.6, so from x = 0 the outcome
can reach [0, 0.4] in one round, and from x = 0.9 it can reach [0.54, 0.94].

| Question | Answer | How to see it |
|---|---|---|
| BIG without inertia: what do players do? (`one_shot`) | actions (0, 0, 1), outcome 1/3 | Every player pushes to an edge: exaggeration |
| Plain median, truthful reports, from x = 0 | 0.6·0 + 0.4·0.2 = 0.08 | The receiver moves only 40% of the way |
| Plain median in equilibrium, from x = 0 | the median player reports 0.5, outcome 0.4·0.5 = 0.2 | It exaggerates to make up for the inertia (Prop. 3) |
| Compensated median, truthful, from 0 and from 0.9 | 0.2 and 0.54 | clip(0.2, 0, 0.4) and clip(0.2, 0.54, 0.94) |
| Compensated averaging, truthful, from 0 | 1/3 | median(0.1, 0.2, 0.9, 1/3, 2/3) = 1/3, inside [0, 0.4] |
| Planner from 0 | 0.4 | clip(0.4, 0, 0.4) |

Total discounted loss of all players, averaged over 101 evenly spread starts (`social_loss`):

| Receiver | Loss |
|---|---|
| Planner (best possible) | 3.8138 |
| Compensated averaging, truthful reports | 3.9413 |
| Plain averaging, at its equilibrium | 4.0264 |
| Plain median, at its equilibrium | 4.9182 |
| Compensated median, truthful reports | 4.9182 (same path as the line above) |

Largest gain from a one-round deviation (`deviation_gain`; 0 means truthful reporting is an equilibrium):

| Receiver with truthful reports | Gain |
|---|---|
| Plain median | 0.0544: not an equilibrium (Proposition 3) |
| Compensated median | 0: an equilibrium (Proposition 9) |
| Compensated averaging (phantoms 1/3, 2/3) | 0: an equilibrium (Proposition 9) |
| Compensated mean of the reports | 0.0322: not an equilibrium (the mean is manipulable) |

## 4. The functions, one by one

All in `doggame/mechanisms.py` unless named otherwise.

- **`one_shot(houses, weights)`** (`many_houses.py`). BIG's static equilibrium. It walks down from the
  highest target; players above the stopping point push to 1, players below push to 0, and at most one
  player (the pivot) ends inside. Returns (actions, outcome, pivot).
- **`next_median`, `next_phantom`, `step`**. The next outcome when every player aims the next outcome at
  its target. They take *targets*, not reports: given a target t, a player's report is
  (t - w0 x) / (1 - w0), clipped to [0, 1] (`reports`).
- **`target_equilibrium(rule, houses, weights, w0)`**. Finds equilibrium targets by repetition. Start with
  targets = houses. For each player, compute on a grid of 801 next outcomes "this round's loss plus the
  discounted loss of the path that follows", and take the best grid point as its new target. Average old and
  new targets (damping 0.5), and stop when nothing changes. The grid limits accuracy to about 0.015 in loss
  (the tolerance used in the tables). In the example the median rule's targets are (0.1, 0.2, 1.0).
- **`compensated(x, reports, phantoms, w0)`**. The compensated receiver: the median of the reports and the
  phantoms, clipped to the reachable interval. Three lines.
- **`social_loss(next_fn, houses)`**. Follows the outcome for 200 rounds from 101 evenly spread starts and
  adds up every player's discounted squared distance. With `shock` it solves a Bellman equation on a grid
  instead (the noise results only).
- **`compensated_losses(houses, shapes, w0)`**. The same loss for many phantom shapes at once, for the
  compensated receiver with truthful reports. It is only a faster version, and a test checks that it
  matches `social_loss`.
- **`deviation_gain(rule, policy, houses, weights, w0, compensate=False)`**. The direct check of an
  equilibrium. At 21 outcomes, for each player, try every report on a 201-point grid for one round and
  then go back to `policy`, and compare the loss with following `policy` throughout. Returns the largest
  improvement. By the one-shot deviation principle, 0 at every state means the policy is an equilibrium,
  up to the grid. `compensate` can also be a function giving the reachable interval from each outcome, for
  receivers other than proportional inertia (Proposition 11).
- **`planner_step`**. Moves as close to the mean target as it can each round. Without shocks this is
  optimal: it is the compensated receiver aimed at the mean target (Proposition 12).
- **`reach_weight(u, w0)`**. W(u), the discounted number of rounds in which u can be reached, averaged over
  the starts of `social_loss`. A compensated receiver's extra loss over the planner is
  2N times the integral of (u - mean target) W(u) from the mean target to its end point (Proposition 12).
- **`limit_coefficients`, `order_statistic_phantoms`, `best_order_statistic`, `compensated_criterion`**.
  Closed-form criteria from the slow-receiver propositions (8 and 10). They are checked against simulation
  in tests.

## 5. Each result: proof, code, data, test

| Result (as stated in the shared summary) | Proof | Study (data) | Table in `dog_game_summary.py` | Test in `tests/test_mechanisms.py` |
|---|---|---|---|---|
| Median receiver better at high inertia (Fig. 3) | numerical | `mechanisms_wide` | `mechanisms_wide_table` | none for the ranking; `test_the_target_form_is_an_equilibrium_under_both_rules_and_the_planner_loses_least` checks the equilibria being compared |
| Median receiver not strategy-proof with inertia | Prop. 3 | `mechanisms` (`truthful_gain`) | `mechanisms_table` | `test_truthful_median_reporting_survives_without_inertia_but_not_with_it` |
| With a pivot the averaging receiver lands exactly on the one-shot outcome in finitely many rounds (closed form); an equilibrium when every edge player is far enough from the pivot | Prop. 18 (target form without the condition: numerical) | | | `test_a_pivot_puts_the_dog_exactly_on_the_one_shot_outcome` |
| A plain receiver reaches its rest point exactly when a target decides it; a phantom gives only a geometric approach | Prop. 19 | | | `test_the_receiver_arrives_exactly_when_a_target_decides_and_not_when_a_phantom_does` |
| Averaging = generalized median with phantoms k/N | Prop. 7 | | | `test_the_mean_rule_is_the_generalized_median_with_evenly_spread_phantoms_and_the_median_rule_with_edge_phantoms` |
| Best phantoms move toward the edges (Fig. 4 left) | numerical | `mechanisms_phantoms` | `mechanisms_phantoms_table` | `test_without_inertia_the_three_player_loss_of_phantoms_p_and_1_minus_p_matches_the_closed_form` |
| Slow receiver: the order statistic nearest the mean target | Prop. 8 (first order) | `mechanisms_limit`, `mechanisms_skewed_seven` | `mechanisms_limit_table`, `mechanisms_skewed_table` | `test_the_high_inertia_coefficients_match_simulation_and_pick_the_median_receiver`, `test_the_best_slow_receiver_is_the_order_statistic_closest_to_the_mean_target` |
| Some inertia helps with noise | numerical | `mechanisms_noise`, `mechanisms_noise_planner` | `mechanisms_noise_table` | `test_the_greedy_planner_loss_falls_with_inertia_while_the_mean_house_stays_reachable` (the mechanism, not the best inertia) |
| Compensation restores truthful reporting (Fig. 6) | Prop. 9 and corollary | `mechanisms_compensated` | `mechanisms_compensated_table` | `test_truthful_reporting_is_an_equilibrium_once_the_receiver_compensates_for_inertia`, `test_compensation_keeps_every_strategy_proof_rule_truthful_but_not_averaging` |
| With compensation and uniform targets the best receiver does not depend on inertia (Fig. 4 right) | Prop. 10 (first order) plus numerical | `mechanisms_compensated_design` | `mechanisms_compensated_design_table` | `test_a_slow_compensated_receiver_is_ranked_by_the_static_criterion` |
| Compensation keeps every strategy-proof rule truthful for any reachable set that allows staying and has nondecreasing ends; neither condition can be dropped | Prop. 11 | | | `test_compensation_keeps_truthful_reporting_for_any_monotone_reachable_set_that_allows_staying`, `test_truthful_reporting_fails_when_the_receiver_cannot_stay_or_its_reach_falls` |
| Staying put is needed: if the receiver is forced away from a point next to one where it can stay, one influencer can gain | Prop. 15 | | | `test_a_receiver_that_cannot_stay_put_lets_a_single_influencer_gain` |
| The reach must not fall: if the upper reach dips after a peak that can be passed, one influencer can gain | Prop. 16 | | | `test_a_falling_upper_reach_lets_a_single_influencer_gain` |
| A dip in the lower reach can be harmless for squared loss but not for every loss | example | | | `test_a_dip_in_the_lower_reach_is_harmless_for_squared_loss_but_not_for_every_loss` |
| Slowing each phantom separately, instead of the outcome, does not keep reporting truthful | example | | | `test_slowing_each_phantom_separately_does_not_keep_reporting_truthful` |
| A compensated receiver loses the planner's loss plus a weighted static criterion | Prop. 12 | | | `test_a_compensated_receiver_loses_the_planners_loss_plus_a_weighted_static_criterion` |
| With compensation each phantom's criterion keeps its static sign pattern, so for uniform targets the optimal phantoms are k/N at every inertia; with clustered targets inertia can switch between static local optima | Prop. 13 | | | `test_moving_a_phantom_changes_the_loss_at_the_static_rate_times_the_weight`, `test_for_uniform_targets_the_phantom_condition_has_its_closed_form_and_vanishes_at_k_over_n`, `test_with_clustered_targets_inertia_can_move_the_best_compensated_phantom` |
| A compensated receiver's price of anarchy is at most its static value and falls as inertia grows | Prop. 14 | | | `test_inertia_never_raises_a_compensated_receivers_price_of_anarchy` |
| With noise, symmetric adjustment keeps truthful reporting; asymmetric adjustment or clipping breaks it | Prop. 17 | | | `test_noise_breaks_truthful_reporting_only_through_asymmetric_adjustment` |
| With clipped shocks truthful reporting can lose a little | numerical | | | `test_with_clipped_shocks_the_median_influencer_gains_a_little_by_shading` |
| PG learns the one-shot equilibrium | numerical | `algorithms`, `walking` | `algorithms_table`, `walking_table` | `test_training_moves_policy_closer_to_nash` (`tests/test_train.py`) |
| PG learns the static, not the dynamic, answer; the fix | numerical | `feedback_learning`, `learning_fix` | `feedback_learning_table`, `learning_fix_table` | `test_the_explicit_equilibrium_has_no_feedback_gain_and_holding_does`, `test_a_table_of_actions_moves_toward_the_feedback_equilibrium` |

"Numerical" means there is no proof; the claim rests on the saved draws.

## 6. How a number gets from the code to a document

1. A study function in `dog_game_studies.py` lists runs as (key, function) pairs. Running
   `python3 -m doggame.dog_game_studies mechanisms_compensated` executes the runs not yet saved and appends
   one row per run to `results/dog_game_studies.jsonl`, for example
   `{"study": "mechanisms_compensated", "key": {"n": 3, "w0": 0.0, "draw": 5}, ...}`.
2. `python3 -m doggame.dog_game_summary` reads every row and prints all the tables.
3. The numbers in the docs are copied from those tables.

So to trace a number in a document: find its table, then the study, then the function. To check whether a
code change moved it, delete that study's rows from a copy of the results file, rerun the study, and compare
the tables.

The worked example in Section 3 is reproduced by:

```bash
cd dog-game && /opt/anaconda3/bin/python3 -c "
import numpy as np; from doggame import mechanisms as M
h, w, w0 = np.array([.1, .2, .9]), np.ones(3) / 3, 0.6
for r in ('mean', 'median'):
    t, _ = M.target_equilibrium(r, h, w, w0); print(r, M.social_loss(lambda x: M.step(r, x, t, w, w0), h))
print('compensated', M.compensated_losses(h, [(1/3, 2/3), (0., 1.)], w0))
print('planner', M.social_loss(lambda x: M.planner_step(x, h, w0), h))
print('gain plain median', M.deviation_gain('median', lambda x: h, h, w, w0))
print('gain compensated median', M.deviation_gain((0., 1.), lambda x: h, h, w, w0, compensate=True))"
```

The figures of the key-results summary PDF are drawn by `doggame/summary_figures.py`
(`python3 -m doggame.summary_figures`, written to `build/summary/`). Figures 1, 2 and 7 solve small games
directly; the others read the saved runs. `tests/test_summary_figures.py` checks that the numbers quoted
for Figures 2 and 5 come out of the data. `python3 -m doggame.summary_pdf` rebuilds the whole PDF, figures
included, as `build/summary/Dog_Game.pdf`. `python3 -m doggame.results_report` builds the longer results report
as `build/report/Dog_Game_Results.pdf`; `tests/test_results_report.py` checks that it builds on six pages with
its five figures.

## 7. Problems found and fixed so far

Each was diagnosed and committed as `debug: ...`:
- A study runner re-read the whole results file before every run (`1f4a0e0`).
- The Lemke-Howson solver returned invalid answers or hung on some games (`901f8b3`, `d1354db`).
- Nash-Q results depended on machine speed because of a solver time limit (`161f7ca`).
- Interrupted runs lost their results (`7f775d1`, `14b0719`, `dd81451`).
- A module was never committed, so fresh clones broke (`7d144a5`).

Results that were corrected:
- The equilibrium-selection result (`5d11283`).
- A 10-seed learning result that did not replicate at 50 seeds.

## 8. The archive

Threads the current results do not use are in `archive/` (see `archive/README.md`): the report builders,
Nash-Q on a grid, the polar-action and walking-equilibrium studies, the soccer extras and the extra inertia
checks. They are frozen at the git tag `pre-archive`, where they still run; their saved results stay in
`results/` and their reports in the repository. The PDFs of the Markdown notes (for example
`docs/ASSUMPTIONS.pdf`) are built with the archived `research_report_pdf`, run from that tag:

```bash
git worktree add ../dog-game-pre-archive pre-archive
cd ../dog-game-pre-archive && python3 -m doggame.research_report_pdf ../dog-game/ASSUMPTIONS.md ../dog-game/docs/ASSUMPTIONS.pdf
```

## 9. Known limits of the numbers

- Equilibria of the plain receivers come from `target_equilibrium` on an 801-point grid, accurate to about
  0.015 in loss. Differences smaller than that are treated as ties.
- Losses use 200 rounds and 101 starts. With discount 0.9, the rounds left out weigh 0.9^200, which is
  negligible.
- `deviation_gain` checks 21 outcomes and 201 reports per player, so "gain 0" means no gain on that grid.
- Propositions 8 and 10 are first-order (slow-receiver) statements. Their numerical checks hold to about 4
  to 15% at inertia 0.99 to 0.995, and less well at 0.9.
