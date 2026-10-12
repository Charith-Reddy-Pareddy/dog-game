# Saved results

Every table, chart and number in the [research report](../RESEARCH_REPORT.md), the
[website](https://charith-reddy-pareddy.github.io/dog-game/) and the PDFs is built from the
files here. Each `.jsonl` file has one JSON object per line, one per finished run, so a stopped
experiment keeps what it finished and a rerun skips it. Print every table with
`python3 -m doggame.dog_game_summary`.

## The dog game

- **`dog_game_runs.jsonl`**, from `python3 -m doggame.dog_game_study`.
  - `kind: "convergence"`: REINFORCE self-play on the main game. Fields: `architecture`
    (`separate`, `shared` or `partial`), `seed`, `config` (the house setup, 0 to 4, from
    `doggame.experiments.DEFAULT_CONFIGS`) and `exploitability`, the larger of the two
    players' gains from deviating after 1500 episodes. Below 0.004 counts as converged.
  - `kind: "walking"`: the walking game, REINFORCE or the 10-direction Nash-Q network
    (`architecture: "reinforce"` or `"nash-q"`). Fields: `seed`, `before` and `after`, the
    mean over 10 sampled games of the players' summed distance from the one-shot Nash corners.
- **`dog_game_studies.jsonl`**, from `python3 -m doggame.dog_game_studies`. Each line has
  `study`, `key` (what was varied) and the measured fields:
  - `algorithms`: REINFORCE, A2C and PPO on the main game. `curve` maps episodes trained to exploitability.
  - `walking`: the same three algorithms on the walking game. `start` is the distance before
    training and `curve` maps episodes to distance, every 25 episodes.
  - `ppo_variants`: PPO with 1 or 2 epochs (main game) or a smaller learning rate / fewer epochs (walking game).
  - `ppo_long`: A2C and PPO on the walking game for 900 episodes.
  - `walking_settings`: A2C on the walking game with another maximum step or discount.
  - `walking_houses`: the three algorithms on the walking game with three other house setups.
  - `walking_inertia`: A2C on the walking game with dog inertia (`w0` of 0.3 and 0.6).
  - `walking_equilibrium`: for each house setup and start, how much either player could gain by deviating from the walk to the one-shot corners (`red` and `blue`, each with `followed`, `best` and `gain`).
  - `selection`: where self-play ends when the houses share a y-coordinate (`red`, `blue`, `exploitability`).
  - `inertia`: training with the dog's own weight `w0` set to 0, 0.3, 0.6 or 0.9.
  - `pretraining`: training on a setup after starting from another one (`curve`).
  - `stage_solver`: the Nash-Q fixed-point error (`q_error`) and strategy `gap` against the exact solver.
  - `resolution`: the exact solution of the discretized game on 3 x 3 to 17 x 17 grids.
  - `noise`: fictitious play's indifference `spread` against the exact equilibrium.
- **`dog_game_trajectories.json`**: the paths of red, blue and the dog in one sampled 15-step walking
  game, before and after training, for five seeds and both methods. It feeds the website's
  trajectory plot and the results PDF.

## Companion code (grid soccer)

`soccer_5_seeds.jsonl`, `soccer_fp_diagnosis_runs.jsonl` and `soccer_averaging_round_*.txt` come from
the separate grid-soccer experiments (`doggame/soccer*.py`). They are not used by the report, the
website or the dog-game PDFs.
