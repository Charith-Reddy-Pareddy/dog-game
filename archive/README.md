# Archive

Code from research threads that the current results do not use. It is kept for reference and is not
imported, run or tested from here: the tests collect only `tests/` (see `pytest.ini`). Every file here is
exactly as it was at the git tag `pre-archive`, where it still runs:

```bash
git checkout pre-archive
```

Saved results stay in `results/` and the reports built from them stay in the repository root and `docs/`.

| Thread | Files | Results and reports |
|---|---|---|
| Report builders | `dog_game_report`, `figures`, `research_report_pdf`, `pdf_kit`, `walking_figures`, `polar_action_report`, `soccer_report` | `RESEARCH_REPORT.md`, `PG_RESEARCH_UPDATE.md` and their PDFs |
| Nash-Q on a grid | `dqn`, `fictitious_play`, `continuous_q`, `continuous_br`, `continuous_q_figure`; studies `stage_solver`, `resolution`, `noise`, `continuous_q`, `continuous_q_tools` | `RESEARCH_REPORT.md` |
| Polar actions and the walking equilibrium | `polar_action_study`, `polar_action_robustness`, `walking_equilibrium`; study `walking_equilibrium` | `PG_RESEARCH_UPDATE.md`, `WALKING_GAME_REPORT.md` |
| Soccer extras | `soccer_averaging`, `soccer_eval`, `soccer_experiments`, `soccer_fp_diagnosis`, `soccer_pg`, `soccer_seeds` | `results/soccer_*` |
| Extra inertia checks | `inertia_equilibrium`, `markov_potential`, `potential`; studies `inertia_equilibrium`, `markov_potential` | `RESEARCH_REPORT.md` |

Functions and tests that were cut out of shared files are in `doggame/dog_game_studies_archived.py`,
`tests/test_dog_game_studies_archived.py`, `tests/test_verify_archived.py` and
`tests/test_inertia_agreement_archived.py`.
