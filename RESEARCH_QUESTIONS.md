# Research Questions

These are the questions that motivated this repository, drawn from the
project's planning notes on the dog game (the "AI dog" / "Battling
Influencers Game"). They are grouped by the two solution methods the project
compares, plus the questions common to both, and then a set of new questions
that the experiments here raised.

Each question has a status. **Answered** means a result file in
[`results/`](results/) settles it for the dog game, with the number of runs
behind it. **Partly answered** means there is evidence but it leaves
something untested. **Open** means nothing here tests it yet. The answers,
with the numbers and the caveats, are in [RESEARCH_REPORT.md](RESEARCH_REPORT.md)
([PDF](research_report.pdf)).

## Terms used

- **The dog game.** Two players each own a house. Each round both pick a point in the unit
  square, and a shared dog moves to the weighted average of the two points. Each player is
  rewarded for the dog being close to their own house.
- **The walking game** (the angle-radius version). Each player has a position and each
  round picks a direction and a step length of up to 0.2, then walks that way. The dog moves
  to the average of the two new positions.
- **Exploitability.** How much either player could gain by switching to a better action. It
  is 0 at an equilibrium. A run is called *converged* when it is below 0.004.
- **Stage game.** The one-round game at each step of a learning method such as Nash-Q, whose
  equilibrium the method needs to solve.
- **Fictitious play and Lemke-Howson.** An approximate and an exact way to find a stage-game
  equilibrium.
- **Inertia (`w0`).** The share of the dog's old position the dog keeps each round.

## Policy gradient (REINFORCE, A2C, PPO)

1. **Can self-play policy gradient converge to a Nash equilibrium in a
   general-sum game?** Policy gradient is a single-agent algorithm; two
   copies played against each other are best-response dynamics, not a game
   solver, and nothing guarantees mutual improvement converges.
   *Status: answered for the dog game (74 of 75 REINFORCE runs, 25 of 25
   PPO runs); open for general-sum games in general.*
2. Does the number of episodes needed depend on the house and weight
   configuration, or is there one number that works everywhere?
   *Status: answered. It depends on the configuration and on the algorithm.*
3. Does pre-training help, and what does the benefit depend on?
   *Status: answered for one target setup. It helps a lot when the source
   setup has the same equilibrium corners and hurts when it does not.*
4. When the game has more than one equilibrium, which one does self-play
   find, and is it consistent across seeds and architectures?
   *Status: answered for the shared-coordinate setup (30 runs). Never the
   solver's interior point; always a boundary equilibrium, one of two
   mirror images, about 3 to 1.*

## Nash-Q on a discretized grid

5. **How precisely must the stage game inside the Nash-Q backup be
   solved?** An exact solver gives an exact equilibrium; fictitious play only
   approximates one. How much stage-game error can the outer Bellman
   iteration tolerate?
   *Status: answered for the dog game. The error of the fixed point falls
   about as 1 over the number of fictitious-play iterations.*
5b. Is there literature on how precisely the stage game must be solved for
    the outer loop to converge?
    *Status: open as a literature question; this repository measures it
    for one game but does not survey the literature.*
6. How does grid resolution trade off against solution quality, and is a
   neural network needed as the grid grows?
   *Status: answered for the dog game. Quality stops improving at 5 x 5;
   cost keeps growing steeply. The network question is partly answered by the
   walking game's deep Nash-Q.*

## Common to both methods

7. **What is the right multi-agent network architecture?** Separate
   networks, a shared trunk, or a partly shared trunk.
   *Status: partly answered. All three converge (24, 25 and 25 of 25 runs);
   cross-play between independently trained architectures is untested.*
8. **When is the equilibrium unique?** The conjecture is that the houses must
   differ in both coordinates.
   *Status: answered. Confirmed, and every shared coordinate opens a
   continuum of equilibria.*
9. **How do you verify, after the fact, that a policy is an equilibrium?**
   *Status: answered. `doggame/verify.py` implements exploitability and the
   indifference check.*
10. Practical solvers rarely land on clean probabilities. How much of the
    non-clean mixing is real and how much is numerical noise?
    *Status: answered for the dog game. The exact equilibrium is a single
    action pair, so all the mixing fictitious play shows is noise.*
11. Which findings are expected to transfer to a harder game (one with many
    states and no unique equilibrium), and which depend on the dog game's
    simple structure?
    *Status: open. The state-independent transition, the unique equilibrium
    and the one-shot payoffs are all dog-game specific.*

## From the follow-up planning notes

12. Should the dog have momentum (`w0 * dog + w1 * red + w2 * blue`), and does
    inertia change anything besides how long it takes?
    *Status: partly answered. Inertia leaves the steady state unchanged but
    slows policy-gradient training sharply on the harder setup; whether it
    changes the equilibrium of the dynamic game is untested.*
13. Can a "vertex Nash" choice (the equilibrium with the largest sum of values)
    give a more useful equilibrium selection for multi-state games?
    *Status: open. The dog game has one state.*
14. (Adjacent thread.) Can an LLM's behavior in a repeated game be modeled as
    bounded-rationality play over a continuous action space, and can its
    choices be inverted into a reward function?
    *Status: out of scope for this repository.*
15. (Adjacent thread.) For a continuous-action grid player, is the natural
    action space a disk (an angle and a radius) with a separate heading-only
    action?
    *Status: partly addressed by the walking version, which uses an angle and
    a step length.*

## New questions from this work

16. **Do A2C and PPO reach the equilibrium faster than REINFORCE, and is the
    speed-up from the baseline or from reusing each episode?**
    *Status: answered. On the main game PPO is much faster per episode (19 of 25
    runs converged after 100 episodes, against 0 of 25) and A2C is not faster than
    REINFORCE. With one epoch (one gradient step per episode, like the others) PPO is
    no faster than REINFORCE, so the speed-up comes from reusing each episode (Q24).
    On the walking game the order changes (Q26).*
17. **What decides which of the two mirror equilibria self-play finds, and
    why does it never find the interior solution?**
    *Status: open on the mechanism. The outcome is measured (23 of 30
    against 7 of 30, no interior); the explanation (a wide initial
    exploration noise plus clipping at the boundary) is a hypothesis.*
18. **How strongly does dog inertia slow policy-gradient training?**
    *Status: answered for two setups. On the easy setup every run converges
    up to `w0 = 0.9`; on the hard one 4 of 5 converge at `w0 = 0`, 5 of 5 at
    0.3, 1 of 5 at 0.6 and 0 of 5 at 0.9 within 1500 episodes.*
19. **Does the one-shot equilibrium remain the equilibrium of the game with
    inertia?** Policies trained with inertia are scored against the one-shot
    game; the equilibrium of the full dynamic game has not been solved.
    *Status: open.*
20. **Why does pre-training from a different setup hurt?**
    *Status: open on the mechanism. The measured pattern (same corners help,
    flipped corners hurt) suggests the network has to un-learn a boundary
    direction, but that is a hypothesis.*
21. **Does re-solving the stage game less often change the Nash-Q fixed
    point?**
    *Status: answered for the dog game. No: with a state-independent
    transition, every Q matrix differs from the payoffs by a constant, so the
    equilibrium never moves between iterations.*
22. **Is the exact equilibrium of the discretized game always a single
    action pair?**
    *Status: answered for the five setups and five grid sizes tested (25
    games). Yes.*
23. **Is the one-shot corner pair also an equilibrium of the walking game?**
    The walking experiments check that players head to the one-shot corners,
    not that those corners are an equilibrium of the walking game.
    *Status: open.*
24. **Does PPO stay ahead when matched on gradient steps rather than episodes?**
    *Status: answered for the main game (75 runs of PPO with 1, 2 and 4 epochs).
    No clear lead per gradient step. After 100 episodes 1, 9 and 19 of 25 runs had
    converged with 1, 2 and 4 epochs, and one-epoch PPO is no faster than REINFORCE.*
25. **How does the walking game change with the maximum step length and with
    the discount?** *Status: open; only a step of 0.2 and a discount of 0.9
    were run.*

26. **Do A2C and PPO beat REINFORCE on the walking game?**
    *Status: answered for one house setup (15 runs). A2C is the most reliable
    (5 of 5 seeds at the corners at the end, 34 of 35 later checkpoints under
    0.1); REINFORCE is slower (4 of 5, 24 of 35); PPO gets there fastest but
    does not stay (2 of 5, 14 of 35).*
27. **Why is PPO unstable on the walking game?**
    *Status: open on the mechanism. A smaller learning rate does not help and fewer
    epochs help partly, so it is not simply the step size; reusing each episode seems
    to contribute, but that is a hypothesis.*
28. **Does a smaller learning rate or fewer epochs make PPO reliable on the walking game?**
    *Status: answered for the settings tried (15 runs). Fewer epochs help partly (4 of 5
    seeds at the corners, 21 of 35 later checkpoints, against 2 of 5 and 14 of 35 for the
    default), a smaller learning rate does not (1 of 5 and 2 of 5), and nothing matches A2C
    (5 of 5, 34 of 35).*
29. **Do the slower PPO settings settle with more than 300 episodes?** Several runs
    were still far from the corners when training stopped. *Status: open.*

## Where each question is answered in this repository

| Questions | Evidence |
|---|---|
| 1, 2, 7, 8 | `results/dog_game_runs.jsonl` (75 REINFORCE runs), `doggame/experiments.py` |
| 1, 2, 16 | `results/dog_game_studies.jsonl`, study `algorithms` (75 runs), `doggame/train_ac.py` |
| 3, 20 | study `pretraining` (15 runs) |
| 4, 17 | study `selection` (30 runs) |
| 5, 21 | study `stage_solver` (90 runs) |
| 6, 22 | study `resolution` (25 games) |
| 9 | `doggame/verify.py` |
| 10 | study `noise` (31 runs) |
| 12, 18, 19 | study `inertia` (40 runs) |
| 26 | study `walking` (15 runs) |
| 24, 27, 28 | study `ppo_variants` (65 runs) |
| walking game | `results/dog_game_runs.jsonl`, `results/dog_game_trajectories.json` |

Questions 5b, 11, 13, 14, 19, 23, 25, 27 and 29 are open. The studies are in
[`doggame/dog_game_studies.py`](doggame/dog_game_studies.py) and the tables are
produced by [`doggame/dog_game_summary.py`](doggame/dog_game_summary.py).
