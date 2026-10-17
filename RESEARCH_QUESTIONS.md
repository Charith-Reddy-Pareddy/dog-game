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
   *Status: answered for the shared-coordinate setup. Never the solver's
   interior point; always a boundary equilibrium, one of two mirror images.
   Which one is consistent with a fair coin (31 to 19 over 50 seeds at the
   default noise, p = 0.12); the earlier 23 to 7 was a small-sample effect.*

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
    *Status: answered for the shared-y setup. The one-shot potential is exactly
    flat along the continuum of equilibria, so the expected policy gradient is
    zero there and the players' means random-walk; nothing pulls them back to the
    interior, and the box stops the walk at an edge. Without clipping, 41 of 50
    runs still end on a mirror boundary; with an unbounded mean in y as well, the
    walk does not stop (red's mean ends outside the box in 38 of 50 runs). Which
    mirror image wins is consistent with a fair coin at every noise level over 50
    seeds; an earlier 10-seed result that the noise flips it did not hold up.*
18. **How strongly does dog inertia slow policy-gradient training?**
    *Status: answered for two setups. On the easy setup every run converges
    up to `w0 = 0.9`; on the hard one 4 of 5 converge at `w0 = 0`, 5 of 5 at
    0.3, 1 of 5 at 0.6 and 0 of 5 at 0.9 within 1500 episodes.*
19. **Does the one-shot equilibrium remain an equilibrium of the game with
    inertia?** *Status: answered.* The exact feedback (Markov-perfect)
    equilibrium plays the one-shot Nash actions along the equilibrium path but
    not off it. In closed form, holding the one-shot actions at every dog
    position is Markov-perfect if and only if each player's slack,
    `(h - d*) / (1 - d*)` for a player at the top of the box, is at least
    `w0 (1 - g w0) / (1 - g w0^2)`, and no one-shot action is interior. In the
    five setups that holds only in setup 1 up to `w0 = 0.6`, matching the
    dynamic program.*
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
    *Status: answered for the resting state and for the straight-walk profile (8 checks, four house
    setups). Both players resting on the corners is an exact equilibrium (gain 0 in all four setups).
    Walking straight to the corners at full speed is a near-equilibrium: the best deviation gains
    0.13% to 4.7% of the reward. The equilibrium of the full dynamic game with feedback strategies
    was not computed.*
24. **Does PPO stay ahead when matched on gradient steps rather than episodes?**
    *Status: answered for the main game (75 runs of PPO with 1, 2 and 4 epochs).
    No clear lead per gradient step. After 100 episodes 1, 9 and 19 of 25 runs had
    converged with 1, 2 and 4 epochs, and one-epoch PPO is no faster than REINFORCE.*
25. **How does the walking game change with the maximum step length and with the discount?**
    *Status: partly answered for A2C on one house setup (20 runs). A lower discount
    makes learning faster (0.5: 5 of 5 seeds, 35 of 35 later checkpoints at the corners)
    and a higher one slower (0.99: 4 of 5, 22 of 35); a longer step (0.4) costs nothing, and
    a shorter one (0.1) is slower (3 of 5, 20 of 35) within 300 episodes. The one-shot corners
    do not depend on either, so this is about how fast the players get there, not where.*
26. **Do A2C and PPO beat REINFORCE on the walking game?**
    *Status: answered for one house setup (15 runs). A2C is the most reliable
    (5 of 5 seeds at the corners at the end, 34 of 35 later checkpoints under
    0.1); REINFORCE is slower (4 of 5, 24 of 35); PPO gets there fastest but
    does not stay (2 of 5, 14 of 35).*
27. **Why is PPO unstable on the walking game?**
    *Status: open on the mechanism. A smaller learning rate does not help, fewer epochs
    help, and default PPO's failures persist (two of five seeds never return to the corners
    in 900 episodes), so it is not just slow; reusing each episode seems to contribute, but that
    is a hypothesis.*
28. **Does a smaller learning rate or fewer epochs make PPO reliable on the walking game?**
    *Status: answered for the settings tried. Fewer epochs help (4 of 5 seeds after 300
    episodes, 5 of 5 after 900), a smaller learning rate does not (1 of 5 and 2 of 5 after 300),
    and one-epoch PPO is still much slower than A2C (0.12 against 0.011 mean distance after 300
    episodes).*
29. **Do the slower PPO settings settle with more than 300 episodes?**
    *Status: answered for one-epoch PPO and the default (30 runs with A2C). One-epoch PPO
    settles (5 of 5 seeds at the corners after 900 episodes) and default PPO does not (3 of 5,
    with two seeds stuck far away). The smaller learning rate settings were not run longer.*

30. **Do the walking-game findings hold on other house setups and with dog inertia?**
    *Status: partly answered (45 runs on three other setups, 10 with inertia). A2C is again the
    most reliable wherever any algorithm learns the game, and PPO is the least reliable or tied
    for least, with no early lead on the other setups. One setup is much harder: no algorithm
    gets more than 1 of 5 seeds to the corners in 300 episodes. Inertia slows A2C (4 of 5 seeds
    at 0.6). PPO with inertia and longer training on the hard setup extend the
    evidence, but five seeds and one hard configuration do not resolve
    generalization.*
31. **Does an entropy bonus or a larger network change PPO's instability on the walking game?**
    A 64-unit network was tried with one and four PPO epochs; results differed
    substantially across those settings and are based on five seeds. Entropy
    bonus was not tested. *Status: partly answered for network capacity.*

32. **Does a polar absolute-action policy improve policy-gradient learning on
    the original dog game?** The angle is von Mises-distributed and the radius
    is a Beta fraction of the ray from the square center to its boundary.
    The primary comparison has clipped-Gaussian Cartesian and bounded Beta
    Cartesian controls, REINFORCE/A2C/PPO, one house setup and ten seeds. Polar
    has higher expected gain than Cartesian Beta in all 30 paired seeds. The
    exploratory REINFORCE follow-up on four other setups is mixed: two
    intervals favor Cartesian Beta and two include zero. The metric is a
    one-shot score at the reset state, and the policy-family comparison does
    not isolate coordinate geometry. *Status: partly answered for the tested
    setups; no general improvement is established.* See Section 5.18 of the
    [research report](RESEARCH_REPORT.md) and the concise
    [PG update](PG_RESEARCH_UPDATE.md).

33. **Is the dog game a Markov potential game?** In a Markov potential game every
    4-cycle of unilateral policy changes sums to zero.
    *Status: answered: no. With policies that react to the dog's position the cycle
    sums are +7.90, +7.12, +5.63 and +1.85 at `w0` of 0, 0.3, 0.6 and 0.9, so it is
    not one, even without inertia. With policies that ignore the dog every cycle is
    zero, so it is a potential game in open-loop strategies at any inertia. The
    convergence theorem for independent policy gradient in Markov potential games
    therefore does not explain the convergence seen here.*
34. **Does an explicit piecewise-affine solution of the box-constrained game reproduce the
    dynamic-programming feedback equilibrium exactly?** Constrained linear-quadratic
    games have feedback solutions that are piecewise affine over polyhedral regions.
    *Status: partly answered. When the houses lie on either side of the dog, the policy in
    which each player aims the next dog position at `d* + k (h - d*)` is a feedback
    equilibrium (proved in docs/INERTIA_PROOFS.md; 77 of 77 random games, all such setups, within one grid step
    of the dynamic program). When a one-shot action is interior it fails in 53 of 123
    random games, and that case is open.*
35. **What happens in a noisy, continuous-time dog game?** With
    `dX = (w a_red + (1 - w) a_blue - X) dt + sigma dW`, a Riccati benchmark needs a
    quadratic action cost: without one, each player's Hamiltonian is linear in its own
    action and, without the box, no equilibrium exists. Does continuous-time
    q-learning recover the box-constrained equilibrium as that cost goes to zero?
    *Status: partly answered. As the time step shrinks, Proposition 2's equilibrium becomes
    bang-bang with switching points d* + (r + 2)/(r + 1) (h - d*), and holding needs slack at
    least (1 + r)/(2 + r). With a noisy dog the equilibrium stays bang-bang but the switching
    points move (red's from 0.725 to 0.765 at sigma = 0.2), also without clipping, unlike the
    linear-quadratic benchmark. Continuous-time q-learning is open.*
36. **Does the many-house game have a mean-field limit, and what do learners find there?**
    *Status: partly answered. With equal weights the one-shot dog converges to the generalized
    median d = 1 - F(d) of the houses' distribution, the known Nash outcome of the average voting
    rule. With inertia, holding the one-shot actions is Markov-perfect only when no pivot player is
    inside the box (about one draw in three) and the inertia is below roughly 0.35/N. The
    many-house feedback equilibrium and what learners find are open.*
37. **Does policy-gradient self-play learn a feedback equilibrium of the game with inertia?**
    *Status: answered for setup 3 at w0 = 0.6 and setup 5 at w0 = 0.9: no. Starting every
    episode from a random dog position, A2C and PPO learn to hold the one-shot actions at every
    position (19 of 20 Gaussian runs), which is exploitable off the path. Five times longer
    training and a Beta policy do not change this, and REINFORCE does not do better. Exact,
    noise-free gradient play finds the equilibrium with a table of actions (4 of 4) but not with a
    shared network (0 of 20), so function approximation, not noise, is what stops it.*

## Where each question is answered in this repository

| Questions | Evidence |
|---|---|
| 1, 2, 7, 8 | `results/dog_game_runs.jsonl` (75 REINFORCE runs), `doggame/experiments.py` |
| 1, 2, 16 | `results/dog_game_studies.jsonl`, study `algorithms` (75 runs), `doggame/train_ac.py` |
| 3, 20 | study `pretraining` (15 runs) |
| 4, 17 | studies `selection` (30 runs), `selection_noise` (200), `selection_clip` (50) and `selection_linear` (50), `doggame/potential.py` |
| 5, 21 | study `stage_solver` (90 runs) |
| 6, 22 | study `resolution` (25 games) |
| 9 | `doggame/verify.py` |
| 10 | study `noise` (31 runs) |
| 12, 18 | study `inertia` (40 runs) |
| 19, 34 | studies `inertia_dp` (15 solves), `inertia_explicit` (15) and `inertia_random` (200 games), `doggame/inertia_dp.py`, `doggame/inertia_mpe.py`, `doggame/inertia_theory.py` |
| 33 | study `markov_potential` (8 cycles), `doggame/markov_potential.py` |
| 36 | study `many_houses` (2,000 draws), `doggame/many_houses.py` |
| 37 | studies `feedback_learning` (55 rows) and `exact_play` (24 runs), `doggame/feedback_learning.py`, `doggame/exact_play.py` |
| 35 | study `noisy_inertia` (9 solves), `continuous_limit` in `doggame/inertia_theory.py`, `noise` in `doggame/inertia_dp.py` |
| 26 | study `walking` (15 runs) |
| walking game, all questions | [WALKING_GAME_REPORT.md](WALKING_GAME_REPORT.md) ([PDF](docs/walking_game_report.pdf)) |
| 24, 27, 28 | study `ppo_variants` (65 runs) |
| 27, 28, 29 | study `ppo_long` (15 runs) |
| 25 | study `walking_settings` (20 runs) |
| 30 | studies `walking_houses` (45 runs) and `walking_inertia` (10 runs) |
| 23 | study `walking_equilibrium` (8 checks), `doggame/walking_equilibrium.py` |
| walking game | `results/dog_game_runs.jsonl`, `results/dog_game_trajectories.json` |
| 32 | `docs/POLAR_ACTION_PROTOCOL.md`, `docs/POLAR_ACTION_ROBUSTNESS_PROTOCOL.md`, `doggame/polar_action_study.py`, `doggame/polar_action_robustness.py`, `results/polar_action_pg.jsonl`, `results/polar_action_pg_robustness.jsonl` |

Questions 5b, 11, 13, 14, 20 and 27 are open, and 7, 12, 15, 25, 30, 31, 32, 34, 35 and 36 are partly answered. The studies are in
[`doggame/dog_game_studies.py`](doggame/dog_game_studies.py) and the tables are
produced by [`doggame/dog_game_summary.py`](doggame/dog_game_summary.py).
