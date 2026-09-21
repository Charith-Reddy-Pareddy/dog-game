# Research Questions

These are the open questions that motivated this repository, drawn from
the research group's planning meeting on the dog game (the "AI dog" /
"Battling Influencer Game") and the preceding grid-soccer exact-solver
work. They're grouped by the two subgroups the meeting split into, plus
the questions common to both.

## Policy gradient (REINFORCE / A2C-style self-play)

1. **Can self-play policy gradient converge to a Nash equilibrium at
   all in a general-sum game?** Policy gradient is fundamentally a
   single-agent learning algorithm; running two copies of it against
   each other is best-response dynamics, not a direct game solver, and
   there is no guarantee that mutual improvement converges anywhere.
   This was posed as *the* open question for this subgroup.
2. Does the number of self-play episodes needed to converge depend on
   the specific house/weight configuration, or is there a single
   number that works across the board?
3. Does pre-training help, and if so, on what does the benefit depend?
4. When the game has more than one Nash equilibrium (see question 8
   below), which one does self-play find, and is it consistent across
   runs, seeds, or network architectures?

## DQN / Nash-Q (discretized-action side)

5. **How should the per-state stage game inside the Nash-Q Bellman
   backup `Q = R + discount * NashV(Q)` actually be solved?** An LP
   solver gives an exact stage-game equilibrium; best-response
   dynamics (fictitious play) only approximates one. How much stage
   game error is tolerable before the outer Bellman iteration stops
   converging to the right fixed point?
5b. Is there existing literature on how precisely the stage game must
   be solved for the outer dynamic-programming loop to converge, given
   how commonly Nash-Q-style methods are used in practice?
6. Continuous actions have to be discretized into a finite grid for
   DQN. How does the grid resolution trade off against solution
   quality, and is a neural-network function approximator viable in
   place of a full grid as resolution grows?

## Common to both subgroups

7. **What is the right multi-agent policy network architecture?**
   Fully separate per-player networks, a fully shared trunk, or a
   partially shared trunk with separate per-player branches — do they
   differ in training stability, final policy quality, or how they
   perform against each other (not just against themselves)?
8. **When is the dog game's Nash equilibrium unique?** The conjecture
   from the meeting is that uniqueness requires the two houses to
   differ in *both* coordinates. What happens at the degenerate case
   (a shared coordinate), and do different solvers reliably pick
   different-but-equally-valid equilibria there?
9. **How do you verify, after the fact, that a solved or learned
   policy actually is a Nash equilibrium?** For a mixed strategy: are
   all actions played with positive probability equally good, and is
   no unplayed action strictly better? For a continuous action pair:
   is there no unilateral deviation that improves either player's
   payoff?
10. Practical solvers (fictitious play, best-response iteration)
    rarely land on "clean" probabilities (50/50, 1/3-2/3, etc.) the
    way a hand-solved game would. How much of that is real mixing
    versus numerical/approximation noise, and how should a verification
    check be tuned to tell the two apart?
11. Everything above is being developed on the dog game specifically
    because it is simpler than the soccer game (general-sum vs.
    zero-sum, provably unique equilibrium under mild conditions vs.
    many possible equilibria). Which of the findings here — on
    architecture, verification, and stage-game solving — are expected
    to transfer to the soccer game, and which are artifacts of the dog
    game's simpler structure?

## Status of these questions in this repository

- Questions 1-2 and 7 are addressed empirically in
  [`doggame/experiments.py`](doggame/experiments.py) and summarized in
  the [README](README.md#findings) and on the
  [live results page](https://charith-reddy-pareddy.github.io/dog-game/#findings).
- Questions 5 and 6 are addressed by
  [`doggame/fictitious_play.py`](doggame/fictitious_play.py) and
  [`doggame/dqn.py`](doggame/dqn.py), which solve the discretized stage
  game by best-response dynamics rather than an LP solver, then check
  the resulting fixed point against the closed-form value the
  state-independent transition permits (see `doggame/dqn.py`'s module
  docstring).
- Question 8 has a direct, reproducible example in the README and the
  live page's "Finding 2".
- Question 9 is implemented directly in
  [`doggame/verify.py`](doggame/verify.py).
- Questions 3, 4, 10, and 11 are open; see
  [RESEARCH_REPORT.md](RESEARCH_REPORT.md#open-questions) for where
  this repo's results leave them.
