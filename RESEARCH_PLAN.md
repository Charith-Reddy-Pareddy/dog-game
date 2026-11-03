# Research plan

Working title: *Influence with persistence: feedback equilibria, exaggeration, mechanism choice and
learning in a potential influence game.* Target venue: ACM EC (economics and computation), with the
learning result as a separate machine-learning workshop paper.

Each item lists the question, what counts as done, and the literature it must be checked against.

## Done (see RESEARCH_REPORT.md, Sections 5.15 to 5.24)

- When the one-shot equilibrium stays Markov-perfect under inertia (Proposition 1, proved) and an
  explicit feedback equilibrium when the houses straddle the dog (Proposition 2, proved).
- Equilibria in target form beyond Proposition 2 (numerical; 199 of 200 two-player games, 55 of 60
  games with 3 to 8 players).
- Many houses, continuous time and a noisy dog; the game is not a Markov potential game.
- Learning: self-play learns the static profile, not the feedback equilibrium; exact gradient play
  finds it with a table of actions but not with a shared network (dog game and soccer).

## Done since (Sections 5.25 to 5.27)

- Item 1: the overshoot in closed form for two players (largest at inertia 1 / (1 + sqrt(1 - g)),
  where a target is at most twice as far from the long-run dog as the house); the many-player
  version is numerical.
- Item 2: Propositions 3 and 4, with numbers for 3, 5 and 7 players.
- Item 3: price of anarchy by inertia and players, and with shocks; a welfare-maximizing inertia
  exists when the shock comes with the aggregate (Proposition 5 for the planner).
- Items 4 and 5: the duopoly's exact equilibrium; per-state regression fixes the network on all
  three games, five seeds each. State-local features were not tried.

Next: a sampled (model-free) version of the fix; item 6; and the three items toward a paper:

- **A general reversal theorem** (started, [docs/REVERSAL_NOTES.md](docs/REVERSAL_NOTES.md)): in the
  continuous-time limit the median rule runs at full speed and the mean rule is a tug of war; a
  candidate proposition and a threshold conjecture.
- **Robustness** (started, study `mechanisms_robust`): discount 0.95 and 0.99, 9 and 15 players. Not
  started: unequal weights (needs a weighted median rule for a fair comparison) and two or more
  dimensions (coordinate-wise rules).
- **Related work** (started, [docs/RELATED_WORK.md](docs/RELATED_WORK.md)): a draft list, with the
  areas not yet covered.

**The receiver as a mechanism** (the paper's main thread, following BIG's call for mechanism design):
- Proposition 7 (proved): BIG's averaging receiver is the generalized median with phantoms k/N, with
  or without inertia; the phantoms set both where the receiver settles and how fast it moves.
- Without inertia the uniform-phantom (averaging) receiver is welfare-best among strategy-proof
  receivers (known: Jennings, Laraki, Puppe and Varloot, 2021).
- Proposition 8 (first-order proof): as inertia tends to 1 the welfare-best receiver is the order
  statistic of the targets closest to the mean target in mean square: the median for uniform targets
  (checked at inertia 0.99 and 0.995, study `mechanisms_limit`), a higher order statistic for strongly
  skewed ones (confirmed in direction with 7 players, studies `mechanisms_skewed`, `mechanisms_skewed_seven`).
- In between (studies `mechanisms_phantoms`, `mechanisms_asymmetric`): the best phantoms move from
  averaging toward the edges as inertia grows.
- Proposition 9 (proved): a receiver that compensates for its inertia (moves as close to the generalized
  median of the reports as it can) makes truthful reporting a feedback equilibrium for any phantoms and
  inertia. With truthful reports the compensated averaging receiver beats both plain receivers' equilibria on
  average at every positive inertia (study `mechanisms_compensated`).
  The corollary extends it to every strategy-proof rule (not to averaging the reports).
- Proposition 10 (first-order proof): a slow compensated receiver keeps the static optimum, E[(s - hbar)^2],
  so averaging for uniform targets; numerically the best compensated shape is the same at every inertia from
  0 to 0.995 for 3, 5 and 7 players (study `mechanisms_compensated_design`).
- Proposition 11 (proved): Proposition 9 holds for every receiver whose reachable set contains the current
  position and has nondecreasing endpoints (speed limits, asymmetric adjustment); neither condition can be dropped.
- Proposition 12 (proved): with truthful reports a compensated receiver's loss is the planner's plus
  2N integral (u - hbar) W(u) du, with W the discounted time a point is reachable; the planner is the compensated
  receiver aimed at the mean target.
- Proposition 13 (proved): the loss is a sum of one term per phantom with the static sign pattern, so for uniform
  targets the optimal compensated phantoms are k/N at every inertia; with clustered targets inertia can switch
  between the static local optima (counterexample found in an independent proof check).
- Open: make Proposition 8's remainder uniform; unequal weights; whether truthful reporting is the only
  equilibrium of the compensated receiver; shocks (a small gain from shading, see the proof notes).

## Economics

1. **Exaggeration under persistence.** In the one-shot game players exaggerate (push to the edges).
   With inertia, how far past their houses do players aim, and how does that depend on inertia and
   patience? Done when: a closed form, verified against the dynamic program.
2. **Mean against median under persistence.** The median rule is strategy-proof in one shot and the
   mean rule is not (Moulin 1980; Renault and Trannoy 2005). Does truthful reporting under the
   median rule stay a feedback equilibrium when the outcome has inertia? How do the two rules
   compare in outcome, manipulation and welfare? Done when: a statement with a proof where possible,
   and numbers for 3 and 5 players. Check against: Dziuda and Loeper (JPE 2016), dynamically
   consistent voting rules (JET 2015), Bailey and Tovey (AAMAS 2025).
3. **Price of anarchy.** Social loss of the equilibrium against a planner, as a function of inertia
   and the number of players, deterministic and with shocks; is there a welfare-maximizing inertia
   ("cooling off")? Done when: tables with both, and a statement of what drives the dependence.

## Learning

4. **A third, economic game.** Sticky-price duopoly in discrete time (Fershtman and Kamien 1987;
   Heston and Hu 2025): compute its feedback equilibrium exactly and test whether self-play and exact
   gradient play learn it or the static Cournot outcome.
5. **A fix that works.** State-local features (radial-basis inputs) and training on off-path
   deviations, tested on the dog game, soccer and the duopoly. Done when: a fix reaches the
   feedback equilibrium where a plain network does not, on all three games, with 5 or more seeds.

## Open theory

6. Existence (and uniqueness) of the target-form equilibrium for interior one-shot actions.
