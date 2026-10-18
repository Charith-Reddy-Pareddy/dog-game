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
