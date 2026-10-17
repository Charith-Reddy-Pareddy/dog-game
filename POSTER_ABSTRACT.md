# Poster abstract

**When the Dog Has Inertia: Exact Feedback Equilibria and Equilibrium Selection in a Potential Influence Game**

Charith Reddy Pareddy, University of Wisconsin–Madison

In the dog game, two players repeatedly pull a shared point, the dog, toward their own houses in the unit square.
Without inertia it is an instance of the Battling Influencers Game, an exact potential game whose equilibria maximize
a known potential. We ask what survives when the dog keeps part of its position each round, and what
policy-gradient learners find when equilibria form a continuum. We prove two results for the game with inertia.
First, holding the one-shot equilibrium actions at every dog position is a Markov-perfect equilibrium if and only if
each player's slack beyond the dog is at least ρ = w0(1 − γw0)/(1 − γw0²), where w0 is the inertia and γ the
discount, so more patient players tolerate more inertia. Second, when the houses lie on either side of the dog, an
explicit piecewise-affine policy, in which each player aims the next dog position at a closed-form target, is a
feedback equilibrium. Both match an exact dynamic program to one grid step; the second holds in all 77 such games
among 200 random ones, and fails in 53 of the 123 games where a one-shot action is interior. We also show that the
dynamic game is not a Markov potential game once policies react to the dog. Finally, with 250 self-play runs and two
ablations, we find that along the potential's flat direction learning is a random walk that the action box stops:
no run reaches the interior equilibrium, and which of two mirror equilibria wins is a fair coin, correcting an
earlier small-sample result.

Poster: [docs/poster.pdf](docs/poster.pdf) (48 x 36 inches), built by `python3 -m doggame.poster`.
Full results: [RESEARCH_REPORT.md](RESEARCH_REPORT.md), Sections 5.15, 5.19 and 5.20.
