# Toward a general reversal theorem (working notes, in progress)

Section 5.25 of the research report finds numerically that the mean rule loses less than the median
rule without inertia and more with high inertia. These notes work toward a general statement. Nothing
here is finished: each item says whether it is proved, checked numerically, or conjectured.

## Continuous-time limit

Take inertia w0 = 1 - dt and discount g = exp(-rho dt) and let dt -> 0, as in the continuous-time
section of [INERTIA_PROOFS.md](INERTIA_PROOFS.md). The outcome then follows dx/dt = a(t) - x, where
a(t) in [0, 1] is the rule applied to the reports, and the social loss is

  L = integral_0^inf exp(-rho t) sum_i (x(t) - h_i)^2 dt.

The ratio of the discount rate to the speed of adjustment is what matters. In discrete time this is
roughly (1 - g) / (1 - w0), so more inertia acts like a larger rho, and more patience like a smaller rho.

**Median rule (follows from Proposition 4).** The median player reports 1 while the outcome is below
the median house h_m (0 while above), so dx/dt = 1 - x until x reaches h_m, which it does at
t = ln((1 - x0) / (1 - h_m)), and stays there. This is the fastest any rule can move the outcome.

**Mean rule (follows from the corollary to Proposition 7, given target form).** By Proposition 7 the
mean rule is the generalized median with phantoms k/N, and while no player is pivotal the aggregate is
the A-th phantom, A/N, with A the number of players pushing up. Away from its long-run point, every
player is at an edge in the target-form equilibrium, and

  dx/dt = U(x) - x,   U(x) = total weight of the players whose equilibrium targets lie above x,

so the outcome is pulled up only by the players above it, a tug of war. At a target where U jumps
across x the outcome stops there (the pivot). The discrete equilibrium paths match this to within
0.0012 at dt = 0.01 in three games with 3 and 5 players (test
`test_in_the_continuous_limit_the_median_rule_runs_and_the_mean_rule_is_a_tug_of_war`). The
targets come from the discrete solver; a closed form for them is open.

## A candidate proposition

**Candidate Proposition 6.** Suppose both rules end at the median house: the mean rule's pivot is the
median player, with target h_m. Start below h_m, and suppose h_m <= hbar (the mean house); the mirror
case is the same. Then for every rho the median rule's loss is at most the mean rule's.

*Proof sketch.* U(x) - x <= 1 - x, so by the comparison theorem for ODEs the mean-rule path never
overtakes the median-rule path, and both stay below h_m <= hbar. The summed loss
sum_i (x - h_i)^2 decreases in x below hbar, so at every time the median rule's loss is no larger. ∎

To finish: make the comparison argument rigorous for the discontinuous right-hand side (Filippov
solutions), say when the pivot is the median player, and treat h_m > hbar.

## Conjecture: a threshold in rho

Let D(rho) = L_median - L_mean, for fixed houses and start.
- As rho -> 0, rho L tends to the long-run loss, so D > 0 whenever the mean rule's long-run point is
  closer to hbar than h_m is (the usual case, Section 5.25).
- As rho -> infinity, only the start matters: the loss is about l(x0) / rho + l'(x0) x'(0) / rho^2,
  and the median rule moves faster at the start (1 - x0 against U(x0) - x0). So D < 0 when the first
  move is toward hbar.

Hence D changes sign at least once (if the targets move continuously with rho). **Conjecture:**
exactly once, at a threshold rho* that can be computed from the houses, so the median rule wins
exactly when the adjustment is slow relative to discounting. Both limits use targets that themselves
depend on rho; this needs care.

**Testable prediction for the discrete game:** the inertia at which the median rule starts winning
should rise with patience. First results from the study `mechanisms_robust` (100 draws per cell,
table `mechanisms robust` in `doggame/dog_game_summary.py`) agree:
- The reversal holds in every cell: the median rule loses less in 17 to 19% of draws without inertia
  and in a majority at high inertia, for discount 0.9, 0.95 and 0.99 and for 3 to 15 players.
- The first inertia at which it wins in more than half the draws rises with patience: for 5 players
  0.6, 0.6 and 0.75, for 7 players 0.6, 0.75 and 0.9 (discount 0.9, 0.95, 0.99; the 0.9 values come
  from `mechanisms_wide`). For 3 players it is 0.3, 0.45 and 0.45.
- With 9 and 15 players the median rule wins in 98 and 100% of draws at inertia 0.95. There the
  target-form equilibria are not exact: the largest one-round deviation gain is about 0.011 (checked
  on 3 draws per inertia), so these cells need a better equilibrium solver before they are reported.

## The welfare-best receiver moves toward the median (study `mechanisms_phantoms`)

Symmetric phantoms on a grid (3 influencers: (p, 1 - p), p in steps of 0.025, 100 draws; 5 influencers:
(p1, p2, 1 - p2, 1 - p1) in steps of 0.1, 50 draws), expected discounted summed loss over evenly spread
starts, table `mechanisms phantoms` in `doggame/dog_game_summary.py`:
- Without inertia the best grid shape is the one nearest averaging (p = 0.325 next to 1/3 for 3
  influencers; p2 = 0.4 for 5), as the static theory says.
- As inertia grows the best phantoms move toward the edges: for 3 influencers p = 0.3 at w0 = 0.45 and
  0.6, 0.275 at 0.75, and 0 (the median receiver) at 0.9 and 0.95; for 5 influencers p2 = 0.3 at 0.75,
  0.2 at 0.9 and (0, 0), the median receiver, at 0.95. The best receiver saves 3.4 to 6.1% over
  averaging at inertia 0.9 to 0.95, and at most 1.2% up to 0.75.
- Caveats: 105 of 25,200 equilibrium solves stopped at the 300-iteration cap, mostly at inertia 0.9 and
  0.95; dropping those draws leaves every best shape unchanged except 5 influencers at 0.75 (0.4
  instead of 0.3, a neighbouring grid point). Only symmetric phantoms were searched, and the 5-player
  grid is coarse; near the optimum several shapes tie closely.

## Next steps

1. The tug-of-war dynamics now follow from Proposition 7; what is left is the general receiver: with
   phantoms alpha the drift is alpha_A - x, so the comparison of any two receivers reduces to their
   phantoms (study `mechanisms_phantoms`).
2. Finish Candidate Proposition 6.
3. Compute D(rho) in closed form for three players (the paths are piecewise exponential) and check
   the single crossing.
4. Decide whether the paper states its results in continuous time or keeps discrete time.
