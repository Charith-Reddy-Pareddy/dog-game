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

**Mean rule (checked numerically, not proved).** Away from its long-run point, every player is at an
edge in the target-form equilibrium, and

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
should rise with patience. The study `mechanisms_robust` (discount 0.95 and 0.99) tests this.

## Next steps

1. Prove the tug-of-war dynamics for the mean rule from the target-form equilibrium.
2. Finish Candidate Proposition 6.
3. Compute D(rho) in closed form for three players (the paths are piecewise exponential) and check
   the single crossing.
4. Decide whether the paper states its results in continuous time or keeps discrete time.
