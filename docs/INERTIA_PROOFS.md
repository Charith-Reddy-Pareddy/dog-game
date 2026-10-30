# Proofs for the dog game with inertia

This note proves the two propositions of Section 5.20 of the research report. The proofs are mine and have not
been refereed; every step is also checked numerically in `tests/test_inertia_theory.py` and the studies
`inertia_explicit` and `inertia_random`.

## Setting

Rewards are sums over the two coordinates, the dog moves coordinate by coordinate, and the action set is the
product of two intervals. So a profile is a feedback equilibrium of the two-dimensional game exactly when it is one
in each coordinate: a two-dimensional deviation is a pair of one-coordinate deviations whose payoffs add. Work in
one coordinate from now on.

The state is the dog's position x in [0, 1]. Red and blue choose actions a_red and a_blue in [0, 1], and the next
position is

    f(x, a_red, a_blue) = w0 x + (1 - w0) (w a_red + (1 - w) a_blue),

with inertia w0 in [0, 1), red's weight w in (0, 1) and blue's weight 1 - w. Player i receives
-(f - h_i)^2, the squared distance of the next position from its house h_i, discounted by g in [0, 1). A stationary
Markov profile pi maps positions to actions, and V_i(x) is player i's discounted payoff from x under pi. The profile
is a feedback (Markov-perfect) equilibrium if, at every x, each player's action maximizes
-(f - h_i)^2 + g V_i(f) given the other's action. By the one-shot deviation principle (Fudenberg and Tirole, 1991),
which applies because payoffs are bounded and discounted, this is the same as having no profitable one-round
deviation at any state.

Let (a_red*, a_blue*) be the one-shot Nash actions and d* = w a_red* + (1 - w) a_blue* the one-shot dog. Define

    k = (1 - g w0^2) / (1 - g w0),      T_i = d* + k (h_i - d*),      z_h(x) = w0 x + (1 - w0) d*.

k is at least 1 because w0^2 <= w0. T_i is player i's target and z_h(x) is the next position when both hold.

## Lemma 1 (the value of holding)

If both players always play the one-shot actions, then z_h(x) - d* = w0 (x - d*), so the position after t + 1 rounds
is d* + w0^(t+1) (x - d*), and summing the geometric series gives, with u = x - d* and e = d* - h_i,

    V_i(x) = -[ e^2 / (1 - g) + 2 e w0 u / (1 - g w0) + w0^2 u^2 / (1 - g w0^2) ].

## Lemma 2 (the one-round objective under holding)

If player i plays something else for one round and both hold afterwards, its payoff as a function of the next
position z is G_i(z) = -(z - h_i)^2 + g V_i(z). Its second derivative is -2 - 2 g w0^2 / (1 - g w0^2) < 0, so it
is strictly concave, and

    G_i'(z) = -2 [ (z - d*) / (1 - g w0^2) + (d* - h_i) / (1 - g w0) ],

which is zero exactly at z = T_i. So G_i increases up to T_i and decreases after it. With the other player at its
one-shot action, player i reaches the next positions in an interval of length (1 - w0) w_i (w_red = w,
w_blue = 1 - w), which is increasing in its action, and holding gives z_h(x).

## Proposition 1

Holding the one-shot actions at every position is a feedback equilibrium if and only if each player at the top of
the box (a_i* = 1) has k (h_i - d*) >= w0 (1 - d*), each player at the bottom (a_i* = 0) has
k (d* - h_i) >= w0 d*, and, when w0 > 0, no one-shot action is strictly inside the box.

**Proof.** By Lemma 2 and the one-shot deviation principle, holding is optimal for player i at x exactly when z_h(x)
maximizes the strictly concave G_i over i's reachable interval. For a player at the top, z_h(x) is the right end of
that interval, so it is optimal iff G_i'(z_h(x)) >= 0, that is z_h(x) <= T_i, that is

    w0 (x - d*) <= k (h_i - d*).

The left side is largest at x = 1, which gives the stated condition (for w0 = 0 it always holds, since a one-shot
best response at the top has h_i >= d*). A player at the bottom is the mirror image, with the worst case at x = 0.
For a player strictly inside, z_h(x) is inside its interval, so holding is optimal iff z_h(x) = T_i. A one-shot best
response strictly inside the box puts the dog on the player's house, h_i = d*, so T_i = d*, and the condition becomes
w0 (x - d*) = 0, which fails at every x other than d* when w0 > 0. Optimality at every x for every player is the
stated set of conditions. ∎

Dividing by 1 - d* (or by d*), the conditions say that the player's slack s = (h_i - d*) / (1 - d*) (or
(d* - h_i) / d*) is at least rho(w0, g) = w0 (1 - g w0) / (1 - g w0^2). rho rises from 0 to 1 as w0 goes from 0 to
1 (its derivative in w0 is (1 - g w0 (2 - w0)) / (1 - g w0^2)^2, at least (1 - g) / (1 - g w0^2)^2 > 0), and its
derivative in g is w0 (w0^2 - w0) / (1 - g w0^2)^2 <= 0, so more patient players tolerate more inertia.

## Proposition 2

Suppose the coordinate is straddling: after relabeling, h_blue <= w <= h_red, so the one-shot actions are
a_red* = 1, a_blue* = 0 and d* = w. Then the policy

    a_red(x)  = clip(1 + (T_red  - z_h(x)) / ((1 - w0) w),       0, 1),
    a_blue(x) = clip(0 + (T_blue - z_h(x)) / ((1 - w0) (1 - w)), 0, 1)

is a feedback equilibrium for every w0 in [0, 1) and g in [0, 1).

In words, each player aims the next position at its target, as far as its weight allows, as if the other held its
one-shot action.

**Proof.** If w0 = 0 then z_h = d*, both clips return the one-shot actions, and the game is the one-shot game in
every round, so assume w0 > 0. Note T_blue <= d* <= T_red, because k >= 1 and h_blue <= d* <= h_red.

**Regions.** Let X_red = d* + (T_red - d*) / w0 and X_blue = d* - (d* - T_blue) / w0, so that z_h(X_red) = T_red and
z_h(X_blue) = T_blue. Red eases off its edge exactly on R = {x : z_h(x) > T_red} = (X_red, 1], blue exactly on
B = [0, X_blue), and both hold on H = [X_blue, X_red] intersected with [0, 1]. R and B do not meet, because
X_blue <= d* <= X_red, so at most one player leaves its edge at any x. The next position under the policy is

    N(x) = z_h(x) on H,      max(T_red, z_h(x) - (1 - w0) w) on R,      min(T_blue, z_h(x) + (1 - w0)(1 - w)) on B.

N is nondecreasing; on R, T_red <= N(x) < x (since x > X_red >= T_red and z_h(x) < x when x > d*); on B,
x < N(x) <= T_blue. H is invariant: for x in H, N(x) = z_h(x) lies between x and d*, both in H. The targets lie in H
when they lie in [0, 1]: T_red <= X_red because T_red - d* >= 0 and w0 <= 1, and T_red >= d* >= X_blue; similarly
for T_blue.

**Values.** Because H is invariant and both hold on it, V_i equals the holding value of Lemma 1 on H. Let
g_i(z) = -(z - h_i)^2 + g V_i(z) be player i's one-round objective as a function of the next position. On H it is
G_i of Lemma 2. At any z the policy moves the dog to N(z), so V_i(z) = g_i(N(z)), and therefore

    g_i(z) = -(z - h_i)^2 + g g_i(N(z))    for every z.                                         (*)

**Claim: g_red is nondecreasing on [0, T_red] and nonincreasing on [T_red, 1]** (each part when nonempty).

**Above the target.** Let M be the set of bounded functions phi on [T_red, 1] that equal G_red on [T_red, X_red] and
are nonincreasing. Define (A phi)(z) = G_red(z) on [T_red, X_red] and (A phi)(z) = -(z - h_red)^2 + g phi(N(z)) on
R; this is well defined because N maps R into [T_red, 1). A is a contraction with modulus g in the sup norm, and by
(*) the restriction of g_red to [T_red, 1] is its fixed point. A maps M into M:

- on R, -(z - h_red)^2 decreases, because z > X_red >= T_red >= h_red (T_red - h_red = (k - 1)(h_red - d*) >= 0),
  and phi(N(z)) does not increase, because N does not decrease and phi does not increase;
- across the boundary, for z1 in [T_red, X_red] and z2 in R,
  (A phi)(z2) <= -(z2 - h_red)^2 + g phi(T_red) < -(X_red - h_red)^2 + g G_red(T_red) = G_red(X_red) <= G_red(z1),
  using N(z2) >= T_red, phi(T_red) = G_red(T_red), the identity G_red(X_red) = -(X_red - h_red)^2 +
  g G_red(z_h(X_red)) with z_h(X_red) = T_red (this is (*) on H), and that G_red decreases after T_red.

M is closed under uniform limits and contains, for example, G_red on [T_red, X_red] extended by the constant
G_red(X_red), so iterating A from it converges to the fixed point, which therefore lies in M (Banach's fixed point
theorem).

**Below the target.** The same argument on [0, T_red] with the set of functions that equal G_red on [X_blue, T_red]
and are nondecreasing: on B, -(z - h_red)^2 increases because z < X_blue <= d* <= h_red, and N maps B into
(z, T_blue], which lies in [0, T_red]; across the boundary, (A phi)(z1) <= -(z1 - h_red)^2 + g G_red(T_blue) <
-(X_blue - h_red)^2 + g G_red(T_blue) = G_red(X_blue) <= G_red(z2) for z1 in B and z2 in [X_blue, T_red], using
z_h(X_blue) = T_blue, T_blue in [X_blue, T_red] and that G_red increases up to T_red. This proves the claim; the
claim for blue is the mirror image, with its peak at T_blue.

**Best replies.** A function that does not decrease up to T and does not increase after it is maximized over any
interval [L, U] at clip(T, L, U). Given blue's action at x, red reaches the next positions [U - (1 - w0) w, U], where
U is the position with a_red = 1. On H and R, blue is at 0, so U = z_h(x) and red's best reply is
clip(T_red, U - (1 - w0) w, U), which is exactly the policy (it holds on H, where z_h(x) <= T_red, and moves to
max(T_red, z_h(x) - (1 - w0) w) on R). On B, blue moves the position up to N(x) <= T_blue <= T_red with red at 1,
so U = N(x) <= T_red and the best reply is red's edge, as in the policy. Blue's case is the mirror image. By the
one-shot deviation principle the policy is a feedback equilibrium. ∎

## Many houses

With N players, weights w_i summing to 1 and the dog at sum_i w_i a_i, Lemmas 1 and 2 hold unchanged for each player
(the other players' held actions only enter through d*, and player i reaches an interval of length (1 - w0) w_i).
So Proposition 1 holds verbatim: holding the one-shot actions is a feedback equilibrium iff every player at an edge
has slack at least rho(w0, g) and, when w0 > 0, no player (the pivot) is strictly inside the box. In the one-shot
equilibrium, which is the generalized median of Renault and Trannoy (2005), at most one player is inside. The
study `many_houses` measures how often there is none, and how the largest admissible inertia scales with N.
The target form also carries over numerically: if every player aims at a target, each round is the one-shot game
with the targets as houses, and targets from a fixed point give a feedback equilibrium whenever every player's
one-round objective is single-peaked (study `many_targets`); there is no proof for many houses.

## Continuous-time limit

With time step dt, w0 = 1 - dt and g = exp(-r dt), 1 - g w0 = (r + 1) dt + O(dt^2) and
1 - g w0^2 = (r + 2) dt + O(dt^2), so k -> (r + 2) / (r + 1) and rho -> (r + 1) / (r + 2). In Proposition 2,
(T_i - z_h(x)) / ((1 - w0) w_i) = (T_i - x + dt (x - d*)) / (dt w_i), which tends to +infinity where x < T_i and to
-infinity where x > T_i, so the policy tends to the bang-bang rule "push the edge toward T_i", with the interior band
of width O(dt). This describes the limit of the discrete-time equilibria; that it is an equilibrium of the
continuous-time game is not proved here.

## Overshooting

In Proposition 2 a player aims the next dog position at T_i = d* + k (h_i - d*), so it aims past its
own house by (k - 1)(h_i - d*). Since k = (1 - g w0^2) / (1 - g w0), dk/dw0 has the sign of
g w0^2 - 2 w0 + 1, which vanishes at w0 = (1 - s) / g = 1 / (1 + s) with s = sqrt(1 - g). So k = 1 at
w0 = 0 and at w0 = 1, and the overshoot is largest at the intermediate inertia 1 / (1 + s), where
k = 2 / (1 + s): at most 2, approached by patient players.

## Persistence and the median rule

Now N players (N odd) with distinct houses h_1 < ... < h_N report points r_i in [0, 1], and the outcome
moves to z = w0 x + (1 - w0) median(r); losses are (z - h_i)^2, discounted. Let h_m be the median house.

**Proposition 3.** For w0 = 0 truthful reporting (r_i = h_i at every outcome) is a feedback equilibrium;
for any w0 in (0, 1) it is not.

*Proof.* With w0 = 0 the round does not depend on the outcome, so each round is the one-shot median
game, where truthful reporting is dominant (Moulin, 1980). With w0 > 0 and everyone truthful, the
outcome moves to w0 x + (1 - w0) h_m and converges to h_m, so by Lemma 2 (with d* = h_m, the median
player's house) the median player's one-round objective is a concave quadratic in the next outcome
peaked at T_m = h_m. By changing its report within [h_(m-1), h_(m+1)] the median player moves the next
outcome over an interval that contains w0 x + (1 - w0) h_m strictly inside, and that contains h_m itself
whenever x is close enough to h_m. At such an x other than h_m, reporting (h_m - w0 x) / (1 - w0), not
h_m, puts the outcome on its peak, a strict gain. ∎

**Proposition 4.** For every w0 in [0, 1) the following profile is a feedback equilibrium: the median
player reports clip((h_m - w0 x) / (1 - w0), 0, 1), players with houses above h_m report 1 and players
below report 0. Under it the outcome moves to G(x) = clip(h_m, w0 x, w0 x + 1 - w0) each round: as close
to the median house as can be reached, the path a planner minimising (z - h_m)^2 would choose.

*Proof.* The median of the reports is the median player's report, which gives G. By the one-shot
deviation principle it suffices to check one-round deviations.
- The median player can move the median, hence the next outcome, anywhere in [w0 x, w0 x + 1 - w0]. Its
  problem is then a one-player control problem with loss (z - h_m)^2: if h_m is reachable it stays
  reachable forever after, and otherwise the reachable interval moves monotonically with the outcome, so
  moving as close to h_m as possible each round is optimal.
- A player with house h_i > h_m can only lower the median (by reporting below the median player's
  report), never raise it. If G(x) <= h_m, lowering the next outcome brings it further from h_i and, since
  G is nondecreasing, lowers the whole future path, which stays at or below h_m < h_i: a loss. If
  G(x) > h_m, the median player is reporting 0, so G(x) = w0 x is already the lowest reachable outcome
  and nothing can be lowered.
- Players below the median are the mirror image. ∎

So inertia costs the median rule its truthfulness but not its speed: the outcome still moves to the
median house as fast as any mechanism could move it there. The mean rule has no such property; in the
study `mechanisms` its outcome moves more slowly, because players on the far side of the outcome keep
pulling against it.

## Cooling off

Add a shock to each round: the outcome lands at y = clip(z + e, 0, 1), where z is the outcome the
reports aim at and e has a symmetric distribution with e != 0 with positive probability. The shock
either comes with the new aggregate, e = (1 - w0) s eps (the inertia filters it), or hits the outcome
itself, e = s eps. The greedy planner aims at the mean house hbar of the N houses each round, as close
as it can reach.

**Proposition 5.** Let wbar = min(hbar, 1 - hbar). For w0 in [0, wbar] the greedy planner's expected
discounted summed loss, from any start, is

  L(w0) = (sum_i (h_i - hbar)^2 + N phi(sigma)) / (1 - g),  phi(sigma) = E[(clip(hbar + sigma eps, 0, 1) - hbar)^2],

with sigma = (1 - w0) s when the shock comes with the aggregate and sigma = s when it hits the outcome.
So with a filtered shock L is strictly decreasing on [0, wbar]: some inertia is strictly better than
none, and the welfare-maximizing inertia is at least wbar. With a shock on the outcome, L is constant
on [0, wbar].

*Proof.* From any y in [0, 1] the reachable interval [w0 y, w0 y + 1 - w0] contains hbar when
w0 <= hbar and hbar <= 1 - w0, so the planner aims at hbar in every round and every round's outcome is
clip(hbar + e, 0, 1). The summed loss at an outcome y is sum_i (h_i - hbar)^2 + N (y - hbar)^2, which
gives L. Each |clip(hbar + sigma eps, 0, 1) - hbar| is nondecreasing in sigma and strictly increasing
where it is not clipped, which has positive probability, so phi is strictly increasing. ∎

Above wbar the planner can no longer reach hbar from every outcome, and slow adjustment starts to
cost; the study `mechanisms_noise` finds the loss U-shaped in w0, lowest at w0 = 0.6 to 0.75 for the
planner and at 0.45 to 0.9 for the equilibria of the mean and the median rule (which the proposition
does not cover), the optimum rising with the size of the shock. With a
shock on the outcome the planner is best off with no inertia. (The greedy planner is not optimal when
clipping is frequent; the study also computes the optimal planner.)

## The receiver as a mechanism

In the Battling Influencers Game (Wu, Zhu, Cai and Zhu, 2025) the receiver averages the
influencers' actions, and at every pure Nash equilibrium all influencers but at most one exaggerate to
the boundary, the remaining one putting the receiver on its own target (their Theorem 4). The paper's
conclusion asks for mechanism design that removes this incentive. With inertia the question becomes:
which receiver should aggregate the influencers' actions round after round?

On one coordinate with action space [0, 1] and equal weights, a natural family of receivers is the
generalized medians of Moulin (1980): the receiver moves to z = w0 x + (1 - w0) a, where a is the median
of the N actions and N - 1 fixed phantoms alpha_1 <= ... <= alpha_{N-1}. In one shot these are exactly
the anonymous strategy-proof rules for single-peaked preferences (with Pareto efficiency), and they
include the median receiver (phantoms (N - 1)/2 at 0 and (N - 1)/2 at 1).

**Proposition 7.** With equal weights, the averaging receiver of BIG and the generalized median
receiver with phantoms alpha_k = k/N have the same equilibria in target form, for every inertia w0
and discount g, and hence the same paths and the same losses.

*Proof.* In target form each influencer aims the next position at its target T_i; let
u_i = (T_i - w0 x) / (1 - w0) be the aggregate that would put the receiver there. Under averaging, the
round is BIG with targets u_i (Section "Many houses"): influencers with u_i > a play 1, those with
u_i < a play 0, and at most one with u_i = a plays inside, so with A = #{u_i > a} and B = #{u_i < a},
A / N <= a <= (N - B) / N. Among the 2N - 1 values {u_i clipped to [0, 1]} and {k / N}, the phantoms
above a number at most N - 1 - A (they are k / N > a >= A / N, so k >= A + 1), so at most N - 1
values lie above a; symmetrically at most N - 1 lie below. So a is their median, which is what the
generalized median receiver produces when every influencer reports its u_i, a dominant action because
its one-round objective is single-peaked in the next position (the assumption of target form). The
map from targets to next position is the same for both receivers, so are the fixed-point conditions
for the targets. ∎

For w0 = 0 this is the static statement that the average voting rule's equilibrium outcome is a
generalized median (Renault and Trannoy, 2005), and BIG's Theorem 4 is its form for the actions.
Numerically, the two receivers' targets and losses agree to machine precision, and the generalized
median equilibria pass the one-round deviation check (test
`test_the_mean_rule_is_the_generalized_median_with_evenly_spread_phantoms_and_the_median_rule_with_edge_phantoms`).

**Corollary (the phantoms are the speed).** While no influencer is pivotal, every influencer plays an
edge, and with A of them pushing up the aggregate is the A-th smallest phantom (alpha_0 = 0,
alpha_N = 1): the N - A zeros fill the bottom of the 2N - 1 values and the median is the next one.
So z = w0 x + (1 - w0) alpha_A. The phantoms fix both where the receiver settles (the generalized
median of the targets) and how fast it gets there: averaging moves with drift A / N - x, a tug of
war, and the median receiver at full speed, 0 or 1, as soon as a majority pushes one way. Choosing a
receiver means choosing this trade-off, which is what the study `mechanisms_phantoms` optimizes.

Without inertia the trade-off has a known answer: for squared distance and targets drawn uniformly,
the uniform median (phantoms k/N) is the welfare-maximizing strategy-proof rule (Jennings, Laraki,
Puppe and Varloot, 2021). With three influencers and phantoms (p, 1 - p) the expected summed loss is
(1 - 10p^3 + 30p^4 - 18p^5)/5, whose derivative -6p^2(3p - 1)(p - 1) makes p = 1/3 the unique
minimizer on [0, 1/2] (5/27 against 1/5 for the median receiver). So in the static game BIG's
averaging receiver, for all its exaggeration, is the best strategy-proof receiver.

**Proposition 8 (slow receivers should use the median).** Take N odd, equal weights, targets drawn
independently and uniformly on [0, 1], the start x uniform on [0, 1], and a fixed discount g. For a
generalized median receiver with phantoms alpha (any monotone alpha, not only symmetric ones), the
expected discounted summed loss is

  L(alpha; w0) = L_0(w0) + (1 - w0) (2N / (1 - g)^2) (sum_{k=1}^{N-1} c_k alpha_k + c_N) + O((1 - w0)^2),

where L_0(w0) does not depend on alpha and c_k = (N - 2k) / (2N(N + 2)). The first-order term is uniquely
minimized by alpha_k = 0 for k < N/2 and alpha_k = 1 for k > N/2, the median receiver. So as w0 tends
to 1 the welfare-best generalized median receiver tends to the median receiver, whereas at w0 = 0 it
is BIG's averaging receiver (above).

*Proof (first-order argument).*
1. *Influencers sit at edges.* As w0 -> 1 the position barely moves within the discount horizon, so an
   influencer's one-round objective tends uniformly to -(z - h_i)^2 / (1 - g) and its target T_i to h_i.
   The aggregate it would need, u_i = (T_i - w0 x)/(1 - w0), then diverges unless x = T_i, so every
   influencer reports 0 or 1 and, by the corollary, the aggregate is alpha_{A(x)} with A(x) the number of
   targets above x. Starts within O(1 - w0) of a target change this on a set of probability O(1 - w0),
   which only affects the O((1 - w0)^2) term.
2. *Expansion.* With e = 1 - w0, x_{k+1} = x_k + e (alpha_{A(x_k)} - x_k), so x_k = x + k e (alpha_{A(x)} - x)
   + O(k^2 e^2), and with l(x) = sum_i (x - h_i)^2,
   sum_k g^k l(x_{k+1}) = l(x) / (1 - g) + e l'(x) (alpha_{A(x)} - x) / (1 - g)^2 + O(e^2),
   using sum_k (k + 1) g^k = 1 / (1 - g)^2. Since l'(x) = 2N (x - hbar), the part that depends on alpha is
   e (2N / (1 - g)^2) E[(x - hbar) alpha_{A(x)}] = e (2N / (1 - g)^2) sum_k c_k alpha_k, with
   c_k = E[(x - hbar) 1{A(x) = k}].
3. *The coefficients.* Given x and A = k, the k targets above are uniform on (x, 1) and the others on
   (0, x), so E[hbar | x, A = k] = x/2 + k/(2N). Given A = k, x has the Beta(N - k + 1, k + 1) density
   with mean (N - k + 1)/(N + 2), and P(A = k) = 1/(N + 1). So
   c_k = (1/(N + 1)) ((N - k + 1)/(2(N + 2)) - k/(2N)) = (N - 2k) / (2N(N + 2)).
4. *The minimizer.* The objective is linear in alpha over the monotone set 0 <= alpha_1 <= ... <=
   alpha_{N-1} <= 1, whose extreme points are the thresholds alpha_k = 1{k >= m}, with value
   sum_{k >= m} c_k. Since c_k > 0 for k < N/2 and c_k < 0 for k > N/2 (never 0 for N odd), the tail sum is
   smallest exactly at m = (N + 1)/2. ∎

The remainder in step 2 is bounded for each fixed alpha but the argument does not make it uniform over
alpha, and the regime needs 1 - w0 small against 1 - g, not just small. Numerically (study
`mechanisms_limit`, three influencers, discount 0.9), at w0 = 0.99 and 0.995 the loss rises with p from
the median receiver p = 0, with slopes close to the first-order prediction (table `mechanisms limit`).
Together with the static result, the welfare-best strategy-proof receiver runs from BIG's averaging
receiver without inertia to the median receiver with slow adjustment; the study `mechanisms_phantoms`
traces the path in between.

## What the propositions do not cover

- **Interior one-shot actions.** If a player's one-shot action is strictly inside the box, its hold region is the
  single point d*, and off it that player pushes at an edge; the dog then returns faster than under holding, the
  continuation in Lemma 2 is wrong, and the other player's best target moves. In one random game
  (houses 0.754 and 0.538, w = 0.347, w0 = 0.767, g = 0.335) the dynamic program's red aims at about 0.785 where the
  policy of Proposition 2 aims at 0.771. The policy fails in 53 of 123 such random games. Because a player's one-round objective depends only on the next
  position, an equilibrium in target form still exists whenever the objectives stay single-peaked; with the
  targets from a two-dimensional fixed point (`target_equilibrium`), this holds numerically in 122 of the 123
  games. A proof is open.
- **Uniqueness.** The propositions give one feedback equilibrium; they do not rule out others.
- **Shared coordinates.** When the houses coincide, both one-shot actions can be interior and the one-shot game has a
  continuum of equilibria.
