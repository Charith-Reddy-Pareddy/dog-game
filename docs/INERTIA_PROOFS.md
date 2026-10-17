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

## What the propositions do not cover

- **Interior one-shot actions.** If a player's one-shot action is strictly inside the box, its hold region is the
  single point d*, and off it that player pushes at an edge; the dog then returns faster than under holding, the
  continuation in Lemma 2 is wrong, and the other player's best target moves. In one random game
  (houses 0.754 and 0.538, w = 0.347, w0 = 0.767, g = 0.335) the dynamic program's red aims at about 0.785 where the
  policy of Proposition 2 aims at 0.771. The policy fails in 53 of 123 such random games; the equilibrium there is
  open.
- **Uniqueness.** The propositions give one feedback equilibrium; they do not rule out others.
- **Shared coordinates.** When the houses coincide, both one-shot actions can be interior and the one-shot game has a
  continuum of equilibria.
