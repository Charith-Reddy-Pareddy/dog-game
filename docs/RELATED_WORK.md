# Related work for the mechanism results (draft, in progress)

A starting list for the paper's related-work section on Sections 5.25 and 5.26 of the research report:
the mean and median rules when the outcome has inertia, and the welfare-maximizing inertia. Each entry
says what the work does, how ours differs, and how far it was checked. "Verified" means the title, venue
and main claim were confirmed from the publisher's page or abstract, not that the full paper was read.

## Mean against median aggregation (static)

- **Moulin (1980)**, On strategy-proofness and single peakedness, *Public Choice* 35, 437-455. The
  median and generalized median rules are strategy-proof for single-peaked preferences. *Ours:*
  persistence breaks this (Proposition 3). Verified.
- **Renault and Trannoy (2005)**, Protecting minorities through the average voting rule, *Journal of
  Public Economic Theory* 7, 169-199. The equilibrium outcome of the average rule is a generalized
  median. *Ours:* the one-shot dog game is this rule; we add inertia. Verified.
- **Rosar (2015)**, [Continuous decisions by a committee: median versus average
  mechanisms](https://ideas.repec.org/a/eee/jetheo/v159y2015ipap15-65.html), *Journal of Economic
  Theory* 159, 15-65. With private information and an optimally designed report set, the average
  beats the median for a utilitarian planner (uniform information, any number of agents). *Ours:*
  agrees with our ranking without inertia; with persistence the ranking reverses. The closest paper
  found so far; read in full. Verified (abstract).
- **Freeman, Pennock, Peters and Wortman Vaughan**, truthful aggregation of budget proposals (moving
  phantom mechanisms). Strategy-proof aggregation beyond one dimension; relevant if the paper goes
  multi-dimensional. *Not yet verified.*

## Dynamic collective choice and the status quo

- **Dziuda and Loeper (2016)**, Dynamic collective choice with endogenous status quo, *Journal of
  Political Economy* 124(4). Today's agreement is tomorrow's status quo, which creates inefficiency and
  status-quo inertia. *Ours:* inertia is a fixed physical feature of the outcome, not a voting rule,
  and we compare aggregation rules. Verified.
- **Duggan and Kalandrakis (2012)**, Dynamic legislative policy making, *Journal of Economic Theory*
  147(5), 1653-1688. Existence of stationary Markov perfect equilibria with an endogenous status quo,
  multidimensional policy and small shocks. *Ours:* a much simpler game with explicit equilibria.
  Verified (abstract).
- **Bowen, Chen and Eraslan (2014)**, Mandatory versus discretionary spending: the status quo effect,
  *American Economic Review* 104(10), 2941-2974. A persistent status quo (mandatory programs) can
  Pareto-dominate when parties are patient. *Ours:* a related "persistence can help" message, with
  noise, not power shifts, as the reason (Section 5.26). Verified (abstract).
- **de Keijzer and Wojtczak (2018)**, [Facility reallocation on the
  line](https://www.ijcai.org/proceedings/2018/26), IJCAI. A facility relocated each stage at a moving
  cost; optimal mechanisms are not strategy-proof. *Ours:* inertia instead of a moving cost, strategic
  play in a Markov game. Verified.

## Repeated strategic reporting and convergence

- **Meir, Polukarov, Rosenschein and Jennings (2010)**, [Convergence to equilibria in plurality
  voting](https://ojs.aaai.org/index.php/AAAI/article/view/7624), AAAI. Iterative voting: voters
  respond to the current outcome, and convergence depends on tie-breaking and update rules. *Ours:*
  forward-looking players and a continuous outcome with inertia. Verified.
- **Bindel, Kleinberg and Oren (2015)**, How bad is forming your own opinion?, *Games and Economic
  Behavior* (FOCS 2011). Price of anarchy of opinion averaging with intrinsic beliefs, at most 9/8 on
  undirected graphs. *Ours:* a price of anarchy for aggregation with inertia, with a planner that is
  limited by the same inertia. Verified (abstract).

## Smoothing and caution

- **Brainard (1967)**, Uncertainty and the effectiveness of policy, *American Economic Review* 57(2),
  411-425. Attenuation: with uncertainty about the effect of policy, respond more cautiously. *Ours:*
  the cooling-off result is a related trade-off (filtering noise against slow adjustment), from
  inertia in the outcome and with strategic players. Verified (secondary sources).

## Not yet covered

1. Aggregation rules (mean, median, generalized medians) when the outcome adjusts only partly each
   period, for instance in partial-adjustment or gradualism models of committees.
2. Differential games in the style of Fershtman and Kamien, for the continuous-time framing in
   [REVERSAL_NOTES.md](REVERSAL_NOTES.md).
3. Strategy-proofness in dynamic or repeated settings, in case an existing result already implies
   Proposition 3.
