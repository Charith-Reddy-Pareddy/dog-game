# Related work for the mechanism results (draft, in progress)

A starting list for the paper's related-work section on Sections 5.25 and 5.26 of the research report:
the mean and median rules when the outcome has inertia, and the welfare-maximizing inertia. Each entry
says what the work does, how ours differs, and how far it was checked. "Verified" means the title, venue
and main claim were confirmed from the publisher's page or abstract, not that the full paper was read.

## The source model

- **Wu, Zhu, Cai and Zhu (2025)**, [The Battling Influencers Game: Nash equilibria structure of a
  potential game and implications to value alignment](https://arxiv.org/abs/2502.01127), arXiv
  2502.01127. BIG is a potential game; at every pure equilibrium all influencers but at most one
  exaggerate to the boundary (Theorem 4); the conclusion names mechanism design against this
  exaggeration as future work. *Ours:* BIG with a persistent receiver, and the receiver as the
  mechanism (Proposition 7 and the study `mechanisms_phantoms`). Verified (read in full).

## Mean against median aggregation (static)

- **Moulin (1980)**, On strategy-proofness and single peakedness, *Public Choice* 35, 437-455. The
  median and generalized median rules are strategy-proof for single-peaked preferences. *Ours:*
  persistence breaks this (Proposition 3). Verified.
- **Border and Jordan (1983)**, Straightforward elections, unanimity and phantom voters, *Review of
  Economic Studies* 50(1), 153-170. Uncompromising (strategy-proof) choice rules are characterized through
  phantom voters. *Ours:* the corollary to Proposition 9 needs only this property, so compensating for inertia
  keeps every such rule truthful. Verified (abstract).
- **Renault and Trannoy (2005)**, Protecting minorities through the average voting rule, *Journal of
  Public Economic Theory* 7, 169-199. The equilibrium outcome of the average rule is a generalized
  median. *Ours:* the one-shot dog game is this rule; we add inertia. Verified.
- **Rosar (2015)**, [Continuous decisions by a committee: median versus average
  mechanisms](https://ideas.repec.org/a/eee/jetheo/v159y2015ipap15-65.html), *Journal of Economic
  Theory* 159, 15-65. With private information and an optimally designed report set, the average
  beats the median for a utilitarian planner (uniform information, any number of agents). *Ours:*
  agrees with our ranking without inertia; with persistence the ranking reverses. The closest paper
  found so far; read in full. Verified (abstract).
- **Freeman, Pennock, Peters and Wortman Vaughan (2021)**, [Truthful aggregation of budget
  proposals](https://arxiv.org/abs/1905.00457), *Journal of Economic Theory* 193, 105234. Moving
  phantom mechanisms; with two alternatives the independent markets mechanism is the uniform phantom
  rule, the unique anonymous, continuous and proportional one. *Ours:* the uniform phantom rule is
  BIG's averaging receiver (Proposition 7). Verified (abstract).

- **Walsh (2021)**, [Strategy proof mechanisms for facility location at limited
  locations](https://arxiv.org/abs/2009.07982), arXiv 2009.07982. Placing the facility at the feasible
  location nearest the median agent is strategy-proof (Theorem 3). *Ours:* the one-round step of
  Proposition 9 is this fact with the feasible set [w0 x, w0 x + 1 - w0]; the dynamic step is new as far as
  checked. Verified (abstract and Theorem 3).

- **Barberà, Massó and Neme (1997)**, [Voting under constraints](https://ideas.repec.org/a/eee/jetheo/v76y1997i2p298-321.html),
  *Journal of Economic Theory* 76(2), 298-321. Strategy-proof rules on a constrained feasible set. *Ours:* the
  feasible set of Proposition 11 changes with the state; the one-round step is the fixed-set fact. Verified
  (citation page; no abstract available).
- **Chandrasekher (2015)**, [Dynamically consistent voting rules](https://ideas.repec.org/a/eee/jetheo/v160y2015icp175-187.html),
  *Journal of Economic Theory* 160, 175-187. Strategy-proof and dynamically consistent voting rules when menus
  shrink over stages, on single-peaked domains. *Ours:* a persistent outcome and Markov equilibrium instead of
  shrinking menus. Verified (abstract).

## Welfare-optimal strategy-proof rules (static)

- **Jennings, Laraki, Puppe and Varloot (2021)**, [New characterizations of strategy-proofness under
  single-peakedness](https://arxiv.org/abs/2102.11686), arXiv 2102.11686. For squared distance and a
  uniform prior, the uniform (linear) median, phantoms at k/N, is the welfare-maximizing
  strategy-proof rule. *Ours:* so without inertia BIG's averaging receiver is welfare-optimal among
  strategy-proof receivers (our exact three-player formula, (1 - 10p^3 + 30p^4 - 18p^5)/5 with its
  minimum at p = 1/3, is a special case); the question is whether persistence changes the optimum.
  Verified (abstract and introduction).
- **Caragiannis, Procaccia and Shah (2016)**, [Truthful univariate
  estimators](https://proceedings.mlr.press/v48/caragiannis16.html), ICML. Truthful estimators of a
  population mean are generalized medians; worst-case optimal ones beat the median. Verified (abstract).

- **Gershkov, Moldovanu and Shi (2017)**, [Optimal voting rules](https://ideas.repec.org/a/oup/restud/v84y2017i2p688-717..html),
  *Review of Economic Studies* 84(2), 688-717. The utilitarian-optimal dominant-strategy mechanism with
  single-peaked or single-crossing preferences, implemented by sequential voting with alternative-specific
  thresholds. *Ours:* Proposition 13 shows the static optimum stays optimal for compensated receivers at every
  inertia. Verified (abstract).

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
  The full version (arXiv 2103.12894) shows that the optimal offline and online mechanisms, which depend on the
  previous location, are not strategy-proof (Corollary 4), and offers the median agent in every stage, which ignores
  history, as the strategy-proof mechanism (Theorem 4). *Ours:* Proposition 11 gives history-dependent receivers
  that keep truthful reporting an equilibrium. Verified (full text, Section 6).
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
3. Strategy-proofness in dynamic or repeated settings: checked for Proposition 11 (Chandrasekher 2015, de Keijzer
   and Wojtczak 2018 and its full version, Barberà, Massó and Neme 1997); none implies it, as far as checked.
   Google Scholar blocked automated searches, so the search used publisher pages, IDEAS/RePEc and arXiv.
