"""Experiments that answer open research questions about the dog game.

Each study is a generator of (key, run) pairs; `run()` executes the ones missing
from the results file and appends each finished row at once, so a stopped run
keeps what it did and a rerun skips it. Run everything with
`python3 -m doggame.dog_game_studies`, or pick studies by name; add a last argument like `1/4`
to do only the second of four shards.

- algorithms:   REINFORCE vs A2C vs PPO, how fast each gets near the equilibrium
- walking:      the same three algorithms on the angle-radius (walking) game
- ppo_variants: PPO with fewer epochs or a smaller learning rate, on both games
- ppo_long:     the PPO settings trained three times longer on the walking game
- walking_entropy: default PPO on the walking game with an entropy bonus
- walking_clip: PPO on the walking game with a tighter clipping range
- continuous_q: Nash-Q with continuous moves on the walking game, three move encodings
- mechanisms: mean against median aggregation with inertia, outcome, exaggeration and welfare
- mechanisms_wide: the same welfare comparison on more draws, players and inertias
- mechanisms_robust: the same with more patient players and with 9 and 15 players
- mechanisms_phantoms: the welfare-best generalized median (phantom positions) at each inertia
- mechanisms_limit: Proposition 8 at inertia 0.99 and 0.995, where the median receiver should be best
- mechanisms_asymmetric: whether the welfare-best phantoms are symmetric (3 players, all pairs on a grid)
- mechanisms_skewed: with skewed targets the best slow receiver is an order statistic other than the median
- mechanisms_skewed_seven: the same with 7 players and Beta(0.1, 1) targets, where the predicted gap is larger
- mechanisms_compensated: receivers that compensate for inertia, under truthful reports, on the mechanisms_wide draws
- mechanisms_compensated_design: the welfare-best compensated receiver on fine phantom grids, 3, 5 and 7 players
- mechanisms_noise: the same with shocks on the outcome, to look for a welfare-maximizing inertia
- mechanisms_noise_planner: the optimal planner for those games
- continuous_q_tools: best-response methods, simultaneous gradient play and paths on those networks
- walking_settings: the walking game with a shorter or longer step and a lower or higher discount
- walking_houses: the three algorithms on three other house setups
- walking_inertia: A2C on the walking game with dog inertia
- walking_capacity: PPO on the walking game with a larger network
- walking_ppo_inertia: PPO on the walking game with dog inertia
- walking_hard_long: the three algorithms on the hardest walking setup for 900 episodes
- inertia_equilibrium: is the one-shot Nash profile an equilibrium of the main game with dog inertia?
- selection_clip: the selection study with actions and the dog left unclipped
- selection_linear: the same with an unbounded linear policy mean in y
- many_houses: the one-coordinate game with N houses, its one-shot dog and how much inertia it tolerates
- many_targets: feedback equilibria in target form with 3, 5 and 8 players
- feedback_learning: whether self-play from random dog positions learns a feedback equilibrium with inertia
- exact_play: noise-free gradient play on feedback policies, a table against a shared network
- exact_play_many: the same with 3 and 5 players
- soccer_exact_play: exact gradient play on the soccer game, a table against a network
- learning_fix: per-state best-response or equilibrium regression on the dog game, the duopoly and soccer
- learning_checks: smaller step sizes for the duopoly table, and fitting the soccer network to the equilibrium
- vertex_selection: whether a largest-total-value equilibrium selection changes the feedback equilibrium
- noisy_inertia: where the equilibrium switches when the dog gets a Gaussian shock, as the time step shrinks
- inertia_explicit: the closed-form hold condition and explicit feedback equilibrium against the dynamic program
- inertia_random: the explicit feedback equilibrium on random one-coordinate games
- target_random: the target fixed point on the same random games
- walking_equilibrium: is walking to the one-shot corners an equilibrium of the walking game?
- selection:    which equilibrium self-play finds when the houses share a coordinate
- cross_play:   setup 1 policies per layout, for pairing players from independently trained runs
- inertia:      does giving the dog momentum (w0) change where training ends up
- pretraining:  does starting from a policy trained on another setup help
- pretraining_mechanism: whether resetting exploration or de-saturating the policy removes the harm of pre-training
- stage_solver: how precisely the Nash-Q stage game has to be solved
- resolution:   how the action-grid resolution affects the discretized solution
- noise:        how much of fictitious play's mixing is just noise
"""

import json
import os
import sys

import numpy as np
import torch

from doggame.agents import TwoPlayerPolicy
from doggame.dog_game_study import WALK_HOUSES, walk_distance
from doggame.env import DogGameEnv, Square, make_inertial_transition
from doggame.experiments import DEFAULT_CONFIGS
from doggame.nash import solve_stage_nash
from doggame.polar import PolarDogGameEnv
from doggame.inertia_dp import policy as dp_policy, solve as solve_dp
from doggame.inertia_equilibrium import check_profile as check_inertia_profile
from doggame.walking_equilibrium import check_profile
from doggame.polar_agent import PolarPolicy, final_positions, train_polar, train_polar_actor_critic
from doggame.train import train_self_play
from doggame.train_ac import train_actor_critic
from doggame.verify import exploitability

RESULTS = "results/dog_game_studies.jsonl"
EPISODES = 1500
CONVERGED = 0.004  # exploitability below this counts as converged
DEGENERATE = 3  # the configuration whose houses share a y-coordinate
SLOW = 1  # the configuration where one run needed more than 1500 episodes


def make_env(config, w0=0.0):
    return DogGameEnv(**config, discount=0.9, transition_fn=make_inertial_transition(w0))


def learned_actions(policy, env):
    """The policy's mean action for each player at the starting state."""
    with torch.no_grad():
        mean_red, mean_blue = policy.means(torch.as_tensor(env.reset(), dtype=torch.float32).unsqueeze(0))
    return mean_red.squeeze(0).numpy(), mean_blue.squeeze(0).numpy()


def score(policy, env):
    """How far the learned policy is from the stage-game equilibrium, and how much either player could gain."""
    red, blue = learned_actions(policy, env)
    nash_red, nash_blue, _ = solve_stage_nash(env.house_red, env.house_blue, env.w, env.domain)
    gap = max(exploitability(red, blue, env.house_red, env.house_blue, env.w, env.domain))
    return dict(exploitability=float(gap), distance=float(np.linalg.norm(red - nash_red) + np.linalg.norm(blue - nash_blue)),
                red=red.tolist(), blue=blue.tolist())


def train_with_curve(env, policy, algorithm, episodes, checkpoints, seed, **options):
    """Train, scoring the policy at each checkpoint. Returns {episodes so far: exploitability}."""
    curve = {}

    def record(done):
        if done in checkpoints:
            curve[done] = score(policy, env)["exploitability"]

    if algorithm == "reinforce":
        train_self_play(env, policy, episodes=episodes, seed=seed, callback=record)
    else:
        train_actor_critic(env, policy, algorithm, episodes=episodes, seed=seed, callback=record, **options)
    return curve


def study_algorithms(seeds):
    checkpoints = (100, 250, 500, 1000, 1500)
    for config_index, config in enumerate(DEFAULT_CONFIGS):
        for algorithm in ("reinforce", "a2c", "ppo"):
            for seed in seeds:
                def run(config=config, algorithm=algorithm, seed=seed):
                    env = make_env(config)
                    policy = TwoPlayerPolicy(env.domain, architecture="separate", seed=seed)
                    return dict(curve=train_with_curve(env, policy, algorithm, EPISODES, checkpoints, seed))
                yield dict(config=config_index, algorithm=algorithm, seed=seed), run


def walking_run(algorithm, seed, episodes=300, every=25, max_step=0.2, discount=0.9, houses=WALK_HOUSES, w=0.5, w0=0.0, hidden=16, **options):
    """Train one walking-game policy, scoring it every `every` episodes (finely, because the walking policies'
    paths swing): how far the players end from the one-shot Nash corners, which do not depend on the step or the discount."""
    env = PolarDogGameEnv(*houses, w=w, w0=w0, max_step=max_step, discount=discount)
    torch.manual_seed(seed)
    policy = PolarPolicy(hidden=hidden, seed=seed)
    distance = lambda: walk_distance(env, lambda: final_positions(env, policy))
    start, curve = distance(), {}

    def record(done):
        if done % every == 0:
            curve[done] = distance()

    if algorithm == "reinforce":
        train_polar(env, policy, episodes=episodes, seed=seed, callback=record, **options)
    else:
        train_polar_actor_critic(env, policy, algorithm, episodes=episodes, seed=seed, callback=record, **options)
    return dict(start=start, curve=curve)


def study_walking(seeds):
    """REINFORCE, A2C and PPO on the walking game."""
    for algorithm in ("reinforce", "a2c", "ppo"):
        for seed in seeds:
            yield dict(algorithm=algorithm, seed=seed), lambda algorithm=algorithm, seed=seed: walking_run(algorithm, seed)


def study_ppo_variants(seeds):
    """Is PPO's speed from reusing each episode, and does a gentler PPO fix its instability on the walking game?"""
    checkpoints = (100, 250, 500, 1000, 1500)
    for epochs in (1, 2):  # 4 epochs, the default, is in the `algorithms` study
        for config_index, config in enumerate(DEFAULT_CONFIGS):
            for seed in seeds:
                def run(config=config, epochs=epochs, seed=seed):
                    env = make_env(config)
                    policy = TwoPlayerPolicy(env.domain, architecture="separate", seed=seed)
                    return dict(curve=train_with_curve(env, policy, "ppo", EPISODES, checkpoints, seed, ppo_epochs=epochs))
                yield dict(game="main", epochs=epochs, lr=0.01, config=config_index, seed=seed), run
    for lr, epochs in ((0.01, 1), (0.003, 4), (0.003, 1)):  # lr 0.01 with 4 epochs is in the `walking` study
        for seed in seeds:
            yield (dict(game="walking", epochs=epochs, lr=lr, seed=seed),
                   lambda lr=lr, epochs=epochs, seed=seed: walking_run("ppo", seed, lr=lr, ppo_epochs=epochs))


def study_ppo_long(seeds):
    """Do the slow or unstable PPO settings settle when trained three times longer (900 episodes)?"""
    for label, algorithm, options in (("a2c", "a2c", {}), ("ppo4", "ppo", {}), ("ppo1", "ppo", {"ppo_epochs": 1})):
        for seed in seeds:
            yield (dict(setting=label, seed=seed),
                   lambda algorithm=algorithm, seed=seed, options=options: walking_run(algorithm, seed, episodes=900, every=25, **options))


def study_walking_entropy(seeds):
    """Does an entropy bonus steady default PPO on the walking game? 900 episodes, as in `ppo_long`."""
    for coefficient in (0.0, 0.01, 0.03, 0.1):
        for seed in seeds:
            yield (dict(entropy=coefficient, seed=seed),
                   lambda coefficient=coefficient, seed=seed: walking_run("ppo", seed, episodes=900, every=25, entropy=coefficient))


def study_walking_clip(seeds):
    """Is PPO unstable on the walking game because four epochs on one episode move the policy too far? A tighter
    clipping range limits each update's step; 900 episodes, four epochs (the default clip, 0.2, is in `ppo_long`)."""
    for clip in (0.05, 0.1):
        for seed in seeds:
            yield dict(clip=clip, seed=seed), lambda clip=clip, seed=seed: walking_run("ppo", seed, episodes=900, every=25, clip=clip)


def _continuous_env():
    return PolarDogGameEnv(*WALK_HOUSES, w=0.5, max_step=0.2, discount=0.9)


def study_mechanisms(seeds):
    """Mean against median aggregation with inertia (doggame.mechanisms): 30 random draws of 3 or 5
    equally weighted houses per inertia. For each rule: the long-run outcome, the players' mean
    exaggeration there (|report - house|), the discounted summed loss over evenly spread starts
    against the planner's, and, on the first 5 draws, the one-round deviation gain of the
    target-form equilibrium and (median rule) of truthful reporting."""
    from doggame import mechanisms as M

    for n in (3, 5):
        for w0 in (0.0, 0.3, 0.6, 0.9):
            for draw in range(30):
                def run(n=n, w0=w0, draw=draw):
                    houses, weights = np.random.default_rng(100 * n + draw).random(n), np.full(n, 1 / n)
                    planner = M.social_loss(lambda x: M.planner_step(x, houses, w0), houses)
                    out = dict(houses=houses.tolist(), planner=planner)
                    for rule in M.RULES:
                        targets, _ = M.target_equilibrium(rule, houses, weights, w0)
                        move = lambda x, rule=rule, targets=targets: M.step(rule, x, targets, weights, w0)
                        x = M.steady_state(move, start=float(np.mean(houses)))
                        entry = dict(targets=targets.tolist(), outcome=x, loss=M.social_loss(move, houses),
                                     exaggeration=float(np.mean(np.abs(M.reports(rule, x, targets, weights, w0) - houses))))
                        if draw < 5:
                            policy = lambda v, rule=rule, targets=targets: M.reports(rule, v, targets, weights, w0)
                            entry["gain"] = M.deviation_gain(rule, policy, houses, weights, w0, states=11, grid=101, rounds=120)
                            if rule == "median":
                                entry["truthful_gain"] = M.deviation_gain(rule, lambda v: houses.copy(), houses, weights, w0, states=11, grid=101, rounds=120)
                        out[rule] = entry
                    return out
                yield dict(n=n, w0=w0, draw=draw), run


def study_mechanisms_wide(seeds):
    """The mean-against-median comparison on 100 draws of 3, 5 or 7 equally weighted houses at seven
    inertias (no equilibrium checks): each rule's discounted summed loss over evenly spread starts and
    its long-run loss, against the planner's."""
    from doggame import mechanisms as M

    for n in (3, 5, 7):
        for w0 in (0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9):
            for draw in range(100):
                def run(n=n, w0=w0, draw=draw):
                    houses, weights = np.random.default_rng(1000 * n + draw).random(n), np.full(n, 1 / n)
                    lr = lambda x: float(((x - houses) ** 2).sum())  # summed loss at a fixed outcome  # noqa: E731
                    out = dict(planner=M.social_loss(lambda x: M.planner_step(x, houses, w0), houses), planner_long=lr(houses.mean()))
                    for rule in M.RULES:
                        targets, _ = M.target_equilibrium(rule, houses, weights, w0)
                        move = lambda x, rule=rule, targets=targets: M.step(rule, x, targets, weights, w0)
                        x = M.steady_state(move, start=float(np.mean(houses)))
                        out[rule] = dict(loss=M.social_loss(move, houses), long=lr(x), outcome=x)
                    return out
                yield dict(n=n, w0=w0, draw=draw), run


def study_mechanisms_robust(seeds):
    """Robustness of the mean-against-median reversal (mechanisms_wide, same draws): more patient players
    (discount 0.95 and 0.99, 3, 5 or 7 players) and more players (9 and 15, discount 0.9), 100 draws at
    eight inertias. The horizon grows with patience so that the cut-off tail is below 1e-6; on the first
    3 draws the one-round deviation gain of each rule's equilibrium is checked where the horizon is at most
    300 rounds (not at discount 0.99, where the check is too slow)."""
    from doggame import mechanisms as M

    cells = [(n, g) for g in (0.95, 0.99) for n in (3, 5, 7)] + [(n, 0.9) for n in (9, 15)]
    for n, g in cells:
        rounds = int(np.ceil(np.log(1e-6) / np.log(g)))
        for w0 in (0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 0.95):
            for draw in range(100):
                def run(n=n, g=g, rounds=rounds, w0=w0, draw=draw):
                    houses, weights = np.random.default_rng(1000 * n + draw).random(n), np.full(n, 1 / n)
                    lr = lambda x: float(((x - houses) ** 2).sum())  # noqa: E731
                    out = dict(planner=M.social_loss(lambda x: M.planner_step(x, houses, w0), houses, g, rounds=rounds), planner_long=lr(houses.mean()))
                    for rule in M.RULES:
                        targets, it = M.target_equilibrium(rule, houses, weights, w0, g, grid=401, rounds=rounds)
                        move = lambda x, rule=rule, targets=targets: M.step(rule, x, targets, weights, w0)
                        x = M.steady_state(move, start=float(np.mean(houses)))
                        out[rule] = dict(loss=M.social_loss(move, houses, g, rounds=rounds), long=lr(x), outcome=x, iterations=it)
                        if draw < 3 and rounds <= 300:
                            policy = lambda v, rule=rule, targets=targets: M.reports(rule, v, targets, weights, w0)
                            out[rule]["gain"] = M.deviation_gain(rule, policy, houses, weights, w0, g, states=11, grid=101, rounds=rounds)
                    return out
                yield dict(n=n, discount=g, w0=w0, draw=draw), run


def study_mechanisms_phantoms(seeds):
    """Which generalized median (Moulin 1980: the median of the reports and N - 1 phantoms) minimizes the
    social loss at each inertia? Symmetric phantoms: (p, 1 - p) for 3 players, p from 0 (the median rule)
    to 0.5 (1/3 is the mean rule), 100 draws; (p1, p2, 1 - p2, 1 - p1) for 5 players on a 0.1 grid
    ((0.2, 0.4) is the mean rule, (0, 0) the median rule), 50 draws. Same draws as mechanisms_wide."""
    from doggame import mechanisms as M

    shapes = [(3, (p, 1 - p)) for p in np.round(np.arange(0, 0.5001, 0.025), 3)]
    shapes += [(5, (p1, p2, 1 - p2, 1 - p1)) for p1 in np.round(np.arange(0, 0.501, 0.1), 1) for p2 in np.round(np.arange(0, 0.501, 0.1), 1) if p1 <= p2]
    for n, phantoms in shapes:
        phantoms = tuple(float(round(v, 3)) for v in phantoms)
        for w0 in (0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 0.95):
            for draw in range(100 if n == 3 else 50):
                def run(n=n, phantoms=phantoms, w0=w0, draw=draw):
                    houses, weights = np.random.default_rng(1000 * n + draw).random(n), np.full(n, 1 / n)
                    targets, it = M.target_equilibrium(phantoms, houses, weights, w0, grid=401)
                    move = lambda x: M.step(phantoms, x, targets, weights, w0)  # noqa: E731
                    return dict(loss=M.social_loss(move, houses), outcome=M.steady_state(move, start=float(np.mean(houses))), iterations=it,
                                planner=M.social_loss(lambda x: M.planner_step(x, houses, w0), houses))
                yield dict(n=n, phantoms=list(phantoms), w0=w0, draw=draw), run


def study_mechanisms_compensated(seeds):
    """Inertia-compensated receivers (the next outcome as close to the generalized median of the reports and
    phantoms as the inertia allows) under truthful reports, on the draws of mechanisms_wide: the social loss of
    the compensated averaging (phantoms k/N) and median receivers, for 3 players also of the symmetric phantoms
    (p, 1 - p) on a 0.025 grid, and on the first 3 draws the largest one-round gain from not reporting truthfully."""
    from doggame import mechanisms as M

    for n in (3, 5, 7):
        averaging, median = tuple(k / n for k in range(1, n)), tuple([0.0] * (n // 2) + [1.0] * ((n - 1) // 2))
        for w0 in (0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9):
            for draw in range(100):
                def run(n=n, averaging=averaging, median=median, w0=w0, draw=draw):
                    houses, weights = np.random.default_rng(1000 * n + draw).random(n), np.full(n, 1 / n)
                    truthful = lambda x: houses.copy()  # noqa: E731
                    loss = lambda ph: M.social_loss(lambda x: M.compensated(x, np.tile(houses, (len(x), 1)), ph, w0), houses)  # noqa: E731
                    out = dict(averaging=loss(averaging), median=loss(median))
                    if n == 3:
                        out["grid"] = [loss((p, 1 - p)) for p in np.round(np.arange(0, 0.5001, 0.025), 3)]
                    if draw < 3:
                        out["gain"] = max(M.deviation_gain(ph, truthful, houses, weights, w0, states=11, grid=101, compensate=True) for ph in (averaging, median))
                    return out
                yield dict(n=n, w0=w0, draw=draw), run


def study_mechanisms_compensated_design(seeds):
    """The welfare-best compensated receiver under truthful reports: every symmetric phantom shape on a grid, plus
    exact averaging (phantoms k/N): (p, 1 - p) in steps of 0.005 for 3 players (discount 0.9 and 0.99, 1000 draws),
    (p1, p2, 1 - p2, 1 - p1) in steps of 0.025 for 5 players and (p1, p2, p3, ...) in steps of 0.05 for 7 players
    (discount 0.9, 500 draws), at eight inertias. One row per cell: each shape's mean social loss over the draws and
    the standard error of its paired difference from averaging."""
    from doggame import mechanisms as M

    def shapes(n):
        half = {3: [(p,) for p in np.round(np.arange(0, 0.5001, 0.005), 3)],
                5: [(a, b) for a in np.round(np.arange(0, 0.5001, 0.025), 3) for b in np.round(np.arange(0, 0.5001, 0.025), 3) if a <= b],
                7: [(a, b, c) for a in np.round(np.arange(0, 0.5001, 0.05), 2) for b in np.round(np.arange(0, 0.5001, 0.05), 2)
                    for c in np.round(np.arange(0, 0.5001, 0.05), 2) if a <= b <= c]}[n]
        half = [tuple(float(v) for v in h) for h in half] + [tuple(k / n for k in range(1, (n + 1) // 2))]  # last: averaging
        return [h + tuple(1 - v for v in reversed(h)) for h in half]

    for n, g, draws in ((3, 0.9, 1000), (3, 0.99, 1000), (5, 0.9, 500), (7, 0.9, 500)):
        rounds = 200 if g == 0.9 else int(np.ceil(np.log(1e-6) / np.log(g)))
        for w0 in (0.0, 0.3, 0.6, 0.75, 0.9, 0.95, 0.99, 0.995):
            def run(n=n, g=g, draws=draws, rounds=rounds, w0=w0):
                cells = shapes(n)
                loss = np.array([M.compensated_losses(np.random.default_rng(100_000 * n + d).random(n), cells, w0, g, rounds=rounds)
                                 for d in range(draws)])
                diff = loss - loss[:, -1:]
                return dict(shapes=[list(c) for c in cells], mean=list(loss.mean(0)), se=list(diff.std(0, ddof=1) / np.sqrt(draws)))
            yield dict(n=n, discount=g, w0=w0, draws=draws), run


def study_mechanisms_asymmetric(seeds):
    """Is the welfare-best generalized median symmetric? All phantom pairs 0 <= p1 <= p2 <= 1 on a 0.1 grid for
    3 players, at inertia 0, 0.6, 0.9 and 0.95, on the first 50 draws of mechanisms_phantoms."""
    from doggame import mechanisms as M

    pairs = [(p1, p2) for p1 in np.round(np.arange(0, 1.001, 0.1), 1) for p2 in np.round(np.arange(0, 1.001, 0.1), 1) if p1 <= p2]
    for p1, p2 in pairs:
        phantoms = (float(p1), float(p2))
        for w0 in (0.0, 0.6, 0.9, 0.95):
            for draw in range(50):
                def run(phantoms=phantoms, w0=w0, draw=draw):
                    houses, weights = np.random.default_rng(3000 + draw).random(3), np.full(3, 1 / 3)
                    targets, it = M.target_equilibrium(phantoms, houses, weights, w0, grid=401)
                    return dict(loss=M.social_loss(lambda x: M.step(phantoms, x, targets, weights, w0), houses), iterations=it)
                yield dict(phantoms=list(phantoms), w0=w0, draw=draw), run


def study_mechanisms_skewed(seeds):
    """Proposition 8 with skewed targets: 5 players with targets from Beta(0.2, 5), where the order statistic that
    best predicts the mean target is the 4th smallest, not the median. At inertia 0.99 and 0.995, 40 draws: the
    order-statistic receivers (phantoms (0, 0, 0, 1), (0, 0, 1, 1) = median, (0, 1, 1, 1)) and averaging."""
    from doggame import mechanisms as M

    receivers = [(0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0), (0.0, 1.0, 1.0, 1.0), (0.2, 0.4, 0.6, 0.8)]
    for w0 in (0.99, 0.995):
        for phantoms in receivers:
            for draw in range(40):
                def run(w0=w0, phantoms=phantoms, draw=draw):
                    houses, weights = np.random.default_rng(5000 + draw).beta(0.2, 5, 5), np.full(5, 0.2)
                    targets, it = M.target_equilibrium(phantoms, houses, weights, w0, grid=401, rounds=300, max_iter=600)
                    return dict(loss=M.social_loss(lambda x: M.step(phantoms, x, targets, weights, w0), houses, rounds=300), iterations=it)
                yield dict(w0=w0, phantoms=list(phantoms), draw=draw), run


def study_mechanisms_skewed_seven(seeds):
    """A stronger test of Proposition 8 with skewed targets: 7 players with targets from Beta(0.1, 1), where the
    predicted best slow receiver (the 5th smallest target) beats the median by about 0.026 at first order, much
    more than in mechanisms_skewed. Inertia 0.99, 60 draws, the median and the 5th-smallest receiver."""
    from doggame import mechanisms as M

    for phantoms in (M.order_statistic_phantoms(7, 4), M.order_statistic_phantoms(7, 5)):
        for draw in range(60):
            def run(phantoms=phantoms, draw=draw):
                houses, weights = np.random.default_rng(7000 + draw).beta(0.1, 1, 7), np.full(7, 1 / 7)
                targets, it = M.target_equilibrium(phantoms, houses, weights, 0.99, grid=401, rounds=300, max_iter=1500)
                return dict(loss=M.social_loss(lambda x: M.step(phantoms, x, targets, weights, 0.99), houses, rounds=300), iterations=it)
            yield dict(phantoms=list(phantoms), draw=draw), run


def study_mechanisms_limit(seeds):
    """Proposition 8 checked: at inertia 0.99 and 0.995 (slow against discount 0.9), the loss of phantoms
    (p, 1 - p) for 3 players, 40 draws, a longer horizon and more fixed-point iterations."""
    from doggame import mechanisms as M

    for w0 in (0.99, 0.995):
        for p in (0.0, 0.1, 0.2, 0.3, 0.325, 0.4, 0.5):
            for draw in range(40):
                def run(w0=w0, p=p, draw=draw):
                    houses, weights, phantoms = np.random.default_rng(3000 + draw).random(3), np.full(3, 1 / 3), (p, 1 - p)
                    targets, it = M.target_equilibrium(phantoms, houses, weights, w0, grid=401, rounds=300, max_iter=600)
                    return dict(loss=M.social_loss(lambda x: M.step(phantoms, x, targets, weights, w0), houses, rounds=300), iterations=it)
                yield dict(w0=w0, p=p, draw=draw), run


def study_mechanisms_noise(seeds):
    """Cooling off: the mean-against-median comparison with a shock on each next outcome (20 draws of 3
    or 5 equally weighted houses, seven inertias). The shock has standard deviation 0.1, 0.2 or 0.4 and
    comes with the new aggregate, so the inertia filters it, or 0.2 hitting the outcome itself. Each
    rule's equilibrium and the greedy planner's expected discounted summed loss (by quadrature)."""
    from doggame import mechanisms as M

    for n in (3, 5):
        for noise, filtered in ((0.1, True), (0.2, True), (0.4, True), (0.2, False)):
            for w0 in (0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9):
                for draw in range(20):
                    def run(n=n, noise=noise, filtered=filtered, w0=w0, draw=draw):
                        houses, weights = np.random.default_rng(1000 * n + draw).random(n), np.full(n, 1 / n)
                        shock = M.shocks(noise, w0, filtered)
                        out = dict(planner=M.social_loss(lambda x: M.planner_step(x, houses, w0), houses, shock=shock))
                        for rule in M.RULES:
                            targets, it = M.target_equilibrium(rule, houses, weights, w0, grid=401, shock=shock)
                            out[rule] = dict(loss=M.social_loss(lambda x: M.step(rule, x, targets, weights, w0), houses, shock=shock), iterations=it)
                        return out
                    yield dict(n=n, noise=noise, filtered=filtered, w0=w0, draw=draw), run


def study_mechanisms_noise_planner(seeds):
    """The optimal planner for every game of mechanisms_noise (with clipped shocks the greedy planner is not optimal)."""
    from doggame import mechanisms as M

    for key, _ in study_mechanisms_noise(seeds):
        def run(n=key["n"], noise=key["noise"], filtered=key["filtered"], w0=key["w0"], draw=key["draw"]):
            houses = np.random.default_rng(1000 * n + draw).random(n)
            shock = M.shocks(noise, w0, filtered)
            target = M.planner_target(houses, w0, shock=shock)
            return dict(target=target, loss=M.social_loss(lambda x: M.next_median(x, [target], w0), houses, shock=shock))
        yield key, run


def study_walking_settings(seeds):
    """A2C on the walking game with a shorter or longer maximum step, or a lower or higher discount."""
    for max_step, discount in ((0.1, 0.9), (0.4, 0.9), (0.2, 0.5), (0.2, 0.99)):
        for seed in seeds:
            yield (dict(max_step=max_step, discount=discount, seed=seed),
                   lambda max_step=max_step, discount=discount, seed=seed: walking_run("a2c", seed, max_step=max_step, discount=discount))


def study_walking_houses(seeds):
    """The three algorithms on the walking game with three other house setups (the first three of DEFAULT_CONFIGS)."""
    for setup in (0, 1, 2):
        config = DEFAULT_CONFIGS[setup]
        for algorithm in ("reinforce", "a2c", "ppo"):
            for seed in seeds:
                yield (dict(setup=setup, algorithm=algorithm, seed=seed),
                       lambda config=config, algorithm=algorithm, seed=seed:
                       walking_run(algorithm, seed, houses=(config["house_red"], config["house_blue"]), w=config["w"]))


def study_walking_inertia(seeds):
    """A2C on the walking game when the dog keeps part of its old position each round (w0 of 0.3 and 0.6)."""
    for w0 in (0.3, 0.6):
        for seed in seeds:
            yield dict(w0=w0, seed=seed), lambda w0=w0, seed=seed: walking_run("a2c", seed, w0=w0)


def study_walking_equilibrium(seeds):
    """Is walking to the one-shot corners an equilibrium of the walking game? Deterministic, so `seeds` is unused."""
    setups = [("default", WALK_HOUSES, 0.5)] + [(i, (c["house_red"], c["house_blue"]), c["w"]) for i, c in enumerate(DEFAULT_CONFIGS[:3])]
    for setup, houses, w in setups:
        for start in ("houses", "corners"):
            def run(houses=houses, w=w, start=start):
                nash_red, nash_blue, _ = solve_stage_nash(houses[0], houses[1], w, Square())
                result = check_profile(houses, w, (nash_red, nash_blue), starts=(nash_red, nash_blue) if start == "corners" else None)
                return {player: {k: float(v) for k, v in values.items()} for player, values in result.items()}
            yield dict(setup=setup, start=start), run


def study_walking_capacity(seeds):
    """PPO on the walking game with a larger network (64 hidden units instead of 16), with 4 epochs and with 1."""
    for epochs in (4, 1):
        for seed in seeds:
            yield dict(hidden=64, epochs=epochs, seed=seed), lambda epochs=epochs, seed=seed: walking_run("ppo", seed, hidden=64, ppo_epochs=epochs)


def study_walking_ppo_inertia(seeds):
    """PPO on the walking game when the dog keeps part of its old position (w0 of 0.3 and 0.6)."""
    for w0 in (0.3, 0.6):
        for seed in seeds:
            yield dict(w0=w0, seed=seed), lambda w0=w0, seed=seed: walking_run("ppo", seed, w0=w0)


def study_walking_hard_long(seeds):
    """The three algorithms on the hardest walking setup, (0.7, 0.1) / (0.6, 0.9) with w = 0.7, for 900 episodes."""
    config = DEFAULT_CONFIGS[2]
    for algorithm in ("reinforce", "a2c", "ppo"):
        for seed in seeds:
            yield (dict(algorithm=algorithm, seed=seed),
                   lambda algorithm=algorithm, seed=seed: walking_run(algorithm, seed, episodes=900, houses=(config["house_red"], config["house_blue"]), w=config["w"]))


def study_inertia_equilibrium(seeds):
    """Is the one-shot Nash profile an equilibrium of the main game with dog inertia? Deterministic, so `seeds` is unused."""
    for setup, config in enumerate(DEFAULT_CONFIGS):
        for w0 in (0.3, 0.6, 0.9):
            def run(config=config, w0=w0):
                nash_red, nash_blue, _ = solve_stage_nash(config["house_red"], config["house_blue"], config["w"], Square(0.0, 1.0))
                result = check_inertia_profile((config["house_red"], config["house_blue"]), config["w"], w0, (nash_red, nash_blue))
                return {player: {k: float(v) for k, v in values.items()} for player, values in result.items()}
            yield dict(setup=setup, w0=w0), run


def study_selection_noise(seeds):
    """Does the initial exploration noise decide which equilibrium self-play finds when the houses share a y-coordinate?
    50 seeds per noise level; the default (log-std -0.5) seeds 0-9 repeat the `selection` study's separate layout."""
    for log_std in (-2.0, -1.0, -0.5, 0.0):
        for seed in range(50):
            def run(log_std=log_std, seed=seed):
                env = make_env(DEFAULT_CONFIGS[DEGENERATE])
                policy = TwoPlayerPolicy(env.domain, architecture="separate", seed=seed, init_log_std=log_std)
                train_self_play(env, policy, episodes=EPISODES, seed=seed)
                return score(policy, env)
            yield dict(log_std=log_std, seed=seed), run


class Unclipped(Square):
    """The unit square for the policy's means, but sampled actions and the dog are left unclipped."""

    def clip(self, point):
        return np.asarray(point, dtype=float)


def study_selection_clip(seeds):
    """The selection study without clipping: if self-play still ends on the boundary, clipping is not why."""
    for seed in range(50):
        def run(seed=seed):
            env = DogGameEnv(**DEFAULT_CONFIGS[DEGENERATE], domain=Unclipped(), discount=0.9, transition_fn=make_inertial_transition(0.0))
            policy = TwoPlayerPolicy(env.domain, architecture="separate", seed=seed)
            train_self_play(env, policy, episodes=EPISODES, seed=seed)
            red, blue = learned_actions(policy, env)
            return dict(red=red.tolist(), blue=blue.tolist(), log_std=[policy.log_std_red.detach().tolist(), policy.log_std_blue.detach().tolist()])
        yield dict(seed=seed), run


def study_inertia_dp(seeds):
    """The exact feedback equilibrium of the inertia game (doggame.inertia_dp), compared with holding the one-shot Nash
    actions: on the path from the starting dog position, and over all dog positions. Deterministic, so `seeds` is unused."""
    for setup, config in enumerate(DEFAULT_CONFIGS):
        for w0 in (0.3, 0.6, 0.9):
            def run(config=config, w0=w0):
                houses = (np.asarray(config["house_red"]), np.asarray(config["house_blue"]))
                nash_red, nash_blue, _ = solve_stage_nash(houses[0], houses[1], config["w"], Square(0.0, 1.0))
                solution = solve_dp(houses, config["w"], w0)
                dog, path_gap = config["w"] * houses[0] + (1 - config["w"]) * houses[1], 0.0
                for _ in range(30):  # the equilibrium path from the starting dog position
                    red, blue = dp_policy(solution, dog)
                    path_gap = max(path_gap, float(np.abs(red - nash_red).max()), float(np.abs(blue - nash_blue).max()))
                    dog = w0 * dog + (1 - w0) * (config["w"] * red + (1 - config["w"]) * blue)
                differs = [float(np.mean((np.abs(p["red"] - nash_red[c]) > 1e-9) | (np.abs(p["blue"] - nash_blue[c]) > 1e-9))) for c, p in enumerate(solution)]
                return dict(path_gap=path_gap, differs=differs, unresolved=[p["unresolved"] for p in solution], sweeps=[p["sweeps"] for p in solution])
            yield dict(setup=setup, w0=w0), run


def study_markov_potential(seeds):
    """4-cycle sums of value changes (zero in every Markov potential game), with policies that ignore the dog and with
    policies that react to it, at four inertias. Deterministic, so `seeds` is unused."""
    from doggame.markov_potential import cycle_sum

    c = DEFAULT_CONFIGS[0]
    houses, w = (c["house_red"], c["house_blue"]), c["w"]
    start = w * np.asarray(houses[0]) + (1 - w) * np.asarray(houses[1])
    const = lambda a: (lambda x: np.asarray(a, dtype=float))
    red, blue = const((1.0, 0.0)), const((0.0, 1.0))
    alternatives = dict(fixed=(const((0.6, 0.4)), const((0.3, 0.7))),
                        reacting=(lambda x: np.clip(1.0 - x, 0, 1), lambda x: np.clip(x, 0, 1)))
    for w0 in (0.0, 0.3, 0.6, 0.9):
        for kind, (red_alt, blue_alt) in alternatives.items():
            yield (dict(w0=w0, policies=kind),
                   lambda w0=w0, red_alt=red_alt, blue_alt=blue_alt: dict(cycle=cycle_sum(red, red_alt, blue, blue_alt, houses, w, w0, start)))


def study_noisy_inertia(seeds):
    """Setup 3's x coordinate as a time-discretized continuous-time game (w0 = 1 - dt, discount exp(-0.5 dt)) with a
    Gaussian shock of standard deviation sigma sqrt(dt) on the dog: where each player's equilibrium action switches.
    The dog is clipped to [0, 1] like the environment, or lives on [-0.5, 1.5] to rule out the clipping."""
    from doggame.inertia_dp import solve_1d

    variants = [(0.05, sigma, True) for sigma in (0.0, 0.1, 0.2, 0.4)] + [(0.025, sigma, True) for sigma in (0.0, 0.2)]
    variants += [(0.05, sigma, False) for sigma in (0.0, 0.02, 0.05, 0.1, 0.2, 0.4)]
    for dt, sigma, clipped in variants:
        def run(dt=dt, sigma=sigma, clipped=clipped):
            bounds = dict() if clipped else dict(low=-0.5, high=1.5)
            sol = solve_1d(0.7, 0.6, 0.7, 1 - dt, discount=float(np.exp(-0.5 * dt)), states=201 if clipped else 401, actions=21,
                           tol=1e-8, max_iter=8000, noise=sigma * np.sqrt(dt), **bounds)
            z, inside = sol["z"], (sol["z"] >= 0) & (sol["z"] <= 1)
            switch = lambda a: float(z[inside][np.argmax(a[inside] < 0.5)])
            return dict(red=switch(sol["red"]), blue=switch(sol["blue"]), sweeps=sol["sweeps"], unresolved=sol["unresolved"])
        yield dict(dt=dt, sigma=sigma, clipped=clipped), run


def study_vertex_selection(seeds):
    """Does picking each state's pure equilibrium with the largest total value ("vertex Nash") change the feedback
    equilibrium of the game with inertia? Setup 4's shared y, setup 3's x and setup 1's y, at inertia 0.6 and 0.9."""
    from doggame.inertia_dp import solve_1d

    for name, game in (("setup 4 y", (0.5, 0.5, 0.5)), ("setup 3 x", (0.7, 0.6, 0.7)), ("setup 1 y", (0.2, 0.8, 0.5))):
        for w0 in (0.6, 0.9):
            def run(game=game, w0=w0):
                first, vertex = (solve_1d(*game, w0, states=201, actions=51, selection=s) for s in ("first", "vertex"))
                return dict(first=float((first["v_red"] + first["v_blue"]).mean()), vertex=float((vertex["v_red"] + vertex["v_blue"]).mean()),
                            differ=float(np.mean((np.abs(first["red"] - vertex["red"]) > 1e-9) | (np.abs(first["blue"] - vertex["blue"]) > 1e-9))),
                            value_gap=float(max(np.abs(first["v_red"] - vertex["v_red"]).max(), np.abs(first["v_blue"] - vertex["v_blue"]).max())))
            yield dict(coordinate=name, w0=w0), run


def study_exact_play(seeds):
    """Exact gradient play (no sampling noise, doggame.exact_play) on one coordinate: a table of actions against a small
    shared network, on setup 3's x at inertia 0.6 and setup 5's x at inertia 0.9."""
    from doggame.exact_play import play, score

    for setup, w0 in ((2, 0.6), (4, 0.9)):
        c = DEFAULT_CONFIGS[setup]
        game = (c["house_red"][0], c["house_blue"][0], c["w"], w0)
        variants = [("table", squash, 16, 0) for squash in ("sigmoid", "clamp")] + [("network", "sigmoid", hidden, seed) for hidden in (16, 64) for seed in seeds]
        for kind, squash, hidden, seed in variants:
            def run(game=game, kind=kind, squash=squash, hidden=hidden, seed=seed):
                return score(*play(*game, kind=kind, squash=squash, hidden=hidden, seed=seed), *game)
            yield dict(setup=setup, w0=w0, kind=kind, squash=squash, hidden=hidden, seed=seed), run


def study_exact_play_many(seeds):
    """Exact gradient play with 3 and 5 players (doggame.exact_play.play_many), five random games each (uniform houses,
    random weights, inertia 0.6): a table of actions against one small network per player, scored by the largest
    one-round deviation gain, next to holding the one-shot actions and the target-form equilibrium."""
    from doggame.exact_play import GRID, deviation_gain_many, play_many
    from doggame.many_houses import one_shot, target_equilibrium

    for n in (3, 5):
        for game in range(5):
            rng = np.random.default_rng(2000 * n + game)
            houses, weights = rng.random(n), rng.dirichlet(np.ones(n))
            for kind in ("holding", "target", "table", "network"):
                def run(houses=houses, weights=weights, kind=kind):
                    xs = np.linspace(0, 1, GRID)
                    if kind == "holding":
                        actions = np.tile(one_shot(houses, weights)[0][:, None], (1, GRID))
                    elif kind == "target":
                        targets, _ = target_equilibrium(houses, weights, 0.6)
                        actions = np.stack([one_shot((targets - 0.6 * x) / 0.4, weights)[0] for x in xs], 1)
                    else:
                        actions = play_many(houses, weights, 0.6, kind)
                    return dict(gain=deviation_gain_many(actions, houses, weights, 0.6))
                yield dict(n=n, game=game, kind=kind), run


def study_soccer_exact_play(seeds):
    """A second game for exact gradient play (doggame.soccer_exact_play): the 7 x 5 soccer game, whose equilibrium is
    pure in every state, with a table of logits against a network per player (2000 Adam steps)."""
    from doggame.soccer import Soccer
    from doggame.soccer_exact_play import play

    for policy, seed in [("table", 0)] + [("network", s) for s in seeds]:
        def run(policy=policy, seed=seed):
            rows = play(Soccer(), policy, seed=seed)
            return dict(curve=[list(r) for r in rows])
        yield dict(policy=policy, seed=seed), run


def study_learning_fix(seeds):
    """A fix for learning feedback equilibria on three games, five seeds each: the dog game (setup 3's x at
    inertia 0.6, setup 5's x at 0.9) and the sticky-price duopoly (inertia 0.8) with best-response
    regression, soccer with equilibrium regression on value-iteration (or the current policies')
    continuation values; the duopoly also with plain exact gradient play, a table and networks."""
    from doggame import duopoly_learning, exact_play, soccer_exact_play
    from doggame.soccer import Soccer

    seeds = range(5)
    for setup, w0 in ((2, 0.6), (4, 0.9)):
        c = DEFAULT_CONFIGS[setup]
        game = (c["house_red"][0], c["house_blue"][0], c["w"], w0)
        for kind, seed in [("table", 0)] + [("network", s) for s in seeds]:
            def run(game=game, kind=kind, seed=seed):
                return exact_play.score(*exact_play.play(*game, kind=kind, seed=seed, steps=1500, fix=True), *game)
            yield dict(game=f"dog setup {setup + 1}", kind=kind, fix="best response", seed=seed), run
    for kind, fix, seed in [("table", None, 0), ("table", "best response", 0)] + [("network", f, s) for f in (None, "best response") for s in seeds]:
        def run(kind=kind, fix=fix, seed=seed):
            steps = 1500 if fix else 3000
            return duopoly_learning.score(duopoly_learning.play(0.8, kind, seed=seed, steps=steps, fix=bool(fix)), 0.8)
        yield dict(game="duopoly", kind=kind, fix=fix, seed=seed), run
    for kind, fix, seed in [("table", "policy", 0), ("network", "policy", 0)] + [("network", "value", s) for s in seeds]:
        def run(kind=kind, fix=fix, seed=seed):
            return dict(curve=[list(r) for r in soccer_exact_play.play(Soccer(), kind, seed=seed, steps=4000, every=500, fix=fix)])
        yield dict(game="soccer", kind=kind, fix=fix, seed=seed), run


def study_learning_checks(seeds):
    """Two checks for learning_fix: plain exact gradient play with a table on the duopoly at smaller step sizes
    and twice the steps, and whether the soccer network can represent the equilibrium at all (fitted to it
    directly, 64 and 256 units)."""
    from doggame import duopoly_learning
    from doggame.soccer import Soccer
    from doggame.soccer_exact_play import fit
    from doggame.soccer_solver import solve_soccer

    for lr in (0.01, 0.003, 0.001):
        yield dict(check="duopoly table", lr=lr), lambda lr=lr: duopoly_learning.score(duopoly_learning.play(0.8, "table", steps=6000, lr=lr), 0.8)
    for hidden in (64, 256):
        def run(hidden=hidden):
            game = Soccer()
            return fit(game, solve_soccer(game), hidden=hidden)
        yield dict(check="soccer fit", hidden=hidden), run


def study_feedback_learning(seeds):
    """Self-play from random dog positions on setup 3 with inertia 0.6: the learned policy's feedback exploitability
    (doggame.feedback_learning), next to Proposition 2's policy and holding the one-shot actions. Gaussian and Beta
    policies, 3000 episodes, and A2C for 15000; and setup 5 at inertia 0.9."""
    from doggame.feedback_learning import explicit_2d, feedback_gain, holding_2d, learn, mean_policy

    config, w0 = DEFAULT_CONFIGS[2], 0.6
    for name, act in (("explicit", lambda: explicit_2d(config, w0)), ("holding", lambda: holding_2d(config))):
        yield dict(algorithm=name, seed=0), lambda act=act: dict(zip(("max_gain", "mean_gain"), feedback_gain(act(), config, w0)))
    variants = [(dict(), algorithm, seed, 3000, "gaussian") for algorithm in ("reinforce", "a2c", "ppo") for seed in seeds]
    variants += [(dict(family="beta"), algorithm, seed, 3000, "beta") for algorithm in ("reinforce", "a2c", "ppo") for seed in seeds]
    variants += [(dict(family=family, episodes=15000), "a2c", seed, 15000, family) for family in ("gaussian", "beta") for seed in range(3)]
    for extra, algorithm, seed, episodes, family in variants:
        def run(algorithm=algorithm, seed=seed, episodes=episodes, family=family):
            policy = learn(config, w0, algorithm, seed, episodes, family)
            return dict(zip(("max_gain", "mean_gain"), feedback_gain(mean_policy(policy), config, w0)))
        yield dict(algorithm=algorithm, seed=seed, **extra), run
    big = dict(setup=4, w0=0.9)  # setup 5 at w0 = 0.9 has the largest off-path incentive of the five setups
    for name, act in (("explicit", lambda: explicit_2d(DEFAULT_CONFIGS[4], 0.9)), ("holding", lambda: holding_2d(DEFAULT_CONFIGS[4]))):
        yield dict(algorithm=name, seed=0, **big), lambda act=act: dict(zip(("max_gain", "mean_gain"), feedback_gain(act(), DEFAULT_CONFIGS[4], 0.9)))
    for algorithm in ("reinforce", "a2c", "ppo"):
        for seed in seeds:
            def run(algorithm=algorithm, seed=seed):
                policy = learn(DEFAULT_CONFIGS[4], 0.9, algorithm, seed, 3000)
                return dict(zip(("max_gain", "mean_gain"), feedback_gain(mean_policy(policy), DEFAULT_CONFIGS[4], 0.9)))
            yield dict(algorithm=algorithm, seed=seed, **big), run


def study_many_houses(seeds):
    """The one-coordinate game with N houses drawn from Beta(2, 5) and equal weights, 200 draws per N: where the
    one-shot dog lands, whether a pivot ends inside the box, and the largest inertia (discount 0.9) at which holding
    the one-shot actions stays Markov-perfect (doggame.many_houses)."""
    from doggame.many_houses import critical_inertia, one_shot

    for n in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
        def run(n=n):
            rng, out = np.random.default_rng(n), dict(dog=[], pivot=[], critical=[])
            for _ in range(200):
                houses, weights = rng.beta(2, 5, n), np.full(n, 1 / n)
                _, d_star, pivot = one_shot(houses, weights)
                out["dog"].append(d_star), out["pivot"].append(pivot is not None), out["critical"].append(critical_inertia(houses, weights))
            return out
        yield dict(n=n), run


def study_many_targets(seeds):
    """The many-house target fixed point (doggame.many_houses.target_equilibrium) on 20 random games for each of 3, 5
    and 8 players (uniform houses, random weights, inertia and discount): is aiming at the targets a feedback equilibrium,
    and would holding the one-shot actions have been?"""
    from doggame.many_houses import critical_inertia, deviation_gain, objective_peaks, target_equilibrium

    for n in (3, 5, 8):
        for game in range(20):
            def run(n=n, game=game):
                rng = np.random.default_rng(1000 * n + game)
                houses, weights, w0, g = rng.random(n), rng.dirichlet(np.ones(n)), rng.uniform(0.1, 0.9), rng.uniform(0.3, 0.95)
                targets, iterations = target_equilibrium(houses, weights, w0, g)
                return dict(gain=deviation_gain(targets, houses, weights, w0, g), iterations=iterations,
                            holds=bool(w0 <= critical_inertia(houses, weights, g)), peaks=max(objective_peaks(targets, houses, weights, w0, g)))
            yield dict(n=n, game=game), run


class LinearY(TwoPlayerPolicy):
    """Sigmoid means in x as usual, but an unbounded linear mean in y (0.5 plus the network's output)."""

    def means(self, state):
        (red, blue), (raw_red, raw_blue) = super().means(state), self.network(state)
        return torch.cat([red[..., :1], 0.5 + raw_red[..., 1:]], -1), torch.cat([blue[..., :1], 0.5 + raw_blue[..., 1:]], -1)


def study_selection_linear(seeds):
    """The no-clipping ablation with a linear mean in y: if self-play still ends on the boundary, the sigmoid is not why."""
    for seed in range(50):
        def run(seed=seed):
            env = DogGameEnv(**DEFAULT_CONFIGS[DEGENERATE], domain=Unclipped(), discount=0.9, transition_fn=make_inertial_transition(0.0))
            policy = LinearY(env.domain, architecture="separate", seed=seed)
            train_self_play(env, policy, episodes=EPISODES, seed=seed)
            red, blue = learned_actions(policy, env)
            return dict(red=red.tolist(), blue=blue.tolist())
        yield dict(seed=seed), run


def study_inertia_explicit(seeds):
    """The closed-form hold share and the explicit feedback equilibrium (doggame.inertia_theory) against the dynamic
    program, per coordinate, with the explicit policy's largest one-round deviation gain over 101 dog positions."""
    from doggame.inertia_dp import solve_1d
    from doggame.inertia_theory import deviation_gain, explicit_policy, hold_share, nash_1d

    for setup, config in enumerate(DEFAULT_CONFIGS):
        for w0 in (0.3, 0.6, 0.9):
            def run(config=config, w0=w0):
                w, out = config["w"], dict(kind=[], theory=[], dp=[], gap=[], gain=[])
                for c in range(2):
                    h_red, h_blue = config["house_red"][c], config["house_blue"][c]
                    a_red, a_blue, kind = nash_1d(h_red, h_blue, w)
                    act, sol = explicit_policy(h_red, h_blue, w, a_red, a_blue, w0), solve_1d(h_red, h_blue, w, w0)
                    red, blue = act(sol["z"])
                    out["kind"].append(kind)
                    out["theory"].append(hold_share(([h_red], [h_blue]), w, w0, ([a_red], [a_blue]))[0])
                    out["dp"].append(float(np.mean((np.abs(sol["red"] - a_red) > 1e-9) | (np.abs(sol["blue"] - a_blue) > 1e-9))))
                    out["gap"].append(float(max(np.abs(red - sol["red"]).max(), np.abs(blue - sol["blue"]).max())))
                    out["gain"].append(max(max(deviation_gain(act, h_red, w, w0, x, True), deviation_gain(act, h_blue, w, w0, x, False))
                                           for x in np.linspace(0, 1, 101)))
                return out
            yield dict(setup=setup, w0=w0), run


def study_target_random(seeds):
    """The target fixed point (doggame.inertia_theory.target_equilibrium) on the same 200 random one-coordinate games
    as `inertia_random`: its largest one-round deviation gain over 41 dog positions and the number of iterations."""
    from doggame.inertia_theory import deviation_gain, nash_1d, target_equilibrium, target_policy

    for game in range(200):
        def run(game=game):
            rng = np.random.default_rng(game)
            h_red, h_blue, w, w0, g = *rng.random(2), rng.uniform(0.05, 0.95), rng.uniform(0.01, 0.97), rng.uniform(0.05, 0.99)
            targets, iterations = target_equilibrium(h_red, h_blue, w, w0, g)
            act = target_policy(*targets, w, w0)
            gain = max(max(deviation_gain(act, h_red, w, w0, x, True, g, rounds=400, grid=1001), deviation_gain(act, h_blue, w, w0, x, False, g, rounds=400, grid=1001))
                       for x in np.linspace(0, 1, 41))
            return dict(kind=nash_1d(h_red, h_blue, w)[2], gain=gain, iterations=iterations, targets=[float(t) for t in targets])
        yield dict(game=game), run


def study_inertia_random(seeds):
    """The explicit feedback equilibrium on 200 random one-coordinate games (houses, weight, inertia and discount drawn
    uniformly): its largest one-round deviation gain over 41 dog positions, by the kind of one-shot equilibrium."""
    from doggame.inertia_theory import deviation_gain, explicit_policy, holds_everywhere, nash_1d

    for game in range(200):
        def run(game=game):
            rng = np.random.default_rng(game)
            h_red, h_blue, w, w0, g = *rng.random(2), rng.uniform(0.05, 0.95), rng.uniform(0.01, 0.97), rng.uniform(0.05, 0.99)
            a_red, a_blue, kind = nash_1d(h_red, h_blue, w)
            act = explicit_policy(h_red, h_blue, w, a_red, a_blue, w0, g)
            gain = max(max(deviation_gain(act, h_red, w, w0, x, True, g, rounds=400, grid=1001), deviation_gain(act, h_blue, w, w0, x, False, g, rounds=400, grid=1001))
                       for x in np.linspace(0, 1, 41))
            return dict(kind=kind, gain=gain, holds=holds_everywhere(([h_red], [h_blue]), w, w0, ([a_red], [a_blue]), g))
        yield dict(game=game), run


def study_cross_play(seeds):
    """Self-play policies on setup 1 (a unique equilibrium) for each layout, kept so that red from one run can be
    paired with blue from another; the shared-y setup's runs are in the `selection` study."""
    for architecture in ("separate", "shared", "partial"):
        for seed in seeds:
            def run(architecture=architecture, seed=seed):
                env = make_env(DEFAULT_CONFIGS[0])
                policy = TwoPlayerPolicy(env.domain, architecture=architecture, seed=seed)
                train_self_play(env, policy, episodes=EPISODES, seed=seed)
                return score(policy, env)
            yield dict(architecture=architecture, seed=seed), run


def study_selection(seeds):
    for architecture in ("separate", "shared", "partial"):
        for seed in range(10):  # equilibrium selection is a question about frequencies, so use more seeds
            def run(architecture=architecture, seed=seed):
                env = make_env(DEFAULT_CONFIGS[DEGENERATE])
                policy = TwoPlayerPolicy(env.domain, architecture=architecture, seed=seed)
                train_self_play(env, policy, episodes=EPISODES, seed=seed)
                return score(policy, env)
            yield dict(architecture=architecture, seed=seed), run


def study_inertia(seeds):
    for config_index in (0, SLOW):
        for w0 in (0.0, 0.3, 0.6, 0.9):
            for seed in seeds:
                def run(config_index=config_index, w0=w0, seed=seed):
                    env = make_env(DEFAULT_CONFIGS[config_index], w0)
                    policy = TwoPlayerPolicy(env.domain, architecture="separate", seed=seed)
                    train_self_play(env, policy, episodes=EPISODES, seed=seed)
                    return score(policy, env)
                yield dict(config=config_index, w0=w0, seed=seed), run


def study_pretraining(seeds):
    checkpoints = (100, 300)
    for source in (None, 0, 4):
        for seed in seeds:
            def run(source=source, seed=seed):
                policy = TwoPlayerPolicy(make_env(DEFAULT_CONFIGS[SLOW]).domain, architecture="separate", seed=seed)
                if source is not None:
                    train_self_play(make_env(DEFAULT_CONFIGS[source]), policy, episodes=EPISODES, seed=seed)
                env = make_env(DEFAULT_CONFIGS[SLOW])
                return dict(curve=train_with_curve(env, policy, "reinforce", 300, checkpoints, seed))
            yield dict(source=source, seed=seed), run


def desaturate(policy, cap=2.0):
    """Scale each player's output layer so its largest pre-sigmoid output over the square is at most `cap`: the policy
    keeps pointing at the same corners, but its sigmoid is no longer saturated."""
    grid = torch.as_tensor(np.stack(np.meshgrid(np.linspace(0, 1, 11), np.linspace(0, 1, 11)), -1).reshape(-1, 2), dtype=torch.float32)
    with torch.no_grad():
        for net in (policy.network.red, policy.network.blue):
            biggest = float(net(grid).abs().max())
            if biggest > cap:
                net.out.weight.mul_(cap / biggest), net.out.bias.mul_(cap / biggest)


def study_pretraining_mechanism(seeds):
    """Why does pre-training on setup 1, whose equilibrium corners are flipped, hurt on setup 2? Pre-train, then
    optionally reset the exploration noise to its initial value, de-saturate the sigmoid (`desaturate`), or both,
    and train on setup 2 as in the `pretraining` study. 10 seeds."""
    checkpoints = (100, 300)
    for variant in ("scratch", "pretrained", "reset noise", "desaturated", "both"):
        for seed in range(10):
            def run(variant=variant, seed=seed):
                policy = TwoPlayerPolicy(make_env(DEFAULT_CONFIGS[SLOW]).domain, architecture="separate", seed=seed)
                saturation = None
                if variant != "scratch":
                    train_self_play(make_env(DEFAULT_CONFIGS[0]), policy, episodes=EPISODES, seed=seed)
                    with torch.no_grad():
                        raw = torch.cat(policy.network(torch.as_tensor(make_env(DEFAULT_CONFIGS[SLOW]).reset(), dtype=torch.float32).unsqueeze(0)), -1)
                    saturation = dict(raw=float(raw.abs().max()), log_std=float(torch.cat([policy.log_std_red, policy.log_std_blue]).mean()))
                    if variant in ("reset noise", "both"):
                        with torch.no_grad():
                            policy.log_std_red.fill_(-0.5), policy.log_std_blue.fill_(-0.5)
                    if variant in ("desaturated", "both"):
                        desaturate(policy)
                env = make_env(DEFAULT_CONFIGS[SLOW])
                return dict(curve=train_with_curve(env, policy, "reinforce", 300, checkpoints, seed), saturation=saturation)
            yield dict(variant=variant, seed=seed), run


STUDIES = dict(mechanisms_compensated_design=study_mechanisms_compensated_design, mechanisms_compensated=study_mechanisms_compensated, mechanisms_skewed_seven=study_mechanisms_skewed_seven, mechanisms_skewed=study_mechanisms_skewed, mechanisms_asymmetric=study_mechanisms_asymmetric, mechanisms_limit=study_mechanisms_limit, mechanisms_phantoms=study_mechanisms_phantoms, mechanisms_robust=study_mechanisms_robust, learning_checks=study_learning_checks, learning_fix=study_learning_fix, mechanisms_noise_planner=study_mechanisms_noise_planner, mechanisms_noise=study_mechanisms_noise, mechanisms_wide=study_mechanisms_wide, mechanisms=study_mechanisms, soccer_exact_play=study_soccer_exact_play, exact_play_many=study_exact_play_many, many_targets=study_many_targets, target_random=study_target_random, cross_play=study_cross_play, vertex_selection=study_vertex_selection, walking_clip=study_walking_clip, walking_entropy=study_walking_entropy, pretraining_mechanism=study_pretraining_mechanism, noisy_inertia=study_noisy_inertia, exact_play=study_exact_play, feedback_learning=study_feedback_learning, many_houses=study_many_houses, selection_linear=study_selection_linear, selection_clip=study_selection_clip, inertia_explicit=study_inertia_explicit, inertia_random=study_inertia_random, selection_noise=study_selection_noise, inertia_dp=study_inertia_dp, markov_potential=study_markov_potential, algorithms=study_algorithms, walking=study_walking, ppo_variants=study_ppo_variants, ppo_long=study_ppo_long,
               walking_settings=study_walking_settings, walking_houses=study_walking_houses, walking_inertia=study_walking_inertia,
               walking_equilibrium=study_walking_equilibrium, walking_capacity=study_walking_capacity, walking_ppo_inertia=study_walking_ppo_inertia,
               walking_hard_long=study_walking_hard_long, inertia_equilibrium=study_inertia_equilibrium, selection=study_selection, inertia=study_inertia, pretraining=study_pretraining)


def finished(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {json.dumps([r["study"], r["key"]], sort_keys=True) for r in map(json.loads, f)}


def run(names=tuple(STUDIES), seeds=range(5), path=RESULTS, shard=(0, 1)):
    """Run the named studies, skipping runs already in `path`, and append each result as it finishes.
    With shard=(i, n), only every n-th run (starting at i) is done, so n processes can split the work."""
    index, count = shard
    done = finished(path)  # read once: shards split the runs, so no other process adds one of ours
    for name in names:
        for position, (key, execute) in enumerate(STUDIES[name](seeds)):
            if position % count != index or json.dumps([name, key], sort_keys=True) in done:
                continue
            row = dict(study=name, key=key, **execute())
            with open(path, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(name, key, flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    os.makedirs("results", exist_ok=True)
    args = sys.argv[1:]
    shard = tuple(int(x) for x in args.pop().split("/")) if args and "/" in args[-1] else (0, 1)
    run(args or list(STUDIES), shard=shard)
