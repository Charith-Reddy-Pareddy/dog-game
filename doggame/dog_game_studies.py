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
- feedback_learning: whether self-play from random dog positions learns a feedback equilibrium with inertia
- exact_play: noise-free gradient play on feedback policies, a table against a shared network
- noisy_inertia: where the equilibrium switches when the dog gets a Gaussian shock, as the time step shrinks
- inertia_explicit: the closed-form hold condition and explicit feedback equilibrium against the dynamic program
- inertia_random: the explicit feedback equilibrium on random one-coordinate games
- walking_equilibrium: is walking to the one-shot corners an equilibrium of the walking game?
- selection:    which equilibrium self-play finds when the houses share a coordinate
- inertia:      does giving the dog momentum (w0) change where training ends up
- pretraining:  does starting from a policy trained on another setup help
- stage_solver: how precisely the Nash-Q stage game has to be solved
- resolution:   how the action-grid resolution affects the discretized solution
- noise:        how much of fictitious play's mixing is just noise
"""

import json
import os
import sys
import time

import numpy as np
import torch

from doggame.agents import TwoPlayerPolicy
from doggame.dog_game_study import WALK_HOUSES, walk_distance
from doggame.discretize import action_grid, payoff_matrices
from doggame.dqn import solve_discretized_nash_q
from doggame.env import DogGameEnv, Square, make_inertial_transition, negative_squared_distance_reward, weighted_average_transition
from doggame.exact_nash import lemke_howson_nash
from doggame.experiments import DEFAULT_CONFIGS
from doggame.fictitious_play import expected_value, fictitious_play
from doggame.nash import solve_stage_nash
from doggame.polar import PolarDogGameEnv
from doggame.inertia_dp import policy as dp_policy, solve as solve_dp
from doggame.inertia_equilibrium import check_profile as check_inertia_profile
from doggame.walking_equilibrium import check_profile
from doggame.polar_agent import PolarPolicy, final_positions, train_polar, train_polar_actor_critic
from doggame.train import train_self_play
from doggame.train_ac import train_actor_critic
from doggame.verify import exploitability, mixed_strategy_indifference_gap

RESULTS = "results/dog_game_studies.jsonl"
EPISODES = 1500
CONVERGED = 0.004  # exploitability below this counts as converged
DEGENERATE = 3  # the configuration whose houses share a y-coordinate
SLOW = 1  # the configuration where one run needed more than 1500 episodes
STAGE_CONFIG = DEFAULT_CONFIGS[0]


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
    variants += [(0.05, sigma, False) for sigma in (0.0, 0.2, 0.4)]
    for dt, sigma, clipped in variants:
        def run(dt=dt, sigma=sigma, clipped=clipped):
            bounds = dict() if clipped else dict(low=-0.5, high=1.5)
            sol = solve_1d(0.7, 0.6, 0.7, 1 - dt, discount=float(np.exp(-0.5 * dt)), states=201 if clipped else 401, actions=21,
                           tol=1e-8, max_iter=8000, noise=sigma * np.sqrt(dt), **bounds)
            z, inside = sol["z"], (sol["z"] >= 0) & (sol["z"] <= 1)
            switch = lambda a: float(z[inside][np.argmax(a[inside] < 0.5)])
            return dict(red=switch(sol["red"]), blue=switch(sol["blue"]), sweeps=sol["sweeps"], unresolved=sol["unresolved"])
        yield dict(dt=dt, sigma=sigma, clipped=clipped), run


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


def stage_game(resolution, config=STAGE_CONFIG):
    domain = Square(0.0, 1.0)
    grid = action_grid(domain, resolution)
    payoff_red, payoff_blue = payoff_matrices(grid, config["house_red"], config["house_blue"], config["w"], domain,
                                              weighted_average_transition, negative_squared_distance_reward)
    return grid, payoff_red, payoff_blue


def nash_gap(payoff_red, payoff_blue, strategy_red, strategy_blue):
    """The most either player could gain by switching to a single best action."""
    value_red = expected_value(payoff_red, strategy_red, strategy_blue)
    value_blue = expected_value(payoff_blue, strategy_red, strategy_blue)
    return float(max((payoff_red @ strategy_blue).max() - value_red, (strategy_red @ payoff_blue).max() - value_blue))


def study_stage_solver(seeds, resolution=9):
    _, payoff_red, payoff_blue = stage_game(resolution)
    exact_q = solve_discretized_nash_q(payoff_red, payoff_blue, 0.9, solver="exact")[0]
    for iterations in (10, 30, 100, 300, 1000, 3000):
        for resync_every in (1, 5, 25):
            for seed in seeds:
                def run(iterations=iterations, resync_every=resync_every, seed=seed):
                    start = time.time()
                    q_red, _, strategy_red, strategy_blue = solve_discretized_nash_q(
                        payoff_red, payoff_blue, 0.9, fictitious_play_iterations=iterations, resync_every=resync_every, seed=seed)
                    return dict(q_error=float(np.abs(q_red - exact_q).max()), gap=nash_gap(payoff_red, payoff_blue, strategy_red, strategy_blue),
                                seconds=time.time() - start)
                yield dict(iterations=iterations, resync_every=resync_every, seed=seed), run


def study_resolution(seeds):
    for config_index, config in enumerate(DEFAULT_CONFIGS):
        nash_red, nash_blue, _ = solve_stage_nash(config["house_red"], config["house_blue"], config["w"], Square(0.0, 1.0))
        for resolution in (3, 5, 9, 13, 17):
            def run(config=config, resolution=resolution, nash_red=nash_red, nash_blue=nash_blue):
                start = time.time()
                grid, payoff_red, payoff_blue = stage_game(resolution, config)
                strategy_red, strategy_blue = lemke_howson_nash(payoff_red, payoff_blue)
                red, blue = grid[strategy_red.argmax()], grid[strategy_blue.argmax()]
                return dict(actions=len(grid), support=[int((strategy_red > 1e-6).sum()), int((strategy_blue > 1e-6).sum())],
                            distance=float(np.linalg.norm(red - nash_red) + np.linalg.norm(blue - nash_blue)), seconds=time.time() - start)
            yield dict(config=config_index, resolution=resolution), run


def study_noise(seeds, resolution=9):
    _, payoff_red, payoff_blue = stage_game(resolution)
    for iterations in (30, 100, 300, 1000, 3000, 10000):
        for seed in seeds:
            def run(iterations=iterations, seed=seed):
                strategy_red, strategy_blue = fictitious_play(payoff_red, payoff_blue, iterations=iterations, seed=seed)
                spread, gap = mixed_strategy_indifference_gap(payoff_red, strategy_red, strategy_blue)
                return dict(spread=float(spread), gap=float(gap), support=int((strategy_red > 1e-3).sum()), top=float(strategy_red.max()))
            yield dict(iterations=iterations, seed=seed), run
    exact_red, exact_blue = lemke_howson_nash(payoff_red, payoff_blue)
    spread, gap = mixed_strategy_indifference_gap(payoff_red, exact_red, exact_blue)
    yield dict(iterations="exact", seed=0), lambda: dict(spread=float(spread), gap=float(gap), support=int((exact_red > 1e-3).sum()), top=float(exact_red.max()))


STUDIES = dict(noisy_inertia=study_noisy_inertia, exact_play=study_exact_play, feedback_learning=study_feedback_learning, many_houses=study_many_houses, selection_linear=study_selection_linear, selection_clip=study_selection_clip, inertia_explicit=study_inertia_explicit, inertia_random=study_inertia_random, selection_noise=study_selection_noise, inertia_dp=study_inertia_dp, markov_potential=study_markov_potential, algorithms=study_algorithms, walking=study_walking, ppo_variants=study_ppo_variants, ppo_long=study_ppo_long,
               walking_settings=study_walking_settings, walking_houses=study_walking_houses, walking_inertia=study_walking_inertia,
               walking_equilibrium=study_walking_equilibrium, walking_capacity=study_walking_capacity, walking_ppo_inertia=study_walking_ppo_inertia,
               walking_hard_long=study_walking_hard_long, inertia_equilibrium=study_inertia_equilibrium, selection=study_selection, inertia=study_inertia, pretraining=study_pretraining,
               stage_solver=study_stage_solver, resolution=study_resolution, noise=study_noise)


def finished(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {json.dumps([r["study"], r["key"]], sort_keys=True) for r in map(json.loads, f)}


def run(names=tuple(STUDIES), seeds=range(5), path=RESULTS, shard=(0, 1)):
    """Run the named studies, skipping runs already in `path`, and append each result as it finishes.
    With shard=(i, n), only every n-th run (starting at i) is done, so n processes can split the work."""
    index, count = shard
    for name in names:
        for position, (key, execute) in enumerate(STUDIES[name](seeds)):
            if position % count != index or json.dumps([name, key], sort_keys=True) in finished(path):
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
