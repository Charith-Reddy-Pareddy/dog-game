"""Archived studies: frozen copies removed from doggame/dog_game_studies.py. They are not imported or tested here; run them at the
git tag pre-archive."""


STAGE_CONFIG = DEFAULT_CONFIGS[0]


def study_continuous_q(seeds):
    """Nash-Q with continuous moves on the walking game (doggame.continuous_q), for each move encoding:
    the corner distance from the houses and from eight fixed random starts, every 25 of 300 episodes."""
    from doggame.continuous_q import ENCODINGS, corner_distance, random_start_distance, train

    for encoding in ENCODINGS:
        for seed in seeds:
            def run(encoding=encoding, seed=seed):
                env, curve = _continuous_env(), {}
                def record(episode, net):
                    curve[episode] = [corner_distance(env, net), random_start_distance(env, net)]
                train(env, encoding, episodes=300, seed=seed, on_checkpoint=record)
                return dict(curve=curve)
            yield dict(encoding=encoding, seed=seed), run


def study_continuous_q_tools(seeds):
    """On the seed-0 network of each encoding: the best-response methods against a grid (50 states),
    simultaneous gradient play by step size (30 states) and greedy paths from four starts."""
    from doggame.continuous_q import ENCODINGS, benchmark, convergence, train, trajectories

    for encoding in ENCODINGS:
        def run(encoding=encoding):
            env = _continuous_env()
            net = train(env, encoding, episodes=300, seed=0)
            return dict(best_response=benchmark(net, env), convergence={str(k): v for k, v in convergence(net).items()},
                        trajectories=trajectories(env, net))
        yield dict(encoding=encoding), run


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
