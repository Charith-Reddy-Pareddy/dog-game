from doggame.experiments import run_convergence_study


def test_convergence_study_returns_one_result_per_config():
    configs = [
        dict(house_red=(0.9, 0.2), house_blue=(0.1, 0.8), w=0.5),
        dict(house_red=(0.7, 0.1), house_blue=(0.6, 0.9), w=0.6),
    ]
    results = run_convergence_study(configs=configs, episodes=50, seed=0)

    assert len(results) == 2
    for result, config in zip(results, configs):
        assert result["config"] == config
        assert result["distance_red"] >= 0.0
        assert result["distance_blue"] >= 0.0
        assert result["red_exploitability"] >= 0.0
        assert result["blue_exploitability"] >= 0.0
