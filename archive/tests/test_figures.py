from doggame.figures import preview_stats


def run(study, key, **fields):
    return dict(study=study, key=key, **fields)


def test_the_preview_numbers_come_from_the_saved_results():
    runs = [dict(kind="convergence", architecture="shared", exploitability=0.001), dict(kind="convergence", architecture="shared", exploitability=0.3)]
    studies = [run("algorithms", dict(config=0, algorithm="ppo", seed=s), curve={"100": e, "1500": 0.0001}) for s, e in enumerate([0.001, 0.5, 0.002])]
    studies += [run("walking", dict(algorithm="a2c", seed=s), start=1.5, curve={"150": 0.02, "300": v}) for s, v in enumerate([0.03, 0.4])]

    stats = dict((label, number) for number, label in preview_stats(studies, runs))

    assert stats["REINFORCE runs converged"] == "1/2"
    assert stats["PPO runs converged"] == "3/3" and stats["PPO runs converged after 100 episodes"] == "2/3"
    assert stats["walking-game seeds at the corners (A2C)"] == "1/2"
