import numpy as np
import pytest
import torch

from doggame.soccer import Soccer
from doggame.soccer_eval import play_games, random_policy
from doggame.soccer_pg import ALGORITHMS, Agent, _tensors, rollout, train_fictitious_play, train_self_play

GAME = Soccer(width=5, height=3, goal_size=1)
NORTH, SOUTH, WEST, EAST = range(4)


def always(move):
    table = torch.zeros(GAME.n_states, 4)
    table[:, move] = 1.0
    return lambda states: table[states]


def test_returns_are_discounted_back_from_the_goal():
    # player 0 walks east along the middle row and scores on its 4th move
    _, _, _, returns = rollout(GAME, _tensors(GAME), always(EAST), always(NORTH), n_games=2, max_steps=100, discount=0.9)
    first_game = returns[0:8:2]  # steps 0-3 of the game that starts with player 0 on the ball
    assert torch.allclose(first_game, torch.tensor([0.9**3, 0.9**2, 0.9, 1.0]))


def test_unknown_algorithm_raises():
    with pytest.raises(ValueError):
        Agent(_tensors(GAME)[0], "nonsense")


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_an_update_changes_the_policy_toward_the_rewarded_move(algorithm):
    torch.manual_seed(0)
    agent = Agent(_tensors(GAME)[0], algorithm, lr=0.05)
    states = torch.zeros(200, dtype=torch.long)
    actions = torch.tensor([EAST] * 100 + [WEST] * 100)
    returns = torch.tensor([1.0] * 100 + [-1.0] * 100)  # east pays, west costs

    before = agent.table()[0].copy()
    for _ in range(20):
        agent.update(states, actions, returns)
    after = agent.table()[0]

    assert after[EAST] > before[EAST] and after[WEST] < before[WEST]


def test_self_play_learns_to_beat_random_play():
    agent0, agent1 = train_self_play(GAME, "reinforce", iterations=150, games=128)
    result = play_games(GAME, agent0.table(), random_policy(GAME), n_games=300)
    assert result["wins"] > 3 * result["losses"]


def test_fictitious_play_returns_a_probability_table_per_player():
    average0, average1 = train_fictitious_play(GAME, "reinforce", rounds=3, iterations=3, games=64)
    for table in (average0, average1):
        assert table.shape == (GAME.n_states, 4)
        assert np.allclose(table.sum(axis=1), 1.0, atol=1e-5)


def test_random_starts_cover_many_different_states():
    states, _, _, _ = rollout(GAME, _tensors(GAME), always(NORTH), always(NORTH), n_games=200, max_steps=1,
                              discount=0.9, random_starts=True)
    assert len(set(states.tolist())) > 50  # the standard starts are only two states


def test_training_counts_how_often_it_visits_each_state():
    standard = torch.zeros(GAME.n_states, dtype=torch.long)
    train_fictitious_play(GAME, "reinforce", rounds=1, iterations=2, games=64, visits=standard)
    spread = torch.zeros(GAME.n_states, dtype=torch.long)
    train_fictitious_play(GAME, "reinforce", rounds=1, iterations=2, games=64, random_starts=True, visits=spread)

    assert (standard[GAME.start_states()] > 0).all()
    # standard starts keep returning to the same two states; random starts spread out
    assert standard.max() / standard.sum() > 2 * spread.max() / spread.sum()


def test_an_interrupted_run_resumes_to_exactly_the_same_result(tmp_path):
    path = str(tmp_path / "run.pt")
    settings = dict(iterations=2, games=64, mix="state", random_starts=True, seed=5)

    straight_through = train_fictitious_play(GAME, "ppo", rounds=6, **settings)

    train_fictitious_play(GAME, "ppo", rounds=3, checkpoint=path, checkpoint_every=1, **settings)  # "killed" after round 3
    resumed = train_fictitious_play(GAME, "ppo", rounds=6, checkpoint=path, checkpoint_every=1, **settings)

    for a, b in zip(straight_through, resumed):
        assert np.array_equal(a, b)


def test_a_checkpoint_is_only_written_every_so_many_rounds(tmp_path):
    path = tmp_path / "run.pt"
    train_fictitious_play(GAME, "reinforce", rounds=2, iterations=1, games=32, checkpoint=str(path), checkpoint_every=5)
    assert not path.exists()
    train_fictitious_play(GAME, "reinforce", rounds=5, iterations=1, games=32, checkpoint=str(path), checkpoint_every=5)
    assert path.exists()
