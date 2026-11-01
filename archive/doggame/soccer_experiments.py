"""Train REINFORCE, A2C and PPO on soccer (by self-play and by fictitious
play) and compare each against the exact solution.

For player 0's learned policy it counts wins, losses and ties over repeated
games against a random opponent, the exact equilibrium, and the exact best
response to that policy. Run it with `python3 -m doggame.soccer_experiments`.
"""

import sys

from doggame.soccer import Soccer
from doggame.soccer_eval import best_response, exploitability, play_games, random_policy, symmetry_gap
from doggame.soccer_pg import ALGORITHMS, train_fictitious_play, train_self_play
from doggame.soccer_solver import solve_soccer

HEADER = f"{'method':26s} {'vs random':>14s} {'vs exact NE':>14s} {'vs exact BR':>14s} {'exploit.':>8s} {'asym.':>6s}"


def counts(result):
    return f"{result['wins']}/{result['losses']}/{result['ties']}"


def evaluate(game, solution, policy0, policy1, n_games=1000):
    best, _ = best_response(game, policy0, player=1)
    return {
        "random": play_games(game, policy0, random_policy(game), n_games),
        "nash": play_games(game, policy0, solution.policy1, n_games),
        "best response": play_games(game, policy0, best, n_games),
        "exploitability": exploitability(game, policy0, policy1),
        "symmetry": symmetry_gap(game, policy0, policy1),
    }


def run(width=5, height=3, goal_size=1, self_play_iterations=500, fp_rounds=40, fp_iterations=20):
    game = Soccer(width, height, goal_size)
    solution = solve_soccer(game)
    print(f"{width}x{height} board, goal {goal_size}: {game.n_states} states, "
          f"{int(solution.mixed.sum())} need a mixed strategy")
    print("each cell is wins/losses/ties for player 0 over 1000 games\n" + HEADER)

    for algorithm in ALGORITHMS:
        agents = train_self_play(game, algorithm, iterations=self_play_iterations)
        rows = [("self-play", agents[0].table(), agents[1].table())]
        rows.append(("fictitious play", *train_fictitious_play(game, algorithm, rounds=fp_rounds, iterations=fp_iterations)))
        for name, policy0, policy1 in rows:
            r = evaluate(game, solution, policy0, policy1)
            print(f"{algorithm + ' ' + name:26s} {counts(r['random']):>14s} {counts(r['nash']):>14s} "
                  f"{counts(r['best response']):>14s} {r['exploitability']:8.3f} {r['symmetry']:6.2f}", flush=True)


if __name__ == "__main__":
    run(*(int(x) for x in sys.argv[1:4]))
