import json

import pytest

from doggame.soccer_report import load_curves, mean_curve, parse_averaging

AVERAGING_OUTPUT = """soccer_fp_checkpoints/random_starts_state_average_800_rounds_8_ppo.pt (round 800): all snapshots 0.13, without the first quarter 0.04, second half only 0.00, last quarter only 0.00, latest only 0.00
soccer_fp_checkpoints/random_starts_state_average_800_rounds_15_ppo.pt (round 800): all snapshots 0.71, without the first quarter 0.67, second half only 0.54, last quarter only 0.32, latest only 0.32

mean over checkpoints
  all snapshots                0.42
  without the first quarter    0.36
  second half only             0.27
  last quarter only            0.16
  latest only                  0.16
"""


def test_parse_averaging_reads_each_seed_and_the_means():
    per_seed, means = parse_averaging(AVERAGING_OUTPUT)

    assert sorted(per_seed) == [8, 15]
    assert per_seed[15] == {"all snapshots": 0.71, "without the first quarter": 0.67, "second half only": 0.54,
                            "last quarter only": 0.32, "latest only": 0.32}
    assert means["all snapshots"] == 0.42 and means["latest only"] == 0.16 and len(means) == 5


def test_curves_are_grouped_by_variant_and_seed_and_averaged(tmp_path):
    path = tmp_path / "runs.jsonl"
    rows = [{"variant": "a", "seed": 0, "curve": [[100, 0.4, 0.2], [200, 0.2, 0.0]]},
            {"variant": "a", "seed": 1, "curve": [[100, 0.2, 0.0], [200, 0.1, 0.0]]},
            {"variant": "b", "seed": 0, "curve": [[100, 0.9, 0.9]]}]
    path.write_text("\n".join(json.dumps(r) for r in rows))

    curves = load_curves(path, "a")

    assert sorted(curves) == [0, 1] and curves[0][200] == (0.2, 0.0)
    assert [r for r, _ in mean_curve(curves, 0)] == [100, 200]
    assert [v for _, v in mean_curve(curves, 0)] == pytest.approx([0.3, 0.15])
    assert [v for _, v in mean_curve(curves, 1)] == pytest.approx([0.1, 0.0])
