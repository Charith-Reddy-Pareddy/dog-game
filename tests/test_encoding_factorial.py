import numpy as np
import pytest
import torch

import doggame.encoding_factorial as E


@pytest.mark.parametrize("inp", ["cart", "polar"])
@pytest.mark.parametrize("opt", ["cart", "polar"])
@pytest.mark.parametrize("region", ["disk", "square"])
def test_moves_stay_in_their_region_and_the_two_coordinates_agree(inp, opt, region):
    torch.manual_seed(0)
    net = E.Net(inp, opt, region)
    v = net.vec(net.sample(500))
    size = v.norm(dim=-1) if region == "disk" else v.abs().max(-1)[0]
    assert size.max() <= 1 + 1e-5
    assert torch.allclose(net.vec(net.from_vec(v)), v, atol=1e-5)


def test_the_exact_paths_show_how_much_the_region_changes_a_trajectory():
    disk = E.exact_path((0.5, 0.5), (1.0, 0.0), "disk")
    square = E.exact_path((0.5, 0.5), (1.0, 0.0), "square")
    assert np.allclose(disk[-1], (1, 0)) and np.allclose(square[-1], (1, 0))
    assert np.allclose(square[1], (0.7, 0.3)) and np.allclose(disk[1], (0.5 + 0.2 / 2 ** 0.5, 0.5 - 0.2 / 2 ** 0.5))


def test_the_saved_runs_give_the_numbers_in_the_notes():
    rows, warm = E.load(E.OUTPUT), E.load(E.OUTPUT_WARM)
    assert len(rows) == 80 and len(warm) == 20
    found = E.summary(rows, warm)
    region = sum(v for k, v in found["shares"].items() if "region" in k)
    assert round(found["shares"]["search"], 3) == 0.74 and round(region, 3) == 0.028
    assert round(found["polar_gap"], 3) == 0.403 and round(found["polar_gap_square"], 3) == 0.406
    assert round(found["region_gap"], 3) == 0.011 and round(found["polar_region_gap"], 3) == 0.03
    assert round(found["exact_region_gap"], 3) == 0.034 and round(found["grid_polar_gap"], 3) == 0.048
    assert round(found["warm_gap"], 3) == 0.034 and (found["warm_seeds_under_01"], found["warm_seeds"]) == (10, 10)
    base = [r for r in warm if r["mode"] == "base"]
    red, blue = np.mean([r["start 0,0"][1:] for r in base], 0)
    assert red < 0.02 and blue > 0.4  # searching from theta = 0, blue stays far from its corner
    red, blue = np.mean([r["start pi,pi"][1:] for r in base], 0)
    assert red > 0.4 and blue < 0.02  # from theta = pi the same networks fail the other way round
    assert np.mean([r["start 0,pi"][0] for r in base]) < 0.05  # and from each player's own side they work
