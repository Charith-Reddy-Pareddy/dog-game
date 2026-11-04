# Why (vx, vy) and (r, theta) moves gave different walking-game paths

Continuous-move Nash-Q on the walking game reached the Nash corners with Cartesian moves and not with polar
moves, although both describe the same game. These notes say why. Everything here is numerical: 10 seeds per cell,
300 episodes, 12 starting positions, 15 rounds. The code is `doggame/encoding_factorial.py`, the runs are
`results/encoding_factorial.jsonl` and `results/encoding_warm_start.jsonl`, and
`tests/test_encoding_factorial.py` recomputes the numbers below from them.

## The earlier runs could not answer the question

`cartesian` and `polar` both use the same disk of moves, and `square` differs from `cartesian` only in the region.
So the gap between the Cartesian and polar panels was never a circle-against-square effect. "Encoding" also mixed two
things: what the network reads, and the coordinates the stage-game search steps in. The study varies each one alone:
network input (vx, vy) or (r, cos theta, sin theta), search coordinates (vx, vy) or (r, theta), and region disk or
square, 2 x 2 x 2.

## What share of the difference is the region?

Shares of the variation between the eight cell mean paths, as run with gradient search:

| Source | Share |
|---|---|
| Search coordinates | 74.0% |
| Search x input | 12.3% |
| Network input | 10.9% |
| Region, with its interactions | 2.8% (0.7% alone) |

Seed noise in the cell means is about 6% of the same total, so the 2.8% is not distinguishable from noise.

The mean distance between paths says the same. Cartesian and polar are 0.403 apart on the disk and 0.406 apart on the
square. Disk and square are 0.011 apart with Cartesian moves and 0.030 with polar moves. So the region is about 3% of
the Cartesian-polar gap with Cartesian moves and about 7% with polar moves. Without any learning, walking to the corners at full speed differs by 0.034
between the regions: a square lets a player move diagonally faster, so it arrives in 5 rounds instead of 7.

## The cause is the search

Replaying the same trained networks with an exhaustive grid search cuts the Cartesian-polar gap from 0.403 to 0.048. So the
polar networks learned nearly the right values, and the search could not find the moves they imply.

The search starts both players at theta = 0, which points east. Red's corner is to the south-east and blue's to the
north-west. From the same networks, trained as before, with only the starting angle changed (final distance from
the corner, from the houses, mean of 10 seeds; the last column is the summed distance over all 12 starts):

| Search starts at | Red | Blue | All starts |
|---|---|---|---|
| Both 0 | 0.01 | 0.46 | 1.05 |
| Both pi | 0.41 | 0.01 | 0.94 |
| Red 0, blue pi | 0.00 | 0.00 | 0.03 |
| Red pi, blue 0 | 0.65 | 0.46 | 1.85 |

Which player fails follows the starting angle. Not separated here: whether the search stops at a local peak or runs out
of its 30 steps.

A warm start, trying 8 directions before the gradient steps, fixes it. Played on the same networks it gives 0.086. Used in
training as well it gives 0.018, every one of the 10 seeds is under 0.1, and the paths are 0.034 from the Cartesian
paths, down from 0.403: the size of the exact difference between the regions.

## Limits

The network input also matters a little. With the grid search, the cell with polar input and Cartesian search is the
weakest (0.354), partly because the search starts at zero move, where direction is undefined. These results are for one
pair of houses, one step length and this network.
