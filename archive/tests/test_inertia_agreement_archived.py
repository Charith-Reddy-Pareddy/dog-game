"""Archived tests: frozen copies removed from tests/test_inertia_agreement.py. They are not imported or tested here; run them at the
git tag pre-archive."""


def test_where_holding_is_an_equilibrium_no_check_finds_a_gain():
    houses, w, nash = setup(0)
    for w0 in (0.3, 0.6):
        assert holds_everywhere(houses, w, w0, nash)
        mpe, start = check_mpe(houses, w, w0, nash, grid=21), check_profile(houses, w, w0, nash, steps=150)
        assert max(mpe[p]["max_gain"] for p in mpe) < 1e-12 and max(start[p]["gain"] for p in start) < 1e-9


def test_the_fixed_opponent_check_gains_at_least_the_one_round_deviation_from_the_start():
    houses, w, nash = setup(0)
    profile = ((0.5, 0.5), nash[1])  # red does not play its one-shot action
    mpe, start = check_mpe(houses, w, 0.6, profile, grid=11), check_profile(houses, w, 0.6, profile, steps=150)
    assert mpe["red"]["start_gain"] > 1e-3 and start["red"]["gain"] >= mpe["red"]["start_gain"] - 1e-9
