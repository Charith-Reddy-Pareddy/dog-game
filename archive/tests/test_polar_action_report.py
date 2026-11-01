from pathlib import Path

from doggame import polar_action_report as R


def test_the_update_tables_reproduce_from_the_saved_runs(tmp_path):
    rows, robustness = R.load_complete(), R.load_robustness_complete()
    R.build_markdown(rows, path=tmp_path / "update.md", robustness_rows=robustness)
    rebuilt = [line for line in (tmp_path / "update.md").read_text().splitlines() if line.startswith("|")]
    committed = [line for line in R.MARKDOWN.read_text().splitlines() if line.startswith("|")]
    assert rebuilt and rebuilt == committed


def test_every_cell_of_the_primary_study_has_ten_seeds_and_the_charts_are_svg():
    rows = R.load_complete()
    _, stats, _ = R._summary(rows)
    assert len(stats) == 9 and all(s["n"] == 10 for s in stats.values())
    assert R._chart_svg(rows).startswith("<svg") and R._robustness_svg(R.load_robustness_complete()).startswith("<svg")
