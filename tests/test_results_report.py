from pypdf import PdfReader

from doggame import summary_figures as F
from doggame.results_report import build


def test_the_report_builds_on_at_most_ten_pages_with_all_twelve_figures(tmp_path):
    reader = PdfReader(build(tmp_path / "report.pdf"))
    text = "".join(page.extract_text() for page in reader.pages)
    assert len(reader.pages) <= 10 and all(f"Fig. {k}." in text for k in range(1, 13))


def test_figure_11_gives_the_deviation_gains_quoted_in_the_report(tmp_path):
    gains = F.fig11(tmp_path / "fig11.png")
    assert gains[0.6][0] < 1e-9 and gains[0.9][0] < 1e-9  # Proposition 2's policy
    assert round(gains[0.6][1], 3) == 0.076 and round(gains[0.9][1], 3) == 0.121  # holding the one-shot actions
