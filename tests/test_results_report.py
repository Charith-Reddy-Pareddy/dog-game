from pypdf import PdfReader

from doggame.results_report import build


def test_the_report_builds_on_six_pages_with_its_five_figures(tmp_path):
    reader = PdfReader(build(tmp_path / "report.pdf"))
    text = "".join(page.extract_text() for page in reader.pages)
    assert len(reader.pages) == 6 and all(f"Fig. {k}." in text for k in range(1, 6)) and "Fig. 6." not in text
