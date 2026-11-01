from pypdf import PdfReader

from doggame.summary_pdf import build


def test_the_summary_builds_on_four_pages_with_all_seven_figures(tmp_path):
    out = build(tmp_path / "summary.pdf")
    reader = PdfReader(out)
    text = "".join(page.extract_text() for page in reader.pages)
    assert len(reader.pages) == 4 and all(f"Fig. {k}." in text for k in range(1, 8))
    assert sorted(p.name for p in tmp_path.glob("fig*.png")) == [f"fig{k}.png" for k in range(1, 8)]
