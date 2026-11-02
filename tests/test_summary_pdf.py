from pypdf import PdfReader

from doggame.summary_pdf import build


def test_the_summary_builds_on_six_pages_with_all_nine_figures(tmp_path):
    out = build(tmp_path / "summary.pdf")
    reader = PdfReader(out)
    text = "".join(page.extract_text() for page in reader.pages)
    assert len(reader.pages) == 6 and all(f"Fig. {k}." in text for k in range(1, 10))
    assert sorted(p.name for p in tmp_path.glob("fig*.png")) == sorted(f"fig{k}.png" for k in (1, 3, 4, 5, 6, 7, 8, 9, 10))
