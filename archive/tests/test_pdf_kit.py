from pypdf import PdfReader
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate

from doggame.pdf_kit import kit


def test_the_shared_styles_build_a_pdf_with_headings_text_and_a_table(tmp_path):
    k, out = kit(), tmp_path / "kit.pdf"
    story = [k.h1("Title"), k.h2("Section"), k.text("Body text."), k.point("A point."), k.table([["a", "b"], ["1", "2"]], [100, 100])]
    SimpleDocTemplate(str(out), pagesize=letter).build(story)
    text = PdfReader(out).pages[0].extract_text()
    assert all(word in text for word in ("Title", "Section", "Body text.", "A point."))
