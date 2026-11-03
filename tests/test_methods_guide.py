from pypdf import PdfReader

from doggame.methods_guide import build


def test_the_guide_builds_on_five_pages_with_all_its_sections(tmp_path):
    reader = PdfReader(build(tmp_path / "guide.pdf"))
    text = "".join(page.extract_text() for page in reader.pages)
    sections = ["1. The model", "2. Definitions", "3. What is implemented", "4. Methods", "5. Logic", "6. A worked example"]
    assert len(reader.pages) == 5 and all(s in text for s in sections)
