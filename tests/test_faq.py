from pypdf import PdfReader

from doggame.faq import build, sections


def test_the_faq_builds_on_three_pages_with_every_question(tmp_path):
    reader = PdfReader(build(tmp_path / "faq.pdf"))
    text = " ".join(page.extract_text().replace("\n", " ") for page in reader.pages)
    questions = [question for _, items in sections for question, _ in items]
    assert len(reader.pages) == 3 and all(question in text for question in questions)
