"""Paragraph styles and tables shared by the PDF report builders (needs reportlab)."""

from types import SimpleNamespace


def kit():
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import Paragraph, Table, TableStyle

    sheet = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=sheet["BodyText"], fontSize=10, leading=14, spaceAfter=6)
    h1 = ParagraphStyle("h1", parent=sheet["Heading1"], fontSize=18, spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=sheet["Heading2"], fontSize=13, spaceBefore=12, spaceAfter=4, keepWithNext=1)
    small = ParagraphStyle("small", parent=body, fontSize=8.5, leading=11, textColor=colors.HexColor("#555555"))
    bullet = ParagraphStyle("bullet", parent=body, leftIndent=14, bulletIndent=2, spaceAfter=3)
    cell = ParagraphStyle("cell", parent=body, fontSize=8.5, leading=10.5, spaceAfter=0)

    def table(data, widths):
        t = Table([[Paragraph(str(c), cell) for c in row] for row in data], colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b8c2cc")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        return t

    return SimpleNamespace(
        h1=lambda t: Paragraph(t, h1), h2=lambda t: Paragraph(t, h2), text=lambda t: Paragraph(t, body),
        small=lambda t: Paragraph(t, small), point=lambda t: Paragraph(t, bullet, bulletText="•"), table=table)
