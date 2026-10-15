"""Builds research_report.pdf from RESEARCH_REPORT.md, so the two never disagree.

Run with `python3 -m doggame.research_report_pdf [source.md output.pdf]` (needs reportlab). The
Markdown it understands is small: `#` title, `##`/`###` headings, paragraphs,
`-` and `1.` lists, `|` tables, `>` quotes, **bold**, `code` and [links](url).
A line `{{figure:name}}` is replaced by the chart of that name from FIGURES.
"""

import json
import re
import sys
from html import escape

import numpy as np

from doggame.pdf_kit import kit

SOURCE = "RESEARCH_REPORT.md"
OUTPUT = "research_report.pdf"
STUDIES = "results/dog_game_studies.jsonl"
CONVERGED = 0.004
ASCII = {"≈": "~", "→": "->", "≥": ">=", "≤": "<=", "−": "-", "θ": "theta", "γ": "gamma", "²": "^2"}


def inline(text):
    """Markdown inline markup to the small HTML subset ReportLab paragraphs understand."""
    for char, plain in ASCII.items():
        text = text.replace(char, plain)
    pieces = re.split(r"(`[^`]+`)", text)
    out = []
    for piece in pieces:
        if piece.startswith("`") and piece.endswith("`") and len(piece) > 2:
            out.append(f'<font face="Courier" size="9">{escape(piece[1:-1])}</font>')
            continue
        piece = escape(piece)
        piece = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", piece)
        piece = re.sub(r"(?<![\w*])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", piece)
        piece = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<link href="\2" color="#2c5c76">\1</link>', piece)
        out.append(piece)
    return "".join(out)


def parse(markdown):
    """Split Markdown into blocks: (kind, payload) with kinds title, h2, h3, para, quote,
    bullets, numbered (a list of items), table (a list of rows) and figure."""
    blocks, lines, i = [], markdown.splitlines(), 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
        elif line.startswith("```"):
            code, i = [], i + 1
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i].rstrip())
                i += 1
            if i >= len(lines):
                raise ValueError("Unclosed fenced code block in Markdown source")
            blocks.append(("code", code))
            i += 1
        elif line.startswith("# "):
            blocks.append(("title", line[2:].strip()))
            i += 1
        elif line.startswith("### "):
            blocks.append(("h3", line[4:].strip()))
            i += 1
        elif line.startswith("## "):
            blocks.append(("h2", line[3:].strip()))
            i += 1
        elif line.startswith("{{figure:") and line.strip().endswith("}}"):
            blocks.append(("figure", line.strip()[len("{{figure:"):-2]))
            i += 1
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            blocks.append(("table", rows))
        elif re.match(r"(- |\d+\. )", line):
            kind = "bullets" if line.startswith("- ") else "numbered"
            items = []
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "|", "{{")):
                match = re.match(r"(- |\d+\. )(.*)", lines[i])
                if match:
                    items.append(match.group(2).strip())
                elif lines[i].startswith("  ") and items:
                    items[-1] += " " + lines[i].strip()
                else:
                    break
                i += 1
            blocks.append((kind, items))
        elif line.startswith("> "):
            quote = []
            while i < len(lines) and lines[i].startswith(">"):
                quote.append(lines[i][1:].strip())
                i += 1
            blocks.append(("quote", " ".join(quote)))
        else:
            paragraph = []
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "|", ">", "{{", "- ")) and not re.match(r"\d+\. ", lines[i]):
                paragraph.append(lines[i].strip())
                i += 1
            blocks.append(("para", " ".join(paragraph)))
    return blocks


def column_widths(rows, total):
    """Split `total` between the columns in proportion to their longest cell: at least 14 characters (20 for the
    first column), room for the column's longest single word so it never breaks mid-word, and at most 28 so a
    long heading wraps instead of squeezing the other columns."""
    weights = []
    for col in range(len(rows[0])):
        longest_cell = max(len(row[col]) for row in rows)
        longest_word = max(len(word) for row in rows for word in row[col].split())
        weights.append(min(28, max(20 if col == 0 else 14, longest_cell, 2.4 * longest_word)))
    return [total * w / sum(weights) for w in weights]


def study_rows(study, path=STUDIES):
    with open(path) as f:
        return [r for r in map(json.loads, f) if r["study"] == study]


def convergence_figure():
    """Share of runs under the convergence threshold, by episodes trained, for each algorithm."""
    from reportlab.graphics.charts.lineplots import LinePlot
    from reportlab.graphics.shapes import Drawing, String
    from reportlab.graphics.widgets.markers import makeMarker
    from reportlab.lib import colors

    rows = study_rows("algorithms")
    checkpoints = sorted({int(e) for r in rows for e in r["curve"]})
    names = (("reinforce", "REINFORCE", "#1f5fa8"), ("a2c", "A2C", "#c0392b"), ("ppo", "PPO", "#2e8b57"))
    d = Drawing(440, 215)
    plot = LinePlot()
    plot.x, plot.y, plot.width, plot.height = 45, 35, 380, 145
    plot.data = [[(e, np.mean([r["curve"][str(e)] < CONVERGED for r in rows if r["key"]["algorithm"] == a])) for e in checkpoints] for a, _, _ in names]
    for i, (_, label, color) in enumerate(names):
        plot.lines[i].strokeColor, plot.lines[i].strokeWidth = colors.HexColor(color), (3.2, 1.6, 1.8)[i]
        plot.lines[i].strokeDashArray = (None, [4, 3], None)[i]  # A2C is dashed so it stays visible where it overlaps REINFORCE
        plot.lines[i].symbol = makeMarker("FilledCircle", size=3.5, fillColor=colors.HexColor(color), strokeColor=colors.HexColor(color))
        d.add(String(70 + i * 110, 200, label, fontSize=8, fillColor=colors.HexColor(color)))
    plot.xValueAxis.valueMin, plot.xValueAxis.valueMax, plot.xValueAxis.valueStep = 0, 1500, 250
    plot.yValueAxis.valueMin, plot.yValueAxis.valueMax, plot.yValueAxis.valueStep = 0, 1, 0.2
    plot.yValueAxis.labelTextFormat = "%.1f"
    plot.xValueAxis.labels.fontSize = plot.yValueAxis.labels.fontSize = 8
    plot.yValueAxis.gridStrokeColor, plot.yValueAxis.visibleGrid = colors.HexColor("#dde3ea"), 1
    d.add(plot)
    d.add(String(235, 8, "training episodes", fontSize=8, textAnchor="middle"))
    d.add(String(10, 185, "share of runs under 0.004", fontSize=8))
    return d


def walking_figure():
    """Share of the five seeds at the Nash corners after each number of training episodes, on the walking game."""
    from reportlab.graphics.charts.lineplots import LinePlot
    from reportlab.graphics.shapes import Drawing, String
    from reportlab.graphics.widgets.markers import makeMarker
    from reportlab.lib import colors

    from doggame import dog_game_summary as summary

    series = summary.walking_share_series(summary.load(STUDIES))
    styles = (("reinforce", "REINFORCE", "#1f5fa8", None, 3.2), ("a2c", "A2C", "#c0392b", [4, 3], 1.6), ("ppo", "PPO", "#2e8b57", None, 1.8))
    d = Drawing(440, 215)
    plot = LinePlot()
    plot.x, plot.y, plot.width, plot.height = 45, 35, 380, 145
    plot.data = [series[name] for name, *_ in styles]
    for i, (_, label, color, dash, width) in enumerate(styles):
        plot.lines[i].strokeColor, plot.lines[i].strokeWidth, plot.lines[i].strokeDashArray = colors.HexColor(color), width, dash
        plot.lines[i].symbol = makeMarker("FilledCircle", size=3, fillColor=colors.HexColor(color), strokeColor=colors.HexColor(color))
        d.add(String(70 + i * 110, 200, label, fontSize=8, fillColor=colors.HexColor(color)))
    plot.xValueAxis.valueMin, plot.xValueAxis.valueMax, plot.xValueAxis.valueStep = 0, 300, 50
    plot.yValueAxis.valueMin, plot.yValueAxis.valueMax, plot.yValueAxis.valueStep = 0, 1, 0.2
    plot.yValueAxis.labelTextFormat = "%.1f"
    plot.xValueAxis.labels.fontSize = plot.yValueAxis.labels.fontSize = 8
    plot.yValueAxis.gridStrokeColor, plot.yValueAxis.visibleGrid = colors.HexColor("#dde3ea"), 1
    d.add(plot)
    d.add(String(235, 8, "training episodes", fontSize=8, textAnchor="middle"))
    d.add(String(10, 185, "share of seeds at the corners", fontSize=8))
    return d


def trajectories_figure():
    """The recorded paths of one walking game, before and after training (doggame.walking_figures)."""
    from doggame.walking_figures import TRAJECTORIES, trajectory_figure

    with open(TRAJECTORIES) as f:
        return trajectory_figure(json.load(f))


FIGURES = {"convergence": convergence_figure, "walking": walking_figure, "trajectories": trajectories_figure}


def build(source=SOURCE, path=OUTPUT):
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import KeepTogether, Preformatted, SimpleDocTemplate, Spacer

    k = kit()
    with open(source) as f:
        blocks = parse(f.read())
    title = next(payload for kind, payload in blocks if kind == "title")
    story = []
    for kind, payload in blocks:
        if kind == "title":
            story += [k.h1(inline(payload)), Spacer(1, 4)]
        elif kind == "h2":
            story.append(k.h2(inline(payload)))
        elif kind == "h3":
            story.append(k.h3(inline(payload)))
        elif kind == "para":
            story.append(k.text(inline(payload)))
        elif kind == "quote":
            story.append(k.quote(inline(payload)))
        elif kind == "bullets":
            story += [k.point(inline(item)) for item in payload]
        elif kind == "numbered":
            story += [k.point(inline(item), mark=f"{n}.") for n, item in enumerate(payload, 1)]
        elif kind == "table":
            story += [k.table([[inline(c) for c in row] for row in payload], column_widths(payload, 6.9 * inch)), Spacer(1, 6)]
        elif kind == "figure":
            story.append(KeepTogether([FIGURES[payload](), Spacer(1, 4)]))
        elif kind == "code":
            code_style = ParagraphStyle("report_code", fontName="Courier", fontSize=7.5, leading=9,
                                        leftIndent=8, rightIndent=8, backColor="#f3f5f7",
                                        borderColor="#d5dbe1", borderWidth=0.4, borderPadding=5,
                                        spaceBefore=3, spaceAfter=8)
            story.append(Preformatted("\n".join(payload), code_style, maxLineLength=108))
    SimpleDocTemplate(path, pagesize=letter, leftMargin=0.8 * inch, rightMargin=0.8 * inch, topMargin=0.7 * inch,
                      bottomMargin=0.7 * inch, title=title.replace("`", ""), author="").build(story)


if __name__ == "__main__":
    build(*sys.argv[1:3])  # optionally: a Markdown source and the PDF to write
