"""Builds the PNG images used by the README and the website's link preview.

Run with `python3 -m doggame.figures` (needs reportlab and poppler's `pdftoppm`).
Charts are drawn with ReportLab, saved as a one-page PDF and rasterized, so no
plotting library is needed. Output goes to docs/preview.png and docs/figures/.
"""

import os
import subprocess

from doggame import dog_game_summary as summary

OUT = "docs"


def preview_stats(studies, runs):
    """The headline numbers on the link preview, as (number, label) pairs, from the saved results."""
    (passed, total), _ = summary.convergence_counts(runs)
    algorithms = summary.study(studies, "algorithms")
    ppo = [r for r in algorithms if r["key"]["algorithm"] == "ppo"]
    walking = {row[0]: row for row in summary.walking_algorithms_table(studies)[1:]}
    last = lambda name: walking[name][-1].split(" of ")
    return [(f"{passed}/{total}", "REINFORCE runs converged"),
            (f"{sum(r['curve']['1500'] < summary.CONVERGED for r in ppo)}/{len(ppo)}", "PPO runs converged"),
            (f"{sum(r['curve']['100'] < summary.CONVERGED for r in ppo)}/{len(ppo)}", "PPO runs converged after 100 episodes"),
            (f"{last('A2C')[0]}/{last('A2C')[1]}", "walking-game seeds at the corners (A2C)")]


def rasterize(pdf, png, dpi):
    """Turn a one-page PDF into a PNG with poppler's pdftoppm."""
    subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-singlefile", pdf, png[:-4]], check=True)


def drawing_to_png(drawing, png, dpi=200):
    from reportlab.graphics import renderPDF

    pdf = png[:-4] + ".pdf"
    renderPDF.drawToFile(drawing, pdf)
    rasterize(pdf, png, dpi)
    os.remove(pdf)


def preview(png, studies, runs):
    """A 1200 x 630 image with the project name and four headline numbers."""
    from reportlab.lib import colors
    from reportlab.pdfgen import canvas

    pdf = png[:-4] + ".pdf"
    c = canvas.Canvas(pdf, pagesize=(1200, 630))
    c.setFillColor(colors.HexColor("#f4f1ec"))
    c.rect(0, 0, 1200, 630, stroke=0, fill=1)
    c.setFillColor(colors.HexColor("#2a2622"))
    c.setFont("Helvetica-Bold", 78)
    c.drawString(70, 495, "Dog Game")
    c.setFont("Helvetica", 30)
    c.setFillColor(colors.HexColor("#5f584d"))
    c.drawString(72, 440, "A general-sum game solved with REINFORCE, A2C, PPO and Nash-Q")
    for i, (number, label) in enumerate(preview_stats(studies, runs)):
        x = 70 + i * 275
        c.setFillColor(colors.white)
        c.setStrokeColor(colors.HexColor("#e4ddd0"))
        c.roundRect(x, 190, 255, 190, 18, stroke=1, fill=1)
        c.setFillColor(colors.HexColor("#2a2622"))
        c.setFont("Helvetica-Bold", 56)
        c.drawString(x + 22, 300, number)
        c.setFont("Helvetica", 20)
        c.setFillColor(colors.HexColor("#5f584d"))
        words, line, y = label.split(), "", 265
        for word in words:
            if c.stringWidth(line + " " + word, "Helvetica", 20) > 215:
                c.drawString(x + 22, y, line.strip())
                line, y = "", y - 25
            line += " " + word
        c.drawString(x + 22, y, line.strip())
    c.setFillColor(colors.HexColor("#d8555f"))
    c.rect(70, 100, 26, 26, stroke=0, fill=1)
    c.setFillColor(colors.HexColor("#3f7d9e"))
    c.rect(106, 100, 26, 26, stroke=0, fill=1)
    c.setFillColor(colors.HexColor("#5f584d"))
    c.setFont("Helvetica", 24)
    c.drawString(152, 106, "charith-reddy-pareddy.github.io/dog-game")
    c.save()
    rasterize(pdf, png, 72)
    os.remove(pdf)


def build(out=OUT):
    from doggame.research_report_pdf import convergence_figure, walking_figure

    studies, runs = summary.load(summary.STUDIES), summary.load(summary.RUNS)
    os.makedirs(f"{out}/figures", exist_ok=True)
    preview(f"{out}/preview.png", studies, runs)
    drawing_to_png(convergence_figure(), f"{out}/figures/algorithms.png")
    drawing_to_png(walking_figure(), f"{out}/figures/walking.png")


if __name__ == "__main__":
    build()
