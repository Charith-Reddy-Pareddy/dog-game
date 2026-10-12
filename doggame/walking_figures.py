"""Figures of the walking game's recorded paths, shared by the PDF reports (needs reportlab)."""

TRAJECTORIES = "results/dog_game_trajectories.json"


def trajectory_figure(walk, seed_index=0):
    """Four panels (REINFORCE and Nash-Q, before and after training) of one recorded game: the houses, the one-shot
    Nash corners and the paths of red, blue and the dog. `walk` is the content of results/dog_game_trajectories.json."""
    from reportlab.graphics.shapes import Circle, Drawing, Line, PolyLine, Rect, String
    from reportlab.lib import colors

    red, blue = colors.HexColor("#c0392b"), colors.HexColor("#1f5fa8")
    example = walk["seeds"][seed_index]

    def panel(d, x, y, size, title, path):
        at = lambda p: (x + p[0] * size, y + p[1] * size)
        d.add(Rect(x, y, size, size, strokeColor=colors.HexColor("#b8c2cc"), fillColor=colors.HexColor("#fafbfc")))
        d.add(String(x, y + size + 5, title, fontSize=8))
        for house, color in ((walk["house_red"], red), (walk["house_blue"], blue)):
            hx, hy = at(house)
            d.add(Rect(hx - 4, hy - 4, 8, 8, fillColor=color, strokeColor=color, fillOpacity=0.25))
        for corner, color in ((walk["nash_red"], red), (walk["nash_blue"], blue)):
            cx, cy = at(corner)
            for dx, dy in ((1, 1), (1, -1)):
                d.add(Line(cx - 4 * dx, cy - 4 * dy, cx + 4 * dx, cy + 4 * dy, strokeColor=color, strokeWidth=1.2))
        for name, color, dash in (("dog", colors.HexColor("#555555"), [2, 2]), ("red", red, None), ("blue", blue, None)):
            d.add(PolyLine([c for p in path[name] for c in at(p)], strokeColor=color, strokeWidth=1.3, strokeDashArray=dash))
            d.add(Circle(*at(path[name][0]), 2.5, strokeColor=color, fillColor=colors.white))
            d.add(Circle(*at(path[name][-1]), 2.5, strokeColor=color, fillColor=color))

    d = Drawing(440, 450)
    for row, (key, label) in enumerate((("reinforce", "REINFORCE"), ("nash_q", "Nash-Q"))):
        for col, when in enumerate(("before", "after")):
            panel(d, 30 + col * 215, 245 - row * 215, 170, f"{label}, {when} training", example[key][when])
    d.add(String(30, 8, "square = house, X = one-shot Nash corner, open circle = start, filled circle = end; dashed gray = dog", fontSize=7.5))
    return d
