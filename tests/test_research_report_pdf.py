from doggame.research_report_pdf import column_widths, inline, parse

SAMPLE = """# Title

Intro line one
continues here.

## Section

- first item
  continues
- second item

1. one
2. two

| a | b |
|---|---|
| 1 | 2 |

> a quoted
> note

### Sub

{{figure:convergence}}
"""


def test_blocks_come_out_in_order_with_their_content():
    kinds = [kind for kind, _ in parse(SAMPLE)]
    blocks = dict(parse(SAMPLE))

    assert kinds == ["title", "para", "h2", "bullets", "numbered", "table", "quote", "h3", "figure"]
    assert blocks["para"] == "Intro line one continues here."
    assert blocks["bullets"] == ["first item continues", "second item"]
    assert blocks["numbered"] == ["one", "two"]
    assert blocks["table"] == [["a", "b"], ["1", "2"]]  # the separator row is dropped
    assert blocks["quote"] == "a quoted note" and blocks["figure"] == "convergence"


def test_inline_markup_becomes_html_and_special_characters_are_escaped():
    assert inline("**bold** and `a < b`") == '<b>bold</b> and <font face="Courier" size="9">a &lt; b</font>'
    assert inline("[site](http://x.y)") == '<link href="http://x.y" color="#2c5c76">site</link>'
    assert inline("R&D") == "R&amp;D"


def test_stars_inside_code_are_left_alone_and_unsupported_symbols_are_replaced():
    assert "0.5*1" in inline("`0.5*1 + 0.5*0`") and "<i>" not in inline("`0.5*1 + 0.5*0`")
    assert inline("a ≈ b → c") == "a ~ b -&gt; c"


def test_column_widths_are_proportional_to_the_longest_cell_and_add_up_to_the_total():
    widths = column_widths([["a", "a much longer heading"], ["1", "2"]], 100)

    assert abs(sum(widths) - 100) < 1e-9
    assert widths[1] > widths[0] and widths[0] == 100 * 20 / (20 + len("a much longer heading"))


def test_italics_are_converted_but_stars_around_spaces_and_inside_code_are_not():
    assert inline("an *Answered.* status") == "an <i>Answered.</i> status"
    assert inline("w0 * dog + w1 * red") == "w0 * dog + w1 * red"
    assert inline("`a*b*c`") == '<font face="Courier" size="9">a*b*c</font>'
    assert inline("**bold *and* more**") == "<b>bold <i>and</i> more</b>"


def test_a_very_long_heading_cannot_take_more_than_its_capped_share():
    widths = column_widths([["x" * 80, "b"], ["1", "2"]], 100)

    assert abs(widths[0] - 100 * 28 / (28 + 12)) < 1e-9
