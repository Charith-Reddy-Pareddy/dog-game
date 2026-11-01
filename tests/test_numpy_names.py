import re
from pathlib import Path

# NumPy 2 removed these from the main namespace; CI installs NumPy 2 while the local setup has 1.26.
REMOVED = ["trapz", "float_", "Inf", "NaN", "product", "cumproduct", "alltrue", "sometrue", "asfarray", "round_"]


def test_no_code_uses_numpy_names_removed_in_numpy_2():
    pattern = re.compile(r"\bnp\.(" + "|".join(REMOVED) + r")\b")
    root = Path(__file__).resolve().parents[1]
    hits = [f"{p.relative_to(root)}:{i}" for folder in ("doggame", "tests") for p in sorted((root / folder).glob("*.py"))
            for i, line in enumerate(p.read_text().splitlines(), 1) if pattern.search(line)]
    assert not hits, hits
