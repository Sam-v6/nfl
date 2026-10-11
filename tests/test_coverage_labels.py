import importlib

import pytest


@pytest.mark.parametrize(("raw", "expected"), [("Cover-3 Seam", 3), ("Cover 6-Left", 6), ("Cover-6 Right", 6), ("Cover-1 Double", 1), ("2-Man", 5), ("Quarters", 4), ("Bracket", None), (None, None)])
def test_canonical_coverage_preserves_fixed_class_order(raw: str | None, expected: int | None) -> None:
	module = importlib.import_module("coverage_labels")
	assert module.canonicalize_coverage(raw) == expected
