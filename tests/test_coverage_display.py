import pandas as pd


def test_play_caption_shows_top_two_specific_coverages() -> None:
	from coverage_display import coverage_caption

	row = pd.Series({f"prob_{i}": p for i, p in enumerate([0.01, 0.01, 0.05, 0.6, 0.02, 0.01, 0.3])})
	assert coverage_caption(row) == "Predicted: Cover-3 60% / Cover-6 30%"
	row["prob_3"] = float("nan")
	assert coverage_caption(row) == "Coverage prediction unavailable"
