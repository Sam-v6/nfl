"""Versioned coverage vocabulary shared by training, prediction and reports."""

CLASSES = ("Cover-0", "Cover-1", "Cover-2", "Cover-3", "Quarters", "2-Man", "Cover-6")
VARIANTS = {
	"Cover-3 Cloud Right": "Cover-3",
	"Cover-3 Cloud Left": "Cover-3",
	"Cover-3 Seam": "Cover-3",
	"Cover-3 Double Cloud": "Cover-3",
	"Cover-6 Right": "Cover-6",
	"Cover 6-Left": "Cover-6",
	"Cover-1 Double": "Cover-1",
}


def canonicalize_coverage(raw: str | None) -> int | None:
	if not isinstance(raw, str):
		return None
	canonical = VARIANTS.get(raw, raw)
	return CLASSES.index(canonical) if canonical in CLASSES else None
