"""Read-only Parquet inventory with original labels and observed history spans."""

import hashlib
import json
from pathlib import Path


def audit_data(data_root: Path, output_dir: Path) -> Path:
	import pandas as pd
	import polars as pl
	import pyarrow.parquet as pq

	from coverage_labels import CLASSES, canonicalize_coverage

	files = sorted(data_root.rglob("*.parquet")) if data_root.is_dir() else []
	if not files:
		raise ValueError(f"No Parquet data found at {data_root}")
	if output_dir.resolve().is_relative_to(data_root.resolve()):
		raise ValueError("Run directory must be outside read-only data root")
	inventory = []
	for path in files:
		parquet = pq.ParquetFile(path)
		with path.open("rb") as stream:
			digest = hashlib.file_digest(stream, "sha256").hexdigest()
		inventory.append({"path": str(path.relative_to(data_root)), "rows": parquet.metadata.num_rows, "columns": parquet.schema_arrow.names, "sha256": digest})
	parts = []
	for season in (2021, 2022):
		base = data_root / str(season)
		if not (base / "plays.parquet").exists():
			continue
		plays = pd.read_parquet(base / "plays.parquet")
		plays["season"] = season
		plays["original_label"] = plays.pff_passCoverage
		plays["target"] = plays.original_label.map(canonicalize_coverage)
		plays["pass_cohort"] = plays.passResult.isin(["C", "I", "IN", "S"])
		spans = []
		for path in sorted(base.glob("tracking_week_*.parquet")):
			spans.append(
				pl.scan_parquet(path)
				.group_by("gameId", "playId")
				.agg(pl.col("frameId").min().alias("first_frame"), pl.col("frameId").filter(pl.col("event") == "ball_snap").min().alias("snap_frame"))
				.collect()
			)
		if spans:
			plays = plays.merge(pl.concat(spans).to_pandas(), on=["gameId", "playId"], how="left", validate="one_to_one")
			plays["history_span"] = plays.snap_frame - plays.first_frame
		parts.append(plays)
	output_dir.mkdir(parents=True, exist_ok=False)
	counts = {}
	if parts:
		plays = pd.concat(parts, ignore_index=True)
		plays.to_parquet(output_dir / "plays.parquet", index=False)
		eligible = plays.loc[plays.pass_cohort & plays.target.notna()]
		counts = {label: int(eligible.target.eq(i).sum()) for i, label in enumerate(CLASSES)}
	report = output_dir / "audit.json"
	report.write_text(
		json.dumps(
			{"data_root": str(data_root), "files": inventory, "coverage_counts": counts, "history_count_note": "Span only; continuous complete-player eligibility is established by prepare"}, indent=2
		)
		+ "\n"
	)
	return report
