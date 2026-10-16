"""Chronological development views of an immutable, verified dataset."""

import hashlib
import json
from pathlib import Path

import pandas as pd

from dataset import file_hash


def build_fold(source: Path, output: Path, tune_week: int, fraction: float = 1.0) -> Path:
	"""Use nested whole-game fit subsets; reserve the following week for scoring."""
	if tune_week not in (3, 4) or fraction not in (0.25, 0.5, 1.0):
		raise ValueError("Only the predeclared development folds and fractions are allowed")
	manifest = json.loads(source.read_text())
	if manifest.get("status") != "complete" or file_hash(source.parent / "examples.parquet") != manifest["examples_sha256"]:
		raise ValueError("Incomplete source or index hash mismatch")
	rows = pd.read_parquet(source.parent / "examples.parquet")
	rows = rows.loc[rows.season.eq(2021) | (rows.season.eq(2022) & rows.week.le(tune_week + 1))].copy()
	rows["split"] = "fit"
	rows.loc[rows.season.eq(2022) & rows.week.eq(tune_week), "split"] = "tune"
	rows.loc[rows.season.eq(2022) & rows.week.eq(tune_week + 1), "split"] = "development"
	rows = rows.loc[rows.split.eq("fit") | ~rows.augmented].copy()
	if rows.groupby(["season", "gameId"]).split.nunique().max() != 1:
		raise ValueError("Game crosses development roles")
	if fraction < 1:
		games = rows.loc[rows.split.eq("fit"), ["season", "gameId"]].drop_duplicates()
		games["rank"] = [hashlib.sha256(f"42:{s}:{g}".encode()).hexdigest() for s, g in games.itertuples(index=False, name=None)]
		games = games.sort_values("rank").head(max(1, int(len(games) * fraction)))
		keys = pd.MultiIndex.from_frame(games[["season", "gameId"]])
		rows = rows.loc[rows.split.ne("fit") | pd.MultiIndex.from_frame(rows[["season", "gameId"]]).isin(keys)].copy()
	if set(rows.split) != {"fit", "tune", "development"}:
		raise ValueError("Empty development role")
	selected = set(rows.shard)
	shards = [{**entry, "path": str((source.parent / entry["path"]).resolve())} for entry in manifest["shards"] if entry["path"] in selected]
	rows["shard"] = rows.shard.map(lambda name: str((source.parent / name).resolve()))
	output.mkdir(parents=True, exist_ok=False)
	rows.to_parquet(output / "examples.parquet", index=False)
	identity = {"source_manifest_sha256": file_hash(source), "tune_week": tune_week, "fraction": fraction, "examples_sha256": file_hash(output / "examples.parquet")}
	view = {**manifest, **identity, "shards": shards, "rows_by_split": rows.split.value_counts().to_dict(), "dataset_hash": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()}
	path = output / "manifest.json"
	path.write_text(json.dumps(view, indent=2) + "\n")
	return path
