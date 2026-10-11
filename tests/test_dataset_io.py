import importlib
from pathlib import Path

import pytest
import torch
from test_dataset import raw_fixture


def prepared(tmp_path: Path) -> Path:
	from dataset import prepare_dataset

	raw_fixture(tmp_path / "raw")
	return prepare_dataset(tmp_path / "raw", {"seasons": [2022], "cutoffs": [-1]}, tmp_path / "prepared")


def test_loader_selects_only_requested_split_and_preserves_keys(tmp_path: Path) -> None:
	manifest = prepared(tmp_path)
	module = importlib.import_module("dataset_io")
	x, mask, y, index = module.load_examples(manifest, "tune", offset=-1)
	assert len(x) == 1
	assert x.shape == (1, 30, 22, 5)
	assert index.gameId.tolist() == [2]
	assert y.tolist() == [3]
	assert mask.sum().item() == 4
	assert x.dtype == torch.float32


def test_corrupt_shard_fails_before_training(tmp_path: Path) -> None:
	manifest = prepared(tmp_path)
	(manifest.parent / "s2022_w5.pt").write_bytes(b"bad")
	with pytest.raises(ValueError, match="hash"):
		importlib.import_module("dataset_io").load_examples(manifest, "tune")


def test_final_loading_requires_selection_receipt(tmp_path: Path) -> None:
	manifest = prepared(tmp_path)
	with pytest.raises(ValueError, match="selection"):
		importlib.import_module("dataset_io").load_examples(manifest, "final")
