from pathlib import Path

import numpy as np


def evidence() -> dict:
	return {
		"ci95": [0.01, 0.08],
		"replicates": 2000,
		"n_games": 4,
		"secondary_candidate_f1": 0.8,
		"secondary_baseline_f1": 0.8,
		"evidence": {
			"seeds": [17, 42, 73],
			"benchmark_verified": True,
			"calibration_split": "calibrate",
			"selection_hash": "frozen",
			"weights_sha256": "weights",
			"dataset_hash": "data",
			"metadata_sha256": "metadata",
			"manifest_sha256": "manifest",
		},
	}


def test_worse_candidate_and_missing_evidence_cannot_promote() -> None:
	from evaluate import check_promotion, score_predictions

	y = np.repeat(np.arange(7), 20)
	baseline = score_predictions(y, np.eye(7)[y] * 0.9 + 0.1 / 7)
	candidate = score_predictions(y, np.full((len(y), 7), 1 / 7))
	assert not check_promotion(candidate, baseline, evidence())["approved"]
	better = check_promotion(baseline, baseline, {**evidence(), "evidence": {"seeds": [42]}})
	assert not better["approved"]
	assert not better["gates"]["evidence"]


def test_support_and_secondary_endpoint_fail_closed() -> None:
	from evaluate import check_promotion, score_predictions

	y = np.repeat(np.arange(7), 20)
	metrics = score_predictions(y, np.eye(7)[y] * 0.9 + 0.1 / 7)
	details = evidence()
	details["secondary_candidate_f1"] = 0.78
	assert not check_promotion(metrics, metrics, details)["gates"]["secondary_endpoint"]
	metrics["per_class"]["2-Man"]["support"] = 1
	assert not check_promotion(metrics, metrics, evidence())["gates"]["class_support"]


def test_fixture_and_rejected_decisions_preserve_incumbent(tmp_path: Path) -> None:
	from evaluate import promote_bundle

	pointer = tmp_path / "incumbent.json"
	pointer.write_text("incumbent")
	assert not promote_bundle(tmp_path / "missing", {"approved": False}, pointer)
	assert not promote_bundle(tmp_path / "missing", {"approved": True}, pointer, fixture=True)
	assert pointer.read_text() == "incumbent"


def test_final_pipeline_requires_unique_output_and_fixed_seeds(tmp_path: Path) -> None:
	import pytest

	from final_report import run_final_comparison

	output = tmp_path / "run"
	output.mkdir()
	with pytest.raises(ValueError, match="exists"):
		run_final_comparison(tmp_path / "manifest.json", {}, output)


def test_promotion_binds_calibration_and_protects_evaluated_bundle(tmp_path: Path) -> None:
	import json

	import pytest
	from test_model_bundle import MODEL_CONFIG

	from dataset import file_hash
	from evaluate import PROMOTION_GATES, promote_bundle
	from model_bundle import save_bundle
	from models.factory import create_coverage_model

	bundle = tmp_path / "bundle"
	save_bundle(create_coverage_model(MODEL_CONFIG), bundle, MODEL_CONFIG, "a" * 64)
	metadata = json.loads((bundle / "metadata.json").read_text())
	metadata.update(temperature=2.0, calibration={"split": "calibrate", "manifest_sha256": "b" * 64})
	(bundle / "metadata.json").write_text(json.dumps(metadata))
	decision = {
		"approved": True,
		"gates": {name: True for name in PROMOTION_GATES},
		"seed_decisions": {str(seed): {name: True for name in PROMOTION_GATES} for seed in (17, 42, 73)},
		"evidence": {"dataset_hash": "a" * 64, "weights_sha256": metadata["weights_sha256"], "metadata_sha256": file_hash(bundle / "metadata.json"), "manifest_sha256": "b" * 64},
	}
	pointer = tmp_path / "accepted-model.json"
	metadata["temperature"] = 3.0
	(bundle / "metadata.json").write_text(json.dumps(metadata))
	with pytest.raises(ValueError, match="calibration"):
		promote_bundle(bundle, decision, pointer)
	assert not pointer.exists()
	metadata["temperature"] = 2.0
	(bundle / "metadata.json").write_text(json.dumps(metadata))
	with pytest.raises(ValueError, match="seed"):
		promote_bundle(bundle, {**decision, "seed_decisions": {}}, pointer)
	with pytest.raises(ValueError, match="gate"):
		promote_bundle(bundle, {**decision, "gates": {"arbitrary": True}}, pointer)
	assert promote_bundle(bundle, decision, pointer)
	accepted = Path(json.loads(pointer.read_text())["bundle"])
	assert accepted != bundle and accepted.name == decision["evidence"]["metadata_sha256"]
	assert (accepted / "metadata.json").stat().st_mode & 0o222 == 0
	metadata["temperature"] = 4.0
	(bundle / "metadata.json").write_text(json.dumps(metadata))
	assert json.loads((accepted / "metadata.json").read_text())["temperature"] == 2.0
