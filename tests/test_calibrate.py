from pathlib import Path

import numpy as np
import pytest


def test_temperature_preserves_argmax_and_improves_overconfidence() -> None:
	from calibrate import apply_temperature, fit_temperature

	logits = np.full((70, 7), -8.0)
	logits[np.arange(70), np.arange(70) % 7] = 8.0
	targets = np.arange(70) % 7
	targets[::4] = (targets[::4] + 1) % 7
	temperature = fit_temperature(logits, targets)
	before = apply_temperature(logits, 1.0)
	after = apply_temperature(logits, temperature)
	assert temperature > 1.0
	assert np.array_equal(before.argmax(1), after.argmax(1))
	assert -np.log(after[np.arange(70), targets]).mean() < -np.log(before[np.arange(70), targets]).mean()


@pytest.mark.parametrize("temperature", [0.0, -1.0, np.nan, np.inf])
def test_invalid_temperature_fails(temperature: float) -> None:
	from calibrate import apply_temperature

	with pytest.raises(ValueError, match="temperature"):
		apply_temperature(np.zeros((3, 7)), temperature)


def test_calibration_refuses_final_ids_before_reading_data(tmp_path: Path) -> None:
	from calibrate import calibrate_bundle

	with pytest.raises(ValueError, match="calibrate"):
		calibrate_bundle(tmp_path / "bundle", tmp_path / "manifest.json", tmp_path / "out", split="final")
	assert not (tmp_path / "out").exists()
