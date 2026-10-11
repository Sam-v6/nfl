from pathlib import Path

import pytest


def test_timer_uses_warmup_and_reports_latency_distribution() -> None:
	from benchmark import measure

	time = [0.0]
	calls = [0]

	def operation() -> None:
		calls[0] += 1
		time[0] += 0.002

	result = measure(operation, lambda: None, warmups=5, repeats=30, clock=lambda: time[0])
	assert calls[0] == 35
	assert result["median_ms"] == pytest.approx(2.0)
	assert result["p95_ms"] == pytest.approx(2.0)
	assert result["repeats"] == 30


def test_benchmark_rejects_overwriting_evidence(tmp_path: Path) -> None:
	from benchmark import benchmark

	output = tmp_path / "existing"
	output.mkdir()
	with pytest.raises(ValueError, match="exists"):
		benchmark({}, output)
