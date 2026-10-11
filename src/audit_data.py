"""Read-only inventory of local Parquet inputs."""

import json
from pathlib import Path


def audit_data(data_root: Path, output_dir: Path) -> Path:
	import pyarrow.parquet as pq

	files = sorted(data_root.rglob("*.parquet")) if data_root.is_dir() else []
	if not files:
		raise ValueError(f"No Parquet data found at {data_root}")
	inventory = []
	for path in files:
		parquet = pq.ParquetFile(path)
		inventory.append({"path": str(path.relative_to(data_root)), "rows": parquet.metadata.num_rows, "columns": parquet.schema_arrow.names})
	output_dir.mkdir(parents=True, exist_ok=False)
	report = output_dir / "audit.json"
	report.write_text(json.dumps({"data_root": str(data_root), "files": inventory}, indent=2) + "\n")
	return report
