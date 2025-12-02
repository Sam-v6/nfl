#!/usr/bin/env python
"""
Converts CSV files (optionally compressed) to Parquet format.
"""

import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CSV_EXTS = {".csv", ".csv.gz", ".csv.bz2", ".csv.xz"}


def csv_to_parquet(in_path: Path, out_path: Path, chunksize: int = 250_000, compression: str = "zstd") -> None:
	"""
	Streams a CSV file to Parquet, writing in chunks to limit memory.

	Inputs:
	- in_path: Path to the CSV file.
	- out_path: Destination path for the Parquet file.
	- chunksize: Rows per chunk to process at a time.
	- compression: Parquet compression codec.

	Outputs:
	- Writes a Parquet file to out_path.
	"""
	out_path.parent.mkdir(parents=True, exist_ok=True)
	writer = None
	try:
		for chunk in pd.read_csv(in_path, chunksize=chunksize, low_memory=False):
			table = pa.Table.from_pandas(chunk, preserve_index=False)

			if writer is None:
				writer = pq.ParquetWriter(where=str(out_path), schema=table.schema, compression=compression, use_dictionary=True)
			writer.write_table(table)

		if writer is None:
			empty_tbl = pa.Table.from_pandas(pd.DataFrame(), preserve_index=False)
			pq.write_table(empty_tbl, out_path, compression=compression)
	finally:
		if writer is not None:
			writer.close()


def main() -> None:
	"""
	CLI entry point to convert all CSV files under a directory tree.

	Inputs:
	- sys.argv[1]: Root directory containing CSVs.
	- sys.argv[2]: Destination root for Parquet outputs.

	Outputs:
	- Writes converted Parquet files mirroring the input tree.
	"""
	if len(sys.argv) != 3:
		print("Usage: python csv_to_parquet.py /path/to/csv_root /path/to/parquet_out")
		sys.exit(1)

	in_root = Path(sys.argv[1]).resolve()
	out_root = Path(sys.argv[2]).resolve()

	if not in_root.exists():
		print(f"Input path does not exist: {in_root}")
		sys.exit(1)

	converted = 0
	for p in in_root.rglob("*"):
		if p.is_file() and any(str(p).lower().endswith(ext) for ext in CSV_EXTS):
			rel = p.relative_to(in_root)
			out_rel = rel.with_suffix(".parquet")
			out_path = out_root / out_rel

			print(f"[convert] {p} -> {out_path}")
			csv_to_parquet(p, out_path)
			converted += 1

	print(f"Done. Converted {converted} file(s) to {out_root}")


if __name__ == "__main__":
	main()
