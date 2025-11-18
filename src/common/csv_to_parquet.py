#!/usr/bin/env python

"""
Transforms raw csv data to raw parquet data

Requires:
- Raw location tracking data in nfl/data/parquet

Outputs parquet data in nfl/data/parquet that is stored in git LFS
"""

import sys
import os
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CSV_EXTS = {".csv", ".csv.gz", ".csv.bz2", ".csv.xz"}

def csv_to_parquet(in_path: Path, out_path: Path, chunksize: int = 250_000, compression: str = "zstd"):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare a writer lazily after first chunk defines schema
    writer = None
    try:
        # Tweak read_csv args as needed (parse dates, dtype, etc.)
        # Example: parse_dates=['timestamp'] if you have a time column
        for i, chunk in enumerate(pd.read_csv(in_path, chunksize=chunksize, low_memory=False)):
            # Convert to Arrow table
            table = pa.Table.from_pandas(chunk, preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(
                    where=str(out_path),
                    schema=table.schema,
                    compression=compression,
                    # Parquet v2 encodings generally smaller/better
                    use_dictionary=True
                )
            writer.write_table(table)

        if writer is None:
            # Empty CSV -> write empty parquet with no rows
            empty_tbl = pa.Table.from_pandas(pd.DataFrame(), preserve_index=False)
            pq.write_table(empty_tbl, out_path, compression=compression)
    finally:
        if writer is not None:
            writer.close()

def main():
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
            # Replace only the final extension with .parquet
            out_rel = rel.with_suffix(".parquet")
            out_path = out_root / out_rel

            print(f"[convert] {p} -> {out_path}")
            csv_to_parquet(p, out_path)
            converted += 1

    print(f"Done. Converted {converted} file(s) to {out_root}")

if __name__ == "__main__":
    main()
