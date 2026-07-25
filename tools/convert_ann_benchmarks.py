#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Convert an ANN-Benchmarks dense HDF5 dataset to fvecs/ivecs files."""

import argparse
from pathlib import Path

import h5py
import numpy as np


def write_fvecs(
    dataset: h5py.Dataset,
    path: Path,
    batch_rows: int,
    row_limit: int | None = None,
    normalize_l2: bool = False,
) -> None:
    if dataset.ndim != 2:
        raise ValueError(f"{dataset.name} must be a two-dimensional dense dataset")
    rows, dimension = dataset.shape
    if row_limit is not None:
        rows = min(rows, row_limit)
    with path.open("wb") as output:
        for start in range(0, rows, batch_rows):
            end = min(start + batch_rows, rows)
            values = np.asarray(
                dataset[start:end], dtype="<f4", order="C"
            )
            if normalize_l2:
                norms = np.linalg.norm(values.astype(np.float64), axis=1)
                invalid = np.flatnonzero(~np.isfinite(norms) | (norms <= 0.0))
                if len(invalid):
                    row = start + int(invalid[0])
                    raise ValueError(
                        f"{dataset.name} row {row} cannot be L2-normalized"
                    )
                values /= norms.astype("<f4")[:, np.newaxis]
            records = np.empty((len(values), dimension + 1), dtype="<u4")
            records[:, 0] = dimension
            records[:, 1:] = values.view("<u4")
            records.tofile(output)


def write_ivecs(
    dataset: h5py.Dataset, path: Path, batch_rows: int, row_limit: int | None = None
) -> None:
    if dataset.ndim != 2:
        raise ValueError(f"{dataset.name} must be a two-dimensional dense dataset")
    rows, width = dataset.shape
    if row_limit is not None:
        rows = min(rows, row_limit)
    with path.open("wb") as output:
        for start in range(0, rows, batch_rows):
            end = min(start + batch_rows, rows)
            values = np.asarray(
                dataset[start:end], dtype="<i4", order="C"
            )
            if np.any(values < 0):
                raise ValueError(f"{dataset.name} contains a negative row ID")
            records = np.empty((len(values), width + 1), dtype="<i4")
            records[:, 0] = width
            records[:, 1:] = values
            records.tofile(output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ANN-Benchmarks train/test/neighbors to fvecs/ivecs"
    )
    parser.add_argument("input", type=Path, help="ANN-Benchmarks HDF5 file")
    parser.add_argument("output_dir", type=Path, help="destination directory")
    parser.add_argument(
        "--prefix",
        help="output filename prefix; defaults to the HDF5 filename before the first dot",
    )
    parser.add_argument("--batch-rows", type=int, default=4096)
    parser.add_argument(
        "--query-limit",
        type=int,
        help="convert only the first N test queries and ground-truth rows",
    )
    parser.add_argument(
        "--normalize-l2",
        action="store_true",
        help=(
            "L2-normalize train and test vectors while preserving published "
            "neighbor IDs; use for angular/cosine datasets"
        ),
    )
    args = parser.parse_args()
    if args.batch_rows <= 0:
        parser.error("--batch-rows must be positive")
    if args.query_limit is not None and args.query_limit <= 0:
        parser.error("--query-limit must be positive")

    prefix = args.prefix or args.input.name.split(".", 1)[0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_path = args.output_dir / f"{prefix}_base.fvecs"
    query_path = args.output_dir / f"{prefix}_query.fvecs"
    truth_path = args.output_dir / f"{prefix}_ground_truth.ivecs"

    with h5py.File(args.input, "r") as source:
        required = {"train", "test", "neighbors"}
        missing = required.difference(source.keys())
        if missing:
            raise ValueError(f"missing HDF5 datasets: {', '.join(sorted(missing))}")
        write_fvecs(
            source["train"],
            base_path,
            args.batch_rows,
            normalize_l2=args.normalize_l2,
        )
        write_fvecs(
            source["test"],
            query_path,
            args.batch_rows,
            row_limit=args.query_limit,
            normalize_l2=args.normalize_l2,
        )
        write_ivecs(
            source["neighbors"],
            truth_path,
            args.batch_rows,
            row_limit=args.query_limit,
        )

    for path in (base_path, query_path, truth_path):
        print(f"{path}\t{path.stat().st_size}")


if __name__ == "__main__":
    main()
