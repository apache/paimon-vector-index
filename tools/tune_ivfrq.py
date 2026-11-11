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
#   Unless required by applicable law or agreed to in writing,
#   software distributed under the License is distributed on an
#   "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#   KIND, either express or implied.  See the License for the
#   specific language governing permissions and limitations
#   under the License.

"""Calibrate IVF-RQ with ann_bench, then verify on disjoint held-out queries.

This is an offline L2 benchmark tool. It prints an explicit build/search
configuration and never changes application configuration or production data.
"""

import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import statistics
import struct
import subprocess


def select_candidate(rows, target_recall, max_bytes_per_vector=None):
    eligible = [row for row in rows
                if math.isfinite(row["recall_at_10"])
                and row["recall_at_10"] >= target_recall
                and math.isfinite(row["steady_batch_qps"])
                and row["steady_batch_qps"] > 0
                and (max_bytes_per_vector is None
                     or row["file_bytes"] / row["n"] <= max_bytes_per_vector)]
    if not eligible:
        raise ValueError("no measured configuration meets the recall and storage targets")
    return min(eligible, key=lambda row: (-row["steady_batch_qps"], row["file_bytes"],
                                          row["nprobe"], row["rq_bits"]))


def select_verified_configuration(default, candidate, target, budget=None, min_qps_gain=0.03):
    def meets(row):
        try:
            select_candidate([row], target, budget)
            return True
        except ValueError:
            return False
    candidate_valid, default_valid = meets(candidate), meets(default)
    same_configuration = all(default[key] == candidate[key] for key in ["rq_bits", "nprobe"])
    base_qps, candidate_qps = default["steady_batch_qps"], candidate["steady_batch_qps"]
    gain = candidate_qps / base_qps - 1.0 if (base_qps > 0 and math.isfinite(base_qps)
        and math.isfinite(candidate_qps)) else None
    performance_valid = (candidate_valid and not same_configuration and gain is not None
        and gain >= min_qps_gain)
    smaller_at_same_speed = (candidate_valid and
        candidate["file_bytes"] < default["file_bytes"] * (1.0 - min_qps_gain) and
        gain is not None and gain >= -min_qps_gain)
    if same_configuration:
        selected = default if default_valid else None
    elif candidate_valid and (not default_valid or performance_valid or smaller_at_same_speed):
        selected = candidate
    elif default_valid:
        selected = default
    else:
        selected = None
    return selected, dict(candidate_recall_validated=candidate_valid,
        default_meets_targets=default_valid, performance_validated=performance_valid,
        validation_qps_gain=gain, min_qps_gain=min_qps_gain,
        kept_default=selected is default, same_configuration=same_configuration)


def record_shape(path):
    with path.open("rb") as source:
        header = source.read(4)
    if len(header) != 4:
        raise ValueError(f"empty or truncated vector file: {path}")
    width, = struct.unpack("<i", header)
    stride = 4 * (width + 1)
    if width <= 0 or path.stat().st_size % stride:
        raise ValueError(f"invalid fixed-width vector file: {path}")
    return path.stat().st_size // stride, stride


def copy_records(source, destination, start, count):
    if source.resolve() == destination.resolve():
        raise ValueError("query split must not overwrite its source")
    rows, stride = record_shape(source)
    if start < 0 or count <= 0 or start + count > rows:
        raise ValueError(f"invalid query split for {source}")
    with source.open("rb") as src, destination.open("wb") as dst:
        src.seek(start * stride)
        remaining = count * stride
        while remaining:
            chunk = src.read(min(remaining, 1024 * 1024))
            if not chunk:
                raise ValueError(f"truncated vector file: {source}")
            dst.write(chunk)
            remaining -= len(chunk)


def write_csv(path, rows):
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-executable", type=Path, required=True)
    parser.add_argument("--base-fvecs", type=Path, required=True)
    parser.add_argument("--query-fvecs", type=Path, required=True)
    parser.add_argument("--ground-truth-ivecs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bits", default="3,4,5,6")
    parser.add_argument("--nprobes", default="16,32,64,128")
    parser.add_argument("--target-recall", type=float, default=0.9)
    parser.add_argument("--min-qps-gain", type=float, default=0.03)
    parser.add_argument("--max-bytes-per-vector", type=float)
    parser.add_argument("--calibration-queries", type=int, default=200)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--nlist", type=int, default=1024)
    parser.add_argument("--train-n", type=int, default=65536)
    parser.add_argument("--threads", type=int, default=12)
    args = parser.parse_args()
    bits = sorted(set(int(value) for value in args.bits.split(",")))
    probes = sorted(set(int(value) for value in args.nprobes.split(",")))
    if (not math.isfinite(args.target_recall) or not 0 <= args.target_recall <= 1
            or any(not 1 <= value <= 8 for value in bits)
            or any(not 1 <= value <= args.nlist for value in probes)
            or args.repetitions <= 0 or args.threads <= 0 or args.train_n <= 0
            or not math.isfinite(args.min_qps_gain) or not 0 <= args.min_qps_gain < 1
            or (args.max_bytes_per_vector is not None and
                (not math.isfinite(args.max_bytes_per_vector) or args.max_bytes_per_vector <= 0))):
        parser.error("invalid recall, width, memory, or repetition setting")
    query_count, _ = record_shape(args.query_fvecs)
    truth_count, _ = record_shape(args.ground_truth_ivecs)
    if query_count != truth_count or not 0 < args.calibration_queries < query_count:
        parser.error("query/truth counts must match and leave a nonempty validation split")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "recommendation.json"
    if report_path.resolve() in {args.base_fvecs.resolve(), args.query_fvecs.resolve(), args.ground_truth_ivecs.resolve()}:
        parser.error("output report must not overwrite an input dataset")
    # A failed rerun must not leave a successful recommendation from an older
    # target or dataset looking like the result of this invocation.
    report_path.write_text(json.dumps(dict(target_recall=args.target_recall,
        validated=False, status="running", options=None), indent=2) + "\n")
    files = {}
    for phase, start, count in [("calibration", 0, args.calibration_queries),
                                 ("validation", args.calibration_queries,
                                  query_count - args.calibration_queries)]:
        query = output / f"{phase}.fvecs"
        truth = output / f"{phase}.ivecs"
        copy_records(args.query_fvecs, query, start, count)
        copy_records(args.ground_truth_ivecs, truth, start, count)
        files[phase] = query, truth
    indexes = {}

    def measure(bits, nprobe, phase, repetition=0):
        label = f"{phase}-b{bits}-p{nprobe}-r{repetition}"
        directory = output / label
        directory.mkdir(exist_ok=True)
        env = {key: value for key, value in os.environ.items() if not key.startswith("ANN_")}
        env.update(ANN_INDEXES="IVF_RQ", ANN_DATASET_NAME=label,
                   ANN_BASE_FVECS=str(args.base_fvecs.resolve()),
                   ANN_QUERY_FVECS=str(files[phase][0]), ANN_GROUND_TRUTH_IVECS=str(files[phase][1]),
                   ANN_RQ_BITS=str(bits), ANN_NLIST=str(args.nlist), ANN_NPROBE=str(nprobe),
                   ANN_TRAIN_N=str(args.train_n), ANN_K="10", ANN_KEEP_INDEXES="1",
                   ANN_STORAGE_CASES="local_ssd_warm_cache", ANN_STEADY_MIN_MS="1000",
                   ANN_OUTPUT_DIR=str(directory), RAYON_NUM_THREADS=str(args.threads))
        if bits in indexes:
            env["ANN_REUSE_INDEX_PATH"] = str(indexes[bits])
        existing_indexes = set(directory.rglob("*.index"))
        with (directory / "stderr.log").open("w") as stderr:
            result = subprocess.run([str(args.benchmark_executable.resolve())], env=env,
                                    text=True, stdout=subprocess.PIPE, stderr=stderr, check=True)
        (directory / "result.csv").write_text(result.stdout)
        (directory / "environment.json").write_text(json.dumps(
            {key: value for key, value in env.items()
             if key.startswith("ANN_") or key == "RAYON_NUM_THREADS"}, indent=2) + "\n")
        parsed = list(csv.DictReader(io.StringIO(result.stdout)))
        if len(parsed) != 1:
            raise ValueError("expected exactly one ann_bench CSV result")
        row = {key: float(parsed[0][key]) for key in ["n", "rq_bits", "nprobe", "file_bytes",
               "recall_at_10", "steady_batch_qps", "steady_sequential_qps", "steady_sequential_p95_us"]}
        if bits not in indexes:
            paths = [path for path in directory.rglob("*.index") if path not in existing_indexes]
            if len(paths) != 1:
                raise ValueError(f"expected one retained index in {directory}")
            indexes[bits] = paths[0]
        print(json.dumps(dict(phase=phase, repetition=repetition, **row)), flush=True)
        return row

    rows = [measure(bits, nprobe, "calibration") for bits in bits for nprobe in probes]
    write_csv(output / "calibration.csv", rows)
    try:
        chosen = select_candidate(rows, args.target_recall, args.max_bytes_per_vector)
    except ValueError as error:
        report_path.write_text(json.dumps(dict(target_recall=args.target_recall,
            validated=False, status="no_calibration_candidate", options=None,
            reason=str(error)), indent=2) + "\n")
        raise SystemExit(2) from error
    chosen_bits, chosen_probe = int(chosen["rq_bits"]), int(chosen["nprobe"])
    # Alternate order to reduce thermal/order bias. Validation queries were not
    # used to select the candidate or construct its index.
    validation = {"default": [], "candidate": []}
    for repetition in range(args.repetitions):
        order = ["default", "candidate"] if repetition % 2 == 0 else ["candidate", "default"]
        for name in order:
            b, p = (4, min(64, args.nlist)) if name == "default" else (chosen_bits, chosen_probe)
            row = measure(b, p, "validation", repetition * 2 + (name == "candidate"))
            validation[name].append(row)
    medians = {name: {key: statistics.median(row[key] for row in samples)
                      for key in samples[0]} for name, samples in validation.items()}
    selected, verification = select_verified_configuration(medians["default"], medians["candidate"],
        args.target_recall, args.max_bytes_per_vector, args.min_qps_gain)
    accepted = selected is not None
    report = dict(target_recall=args.target_recall, validated=accepted,
                  status="validated" if accepted else "validation_failed", **verification,
                  calibration_queries=args.calibration_queries,
                  validation_queries=query_count - args.calibration_queries,
                  options={"index.type": "ivf_rq", "rq.bits": int(selected["rq_bits"]),
                           "nlist": args.nlist, "nprobe": int(selected["nprobe"])} if accepted else None,
                  medians=medians, samples=validation)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
