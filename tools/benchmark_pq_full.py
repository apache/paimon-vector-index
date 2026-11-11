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

"""Run the PQ benchmark suite on two disposable, already-compiled checkouts.

First run benchmark_pr.py, copy candidate benchmark drivers/Cargo.toml to the
baseline, then compile --benches --no-run --message-format=json into
OUTPUT/build-{base,candidate}.jsonl with separate target directories. The
Fashion-MNIST fvecs directory comes from convert_ann_benchmarks.py --query-limit 256.
Production sources are fingerprinted before and after; this tool changes no code.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import sys
import time

from benchmark_pr import library_fingerprint, run_checked


def fingerprint(checkout):
    return library_fingerprint(checkout)


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture_inputs(checkouts, executables, plan, dataset, environment):
    drivers = {side: {str(p.relative_to(checkout)): file_digest(p)
                      for p in sorted((checkout / "core/benches").rglob("*.rs"))}
               for side, checkout in checkouts.items()}
    if drivers["base"] != drivers["candidate"]:
        raise ValueError("Base and candidate benchmark drivers must match")
    targets = {job["bench"] for job in plan}
    binaries = {side: {name: dict(path=str(Path(paths[name]).resolve()),
                                 sha256=file_digest(paths[name])) for name in sorted(targets)}
                for side, paths in executables.items()}
    datasets = {name: file_digest(dataset / name) for name in (
        "fashion_base.fvecs", "fashion_query.fvecs", "fashion_ground_truth.ivecs")}
    # Store only known runtime controls, never credentials from the environment.
    runtime_keys = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "RAYON_RS_NUM_CPUS",
                    "LD_LIBRARY_PATH", "LD_PRELOAD", "DYLD_LIBRARY_PATH",
                    "DYLD_INSERT_LIBRARIES", "MALLOC_CONF", "MallocNanoZone")
    return dict(sources={side: fingerprint(checkout) for side, checkout in checkouts.items()},
                drivers=drivers, binaries=binaries, datasets=datasets,
                runner_sha256=file_digest(__file__),
                runtime=dict(platform=platform.platform(), machine=platform.machine(),
                             processor=platform.processor(), cpu_count=os.cpu_count(),
                             host_sha256=hashlib.sha256(platform.node().encode()).hexdigest(),
                             environment={key: environment.get(key) for key in runtime_keys}))


def job_id(job):
    return f"{job['case']}-r{job['round']}-{job['side']}"


def validate_runs(metadata, complete):
    planned = {job_id(job): job for job in metadata["plan"]}
    if not planned or len(planned) != len(metadata["plan"]):
        raise ValueError("Benchmark plan must contain unique process IDs")
    rounds = {}
    for job in planned.values():
        rounds.setdefault(job["case"], {}).setdefault(job["side"], set()).add(job["round"])
    for case, sides in rounds.items():
        if set(sides) != {"base", "candidate"} or sides["base"] != sides["candidate"]:
            raise ValueError(f"Unbalanced planned rounds for {case}")
    seen = set()
    for run in metadata["runs"]:
        name = run["id"]
        if name in seen or planned.get(name) != run["job"]:
            raise ValueError(f"Duplicate or unexpected benchmark process: {name}")
        seen.add(name)
        if run["status"] not in ("ok", "failed") or (complete and run["status"] != "ok"):
            raise ValueError(f"Unsuccessful benchmark process: {name}")
    if complete and (seen != set(planned) or metadata["failures"]):
        raise ValueError("Missing or failed planned benchmark processes")


def executable_map(path):
    result = {}
    for line in path.read_text().splitlines():
        item = json.loads(line)
        if item.get("reason") == "compiler-artifact" and item.get("executable") and "bench" in item["target"]["kind"]:
            result[item["target"]["name"]] = item["executable"]
    return result


def jobs(dataset):
    result = []

    def paired(case, bench, threads, options=None, rounds=2, args=None):
        for repetition in range(rounds):
            order = ["base", "candidate"] if repetition % 2 == 0 else ["candidate", "base"]
            for side in order:
                result.append(dict(case=case, bench=bench, threads=threads,
                                   round=repetition + 1, side=side, options=options or {}, args=args or []))

    # Full production filter shape, including both sides of the sparse crossover.
    for bits in [4, 8]:
        paired(f"filter-b{bits}", "ivfpq_filter_scan_bench", 1, {"IVFPQ_BENCH_BITS": str(bits)})
    for threads in [1, 4]:
        for bench in ["pq_query_bench", "ivfpq_add_bench"]:
            paired(f"{bench}-t{threads}", bench, threads, rounds=1,
                   args=["--bench", "--save-baseline", "measured"])
        paired(f"reuse-t{threads}", "ivfpq_batch_reuse_bench", threads)
        paired(f"train-t{threads}", "ivfpq_train_bench", threads, {"TRAIN_TARGET": "1"})
        paired(f"recall-t{threads}", "recall_bench", threads)
        paired(f"pq4-t{threads}", "pq4_bench", threads)
        for bits in [4, 8]:
            paired(f"diskann-b{bits}-t{threads}", "diskann_bench", threads, {
                "DISKANN_BENCH_PQ_BITS": str(bits), "DISKANN_BENCH_SMOKE_N": "10000",
                "DISKANN_BENCH_SMOKE_NQ": "64", "DISKANN_BENCH_SMOKE_DIMENSION": "128",
                "DISKANN_BENCH_CONCURRENCY": str(threads), "DISKANN_BENCH_FILTERED_MATRIX": "1"})

    common = dict(ANN_INDEXES="IVF_PQ", ANN_STORAGE_CASES="local_ssd_warm_cache", ANN_NQ="256",
                  ANN_TRAIN_N="16384", ANN_K="10", ANN_NLIST="256", ANN_NPROBE="8",
                  ANN_STEADY_MIN_MS="1000", ANN_SEED="42", ANN_CLUSTERS="32")
    for threads in [1, 4]:
        for dimension in [128, 768]:
            config = dict(common, ANN_DATASET_NAME=f"pq-full-synthetic-{dimension}", ANN_N="100000",
                          ANN_D=str(dimension), ANN_PQ_M=str(dimension // 8), ANN_NOISE_DIMENSIONS=str(dimension))
            paired(f"ann-d{dimension}-t{threads}", "ann_bench", threads, config, rounds=4)
        for probes in [8, 32]:
            config = dict(common, ANN_DATASET_NAME="fashion-mnist-784-euclidean", ANN_N="60000",
                          ANN_D="784", ANN_PQ_M="98", ANN_NPROBE=str(probes),
                          ANN_BASE_FVECS=str(dataset / "fashion_base.fvecs"),
                          ANN_QUERY_FVECS=str(dataset / "fashion_query.fvecs"),
                          ANN_GROUND_TRUTH_IVECS=str(dataset / "fashion_ground_truth.ivecs"))
            paired(f"fashion-p{probes}-t{threads}", "ann_bench", threads, config, rounds=4)
    # Exercise the project's simulated storage profiles separately from warm QPS.
    paired("ann-storage-profiles", "ann_bench", 4, dict(
        ANN_INDEXES="IVF_PQ", ANN_DATASET_NAME="pq-full-storage", ANN_N="10000", ANN_TRAIN_N="4096",
        ANN_NQ="64", ANN_D="128", ANN_PQ_M="16", ANN_K="10", ANN_NLIST="64", ANN_NPROBE="8",
        ANN_SEED="42", ANN_STORAGE_CASES="local_ssd_warm_cache,remote_cache_2ms,object_store_20ms",
        ANN_STEADY_MIN_MS="0"))
    # Independently repeat the unrelated-index variation seen in the CI suite.
    paired("rq-confirmation", "ann_bench", 2, dict(
        ANN_INDEXES="IVF_RQ", ANN_DATASET_NAME="ci-clustered-v1", ANN_N="10000", ANN_TRAIN_N="4096",
        ANN_NQ="2048", ANN_D="64", ANN_PQ_M="8", ANN_K="10", ANN_NLIST="64", ANN_NPROBE="8",
        ANN_CLUSTERS="32", ANN_NOISE_DIMENSIONS="64", ANN_SEED="42", ANN_RQ_BITS="4",
        ANN_STORAGE_CASES="local_ssd_warm_cache", ANN_STEADY_MIN_MS="1000"), rounds=6)
    for metric in ["l2", "ip", "cosine"]:
        for opq in [0, 1]:
            for bits in [4, 8]:
                paired(f"quality-{metric}-opq{opq}-b{bits}", "pq_quality_bench", 4,
                       {"PQ_QUALITY_METRIC": metric, "PQ_QUALITY_OPQ": str(opq), "PQ_QUALITY_BITS": str(bits)}, rounds=1)
    for bits in [4, 8]:
        paired(f"quality-fashion-b{bits}", "pq_quality_bench", 4,
               {"PQ_QUALITY_METRIC": "l2", "PQ_QUALITY_OPQ": "0", "PQ_QUALITY_BITS": str(bits),
                "PQ_QUALITY_DATA_DIR": str(dataset)}, rounds=1)
    paired("add-confirmation-t4", "ivfpq_add_bench", 4, rounds=2,
           args=["--bench", "rows(7|31)$", "--save-baseline", "measured"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=900, help="Per-process timeout seconds")
    args = parser.parse_args()
    root = args.output.resolve()
    raw = root / "suite-raw"
    raw.mkdir(exist_ok=True)
    checkouts = {"base": args.base.resolve(), "candidate": args.candidate.resolve()}
    executables = {side: executable_map(root / f"build-{side}.jsonl") for side in checkouts}
    plan = jobs(args.dataset_dir.resolve())
    clean = {key: value for key, value in os.environ.items()
             if not key.startswith(("ANN_", "DISKANN_", "TRAIN_", "IVFPQ_BENCH_", "RAYON_", "CRITERION_", "PQ_", "PAIMON_VINDEX_LOG_"))}
    inputs = capture_inputs(checkouts, executables, plan, args.dataset_dir.resolve(), clean)
    metadata = dict(platform=platform.platform(), sources=inputs["sources"],
                    drivers=inputs["drivers"]["candidate"], inputs=inputs, plan=plan,
                    command=sys.argv, timeout=args.timeout, runs=[], failures=[])
    metadata_path = root / "suite.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if previous.get("inputs") != inputs or previous["plan"] != plan:
            raise ValueError("Cannot resume with changed or unrecorded benchmark inputs; use a new output directory")
        metadata = previous
    validate_runs(metadata, complete=False)
    completed = {run["id"] for run in metadata["runs"] if run["status"] == "ok"}
    for index, job in enumerate(plan):
        name = job_id(job)
        if name in completed:
            continue
        print(f"[{index+1}/{len(plan)}] {name}", flush=True)
        options = dict(clean, RAYON_NUM_THREADS=str(job["threads"]), LC_ALL="C", **job["options"])
        options["ANN_OUTPUT_DIR"] = str(root / "indexes")
        options["CRITERION_HOME"] = str(root / "criterion" / name)
        exe = executables[job["side"]][job["bench"]]
        command = ["/usr/bin/time", "-l" if platform.system() == "Darwin" else "-v", exe, *job["args"]]
        started = time.monotonic()
        record = dict(id=name, job=job, command=command, status="ok")
        try:
            with (raw / f"{name}.stdout").open("w") as out, (raw / f"{name}.stderr").open("w") as err:
                run_checked(command, deadline=started + args.timeout, timeout=args.timeout,
                            cwd=root, env=options, stdout=out, stderr=err)
        except Exception as error:
            record.update(status="failed", error=str(error))
        record["wall_seconds"] = time.monotonic() - started
        metadata["runs"] = [run for run in metadata["runs"] if run["id"] != name]
        metadata["failures"] = [failure for failure in metadata["failures"] if failure != name]
        if record["status"] == "failed":
            metadata["failures"].append(name)
        metadata["runs"].append(record)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        if record["status"] == "failed":
            print(f"FAILED {name}: {record['error']}", flush=True)
    if capture_inputs(checkouts, executables, plan, args.dataset_dir.resolve(), clean) != inputs:
        raise RuntimeError("Benchmark inputs changed during measurement")
    if any(run["status"] != "ok" for run in metadata["runs"]):
        raise RuntimeError("Some benchmark processes failed; inspect suite.json and raw logs")
    summarize(root)
    print(f"Completed {len(plan)} processes, benchmark inputs unchanged", flush=True)


def summarize(root):
    """Preserve every parsed measurement, then compare like-for-like medians."""
    root = Path(root)
    metadata = json.loads((root / "suite.json").read_text())
    validate_runs(metadata, complete=True)
    observations = []
    process_resources = []
    coverage_by_case = {}
    header_prefixes = ("dataset,", "density_percent,", "mode,", "read_tier,", "filtered_distribution,")
    measurement_fields = {"p50_ms", "wall_ms", "qps", "elapsed_ms", "query_ms", "train_total_s",
                          "mean_ns", "sequential_qps", "batch_qps", "steady_sequential_qps", "steady_batch_qps"}

    def numeric_values(values):
        result = {}
        for key, raw in values.items():
            if key.endswith("_signature"):
                continue
            try:
                value = float(raw)
            except (ValueError, TypeError):
                continue
            if not math.isfinite(value):
                raise ValueError(f"Non-finite benchmark value: {key}={raw}")
            result[key] = value
        return result

    def add(job, subcase, values):
        if not measurement_fields.intersection(numeric_values(values)):
            raise ValueError(f"No usable measurement for {job_id(job)}/{subcase}")
        observations.append(dict(case=job["case"], subcase=subcase, side=job["side"],
                                 round=job["round"], values=values))

    for run in metadata["runs"]:
        if run["status"] != "ok":
            continue
        first_observation = len(observations)
        job = run["job"]
        stdout = (root / "suite-raw" / f"{run['id']}.stdout").read_text()
        stderr = (root / "suite-raw" / f"{run['id']}.stderr").read_text()
        rss = re.search(r"(\d+)\s+maximum resident set size", stderr)
        peak_rss = int(rss[1]) if rss else None
        if peak_rss is None:
            rss = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", stderr)
            peak_rss = int(rss[1]) * 1024 if rss else None
        process_resources.append(dict(id=run["id"], wall_seconds=run["wall_seconds"],
                                      peak_rss_bytes=peak_rss))
        header = None
        scenario = ""
        for line in stdout.splitlines():
            if line.startswith(header_prefixes):
                header = next(csv.reader([line]))
                continue
            if header and "," in line:
                columns = next(csv.reader([line]))
                if len(columns) == len(header):
                    row = dict(zip(header, columns))
                    if "dataset" in row and "index" in row:
                        key = f"{row['index']}/{row['storage']}"
                    elif "dataset" in row:
                        key = f"{row['metric']}/b{row['bits']}/opq{row['opq']}/filter{row['density']}"
                    elif "density_percent" in row:
                        key = row["density_percent"]
                    elif "mode" in row:
                        key = f"{row['mode']}/{row['percentile']}"
                    elif "read_tier" in row:
                        key = f"{row['read_tier']}/l{row['l_search']}"
                    else:
                        key = f"{row['filtered_distribution']}/{row['selectivity_percent']}"
                    add(job, key, row)
            if line.startswith("scenario: "):
                scenario = line.split(",")[0].split(": ", 1)[1]
            if job["bench"] == "recall_bench" and line.startswith(("IVF-PQ ", "IVF-FLAT ", "IVF-SQ ")):
                fields = line.split()
                add(job, f"{scenario}/{fields[0]}/p{fields[1]}",
                    dict(recall_percent=fields[2].rstrip("%"), query_ms=fields[3], us_per_query=fields[4]))
            if job["bench"] == "ivfpq_train_bench" and line.startswith(("small ", "target-768 ")):
                fields = line.split()
                add(job, fields[0], dict(zip(
                    ["n", "d", "nlist", "pq_m", "prep_s", "coarse_s", "pq_s", "train_total_s"], fields[1:])))
        if job["bench"] == "pq4_bench":
            numbers = lambda text: re.findall(r"[0-9]+(?:\.[0-9]+)?", text)
            for line in stdout.splitlines():
                if "Build time:" in line:
                    build = numbers(line)
                elif "Query (nq=" in line:
                    timing = numbers(line)
                elif "Recall@" in line:
                    recall = numbers(line)
            for i, bits in enumerate([8, 4]):
                add(job, str(bits), dict(build_s=build[i], query_ms=timing[1 + i * 2],
                                        us_per_query=timing[2 + i * 2], recall_percent=recall[1 + i]))
        for estimate_path in (root / "criterion" / run["id"]).rglob("measured/estimates.json"):
            estimate = json.loads(estimate_path.read_text())["mean"]
            info = json.loads((estimate_path.parent / "benchmark.json").read_text())
            add(job, info["full_id"], dict(mean_ns=estimate["point_estimate"],
                ci95_lower_ns=estimate["confidence_interval"]["lower_bound"],
                ci95_upper_ns=estimate["confidence_interval"]["upper_bound"]))
        coverage = set()
        for observation in observations[first_observation:]:
            for metric in numeric_values(observation["values"]):
                key = (observation["subcase"], metric)
                if key in coverage:
                    raise ValueError(f"Duplicate measurement in {run['id']}: {key}")
                coverage.add(key)
        if not coverage:
            raise ValueError(f"No measurements parsed for {run['id']}")
        expected = coverage_by_case.setdefault(job["case"], coverage)
        if coverage != expected:
            raise ValueError(f"Inconsistent measurement coverage for {run['id']}")

    grouped = {}
    for observation in observations:
        for metric, value in numeric_values(observation["values"]).items():
            key = (observation["case"], observation["subcase"], metric)
            grouped.setdefault(key, {"base": [], "candidate": []})[observation["side"]].append(value)
    comparisons = []
    for (case, subcase, metric), sides in sorted(grouped.items()):
        if not all(sides.values()):
            raise ValueError(f"Missing side for {case}/{subcase}/{metric}")
        before, after = (statistics.median(sides[side]) for side in ["base", "candidate"])
        comparisons.append(dict(case=case, subcase=subcase, metric=metric, base=before, candidate=after,
                                change_percent=(after / before - 1) * 100 if before else None,
                                base_samples=sides["base"], candidate_samples=sides["candidate"]))
    result = dict(completed=len(metadata["runs"]), planned=len(metadata["plan"]),
                  failures=metadata["failures"], observations=observations,
                  comparisons=comparisons, process_resources=process_resources)
    (root / "measurements.json").write_text(json.dumps(result, indent=2) + "\n")
    with (root / "comparisons.csv").open("w", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["case", "subcase", "metric", "base", "candidate", "change_percent", "base_min", "base_max", "candidate_min", "candidate_max", "samples_per_side"])
        for row in comparisons:
            writer.writerow([row[key] for key in ["case", "subcase", "metric", "base", "candidate", "change_percent"]] +
                            [min(row["base_samples"]), max(row["base_samples"]), min(row["candidate_samples"]),
                             max(row["candidate_samples"]), len(row["base_samples"])])
    return result


if __name__ == "__main__":
    main()
