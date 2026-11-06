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

"""Compare two disposable checkouts using the candidate's ANN benchmark driver."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import statistics
import subprocess
import time


INDEXES = ("IVF_FLAT", "IVF_SQ", "IVF_PQ", "IVF_RQ", "DISKANN")
WORKLOAD = {
    "ANN_DATASET_NAME": "ci-clustered-v1",
    "ANN_N": "10000",
    "ANN_TRAIN_N": "4096",
    "ANN_NQ": "2048",
    "ANN_D": "64",
    "ANN_K": "10",
    "ANN_NLIST": "64",
    "ANN_NPROBE": "8",
    "ANN_PQ_M": "8",
    "ANN_RQ_BITS": "4",
    "ANN_CLUSTERS": "32",
    "ANN_NOISE_DIMENSIONS": "64",
    "ANN_SEED": "42",
    "ANN_DISKANN_L_SEARCHES": "100",
    "ANN_DISKANN_MEMORY_BUDGET_BYTES": str(256 * 1024 * 1024),
    "ANN_DISKANN_BUILD_DISTANCE": "full_precision",
    "ANN_DISKANN_RAW_VECTOR_ENCODING": "f32",
    "ANN_STORAGE_CASES": "local_ssd_warm_cache",
    "RAYON_NUM_THREADS": "2",
    "ANN_STEADY_MIN_MS": "1000",
}
# These identify the workload, not measured results. Never compare unlike cases.
CASE_FIELDS = (
    "dataset", "index", "storage", "n", "train_n", "nq", "d", "k", "nlist",
    "nprobe", "pq_m", "rq_bits", "diskann_build_distance",
    "diskann_raw_vector_encoding", "l_search", "steady_min_ms",
)
METRICS = (
    ("recall_at_10", "Batch Recall@10", "recall"),
    ("steady_sequential_qps", "Warm sequential QPS", "higher"),
    ("steady_batch_qps", "Warm batch QPS", "higher"),
    ("steady_sequential_p95_us", "Warm sequential P95 (µs)", "lower"),
    ("build_ms", "Build (ms)", "lower"),
    ("peak_rss_bytes", "Process peak RSS up to build completion (MiB)", "lower"),
    ("file_bytes", "Index size (MiB)", "lower"),
)


def command_output(command, cwd=None):
    return subprocess.check_output(command, cwd=cwd, text=True, timeout=30).strip()


def run_checked(command, *, deadline, timeout, **kwargs):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Benchmark total time budget exhausted; stopping before next process")
    limit = min(timeout, remaining)
    process = subprocess.Popen(command, start_new_session=os.name == "posix", **kwargs)
    try:
        returncode = process.wait(timeout=limit)
    except subprocess.TimeoutExpired as error:
        # Cargo can leave rustc children alive if only the parent is killed.
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait()
        raise TimeoutError(
            f"{command[0]} exceeded {limit:.1f}s process/remaining total budget"
        ) from error
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def numeric_field(row, field, path, *, integer=False):
    raw = row.get(field)
    if raw is None or not raw.strip():
        raise ValueError(f"{path}: missing or empty metric/parameter {field}")
    try:
        value = int(raw) if integer else float(raw)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{path}: invalid {field}={raw!r}; expected a number") from error
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{path}: invalid {field}={raw!r}; expected finite nonnegative value")
    return value


def read_sample(path, index):
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 1:
        raise ValueError(f"{path}: expected exactly one case, found {len(rows)}")
    row = rows[0]
    if row.get("index") != index:
        raise ValueError(f"{path}: expected index {index}")
    for field in CASE_FIELDS:
        if not row.get(field):
            raise ValueError(f"{path}: missing workload field {field}")
    for field, _, _ in METRICS:
        numeric_field(row, field, path)
    if not 0 <= float(row["recall_at_10"]) <= 1 or numeric_field(row, "nq", path, integer=True) <= 0:
        raise ValueError(f"{path}: invalid recall or query count")
    if float(row["steady_sequential_qps"]) <= 0 or float(row["steady_batch_qps"]) <= 0:
        raise ValueError(f"{path}: throughput must be positive")
    minimum = numeric_field(row, "steady_min_ms", path, integer=True)
    if minimum <= 0:
        raise ValueError(f"{path}: steady_min_ms must be positive")
    for mode in ("sequential", "batch"):
        elapsed = numeric_field(row, f"steady_{mode}_ms", path)
        queries = numeric_field(row, f"steady_{mode}_queries", path, integer=True)
        if elapsed < minimum or queries < int(row["nq"]) or queries % int(row["nq"]):
            raise ValueError(f"{path}: incomplete steady_{mode} measurement")
    return row


def metric_value(row, field):
    value = float(row[field])
    if field in ("peak_rss_bytes", "file_bytes"):
        return value / (1024 * 1024)
    return value


def delta_text(base, candidate, direction):
    if direction == "recall":
        return f"{(candidate - base) * 100:+.2f} pp"
    if base == 0:
        return "0.0%" if candidate == 0 else "n/a (base=0)"
    return f"{(candidate / base - 1) * 100:+.1f}%"


def summarize(samples, rounds):
    results = {}
    for index in INDEXES:
        sides = samples[index]
        if any(len(sides[side]) != rounds for side in ("base", "candidate")):
            raise ValueError(f"{index}: incomplete samples")
        cases = {
            tuple(row[field] for field in CASE_FIELDS)
            for rows in sides.values() for row in rows
        }
        if len(cases) != 1:
            raise ValueError(f"{index}: workload differs between samples")
        result = {}
        for field, label, direction in METRICS:
            entry = {"label": label, "direction": direction}
            for side in ("base", "candidate"):
                values = [metric_value(row, field) for row in sides[side]]
                entry[side] = {
                    "median": statistics.median(values),
                    "min": min(values), "max": max(values), "samples": values,
                }
            entry["delta"] = delta_text(
                entry["base"]["median"], entry["candidate"]["median"], direction
            )
            result[field] = entry
        results[index] = result
    return results


def query_alert_level(entry):
    """Advisory levels for recall and QPS only; other metrics have no thresholds."""
    if entry["direction"] not in ("recall", "higher"):
        raise ValueError("Query alerts only support recall and QPS")
    base, candidate = entry["base"]["median"], entry["candidate"]["median"]
    if entry["direction"] == "recall":
        loss = round((base - candidate) * 100, 6)
        yellow, red = 1, 3
    else:
        loss = round((1 - candidate / base) * 100, 6) if base else 0
        yellow, red = 10, 20
    return 2 if loss > red else 1 if loss > yellow else 0


def render_report(metadata, results):
    focus = (("recall_at_10", "Recall"), ("steady_sequential_qps", "single QPS"),
             ("steady_batch_qps", "batch QPS"))
    levels = {index: max(query_alert_level(metrics[field]) for field, _ in focus)
              for index, metrics in results.items()}
    flagged = sum(level > 0 for level in levels.values())
    icon = ("🟢", "🟡", "🔴")[max(levels.values(), default=0)]
    headline = (f"{flagged}/{len(results)} indexes need a look" if flagged
                else f"No alerts ({len(results)}/{len(results)} indexes)")
    lines = ["## Vector index benchmark", "", f"**{icon} {headline}**", ""]
    if metadata.get("calibration"):
        lines += ["A/A calibration — identical core code; deltas show measurement variation.", ""]
    lines += ["| Index | Recall@10 (Δ) | Single QPS Δ | Batch QPS Δ | Status |",
              "|---|---:|---:|---:|---|"]
    for index, metrics in results.items():
        recall = metrics["recall_at_10"]
        reasons = [label for field, label in focus if query_alert_level(metrics[field])]
        status = ("🟢 OK" if not reasons else
                  f"{('🟢', '🟡', '🔴')[levels[index]]} {', '.join(reasons)}")
        lines.append(
            f"| {index} | {recall['candidate']['median'] * 100:.2f}% ({recall['delta']}) | "
            f"{metrics['steady_sequential_qps']['delta']} | "
            f"{metrics['steady_batch_qps']['delta']} | {status} |"
        )
    lines += ["", "🟡 QPS ↓ >10% or Recall ↓ >1pp · 🔴 QPS ↓ >20% or Recall ↓ >3pp. "
              "Advisory only; QPS is measured after warmup.", "",
              "<details><summary>Absolute metrics & environment</summary>", ""]
    lines += render_details(metadata, results)
    lines += ["", "</details>", ""]
    return "\n".join(lines)


def render_details(metadata, results):
    fields = ("steady_sequential_qps", "steady_batch_qps", "steady_sequential_p95_us",
              "build_ms", "peak_rss_bytes", "file_bytes")
    lines = [
        "Medians, base → PR. QPS is timed after warmup; P95 uses the first warm sweep.", "",
        "| Index | Single QPS | Batch QPS | P95 (µs) | Build (ms) | Process RSS (MiB) | Size (MiB) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for index, metrics in results.items():
        values = [f"{metrics[field]['base']['median']:,.2f} → "
                  f"{metrics[field]['candidate']['median']:,.2f}" for field in fields]
        lines.append(f"| {index} | " + " | ".join(values) + " |")
    lines += [
        "",
        f"Base `{metadata['base_sha']}` · PR merge `{metadata['candidate_sha']}`",
        f"{metadata['rustc'].splitlines()[0]} · {metadata['platform']} · {metadata['cpu']}",
        f"{metadata['rounds']} samples/version/index, alternating execution order; "
        "2 threads; 10k × 64D vectors; 2,048 queries; ≥1s per timed mode.",
        "RSS is the process lifetime peak up to build completion, including dataset/ground truth.",
        "Raw CSVs, sample ranges, workload settings and logs are in the workflow artifact.",
    ]
    return lines


def build(checkout, output, side, env, deadline):
    # Separate target directories prevent artifacts from one revision leaking into the other.
    command = ["cargo", "bench", "--locked", "-p", "paimon-vindex-core", "--bench",
               "ann_bench", "--no-run", "--message-format=json", "--target-dir",
               str(output / "build" / side)]
    print(f"Building {side}", flush=True)
    messages_path = output / f"build-{side}.jsonl"
    with messages_path.open("w") as messages, (output / f"build-{side}.log").open("w") as log:
        run_checked(command, cwd=checkout, env=env, stdout=messages, stderr=log,
                    deadline=deadline, timeout=900)
    executables = []
    for line in messages_path.read_text().splitlines():
        message = json.loads(line)
        if (message.get("reason") == "compiler-artifact"
                and message.get("target", {}).get("name") == "ann_bench"
                and message.get("executable")):
            executables.append(message["executable"])
    if len(executables) != 1:
        raise ValueError(f"{side}: expected one benchmark executable")
    return executables[0]


def library_fingerprint(checkout):
    paths = list((checkout / "core/src").rglob("*"))
    paths += [checkout / name for name in ("Cargo.toml", "Cargo.lock", "core/Cargo.toml", "core/build.rs")]
    paths += list((checkout / ".cargo").rglob("*"))
    digest = hashlib.sha256()
    for path in sorted(path for path in paths if path.is_file()):
        digest.update(str(path.relative_to(checkout)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def run(args):
    deadline = time.monotonic() + args.total_timeout
    output = args.output.resolve()
    base, candidate = args.base.resolve(), args.candidate.resolve()
    if base == candidate:
        raise ValueError("base and candidate must be separate disposable checkouts")
    if command_output(["git", "status", "--porcelain"], base):
        raise ValueError("base checkout must be clean before copying the shared driver")
    metadata = {
        "base_sha": command_output(["git", "rev-parse", "HEAD"], base),
        "candidate_sha": command_output(["git", "rev-parse", "HEAD"], candidate),
        "rustc": command_output(["rustc", "--version", "--verbose"]),
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "rounds": args.rounds, "workload": WORKLOAD,
        "candidate_dirty": bool(command_output(["git", "status", "--porcelain"], candidate)),
        "execution_order": [],
        "total_timeout_seconds": args.total_timeout,
        "base_library_sha256": library_fingerprint(base),
        "candidate_library_sha256": library_fingerprint(candidate),
        "rustflags": "-C target-cpu=x86-64" if platform.machine() == "x86_64" else "",
    }
    metadata["calibration"] = metadata["base_library_sha256"] == metadata["candidate_library_sha256"]
    if shutil.which("lscpu"):
        metadata["cpu"] = next((line.split(":", 1)[1].strip()
                                for line in command_output(["lscpu"]).splitlines()
                                if line.startswith("Model name:")), metadata["cpu"])
    # Use candidate driver on both versions, including its support module. API incompatibility
    # fails the run explicitly instead of silently comparing different benchmark programs.
    digest = hashlib.sha256()
    for relative in ("core/benches/ann_bench.rs", "core/benches/support/ann_bench_support.rs"):
        data = (candidate / relative).read_bytes()
        digest.update(relative.encode() + b"\0" + data)
        (base / relative).parent.mkdir(parents=True, exist_ok=True)
        (base / relative).write_bytes(data)
    metadata["driver_sha256"] = digest.hexdigest()
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("ANN_", "DISKANN_BENCH_", "CARGO_PROFILE_"))
           and key not in ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_RUSTFLAGS")}
    env.update(WORKLOAD)
    env.update({"RUSTFLAGS": metadata["rustflags"], "LC_ALL": "C", "CARGO_INCREMENTAL": "0"})
    executables = {side: build(checkout, output, side, env, deadline)
                   for side, checkout in (("base", base), ("candidate", candidate))}
    samples = {index: {"base": [], "candidate": []} for index in INDEXES}
    raw = output / "raw"
    raw.mkdir()
    for index in INDEXES:
        for repetition in range(args.rounds):
            order = ("base", "candidate") if repetition % 2 == 0 else ("candidate", "base")
            for side in order:
                prefix = raw / f"{index}-{repetition + 1}-{side}"
                print(f"Running {prefix.name}", flush=True)
                sample_env = dict(env, ANN_INDEXES=index, ANN_OUTPUT_DIR=str(output / "indexes"))
                started = time.monotonic()
                with prefix.with_suffix(".csv").open("w") as csv_file, prefix.with_suffix(".log").open("w") as log:
                    run_checked([executables[side]], env=sample_env, cwd=output,
                                stdout=csv_file, stderr=log, timeout=args.timeout, deadline=deadline)
                samples[index][side].append(read_sample(prefix.with_suffix(".csv"), index))
                metadata["execution_order"].append({
                    "index": index, "round": repetition + 1, "side": side,
                    "wall_seconds": round(time.monotonic() - started, 3),
                })
                (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    results = summarize(samples, args.rounds)
    (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    (output / "summary.md").write_text(render_report(metadata, results))
    print(f"Report: {output / 'summary.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="Disposable base checkout (driver is overwritten)")
    parser.add_argument("--candidate", type=Path, required=True, help="Candidate checkout providing the shared driver")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--rounds", type=int, default=6, help="Samples per version/index; even number >= 2")
    parser.add_argument("--timeout", type=int, default=180, help="Timeout in seconds per sample")
    parser.add_argument("--total-timeout", type=int, default=1200, help="Total build/sample budget in seconds")
    args = parser.parse_args()
    if args.rounds < 2 or args.rounds % 2 or args.timeout <= 0 or args.total_timeout <= 0:
        parser.error("rounds must be even and >= 2; timeouts must be positive")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except Exception as error:
        (args.output / "summary.md").write_text(
            "# PR / base benchmark failed\n\n"
            "The comparison is incomplete; no performance conclusion is available. "
            "See the uploaded build/sample logs for details.\n\n"
            f"```text\n{error}\n```\n"
        )
        raise


if __name__ == "__main__":
    main()
