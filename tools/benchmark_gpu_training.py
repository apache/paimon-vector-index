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

"""Compare IVF-SQ CPU Auto, CPU Lloyd, and optional cuVS training.

Source loading, diagnostic hashes and artifact export are outside stage timers.
Training includes native sampling/preprocessing, centroid fit and CPU SQ
calibration. GPU worker setup is reported separately and included in the first
GPU job's inclusive time. No GPU timings or speedups are synthesized.
"""

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import threading
from time import perf_counter

import numpy as np


PRESETS = {
    "smoke": (2048, 16, 8),
    "baseline": (65536, 960, 1024),
    "medium": (262144, 1536, 4096),
    "large": (1048576, 1536, 16384),
}


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def load_matrix(path, *, neighbors=False):
    """Memory-map .npy or ANN-Benchmarks .fvecs/.ivecs matrices."""
    if path.suffix == ".npy":
        result = np.load(path, mmap_mode="r", allow_pickle=False)
    else:
        expected = ".ivecs" if neighbors else ".fvecs"
        if path.suffix != expected:
            raise ValueError(f"expected .npy or {expected}: {path}")
        if path.stat().st_size < 4 or path.stat().st_size % 4:
            raise ValueError(f"invalid record file length: {path}")
        raw = np.memmap(path, mode="r", dtype="<i4")
        width = int(raw[0])
        if width <= 0 or raw.size % (width + 1):
            raise ValueError(f"invalid record dimensions: {path}")
        records = raw.reshape(-1, width + 1)
        if not np.all(records[:, 0] == width):
            raise ValueError(f"inconsistent record dimensions: {path}")
        result = records[:, 1:] if neighbors else records.view("<f4")[:, 1:]
    if result.ndim != 2 or min(result.shape) == 0:
        raise ValueError(f"expected a nonempty matrix: {path}")
    if neighbors:
        if not np.issubdtype(result.dtype, np.integer):
            raise ValueError("neighbors must contain integer, zero-based base-row IDs")
    elif result.dtype != np.float32:
        raise ValueError("vectors must be float32")
    return result


def array_hash(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_index(path, queries, truth, top_k, nprobe):
    from paimon_vindex import SearchParams, VectorIndexReader

    class Input:
        def __init__(self, file):
            self.file = file
            self.bytes_read = 0
            self.lock = threading.Lock()

        def pread_many(self, ranges):
            self.bytes_read += sum(length for _, length in ranges)
            if hasattr(os, "pread"):
                return [os.pread(self.file.fileno(), length, pos) for pos, length in ranges]
            with self.lock:
                chunks = []
                for pos, length in ranges:
                    self.file.seek(pos)
                    chunks.append(self.file.read(length))
                return chunks

    with path.open("rb") as file:
        source = Input(file)
        with VectorIndexReader(source) as reader:
            params = SearchParams.ivf(top_k, nprobe)
            before = source.bytes_read
            start = perf_counter()
            ids, _ = reader.search_batch(queries, params)
            batch_seconds = perf_counter() - start
            batch_bytes = source.bytes_read - before
            recall = sum(len(set(found.tolist()) & set(expected[:top_k].tolist()))
                         for found, expected in zip(ids, truth)) / (len(queries) * top_k)
            latencies = []
            for query in queries:
                start = perf_counter()
                reader.search(query, params)
                latencies.append((perf_counter() - start) * 1000)
    return {"recall_at_k": recall, "top_k": top_k, "nprobe": nprobe,
            "query_count": len(queries), "first_batch_qps": len(queries) / batch_seconds,
            "first_batch_read_bytes": batch_bytes,
            "after_batch_p95_ms": float(np.percentile(latencies, 95))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESETS, default="smoke")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--base", type=Path, help="float32 .npy or .fvecs; all rows enter the native reservoir")
    source.add_argument("--synthetic", action="store_true", help="explicitly generate a training-load matrix")
    parser.add_argument("--backends", nargs="+", choices=["cpu-auto", "cpu-lloyd", "cuvs", "external"], default=["cpu-auto", "cpu-lloyd"])
    parser.add_argument("--centroids", type=Path, help="float32 .npy centers, required only for the external backend; skips fitting for fixed-model builder comparisons")
    parser.add_argument("--repeats", type=positive, default=3)
    parser.add_argument("--threads", type=positive, default=8)
    parser.add_argument("--batch-rows", type=positive, default=8192)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--build-index", action="store_true", help="also encode ALL base rows and serialize an index per run")
    parser.add_argument("--build-backend", choices=["cpu", "cuvs-assign", "cuvs-encode"], default="cpu",
                        help="full-row builder; cuvs-assign retains native SQ encoding")
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--neighbors", type=Path, help="integer .npy or .ivecs exact neighbors for the supplied base")
    parser.add_argument("--query-limit", type=positive, default=1000)
    parser.add_argument("--top-k", type=positive, default=10)
    parser.add_argument("--nprobe", type=positive)
    parser.add_argument("--metric", choices=["l2", "cosine", "inner_product"], default="l2")
    parser.add_argument("--coarse-assignment", choices=["auto", "exact"], default="auto",
                        help="CPU SQ calibration/writer assignment policy; auto can use approximate Vamana")
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory; existing directories are never overwritten")
    args = parser.parse_args()
    if bool(args.queries) != bool(args.neighbors) or (args.queries and not args.build_index):
        parser.error("--queries and --neighbors must be supplied together with --build-index")
    if len(set(args.backends)) != len(args.backends):
        parser.error("duplicate backends")
    if ("external" in args.backends) != bool(args.centroids):
        parser.error("--centroids is required exactly when --backends includes external")
    if args.build_backend != "cpu" and not args.build_index:
        parser.error("--build-backend requires --build-index")
    os.environ["RAYON_NUM_THREADS"] = str(args.threads)
    from paimon_vindex import VectorIndexTrainer, VectorIndexWriter, _ffi

    rows, dimension, nlist = PRESETS[args.preset]
    if args.synthetic:
        # Fixed seed, one contiguous float32 allocation. Synthetic timings are
        # useful for capacity exploration, not evidence of production recall.
        base = np.random.default_rng(1234).standard_normal((rows, dimension), dtype=np.float32)
        source_name = "synthetic_normal_seed_1234"
    else:
        base = load_matrix(args.base)
        source_name = str(args.base.resolve())
    rows, dimension = base.shape
    if rows < nlist:
        parser.error("base must contain at least nlist rows for this comparison")
    if args.build_backend != "cpu" and args.coarse_assignment != "exact" and dimension * nlist >= 1_000_000:
        parser.error("GPU building requires --coarse-assignment exact for this centroid matrix")
    fixed_centers = load_matrix(args.centroids) if args.centroids else None
    if fixed_centers is not None and (fixed_centers.shape != (nlist, dimension) or not np.isfinite(fixed_centers).all()):
        parser.error("external centers must be finite with shape (nlist, dimension)")
    nprobe = args.nprobe or min(nlist, max(1, nlist // 16))
    if nprobe > nlist:
        parser.error("nprobe must not exceed nlist")
    queries = truth = None
    if args.queries:
        queries = load_matrix(args.queries)[:args.query_limit]
        truth = load_matrix(args.neighbors, neighbors=True)[:args.query_limit]
        if queries.shape[1] != dimension or len(queries) != len(truth) or truth.shape[1] < args.top_k:
            parser.error("query/ground-truth shapes do not match the base and top-k")
        if not np.isfinite(queries).all() or np.any(truth < 0) or np.any(truth >= rows):
            parser.error("queries must be finite and ground-truth IDs must refer to the supplied base")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    options = {"index.type": "ivf_sq", "dimension": str(dimension), "nlist": str(nlist),
               "metric": args.metric, "expected-vector-count": str(rows),
               "ivf.coarse-assignment": args.coarse_assignment}
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
                                           text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"],
                                            cwd=Path(__file__).resolve().parent,
                                            stderr=subprocess.DEVNULL))
    except (OSError, subprocess.CalledProcessError):
        revision = None
        dirty = None
    native_library = Path(_ffi.lib._name).resolve()
    manifest = {"schema_version": 1, "preset": args.preset, "source": source_name,
                "base_shape": list(base.shape), "options": options, "backends": args.backends,
                "repeats": args.repeats, "rayon_threads": args.threads,
                "platform": platform.platform(), "python": platform.python_version(),
                "numpy": np.__version__, "git_revision": revision, "git_dirty": dirty,
                "native_library": str(native_library), "native_library_sha256": file_hash(native_library),
                "benchmark_sha256": file_hash(Path(__file__)),
                "gpu_peak_memory": "not measured", "quality_evaluated": queries is not None}
    manifest.update(build_backend=args.build_backend, device=args.device, batch_rows=args.batch_rows,
                    external_centroids=None if args.centroids is None else str(args.centroids.resolve()),
                    external_centroids_sha256=None if args.centroids is None else file_hash(args.centroids))
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    gpu = None
    runs = []
    try:
        with (args.output_dir / "runs.jsonl").open("x") as results:
            for repeat in range(args.repeats):
                # Rotate execution order to avoid always favoring one backend.
                offset = repeat % len(args.backends)
                order = args.backends[offset:] + args.backends[:offset]
                for backend in order:
                    print(f"{backend} repeat={repeat}: {rows}x{dimension}, nlist={nlist}", flush=True)
                    start = perf_counter()
                    with VectorIndexTrainer.create(options) as trainer:
                        for begin in range(0, rows, args.batch_rows):
                            trainer.add_training_vectors(base[begin:begin+args.batch_rows])
                        prepared = trainer.prepare_training()
                    prepare_seconds = perf_counter() - start
                    with prepared:
                        info = asdict(prepared.info)
                        start = perf_counter()
                        sample = prepared.sample
                        sample_export_seconds = perf_counter() - start
                        sample_sha256 = array_hash(sample)
                        start = perf_counter()
                        initial = np.ascontiguousarray(sample[np.random.default_rng(prepared.info.seed).choice(
                            len(sample), size=nlist, replace=False)])
                        initial_prepare_seconds = (sample_export_seconds + perf_counter() - start) if backend == "cpu-lloyd" else 0.0
                        initial_sha256 = array_hash(initial)
                        del sample
                        setup_seconds = 0.0
                        if backend == "cuvs" and gpu is None:
                            from paimon_vindex.gpu import CuvsKMeans
                            gpu = CuvsKMeans(args.device)
                            setup_seconds = gpu.initialization_seconds
                        start = perf_counter()
                        if backend == "cuvs":
                            # The adapter uses the same seed/row-selection rule
                            # as `initial`, and times its own export/initialization.
                            centers = gpu.fit(prepared)
                        elif backend == "external":
                            centers = fixed_centers
                        else:
                            centers = prepared.fit_centroids_cpu(
                                "auto" if backend == "cpu-auto" else "lloyd",
                                initial_centroids=None if backend == "cpu-auto" else initial,
                            )
                        fit_seconds = perf_counter() - start + initial_prepare_seconds
                        start = perf_counter()
                        training = prepared.finish_with_ivf_centroids(centers)
                        sq_seconds = perf_counter() - start
                    prefix = f"{backend}-{repeat}"
                    np.save(args.output_dir / f"{prefix}-centroids.npy", centers, allow_pickle=False)
                    run = {"backend": backend, "repeat": repeat, "training_info": info,
                           "sample_sha256": sample_sha256,
                           "initial_centroids_sha256": None if backend in ("cpu-auto", "external") else initial_sha256,
                           "centroids_sha256": array_hash(centers),
                           "external_initialization_seconds": initial_prepare_seconds,
                           "prepare_seconds": prepare_seconds, "centroid_fit_seconds": fit_seconds,
                           "sq_calibration_seconds": sq_seconds, "gpu_setup_seconds": setup_seconds,
                           "build_backend": args.build_backend,
                           "training_seconds": prepare_seconds + fit_seconds + sq_seconds,
                           "training_including_setup_seconds": prepare_seconds + fit_seconds + sq_seconds + setup_seconds}
                    if backend == "cuvs":
                        run["gpu"] = gpu.last_run
                    try:
                        if args.build_index:
                            with VectorIndexWriter(training) as writer:
                                start = perf_counter()
                                if args.build_backend == "cpu":
                                    builder_context = nullcontext(writer)
                                else:
                                    from paimon_vindex.gpu import CuvsIvfSqWriter
                                    builder_context = CuvsIvfSqWriter(writer, args.device, encode=args.build_backend == "cuvs-encode")
                                    run["gpu_build_setup_seconds"] = builder_context.initialization_seconds
                                stages = {}
                                with builder_context as builder:
                                    for begin in range(0, rows, args.batch_rows):
                                        chunk = base[begin:begin+args.batch_rows]
                                        builder.add_vectors(np.arange(begin, begin+len(chunk), dtype=np.int64), chunk)
                                        if args.build_backend != "cpu":
                                            if "gpu_build" not in run:
                                                run["gpu_build"] = {key: builder.last_run[key] for key in (
                                                    "backend", "assignment", "device", "device_name",
                                                    "cuvs_version", "cupy_version",
                                                )}
                                            for key, value in builder.last_run.items():
                                                if key.endswith("_seconds") or key.endswith("_bytes") or key == "rows":
                                                    stages[key] = stages.get(key, 0) + value
                                add_seconds = perf_counter() - start
                                if stages:
                                    run["gpu_build_stages"] = stages
                                sizes = writer.ivf_sq_partition_sizes()
                                run["partitions"] = {"empty": int((sizes == 0).sum()), "max": int(sizes.max()),
                                                     "p95": float(np.percentile(sizes, 95)), "mean": float(sizes.mean())}
                                index_path = args.output_dir / f"{prefix}.index"
                                start = perf_counter()
                                with index_path.open("xb") as out:
                                    writer.write(out)
                                write_seconds = perf_counter() - start
                            run.update(add_seconds=add_seconds, serialize_seconds=write_seconds,
                                       index_bytes=index_path.stat().st_size,
                                       build_seconds=run["training_seconds"] + add_seconds + write_seconds,
                                       build_including_setup_seconds=run["training_including_setup_seconds"] + add_seconds + write_seconds)
                            if queries is not None:
                                run["query"] = evaluate_index(index_path, queries, truth, args.top_k, nprobe)
                    finally:
                        training.close()
                    runs.append(run)
                    results.write(json.dumps(run, allow_nan=False) + "\n")
                    results.flush()
        summary = {}
        for backend in args.backends:
            selected = [r for r in runs if r["backend"] == backend]
            summary[backend] = {key: float(np.median([r[key] for r in selected]))
                                for key in ("centroid_fit_seconds", "training_seconds", "training_including_setup_seconds",
                                            "build_seconds", "build_including_setup_seconds") if key in selected[0]}
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
    finally:
        if gpu is not None:
            gpu.close()


if __name__ == "__main__":
    main()
