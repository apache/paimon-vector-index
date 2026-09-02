---
title: "Development and benchmarks"
description: "Build, test, benchmark, and validate Apache Paimon Vector Index."
---

<!--
  Licensed to the Apache Software Foundation (ASF) under one or more
  contributor license agreements. See the NOTICE file distributed with
  this work for additional information regarding copyright ownership.
  The ASF licenses this file to you under the Apache License, Version 2.0.
-->

# Development and benchmarks

Run the standard Rust checks, exercise the C/C++, Java, and Python integrations, and measure ANN and filtered-query behavior with reproducible workloads.

## Repository layout {#workspace}

| Directory | Contents | Primary verification |
|----|----|----|
| `core/` | Indexes, quantizers, I/O, format fixtures, and benchmarks | `cargo test -p paimon-vindex-core` |
| `ffi/` | C ABI and shared library | C smoke test |
| `include/` | C++ RAII header | C++ smoke test |
| `jni/` | Rust JNI bridge | Java Maven tests |
| `java/` | Java public API and tests | `mvn test` |
| `python/` | ctypes package and pytest suite | `pytest` |
| `c/` / `cpp/` | Integration smoke-test projects | CMake |
| `docs-src/` | Markdown documentation sources | Docusaurus build and link/anchor checks |
| `docs-site/` | Docusaurus configuration and site assets | Docusaurus production build |

## Standard Rust checks {#rust}

```bash title="Shell"
cargo fmt --all
cargo test --workspace
cargo clippy --workspace --all-targets
```

Before submitting a change, keep formatting, workspace tests, and Clippy across all targets clean. Changes to binary layouts should also run the core storage-format fixtures.

## ANN benchmark {#ann}

`ann_bench` compares IVF-FLAT, IVF-SQ, IVF-PQ, IVF-RQ, and DiskANN under one workload. By default it generates independent clustered queries and computes exact Top-10 ground truth. It can instead read standard `fvecs`/`ivecs` files and use their published ground truth. Every CSV row starts with a dataset label and reports vector and training counts, raw-dataset bytes, total build time plus train/add/write stages, process peak RSS, warm-up time, serialized bytes, Recall@10, first/P50/P95 latency, sequential and batch throughput, positional-read rounds, range count, and bytes.

```bash title="Shell · defaults"
cargo bench -p paimon-vindex-core --bench ann_bench -- --nocapture
```

```bash title="Shell · documented 100k comparison"
ANN_N=100000 ANN_NQ=64 ANN_D=64 ANN_K=10 \
ANN_NLIST=256 ANN_NPROBE=32 ANN_PQ_CODE_RATIO=0.0625 \
ANN_DISKANN_L_SEARCH=200 \
ANN_RQ_BITS=4 \
cargo bench -p paimon-vindex-core --bench ann_bench -- --nocapture
```

```bash title="Shell · 1024 physical dimensions, exactly 10 GiB of raw vectors"
ANN_DATA_GIB=10 ANN_D=1024 ANN_NOISE_DIMENSIONS=64 \
ANN_TRAIN_N=65536 ANN_NQ=8 ANN_K=10 \
ANN_NLIST=1 ANN_NPROBE=1 ANN_PQ_CODE_RATIO=0.0625 ANN_CLUSTERS=64 \
ANN_DISKANN_L_SEARCHES=200,1000 \
ANN_DISKANN_MEMORY_BUDGET_BYTES=34359738368 \
ANN_INDEXES=DISKANN ANN_KEEP_INDEXES=1 \
ANN_OUTPUT_DIR=/path/on/target/ssd \
cargo bench -p paimon-vindex-core --bench ann_bench
```

```bash title="Shell · complete five-index 10 GiB comparison, isolated processes"
for index in IVF_FLAT IVF_SQ IVF_PQ IVF_RQ DISKANN; do
  ANN_DATA_GIB=10 ANN_D=1024 ANN_NOISE_DIMENSIONS=64 \
  ANN_TRAIN_N=65536 ANN_NQ=8 ANN_K=10 \
  ANN_NLIST=1024 ANN_NPROBE=64 ANN_PQ_CODE_RATIO=0.0625 ANN_CLUSTERS=64 \
    ANN_DISKANN_L_SEARCHES=200,1000 \
  ANN_DISKANN_MEMORY_BUDGET_BYTES=34359738368 \
  ANN_INDEXES="$index" ANN_OUTPUT_DIR=/path/on/target/ssd \
  cargo bench -p paimon-vindex-core --bench ann_bench
done
```

### Public SIFT1M, GIST1M, and GloVe-100 data

[ANN-Benchmarks](https://github.com/erikbern/ann-benchmarks) publishes dense HDF5 files with independent test queries and exact Top-100 neighbors. Convert them once to the streaming-friendly `fvecs`/`ivecs` format. `--query-limit=1000` keeps all recorded datasets at the same query count while retaining enough queries for stable Recall@10.

```bash title="Shell · download and convert a public dataset"
curl -LO https://ann-benchmarks.com/sift-128-euclidean.hdf5
python3 -m pip install h5py numpy
python3 tools/convert_ann_benchmarks.py \
  sift-128-euclidean.hdf5 /data/sift1m \
  --prefix sift1m --query-limit 1000
```

```bash title="Shell · convert GloVe angular data for the common L2 benchmark"
curl -LO https://ann-benchmarks.com/glove-100-angular.hdf5
python3 tools/convert_ann_benchmarks.py \
  glove-100-angular.hdf5 /data/glove100 \
  --prefix glove100 --query-limit 1000 --normalize-l2
```

```bash title="Shell · run all public-data indexes in isolated processes"
RAYON_NUM_THREADS=12 ANN_DATASET_NAME=sift1m \
ANN_BASE_FVECS=/data/sift1m/sift1m_base.fvecs \
ANN_QUERY_FVECS=/data/sift1m/sift1m_query.fvecs \
ANN_GROUND_TRUTH_IVECS=/data/sift1m/sift1m_ground_truth.ivecs \
ANN_K=10 ANN_NLIST=1024 ANN_NPROBE=64 ANN_PQ_CODE_RATIO=0.0625 \
ANN_DISKANN_L_SEARCH=100 \
ANN_DISKANN_RAW_VECTOR_ENCODING=f16 \
ANN_DISKANN_MEMORY_BUDGET_BYTES=8589934592 \
ANN_STORAGE_CASES=local_ssd_warm_cache,remote_cache_2ms,object_store_20ms \
ANN_OUTPUT_DIR=/path/on/target/ssd \
cargo bench -p paimon-vindex-core --bench ann_bench
```

The benchmark infers `ANN_N`, `ANN_NQ`, and `ANN_D` from the public files, chooses `ANN_TRAIN_N=min(N, max(65,536, 64 × nlist))`, and automatically starts one child process per selected index so each peak-RSS value is independent. Explicit shape variables remain available as assertions. For GIST1M, download and convert `gist-960-euclidean.hdf5`; for GloVe-100, use the command above. Unit-normalized L2 has the same neighbor ordering as cosine for non-zero vectors; that equivalence was used by the recorded GloVe run. DiskANN now also accepts native cosine and inner-product metrics, which should be benchmarked against ground truth generated with the same production metric. Keep the other recorded settings unchanged. The default `ANN_PQ_CODE_RATIO=0.0625` resolves automatically to `m=32`, `m=240`, and `m=25` for SIFT, GIST, and GloVe respectively. Set `ANN_PQ_M` only to benchmark an explicit expert override.

At 1024 dimensions, `ANN_DATA_GIB=10` derives exactly 2,621,440 vectors. Run large comparisons one index per process by changing `ANN_INDEXES`; this isolates process peak RSS and, unless `ANN_KEEP_INDEXES=1`, deletes that index after all selected serving cases. A comma-separated subset or `all` is convenient for smaller workloads. The benchmark retains only the selected training prefix, then adds the entire data set. The clustered generator is round-robin by cluster, so a prefix still covers every cluster when `ANN_TRAIN_N ≥ ANN_CLUSTERS`.

> **Physical versus intrinsic dimension** — `ANN_NOISE_DIMENSIONS=64` spreads independent per-vector noise evenly over 64 of the 1024 stored dimensions; the other dimensions retain cluster-center signal. This creates a repeatable storage/compute scale workload without pretending that 1024 independent uniform-noise dimensions resemble production embeddings. With all 1024 dimensions independently noisy, same-cluster distances concentrate so strongly that exact Top-10 membership is nearly arbitrary: a 65k diagnostic produced local Recall@10 of 0.0125 at `l_search=200` and only 0.178 at `l_search=5000`. Use a public or production corpus for an algorithm-quality claim.

| Variable | Meaning |
|----|----|
| `ANN_DATASET_NAME` | Comma-free label written to every CSV row |
| `ANN_BASE_FVECS / ANN_QUERY_FVECS / ANN_GROUND_TRUTH_IVECS` | Optional public-data files; set all three together. Their vector count, query count, dimension, and ground-truth width are inspected before the full payload is loaded. |
| `ANN_N / ANN_NQ / ANN_D` | Inferred for public files; an explicit value is validated as a shape assertion. Generated workloads retain defaults of 20,000 / 64 / 64. |
| `ANN_DATA_GIB` | Generated-workload raw float-vector GiB; derives `n = floor(bytes / (4d))` and cannot be combined with public file paths. |
| `ANN_TRAIN_N` | Optional training-sample override; omitted values resolve to `min(N, max(65,536, 64 × nlist))`. |
| `ANN_K` | Top K |
| `ANN_INDEXES` | `all` or a comma-separated subset of the five index names. Public multi-index runs automatically isolate each index in a child process. |
| `ANN_STORAGE_CASES` | `all` or a comma-separated subset of `local_ssd_warm_cache`, `remote_cache_2ms`, and `object_store_20ms`. The default runs all three. |
| `ANN_NLIST / ANN_NPROBE` | IVF list count / probed lists |
| `ANN_PQ_CODE_RATIO` | IVF-PQ and DiskANN PQ-code/raw-vector byte ratio; default 0.0625 and automatically resolves a valid `m` |
| `ANN_PQ_M` | Optional explicit subquantizer-count override; takes precedence over automatic sizing |
| `ANN_RQ_BITS` | Persisted IVF-RQ bit width, 1–8; default 4 |
| `ANN_DISKANN_L_SEARCH / ANN_DISKANN_L_SEARCHES` | One DiskANN search-list size, or a comma-separated sweep over one built index |
| `ANN_DISKANN_BUILD_DISTANCE` | `product_quantized` by default; use `full_precision` as a graph-quality control |
| `ANN_DISKANN_RAW_VECTOR_ENCODING` | `f16` by default; use `f32` as an exact-distance and numeric-range control. The resolved value is written to every CSV row. |
| `ANN_DISKANN_MEMORY_BUDGET_BYTES` | DiskANN's internal preflight build budget; default 8 GiB and excludes the benchmark's source-vector allocation |
| `ANN_NOISE_DIMENSIONS / ANN_CLUSTERS / ANN_SEED` | Generated-workload distribution and reproducibility controls; ignored by public file loading. |
| `ANN_OUTPUT_DIR` | Directory whose filesystem supplies the local-storage path |
| `ANN_KEEP_INDEXES` | `1` retains generated index files; the default deletes each index after all selected serving cases |
| `ANN_REUSE_INDEX_PATH` | Skips training/build and queries one retained index; requires a single `ANN_INDEXES` value |

> **Large-run memory boundary** — The current public writer API accepts an in-memory vector slice and each index writer retains its encoded or raw build state. A raw 10 GiB DiskANN run therefore needs roughly the 10 GiB source allocation plus DiskANN's separately budgeted build state. On a 48 GiB machine, use a 32 GiB DiskANN budget and run one index per process. This benchmark validates large immutable builds; it is not a streaming-ingest benchmark.

The benchmark executes three serving cases by default. `local_ssd_warm_cache` uses the real index file with no artificial delay. `remote_cache_2ms` adds a fixed 2 ms delay per `pread` call while reading all ranges in that call concurrently. `object_store_20ms` uses the `ObjectStore` read plan: open/optimization and sequential-query latency are the measured CPU/I/O time plus 20 ms for every observed dependent read round, avoiding hours of idle wall-clock sleep while retaining the exact fixed-latency model; batch QPS still injects the real 20 ms sleep so concurrent-client overlap is measured rather than inferred. These models do not include bandwidth or operating-system cache misses. DiskANN uses a dedicated range-I/O pool separate from Rayon query workers, preventing a saturated batch from starving nested positional reads; IVF retains the shared compute pool because its batch read scheduling is not nested inside per-query sessions. The unified IVF Reader reuses the 64-byte type-dispatch header and loads resident metadata in one further contiguous read, reducing initialization from three rounds to two without changing any format. DiskANN is opened explicitly as `LocalStorage`, `RemoteStorage`, or `ObjectStore` with the default 4 GiB Reader budget, which is automatically partitioned into resident state and caches. The sequential sweep and batch call use separate opened-and-optimized Readers, preventing all benchmark queries from prewarming their own batch measurement. CSV output records optimization rounds, ranges, and bytes separately from query I/O. Use `ANN_STORAGE_CASES` for a focused rerun against a retained index. See the [recorded public-corpus results and interpretation](index.md#public-corpus-check).

## DiskANN access-pattern benchmark {#diskann-bench}

`diskann_bench` measures the page-oriented search path separately from the IVF benchmark. With no dataset paths it generates a small deterministic correctness workload. SIFT1M mode reads standard `.fvecs` and `.ivecs` files.

```bash title="Shell · generated smoke workload"
RAYON_NUM_THREADS=4 \
DISKANN_BENCH_ACCEPTANCE=1 DISKANN_BENCH_SIMULATED_RTT_MS=1 \
DISKANN_BENCH_CONCURRENCY=4 \
DISKANN_BENCH_STORAGE_LAYOUT=compact \
DISKANN_BENCH_RAW_VECTOR_ENCODING=f32 \
DISKANN_BENCH_BUILD_DISTANCE=product_quantized \
cargo bench -p paimon-vindex-core --bench diskann_bench
```

```bash title="Shell · SIFT1M acceptance"
DISKANN_BASE_FVECS=/data/sift_base.fvecs \
DISKANN_QUERY_FVECS=/data/sift_query.fvecs \
DISKANN_GROUND_TRUTH_IVECS=/data/sift_groundtruth.ivecs \
DISKANN_BENCH_INDEX_DIR=/mnt/local-nvme \
RAYON_NUM_THREADS=16 \
DISKANN_BENCH_CONCURRENCY=16 \
DISKANN_BENCH_READER_MEMORY_BUDGET_BYTES=4294967296 \
DISKANN_BENCH_ACCEPTANCE=1 \
DISKANN_BENCH_FILTERED_MATRIX=1 \
DISKANN_BENCH_SIMULATED_RTT_MS=5 \
cargo bench -p paimon-vindex-core --bench diskann_bench
```

The benchmark writes a temporary `.dann` file under `DISKANN_BENCH_INDEX_DIR` (or the system temporary directory), reopens it through a benchmark-private `SeekRead` adapter, and deletes it on exit. This adapter exercises native positional I/O without adding a local-file type to the public core API. Use `DISKANN_BENCH_STORAGE_LAYOUT=compact|interleaved`, `DISKANN_BENCH_RAW_VECTOR_ENCODING=f32|f16`, and `DISKANN_BENCH_BUILD_DISTANCE=full_precision|product_quantized` for controlled comparisons. PQ defaults to `DISKANN_BENCH_PQ_CODE_RATIO=0.0625`; `DISKANN_BENCH_PQ_M` is an optional explicit override in `1..=dimension` and `DISKANN_BENCH_PQ_BITS` selects 4 or 8 bits. Generated workloads can be scaled with `DISKANN_BENCH_SMOKE_N`, `DISKANN_BENCH_SMOKE_NQ`, and `DISKANN_BENCH_SMOKE_DIMENSION`; exact top-10 truth is generated automatically. CSV rows inject representative latency hints for all four internal read tiers at `l_search` 50, 100, and 200 so timing noise does not make planner comparisons ambiguous. `DISKANN_BENCH_SIMULATED_RTT_MS` adds the same fixed delay to every `pread` round for every tier. Set `DISKANN_BENCH_WARMUP_QUERIES=0` to measure resident-only initialization; its default 8 replays representative queries before latency sampling. `DISKANN_BENCH_READER_MEMORY_BUDGET_BYTES` controls the one public Reader budget; the benchmark records automatically resolved cache behavior through search statistics. Rows additionally identify layout, raw-vector encoding, build distance, selected graph-shard count, and complete parallel-session query counts.

Set `DISKANN_BENCH_FILTERED_MATRIX=1` to add random and contiguous/cluster-correlated filters at 0.01%, 0.1%, 1%, 10%, 50%, and 100% selectivity. The matrix compares adaptive results with a forced exhaustive matching-node baseline. It reports the actual executed strategy and graph fallback count from `diskann_search_stats`, Recall@10, elapsed time, I/O, PQ distance evaluations versus PQ code loads, query chunks, rerank chunks, candidate references, unique raw windows, batch LRU hits/misses/evictions, and parallel exact-rerank work. Under `DISKANN_BENCH_ACCEPTANCE=1`, every graph-enabled cell must stay within one absolute Recall@10 percentage point of the exhaustive baseline.

The benchmark warms one Reader. `DISKANN_BENCH_CONCURRENCY=1` repeatedly invokes the single-query API; larger values invoke the native batch path at that width. Retained sessions run complete queries concurrently for small batches and all interleaved-layout batches. Large compact batches retain shared vector-window rerank. Graph sessions clone only the storage handle and share resident PQ/row-ID/code allocations, validation state, hot prefix, and cold-adjacency LRU. Local results use the operating-system page cache after index creation; set `DISKANN_BENCH_INDEX_DIR` to the production SSD mount and run cold-cache testing with an external, platform-appropriate cache-control procedure.

> **Acceptance scope** — When enabled, the gate requires unfiltered Recall@10 ≥ 0.90, filtered graph Recall@10 within one point of exhaustive matching-node scan, build time below two hours, peak RSS below 4 GiB, and at most seven graph rounds plus one exact-rerank round per `ObjectStore` query. The generated workload validates mechanics only. SIFT1M or a representative production dataset remains a required non-CI release check.

## C FFI smoke test {#c}

Build the release shared library, then compile the C test against the generated header.

```bash title="Shell"
cargo build --release -p paimon-vindex-ffi
cmake -S c -B c/build
cmake --build c/build
LD_LIBRARY_PATH=target/release c/build/test_vindex
```

> **Platform note** — `LD_LIBRARY_PATH` is the Linux loader variable. Use the equivalent loader configuration on other platforms.

## C++ smoke test {#cpp}

The C++ test reuses the same FFI shared library and validates the RAII header.

```bash title="Shell"
cargo build --release -p paimon-vindex-ffi
cmake -S cpp -B cpp/build
cmake --build cpp/build
LD_LIBRARY_PATH=target/release cpp/build/test_vindex_cpp
```

## Java / JNI tests {#java}

The Maven test phase compiles the Java sources and runs `VectorIndexJavaApiTest`. This test covers result and metadata objects, native resource mapping, closed-handle behavior, and API compilation without loading the native library:

```bash title="Shell"
mvn -f java/pom.xml test
```

Native validation, panic-boundary, and handle-safety tests are standalone `main` programs. The following Linux commands mirror the additional JNI checks run by CI:

```bash title="Shell · Linux JNI checks"
cargo build -p paimon-vindex-jni --release

java -cp java/target/test-classes:java/target/classes \
  org.apache.paimon.index.vector.VectorIndexNativeValidationTest \
  "$(pwd)/target/release/libpaimon_vindex_jni.so"
java -cp java/target/test-classes:java/target/classes \
  org.apache.paimon.index.vector.VectorIndexNativePanicBoundaryTest \
  "$(pwd)/target/release/libpaimon_vindex_jni.so"
java -cp java/target/test-classes:java/target/classes \
  org.apache.paimon.index.vector.VectorIndexNativeHandleSafetyTest \
  "$(pwd)/target/release/libpaimon_vindex_jni.so"
```

> **Platform note** — The shared-library filename above is for Linux. Use the native library filename produced in `target/release` on macOS or Windows.

Release CI additionally builds the multi-platform Maven JAR and runs `VectorIndexNativeLoaderSmokeTest` from the final `java-package` artifact with no native path argument on Linux x86-64, Linux aarch64, macOS arm64, and Windows x86-64. This ensures consumers can use every bundled JNI library directly.

## Python ctypes tests {#python}

The Python package must be able to locate the newly built FFI library:

```bash title="Shell"
cargo build --release -p paimon-vindex-ffi
cd python
PAIMON_VINDEX_LIB_PATH=../target/release pip install -e ".[test]"
PAIMON_VINDEX_LIB_PATH=../target/release pytest -v
```

The suite uses NumPy inputs and covers all index types, persisted RQ bit metadata, single/batch shapes, and error paths.

## Storage format and compatibility {#format}

Changes to headers, flags, sections, sort order, or graph encoding must update both format fixtures and documentation. The normative v1 field definitions live in [core/STORAGE_FORMAT.md](https://github.com/apache/paimon-vector-index/blob/main/core/STORAGE_FORMAT.md).

### Reader compatibility

Unknown versions, required flags, non-zero reserved bytes, negative counts, and out-of-bounds sections must fail rather than be guessed.

### Stable fixtures

`core/tests/fixtures` stores byte-exact hexadecimal fixtures for every v1 index family. DiskANN additionally freezes compact multi-page, 4-bit interleaved, FOR row-ID, and raw row-ID variants.

### Outer integrity

Vector-index files do not embed a checksum. Length and checksum validation belong to the outer Paimon file and manifest layer.
