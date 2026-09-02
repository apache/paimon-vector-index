---
title: "API and language bindings"
description: "Use the shared vector-index lifecycle from Rust, C, C++, Java, and Python."
---

<!--
  Licensed to the Apache Software Foundation (ASF) under one or more
  contributor license agreements. See the NOTICE file distributed with
  this work for additional information regarding copyright ownership.
  The ASF licenses this file to you under the Apache License, Version 2.0.
-->

# API and language bindings

The Rust core, C FFI, C++ RAII layer, Java/JNI, and Python ctypes package share one train → write → serialize → detect → search model. Build options select the index type; Readers discover it from the file magic.

## Public integration layers {#workspace}

| Module | Role | Primary entry point |
|----|----|----|
| `core` | Pure Rust indexes, file Readers/Writers, and benchmarks | `paimon_vindex_core::index` |
| `ffi` | C ABI over the Rust core; builds `libpaimon_vindex_ffi` | Generated `include/paimon_vindex.h` |
| `include` | C++ RAII wrapper | `paimon_vindex.hpp` |
| `jni` + `java` | JNI implementation and Java API | `org.apache.paimon.index.vector` |
| `python` | Pure Python package loading the C FFI through ctypes | `paimon_vindex` |

The top-level Cargo workspace contains `core`, `ffi`, and `jni`. The Python package loads the shared FFI library at runtime.

## Shared lifecycle {#lifecycle}

1.  Create a Trainer and parse and validate its options.
2.  Submit one or more training batches.
3.  Finish training and create a one-shot Writer.
4.  Add row IDs and vectors, then write the file.
5.  Detect the file magic and execute searches.

- Vectors are contiguous `f32` values; length must equal `vector_count × dimension`.
- Training data may arrive in batches. Every IVF trainer keeps a deterministic reservoir of at most `max(65,536, 64 × resolved nlist)` vectors. DiskANN starts from a 50,000-row cap and lowers it when necessary so the retained sample, optional cosine-normalized copy, codebook, and parallel PQ-training scratch fit `diskann.memory-budget-bytes`. Sampling is independent of batch boundaries.
- The Python and Java one-shot `train` helpers infer `dimension` from the matrix and use its row count for automatic `nlist`. When the matrix is only a sample, pass the final corpus size as `expected-vector-count`. Streaming Trainer APIs require a concrete dimension before their first batch.
- A Writer may receive production vectors in multiple batches. Row-ID count must equal vector count.
- Readers expose metadata, single-query search, batch search, and Roaring64-filtered variants.
- Files carry their type and resolved model sections. Callers do not pass index options again when opening a Reader.

> **IVF coarse assignment is approximate by default for large centroid matrices** — When `dimension × nlist ≥ 1,000,000`, `ivf.coarse-assignment=auto` uses a Vamana graph while training and adding vectors. Search still selects lists by exact centroid distance, so graph assignment can lower recall at small `nprobe` and does not guarantee that a vector is found by a self-query with `nprobe=1`. Set `ivf.coarse-assignment=exact` to disable the graph, preserve exact nearest-centroid assignment, and avoid graph startup cost for small non-empty batches. Empty batches never build the graph.

## Shared search parameters {#params}

| Parameter | Applies to | Description |
|----|----|----|
| `top_k` | All indexes | Number of nearest neighbors returned for each query. |
| `Auto` width | All indexes | IVF starts at `max(8, ceil(nlist/16))`, adds enough average-list capacity for at least `4 × top_k` candidates, scales for filter selectivity, and progressively doubles when a filtered result is short. DiskANN uses a calibrated width when present, otherwise `max(100, 2 × top_k)`. |
| `max_initial_filter_expansion_factor` | Automatic IVF search in Rust, C, C++, and Java | Optional cap on inverse-selectivity expansion of the initial `nprobe`. Unset preserves the current unlimited behavior. A value of 1 keeps the unfiltered automatic width. Lower factors reduce initial search work but may reduce recall compared with uncapped automatic search. Progressive expansion occurs only when fewer than `top_k` valid results are returned. |
| `nprobe` | IVF families | Explicit expert override for the number of lists probed. A tagged IVF width is rejected by DiskANN rather than silently ignored. |
| `l_search` | DiskANN | Explicit expert override for graph search-list size; it is clamped to at least `top_k`. A tagged DiskANN width is rejected by IVF indexes. |
| `ivfpq_batch_table_reuse` | 8-bit IVF-PQ batch search | Controls reuse of query distance tables across inverted lists: `off`, `on`, or `auto`. The default is `auto`. |
| `ivfpq_batch_table_reuse_max_bytes` | 8-bit IVF-PQ batch search | Positive memory budget for reused query distance tables. The default is 512 MiB. |

Use the binding's `automatic` search-parameter constructor. For DiskANN, `calibrate_search_width` / `calibrateSearchWidth` evaluates widths 100, 200, and 400 on representative queries and remembers the smallest adjacent pair with at least 98% Top-K result overlap. This is a stability proxy, not a ground-truth recall guarantee; explicit widths always win.

> **IVF-RQ width is a build option** — `rq.bits` accepts 1–8 and defaults to 4. It is persisted in the file and reported as `rq_bits` / `rqBits` metadata; search has no separate bit-width switch.

## Reader options for DiskANN {#reader-options}

Reader options are accepted by every binding and affect DiskANN only. Other index families continue to use their existing positional-I/O behavior.

| Concept | Rust / Python | C / C++ / Java | Default |
|----|----|----|----|
| Random-read latency hint | Input capability `estimated_random_read_latency_nanos` | C/C++ field / Java `estimatedRandomReadLatencyNanos()` | 0: reuse the mandatory header read's elapsed time; positive values bypass measurement |
| Total Reader memory | `memory_budget_bytes` | `memory_budget_bytes` / constructor argument | 4 GiB; automatically partitioned among required resident state, a profile-sized adjacency prefix, a bounded cold-adjacency LRU, and a bounded raw-vector LRU |

The Reader has no public storage-tier switch. DiskANN selects an internal tier once during `open` from the latency hint or mandatory header-read timing; its latency, window, and beam policy then remain stable. `read_plan` / `readPlan` exposes that policy together with the current effective preload and shared-cache capacities. Those capacities are zero before resident initialization and may shrink when lazy row-ID lookup state consumes the same total budget. Cache partitioning remains an internal decision. The input callback receives all positional ranges for one round together and should execute them concurrently.

A storage adapter may additionally advertise `preferred_window_bytes` and a maximum range count named `max_ranges_per_pread` in Rust or `max_ranges_per_read` in C/C++/Python. Zero means unspecified. DiskANN rounds the requested window to complete 4 KiB logical pages, bounds it to 1 MiB, limits each `pread` batch to the advertised range count, and keeps physical alignment concerns inside the storage adapter. Build-time `deployment-profile` remains separate because it can select the persisted compact or interleaved layout. See [DiskANN reader tuning](diskann.md#reader-parameters).

## Search warm-up {#warmup}

After opening a Reader and before repeated searches, initialize resident state and, for DiskANN, optionally replay a small representative query set. Warm-up builds process-local caches without changing the file or search results.

| Language | Resident initialization | Representative-query warm-up | DiskANN width calibration |
|----|----|----|----|
| Rust | `optimize_for_search` | `warmup_queries` | `calibrate_search_width` |
| C | `paimon_vindex_reader_optimize_for_search` | `paimon_vindex_reader_warmup_queries` | `paimon_vindex_reader_calibrate_search_width` |
| C++ | `optimize_for_search` | `warmup_queries` | `calibrate_search_width` |
| Java | `optimizeForSearch` | `warmupQueries` | `calibrateSearchWidth` |
| Python | `optimize_for_search` | `warmup_queries` | `calibrate_search_width` |

`optimize_for_search` is optional for correctness: the first search performs the same initialization lazily. It builds IVF-PQ residual-L2 tables or DiskANN resident PQ/row-ID state and its automatically budgeted hot adjacency prefix. DiskANN `warmup_queries` then executes top-1 graph traversal and persisted-vector rerank for each supplied query, priming immutable adjacency and raw-vector LRUs. Other index families treat representative warm-up as resident initialization only.

## Rust {#rust}

```rust title="Rust · train, write, and search"
use std::fs::File;

use paimon_vindex_core::distance::MetricType;
use paimon_vindex_core::index::{
    VectorIndexConfig, VectorIndexReader, VectorIndexTrainer,
    VectorIndexWriter, VectorSearchParams,
};
use paimon_vindex_core::io::PosWriter;

let config = VectorIndexConfig::IvfSq {
    dimension: 128,
    nlist: 1024,
    metric: MetricType::L2,
    use_approximate_coarse_assignment: true,
};

let training = VectorIndexTrainer::train(
    config, &training_vectors, training_count)?;
let mut writer = VectorIndexWriter::new(training);
writer.add_vectors(&row_ids, &vectors, vector_count)?;

let mut file = File::create("vectors.pvindex")?;
let mut out = PosWriter::new(&mut file);
writer.write(&mut out)?;

let file = File::open("vectors.pvindex")?;
let mut reader = VectorIndexReader::open(file)?;
reader.optimize_for_search()?;
let params = VectorSearchParams::automatic(10);
let (ids, distances) = reader.search(&query, params)?;
```

```rust title="Rust · optional filtered-IVF tuning"
// Example only: select the factor using workload-specific
// latency and Recall@K measurements.
let params = VectorSearchParams::automatic(10)
    .with_max_initial_filter_expansion_factor(4);
```

```rust title="Rust · other configurations"
VectorIndexConfig::IvfFlat {
    dimension: 128, nlist: 1024, metric: MetricType::L2,
    use_approximate_coarse_assignment: true,
};
VectorIndexConfig::ivf_pq(
    128, 1024, MetricType::L2, false,
)?;
VectorIndexConfig::IvfRq {
    dimension: 128, nlist: 1024, bits: 4, metric: MetricType::L2,
    use_approximate_coarse_assignment: true,
};
VectorIndexConfig::IvfSq {
    dimension: 128, nlist: 1024, metric: MetricType::L2,
    use_approximate_coarse_assignment: true,
};
```

The IVF-PQ constructor uses the default relative PQ-code budget and resolves a concrete `m`. In every option-map API, `pq.m` is optional: `pq.code-ratio=0.0625` is the default, and an explicit `pq.m` takes precedence. Metadata and the on-disk header expose the resolved value. Rust callers select the policy through `VectorIndexConfig` before training; direct IVF indexes do not expose a post-training policy switch.

## C FFI {#c}

The C ABI builds `libpaimon_vindex_ffi`; cbindgen produces the public header. Status-returning operations return `0` on success and `-1` on failure. Handle-producing functions such as `paimon_vindex_trainer_open()`, `paimon_vindex_trainer_finish()`, `paimon_vindex_writer_open()`, and `paimon_vindex_reader_open()` return a handle on success or `NULL` on failure. The corresponding free functions return `void` and accept `NULL`. After a failure, `paimon_vindex_last_error()` returns the thread-local diagnostic string, or `NULL` if no error has been recorded.

```c title="C · complete lifecycle"
#include "paimon_vindex.h"

const char *keys[] = {"index.type", "dimension", "nlist", "metric"};
const char *values[] = {"ivf_flat", "128", "1024", "l2"};

PaimonVindexTrainerHandle *trainer =
    paimon_vindex_trainer_open(keys, values, 4);
paimon_vindex_trainer_add_training_vectors(
    trainer, training_vectors, training_count);
PaimonVindexTrainingHandle *training =
    paimon_vindex_trainer_finish(trainer);
paimon_vindex_trainer_free(trainer);

PaimonVindexWriterHandle *writer = paimon_vindex_writer_open(training);
paimon_vindex_training_free(training);
paimon_vindex_writer_add_vectors(writer, row_ids, vectors, vector_count);
paimon_vindex_writer_write_index(writer, output_file);
paimon_vindex_writer_free(writer);

PaimonVindexReaderHandle *reader = paimon_vindex_reader_open(input_file);
PaimonVindexMetadata metadata;
paimon_vindex_reader_metadata(reader, &metadata);
paimon_vindex_reader_optimize_for_search(reader);

int64_t ids[10];
float distances[10];
PaimonVindexSearchParamsEx params =
    paimon_vindex_search_params_ex_default();
params.top_k = 10;
params.max_initial_filter_expansion_factor = 4;
paimon_vindex_reader_warmup_queries(reader, representative_queries, 8, 0);
paimon_vindex_reader_search_ex(reader, query, &params, ids, distances, 10);
paimon_vindex_reader_free(reader);
```

`PaimonVindexSearchParamsEx` is append-only and passed by pointer. Prefer `paimon_vindex_search_params_ex_default()`, which fills the semantic defaults and sets `struct_size` to `PAIMON_VINDEX_SEARCH_PARAMS_EX_V1_SIZE`. The size is the exact end of the last initialized field, not `sizeof(PaimonVindexSearchParamsEx)`, because structure sizes may include tail padding. Newer libraries default fields outside an older caller's initialized prefix and older libraries ignore trailing fields from newer callers. The original by-value search functions remain available for ABI compatibility.

`PaimonVindexOutputFile` and `PaimonVindexInputFile` are callback structures. The input callback receives every positional range in one I/O batch, allowing object-store implementations to issue reads concurrently. DiskANN batch search may also invoke the callback concurrently from separate query workers, so the callback and its context must be thread-safe. The input structure also carries the three optional read-capability hints described above. C callers own result buffers and pass their capacity explicitly. Metadata reports `pq_m`/`pq_bits` for PQ-backed indexes and `rq_bits` for IVF-RQ. IVF-SQ reports `pq_m=0` and `pq_bits=8` to expose its scalar-code width.

## C++ RAII {#cpp}

`include/paimon_vindex.hpp` wraps the C ABI with explicit ownership and automatic cleanup.

```cpp title="C++"
#include "paimon_vindex.hpp"

std::vector<std::pair<std::string, std::string>> options = {
    {"index.type", "ivf_flat"}, {"dimension", "128"},
    {"nlist", "1024"}, {"metric", "l2"},
};

paimon::vindex::Training training =
    paimon::vindex::Trainer::train(
        options, training_vectors.data(), training_count);
paimon::vindex::Writer writer(std::move(training));
writer.add_vectors(row_ids.data(), vectors.data(), vector_count);
writer.write_index(output_file);

paimon::vindex::Reader reader(input_file);
auto metadata = reader.metadata();
reader.optimize_for_search();
reader.warmup_queries(representative_queries.data(), 8);
auto params = paimon::vindex::SearchParams::automatic(10);
params.max_initial_filter_expansion_factor = 4;
params.ivfpq_batch_table_reuse = PAIMON_VINDEX_IVFPQ_BATCH_TABLE_REUSE_AUTO;
auto result = reader.search(query.data(), params);
```

## Java / JNI {#java}

The package is `org.apache.paimon.index.vector`. String options map directly to Paimon table and index properties; Rust parses and validates them when a Trainer is created.

The Maven JAR contains the supported Linux x86-64, Linux aarch64, macOS arm64, and Windows x86-64 JNI libraries. The first native API call selects the current platform, extracts that library to a private temporary file, and loads it automatically. To use a separately built library instead, start the JVM with `-Dpaimon.vindex.native.path=/absolute/path/to/the/library`.

```java title="Java · build and search"
Map<String, String> options = new HashMap<>();
options.put("index.type", "ivf_sq");
options.put("metric", "l2");

try (VectorIndexTraining training =
             VectorIndexTrainer.train(options, trainingVectors, trainingCount);
     VectorIndexWriter writer = new VectorIndexWriter(training)) {
    writer.addVectors(rowIds, vectors, vectorCount);
    writer.writeIndex(vectorIndexOutput);
}

try (VectorIndexReader reader = new VectorIndexReader(vectorIndexInput)) {
    VectorIndexMetadata metadata = reader.metadata();
    reader.optimizeForSearch();
    VectorSearchResult result = reader.search(
            query,
            VectorSearchParams.automatic(10));
}
```

```java title="Java · optional filtered-IVF tuning"
// Example only: select the factor using workload-specific
// latency and Recall@K measurements.
VectorSearchParams params =
        VectorSearchParams.automatic(10)
                .withMaxInitialFilterExpansionFactor(4);
```

### Batching a large training set

```java title="Java · Trainer-owned native staging"
try (VectorIndexTrainer trainer = VectorIndexTrainer.create(options)) {
    for (float[] batch : trainingBatches) {
        trainer.addTrainingVectors(batch, batch.length / dimension);
    }
    try (VectorIndexTraining training = trainer.finishTraining();
         VectorIndexWriter writer = new VectorIndexWriter(training)) {
        writer.addVectors(rowIds, vectors, vectorCount);
        writer.writeIndex(vectorIndexOutput);
    }
}
```

Staging avoids one very large Java `float[]` and its array-length limit. Native reservoir sampling bounds retained training rows, but streaming creation still needs concrete `dimension` and either concrete `nlist` or `expected-vector-count`.

## Python {#python}

The C++, Java, and Python Readers serialize operations on one native handle across threads. Calling another method on that same handle from its input/output callback is rejected as reentrant—even when DiskANN invokes the callback from a native query worker—instead of deadlocking or re-entering a mutable native handle; use a separate handle if a callback needs vector-index work. The Python package loads `libpaimon_vindex_ffi` with ctypes. A single search returns one-dimensional NumPy arrays. `search_batch` accepts a two-dimensional query array and returns arrays shaped `(query_count, top_k)`.

```python title="Python"
from paimon_vindex import (
    SearchParams, VectorIndexReader,
    VectorIndexTrainer, VectorIndexWriter,
)

class VectorIndexInput:
    def __init__(self, data: bytes):
        self.data = data
        self.estimated_random_read_latency_nanos = 20_000_000
        self.preferred_window_bytes = 64 * 1024
        self.max_ranges_per_read = 32

    def pread_many(self, ranges):
        return [self.data[pos : pos + length] for pos, length in ranges]

options = {"index.type": "ivf_sq", "metric": "l2"}
training = VectorIndexTrainer.train(options, training_vectors)
writer = VectorIndexWriter(training)
writer.add_vectors(row_ids, vectors)
writer.write(output)

reader = VectorIndexReader(VectorIndexInput(index_bytes))
reader.optimize_for_search()
reader.warmup_queries(representative_queries)
ids, distances = reader.search(
    query, SearchParams.automatic(top_k=10))
```

## Metadata filter pushdown {#filter}

The query layer can evaluate metadata predicates through Paimon table or scalar indexes, serialize the allowed row IDs as a 64-bit Roaring bitmap, and pass that set into ANN search as a prefilter. Every binding uses the same wire format:

| Language | Single / batch entry points |
|----|----|
| Rust | `search_with_roaring_filter` / `search_batch_with_roaring_filter` |
| C | `paimon_vindex_reader_search_with_roaring_filter` / `...search_batch_with_roaring_filter` |
| Java | `VectorIndexReader.search(..., byte[])` / `searchBatch(..., byte[])` |
| Python | `search(..., filter_bytes=...)` / `search_batch(..., filter_bytes=...)` |

> **Row-ID constraint** — `RoaringTreemap` uses the `u64` domain, so row IDs must be non-negative to match the filter. Filters are query payloads and are never written into index files.

DiskANN maps sparse filters through its persisted row-ID lookup after first use; dense filters scan resident row IDs once per batch. It normally scans PQ codes only for matching nodes, then exactly reranks a bounded candidate set. A quality-gated broad filter may use ordinary graph navigation and post-filter its candidates, with a complete matching-node PQ fallback before raw-vector reads. Benchmark filter-heavy traffic separately because the safe PQ path remains linear in the number of matching nodes.
