---
title: "DiskANN"
description: "Build and tune DiskANN for immutable vectors on local SSD or object storage."
---

<!--
  Licensed to the Apache Software Foundation (ASF) under one or more
  contributor license agreements. See the NOTICE file distributed with
  this work for additional information regarding copyright ownership.
  The ASF licenses this file to you under the Apache License, Version 2.0.
-->

# DiskANN

Keep compact PQ navigation data in memory, read graph and raw-vector records on demand, then rerank the best candidates with persisted F32 or F16 values. Choose an interleaved local-SSD layout or a compact remote-range layout; both are queried through the same abstract storage interface.

## Positioning and trade-offs {#position}

|               |                        |
|---------------|------------------------|
| Navigation    | Global Vamana graph    |
| Resident data | PQ model + IDs + codes |
| Paged data    | Graph + raw vectors    |
| Distance      | L2 / IP / Cosine       |

### Good fit

- The immutable raw-vector set is too large to keep in RAM but fits on local NVMe or SSD.
- High recall is required without scanning multiple complete IVF lists.
- Indexes are built offline and replaced atomically rather than updated node by node.
- Object storage is the system of record and query nodes cache index files on local SSD.
- Direct object-store queries can tolerate a small number of batched range-read rounds.

### Poor fit

- The metric-specific recall target cannot be validated on representative queries before deployment.
- Vectors change frequently and the serving graph must be updated incrementally.
- The full index is already memory-resident and the lowest possible tail latency is the only goal.
- Filtered queries dominate and routinely inspect a large resident ID/PQ population.
- Build peak memory, build duration, or serialized raw-vector size is the primary constraint.

> **Disk-backed does not mean a minimum-size file** — v1 stores every rerank vector in addition to PQ codes and graph edges. The balanced default uses F16, halving that payload and its read bandwidth at the cost of quantized final distances; explicit F32 remains available when exact rerank distances matter. The main advantage is bounded resident data and range-granular query I/O.

> **Measured position** — On the public SIFT1M, GIST1M, and GloVe-100 ANN-Benchmarks corpora, with a PQ code budget equal to 6.25% of raw vector bytes and the balanced F16 default, `l_search=100` reached Recall@10 0.9915 / 0.9336 / 0.8355 at 1.50 / 1.83 / 1.90 ms P95 from a warm local filesystem cache. Under the fixed-2-ms-per-read remote model it reached 0.9808 / 0.8482 / 0.8029 recall at 15.97 / 14.22 / 18.18 ms P95. The trade-off is build time: 74 seconds, 11 minutes 26 seconds, and 2 minutes 33 seconds, much slower than IVF. See the [public-corpus comparison](index.md#public-corpus-check) for the complete matrix and methodology.

### How it compares with this repository's other indexes

| Choice | Prefer it when | Prefer DiskANN when |
|----|----|----|
| IVF-FLAT | You need a simple recall baseline, three metrics, and predictable list scans. | Scanning raw vectors in enough lists exceeds the CPU or I/O budget. |
| IVF-PQ / IVF-RQ | Serialized size and sequential compact-code throughput matter more than reranking persisted vectors. | You can afford rerank vectors on disk and need graph-guided candidate generation plus F32-exact or F16-quantized final distances. |
| IVF-SQ | One-byte residual dimensions and complete selected-list scans meet the recall and bandwidth target. | A global graph and page-granular adjacency/vector reads better match large local-SSD indexes. |

## Build and search architecture {#architecture}

### Build

1.  ### Train PQ

    Split each vector into `pq.m` balanced contiguous chunks and train `2^pq.bits` centroids per chunk. Training uses deterministic reservoir sampling capped at 50,000 vectors, so training memory does not grow with the full corpus.

2.  ### Build Vamana

    Initialize fixed-capacity adjacency in parallel, then navigate build candidates with symmetric PQ centroid-distance tables by default. L2 uses the triangle-inequality robust-prune rule, inner product uses an occluding rule, and cosine normalizes vectors before using the equivalent L2 graph ordering. Reverse-edge overflow pruning, centroid entry selection, and reachability repair remain metric-aware and full precision, so quantization accelerates discovery without becoming the final edge-quality check.

3.  ### Reorder for locality

    Breadth-first reorder compact node blocks in place, remap row IDs, PQ codes, raw vectors, and graph edges together, then sort every adjacency list once for serialization.

4.  ### Write one file

    Serialize a 256-byte header, persistent row-ID/adjacency indexes, and either graph pages plus a dense vector-record section or page-contained `[raw vector][compressed adjacency]` records.

### Unfiltered search

1.  Load the PQ codebook, adaptively packed row IDs, PQ codes, and the block-compressed adjacency locator index into resident memory.
2.  Use the PQ distance table, four-code neighbor batches, and a reusable contiguous heap frontier to traverse the Vamana graph with search-list size `L`.
3.  Read only adjacency windows needed by the selected beam.
4.  Read raw-vector windows for at most `min(L, max(4 × top_k, 64))` candidates.
5.  Compute squared-L2 distances from the persisted F32/F16 records and return the best `top_k`.

Every sorted adjacency list independently uses canonical delta-varint encoding only when it is smaller than fixed-width IDs; otherwise it falls back to raw little-endian `u32`. Locator positions use one `u64` base per 16 nodes plus a `u16` relative offset and `u16` degree/encoding value per node, for `4 × N + 8 × ceil(N / 16)` bytes. Raw fallback guarantees that compression cannot increase adjacency payload bytes or logical page count.

> **Storage choice versus upstream DiskANN** — The legacy Microsoft [PQFlashIndex](https://github.com/microsoft/DiskANN/blob/cpp_main/src/pq_flash_index.cpp) packs fixed-width node records into 4 KiB sectors and keeps PQ navigation data resident. This implementation retains the proven resident-PQ / on-demand-rerank split, but stores variable-length compressed adjacency separately from dense F16/F32 vectors so degree slack does not consume vector-section bytes. It follows current [DiskANN3](https://github.com/microsoft/DiskANN) in keeping storage behind a provider abstraction: query code depends on `SeekRead`, never a local-file type.

> **Third pre-release format review** — The current Microsoft Rust workspace emphasizes provider/accessor-driven asynchronous graph search, while the original SSD design contributes resident PQ navigation, sector reads, and a BFS-derived hot cache. This v1 already has the corresponding abstract `SeekRead` sessions, resident PQ codes, compact 4 KiB adjacency pages, breadth-first locality reorder, hot prefix, bounded LRUs, and persisted-vector rerank. Keeping dense vectors separate in the default compact layout avoids fixed-degree sector slack and remains better suited to remote coalescing. No on-disk change survived this review. Compact batch rerank now hashes candidate windows before sorting only the unique windows into deterministic I/O order; median local batch time changed from 117 / 223 / 162 ms to 111 / 215 / 159 ms on SIFT/GIST/GloVe. A broader Vec sort/dedup replacement for graph window planners was rejected after regressing local SIFT/GIST by 7–13%.

> **F16-quantized or F32-exact rerank** — The balanced default evaluates decoded binary16 values. Explicit F32 returns exact distances for reranked candidates and is also selected by the high-recall preset. Recall remains approximate in both modes because PQ-guided graph traversal and the finite rerank set decide which candidates reach that stage.

## SSD and object-store deployment {#deployment}

### Preferred production path

1.  Publish the immutable index to S3, OSS, HDFS, or another durable store.
2.  Download or cache the complete file on the query node.
3.  Open the Reader and warm the resident sections before serving traffic. The input latency selects the read plan automatically.
4.  Replace files atomically when a new snapshot is ready.

### Direct remote path

1.  Implement concurrent positional reads in the storage callback.
2.  Advertise representative random-read latency when the adapter already knows it; otherwise opening measures the mandatory header read.
3.  Give the Reader one total memory budget; it automatically sizes the adjacency prefix and query caches for the selected internal tier.
4.  Budget P95/P99 for graph rounds plus one persisted-vector rerank round.

| Internal tier | Read window | Unfiltered graph beam | Intent |
|----|----|----|----|
| `Memory` | 4 KiB | 16 nodes per round | Minimize copied bytes while reducing callback rounds for an in-memory immutable source. |
| `LocalStorage` | 16 KiB | 4 nodes per round | Coalesce nearby graph and rerank records while keeping local-SSD byte amplification bounded. |
| `RemoteStorage` | 32 KiB | 16 nodes per round | Balance fixed network latency against bandwidth and byte amplification. |
| `ObjectStore` | 64 KiB | 16 nodes per round | Minimize expensive request rounds when additional transferred bytes are acceptable. |

All tiers use exactly the same file. Quality-gated filtered graph traversal uses the separately validated beam width 4 for every tier; the table's wider beams apply to ordinary unfiltered graph search. The Reader selects exactly once while opening: below 50 µs selects `Memory`, below 750 µs selects `LocalStorage`, below 3 ms selects `RemoteStorage`, and 3 ms or more selects `ObjectStore`. `SeekReadCapabilities::estimated_random_read_latency_nanos` bypasses measurement when an adapter already knows representative latency; zero reuses the mandatory header read's elapsed time and adds no probe I/O. Window and range-count capabilities can refine the selected plan; physical alignment remains encapsulated by the storage adapter. A remote callback receives all ranges for a round together and should issue them concurrently.

> **Measured local-window choice** — In a window-only A/B on the same retained SIFT index, 4 / 8 / 16 / 32 KiB produced Recall@10 0.9910 / 0.9912 / 0.9913 / 0.9913, P95 1.87 / 1.71 / 1.41 / 1.39 ms, and batch throughput 6,649 / 6,931 / 8,127 / 9,587 QPS. The 32 KiB plan read 1.25 GiB across 1,000 sequential queries versus 0.69 GiB at 16 KiB for an 18% batch gain and negligible P95 gain, so the local default stops at 16 KiB. The later F16 SIMD optimization lifts the final 16 KiB batch result further without changing that read-amplification comparison. The remote and object-store internal tiers keep their latency-oriented 32 / 64 KiB plans.

## Usage {#usage}

```text title="Properties · recommended baseline"
index.type=diskann
dimension=128
metric=l2
deployment-profile=local_storage
diskann.build-preset=balanced
```

```rust title="Rust · abstract storage reader"
use paimon_vindex_core::index::{
    VectorIndexReader, VectorIndexReaderOptions, VectorSearchParams,
};

let options = VectorIndexReaderOptions::new(4 * 1024 * 1024 * 1024);
// `input` comes from the storage/cache layer and implements SeekRead.
let mut reader = VectorIndexReader::open_with_options(input, options)?;
reader.optimize_for_search()?;
let plan = reader.read_plan(); // Current effective DiskANN I/O/cache diagnostics.
reader.warmup_queries(&representative_queries, representative_query_count, 100)?;
reader.calibrate_search_width(&representative_queries, representative_query_count, 10)?;
let params = VectorSearchParams::automatic(10);
let (ids, squared_l2) = reader.search(&query, params)?;
```

The core API depends only on `SeekRead`; the application storage adapter owns local-cache, object-store, and transport details. Its `pread` implementation should issue all ranges in a search round concurrently and may report immutable-source capabilities so DiskANN does not overfill a callback or choose the wrong coalescing size. Its I/O executor must remain runnable when CPU query workers are saturated: recursively scheduling range reads onto that same fully occupied worker pool can starve the reads while every query waits on single-flight cache entries. The benchmark therefore uses an independent range-I/O pool for DiskANN. The generic `Read + Seek` adapter is a sequential compatibility path, not the recommended production integration.

The parent Reader and retained search-session clones share one immutable raw-vector LRU. Query-local caches hold borrowed `Arc` windows, so concurrent reranks for the same range are single-flighted instead of reading duplicate bytes. Cache hits and oldest eviction update linked recency metadata in constant time, and capacity is charged by retained `Vec::capacity()`. Misses in one search are read together; a query whose working set is larger than its local-reference allowance bypasses local retention, while batch rerank processes bounded chunks. `DiskAnnSearchStats` resets per top-level search and exposes raw-vector hits, storage misses, and evictions across both cache layers.

F16 compact-vector rerank on AArch64 loads unaligned half-precision lanes, rejects non-finite values, converts them to F32, and accumulates L2 in the same NEON loop. This removes the temporary decode buffer and the second SIMD pass while preserving the scalar fallback and sentinel behavior on other targets.

Cold adjacency windows outside the automatically sized hot prefix use a separate bounded LRU shared by the Reader and its batch workers. Concurrent misses for the same window are single-flighted: one worker performs the positional read while the others wait and then borrow the immutable payload. Successful reads move their existing Vec allocation into shared ownership without copying the payload. Capacity is charged by actual retained allocation capacity and clipped to both the adjacency section and the remaining total memory budget. Budgets of at least 1 MiB use 16 independently locked capacity shards. Every graph worker also bounds query-local cold-window references to 8 MiB and releases least-recently-used windows between beam rounds.

DiskANN retains cloned storage handles as reusable search sessions. Small batches (up to four queries per worker) run graph traversal and persisted-vector rerank concurrently end to end, avoiding a serial parent-rerank bottleneck; the interleaved layout always uses this path because the graph read already contains the vector. Larger compact-layout batches keep the object-store-friendly path: workers return node IDs, the parent unions raw-vector windows, and rerank streams chunks of at most 64 MiB and 1024 ranges. Unsupported sources fall back to complete serial queries. Single and batch rerank retain only the best `top_k` results per query in bounded heaps. Rust storage adapters, C/Python range callbacks, and JNI inputs opt into reusable sessions through `try_clone_reader`. Every clone must expose the same immutable byte sequence and be safe for concurrent calls.

```python title="Python · direct object-store reader"
reader = VectorIndexReader(
    input_with_concurrent_pread_many,
    memory_budget_bytes=4 * 1024 * 1024 * 1024,
)
reader.optimize_for_search()
plan = reader.read_plan()
reader.warmup_queries(representative_queries, l_search=100)
reader.calibrate_search_width(representative_queries, top_k=10)
ids, distances = reader.search(
    query, SearchParams.automatic(top_k=10)
)
```

The C, C++, Java/JNI, and Python APIs expose the same latency hint, total Reader memory budget, and concrete read-plan diagnostics. See the [shared API guide](api.md) for lifecycle and callback ownership.

## Build parameters {#build-parameters}

| Parameter | Default / validation | What it controls | When increased |
|----|----|----|----|
| `dimension` | Inferred by Java/Python one-shot training; required by streaming APIs; `1..=1024` | Raw vector width and PQ codebook size. | Raw-vector payload and rerank work grow linearly. |
| `metric` | Required: `l2`, `inner_product`, or `cosine` | Distance used by graph build, PQ navigation, and exact reranking. Inner-product scores are negative dot products; cosine scores are `1 - cosine`. | Semantic, not inferred. Cosine vectors and queries are normalized internally; zero vectors retain distance 1. |
| `target-recall` | Optional; `0..=1` | Selects `fast_build` at ≤ 0.85, `high_recall` at ≥ 0.97, and `balanced` between them when no preset is pinned. | This is a deterministic starting policy, not a measured guarantee; validate on held-out queries. |
| `max-bytes-per-vector` | Optional positive persisted-size budget | Guides PQ width and may select 4-bit PQ and F16 rerank records. Configuration is rejected before training when the estimated row bytes plus amortized fixed data exceed the budget. | A larger allowance preserves more PQ/raw precision. The check is conservative; compression, headers, and alignment mean it remains a per-vector sizing bound rather than an exact final-file-size promise. |
| `deployment-profile` | `auto`, `memory`, `local_storage`, `remote_storage`, or `object_store` | Selects an interleaved local layout when one record fits 4 KiB; remote/object profiles select compact layout. | It describes the intended serving medium, not a local-file implementation. |
| `diskann.build-preset` | `fast_build`, `balanced`, or `high_recall`; inferred from target recall when omitted | Resolves `R`, `Lbuild`, alpha, raw encoding, and build distance as one coherent baseline. | Higher-recall presets increase graph build work, file bytes, and memory. |
| `pq.code-ratio` | 0.0625; finite and in `(0, 0.25]` for 8-bit or `(0, 0.125]` for 4-bit | Target ratio between resident PQ-code bytes and raw `f32`-vector bytes. The builder selects the nearest `m` and distributes dimensions across balanced chunks. | Usually reduces PQ error when increased, but grows resident memory and per-candidate lookup work. |
| `pq.m` | Optional expert override; `1..=dimension` | Concrete PQ chunk count. Explicit values take precedence over `pq.code-ratio`; exact chunk offsets are persisted in the self-describing codebook. | Use only for a measured override of automatic sizing. Non-divisible dimensions and odd 4-bit values are valid. |
| `pq.bits` | 8; must be 4 or 8 | Centroids and stored bits per PQ chunk. Four-bit codes pack two chunks per byte, use 16-entry query tables, and require a zero high padding nibble when `m` is odd. | Eight bits generally improve graph-navigation recall; four bits reduce codebook, resident codes, training work, and lookup-table size. Rebuild and benchmark both. |
| `diskann.max-degree` | 64; `1..=1023`, preserving page-contained raw fallback | Maximum graph out-degree `R`. | May improve connectivity and recall; increases graph bytes, build work, and page density cost. |
| `diskann.build-search-list-size` | Omitted: `max(100, R)`; explicit values must be ≥ `R` | Candidate width `Lbuild` during Vamana construction. | Usually improves graph quality while increasing build CPU and per-worker scratch. |
| `diskann.alpha` | 1.2; finite and ≥ 1 | Second-pass robust-prune threshold. | Higher values prune candidates less aggressively; validate degree, recall, and graph behavior empirically. |
| `diskann.seed` | 42 | Random neighbor initialization and build order. | It is not a quality knob; fix it for reproducible comparisons. |
| `diskann.memory-budget-bytes` | 8 GiB; \> 0 | Selects the normal parallel graph build when it fits. Otherwise DiskANN automatically chooses 2–64 coarse, overlapping shards, builds and merges one shard at a time, then performs global robust pruning and connectivity repair. It rejects only when neither plan fits the estimate. The 8 GiB default also leaves enough internal build-state budget for the standard one-million-vector, 960-dimensional GIST corpus; the source-vector slice supplied by the caller is separate. | A larger budget reduces or avoids sharding and usually shortens build time. This is an internal peak estimate, not an operating-system memory limit. |
| `diskann.storage-layout` | `auto` or omitted; explicit `compact`/`interleaved` overrides | `compact` stores graph pages separately from densely packed vector records; `interleaved` stores each vector immediately before its adjacency list and requires `E × dimension + 4 × R ≤ 4096`, where `E` is 4 for F32 and 2 for F16. | Automatic selection follows `deployment-profile`; pin only after measuring a deployment-specific exception. |
| `diskann.raw-vector-encoding` | `auto` or omitted; preset/budget resolves F32 or F16 | Controls the persisted rerank-vector element width for both layouts. Compact F32/F16 payloads are exactly `4 × d × N` / `2 × d × N` bytes with no per-page padding. | Explicit F32 preserves original rerank distances. Explicit F16 halves raw-vector I/O but must be recall-tested. |
| `diskann.build-distance` | `auto` or omitted; preset resolves PQ or full precision | Selects build-traversal distance. Both modes use full precision for robust pruning and connectivity repair. | `high_recall` uses full precision; balanced/fast presets use PQ guidance. |

### Starting values

- Start with `diskann.build-preset=balanced`, the automatic `pq.code-ratio=0.0625`, and the intended `deployment-profile`. The balanced preset resolves to 8-bit PQ, `R=64`, `Lbuild=100`, alpha 1.2, F16, and PQ-guided construction unless an explicit option changes the representation.
- For 8-bit PQ, the default resolves to `m=32` at 128 dimensions and `m=240` at 960 dimensions. Other shapes use the nearest count and balanced chunks whose widths differ by at most one. Inspect metadata for the resolved value and use explicit `pq.m` only after a representative recall measurement.
- Test `pq.bits=4` only as a separate build. Accept it when the smaller resident footprint and faster lookup tables preserve the required Recall@K across the full `l_search` sweep.
- Use the F16 default for the storage/I/O baseline, but compare `diskann.raw-vector-encoding=f32` when exact final distances or an unusual numeric range matters. F16 can change top-k ordering even when graph candidates are identical and rejects values outside the finite binary16 range.
- If online `l_search` must become very large to reach the target, test `Lbuild=150/200` before increasing `R`.
- Increase `R` only after recording file size, build peak RSS, graph-page bytes, and recall. Rebuilds are required for every build-parameter change.

> **Why automatic relative sizing matters** — On public GIST1M, a fixed `m=64` uses only 1.67% of the raw 960-dimensional vector bytes and DiskANN saturated near 0.74 Recall@10. The default ratio instead resolves to `m=240`, the same relative budget as `m=32` on SIFT1M; local Recall@10 then reached 0.973 at `l_search=200` and 0.991 at `l_search=500`.

## Query parameters {#query-parameters}

| Parameter | DiskANN meaning | Guidance |
|----|----|----|
| `top_k` | Requested result count. | Results are padded with `-1` and `f32::MAX` if fewer candidates survive. |
| `Auto` | Uses a Reader-calibrated width when available; otherwise `max(100, 2 × top_k)`. Adaptive filtered graph candidates require an effective value of at least 200 in the initial quality-gated release. | Preferred production default. Calibrate on representative queries, then validate the selected width against exact ground truth. |
| `l_search` | Explicit graph search-list size `L=max(top_k, l_search)`. | Expert recall/latency override. Sweep 50, 100, 200, and 400 when the automatic or calibrated result misses the workload target. |
| `nprobe` | Invalid for DiskANN. The tagged search API rejects an IVF width instead of ignoring it. | DiskANN has one global graph and reports logical `nlist=1`. |

> **Large l_search has two costs** — It expands more graph nodes and can read more adjacency pages. Unfiltered graph search uses `max(4 × top_k, 64)` as the exact-rerank seed count, then also reranks later graph candidates whose raw vectors fall in the same already selected read windows. The number of exact distance evaluations can therefore exceed the seed count without adding vector-read windows; candidates in any other window remain approximate-only. Increasing `l_search` still improves discovery, but does not imply exact reranking of every visited candidate.

## Reader and warm-up parameters {#reader-parameters}

| Parameter | Default | Purpose and guidance |
|----|----|----|
| `estimated_random_read_latency_nanos` (input capability) | 0 | Optional representative random-read latency. Zero measures the mandatory header read; a positive value selects the same internal plan without timing noise. |
| `memory_budget_bytes` | 4 GiB | Total Reader budget. Required PQ/model/row-ID state is reserved first; the remainder is partitioned across an automatically sized hot adjacency prefix, a cold-adjacency LRU, and a raw-vector LRU, each clipped to its actual section size. |

Read-plan selection is part of `open`, not `optimize_for_search`. Call `optimize_for_search` before accepting traffic when predictable first-query latency matters: it loads resident data, preloads the automatically sized adjacency prefix, and validates every covered 4 KiB logical page in parallel before publishing that prefix. Follow it with `warmup_queries` to populate query-dependent caches and `calibrate_search_width` to select an automatic width. None is required for correctness; the first non-empty query initializes resident data lazily.

> **Resident size approximation** — Let `K=2^pq.bits` and `C=pq.m` for 8-bit or `ceil(pq.m / 2)` for 4-bit. The steady required state is approximately `4 × K × dimension + ceil(N × row_id_bit_width / 8) + C × N + 4 × N + 8 × ceil(N / 16) + adjacency_page_count + 4 × K × pq.m` bytes, plus headers, decode scratch, and allocator overhead. The one public `memory_budget_bytes` value covers this steady state and the automatically retained adjacency/raw-vector caches. Query-local and active batch scratch remain transient and are separately bounded by the implementation.

## Filtered search behavior {#filtering}

Every Roaring64 filter is first translated into a bitmap of matching internal nodes. Filters with at most `N / 16` values lazily load the persisted `(row_id, node_id)` order and resolve duplicate row IDs with binary-search ranges. Dense filters use a sequential decoder over resident raw or FOR-bitpacked row IDs. If the optional lookup would exceed the Reader memory budget, sparse queries transparently use the sequential path. A filtered batch performs this translation once and shares the bitmap with every candidate worker.

The safe baseline scans matching resident PQ codes and keeps `T=min(M, max(4 × top_k, 64))` candidates. A filtered batch evaluates this path in node-major tiles of up to four queries: each matching PQ code is loaded once per tile while each query keeps an independent bounded heap. The tile shrinks automatically when necessary so active PQ distance tables occupy at most 2 MiB. The resident scan does not clone or read the storage source. The graph path traverses the ordinary unfiltered graph, post-filters its candidates, and falls back to that complete matching-node PQ scan before any raw-vector read when fewer than `T` matches survive. It never filters graph edges, so the filter cannot disconnect traversal.

> **Initial graph quality gate** — The checked formula is `adaptive_l=max(resolved l_search, 2 × ceil(T × N / M))`, capped at `N`, with estimated graph work `min(N, adaptive_l × (R + 1))`. Graph work must be at most half the matching-node scan. Recall-matrix evidence additionally keeps 50% filters and effective `l_search<200` on scan; the initial graph-enabled cell is 100% matching with effective `l_search≥200`. The coalesced `RemoteStorage` and `ObjectStore` plans also require the complete adjacency section to be preloaded.

### Benefit

- Sparse filters avoid an `O(N)` row-ID scan after first use.
- Dense batch filters scan row IDs once per batch rather than once per query.
- Quality-gated all-row filters can reuse normal graph navigation.
- Node-major PQ tiles reuse code loads across up to four queries.
- Batch rerank reads each shared vector window once per chunk.
- F32-exact or F16-quantized final distances with unchanged sentinel behavior.

### Cost

- The optional lookup occupies `4 × N` resident bytes.
- The first sparse query reads and validates that section.
- A filtered batch retains its internal-node bitmap and at most one 1024-query chunk of candidate sets until rerank completes.
- Most filtered cells intentionally remain linear in matching PQ codes to protect recall.

Negative stored row IDs and bitmap values above `i64::MAX` never match. Duplicate row IDs return every matching internal node before bounded PQ candidate selection.

## Recommended tuning order {#tuning}

1.  **Fix acceptance criteria.** Use production queries and filters; record Recall@K, P50/P95/P99, QPS, index bytes, peak RSS, build duration, read rounds, and bytes read.
2.  **Choose storage layout and characterize read latency.** Benchmark `interleaved` on the cached local path, exercise the latency-hint or measured remote path for bandwidth-balanced network reads, and test `compact` when minimizing object-store request rounds matters most. Include file-size and byte-amplification results.
3.  **Calibrate, then verify search width.** Run `calibrate_search_width` on representative queries, validate the chosen 100/200/400 width against exact ground truth, and use an explicit `l_search` only when the workload target requires it.
4.  **Tune PQ.** Sweep `pq.code-ratio` around 0.0625 and record the resolved `pq.m`, then compare a separate 4-bit build. Both the ratio and bit width change resident memory and traversal accuracy.
5.  **Improve graph quality.** Compare `product_quantized` against the `full_precision` control, then raise `Lbuild` and consider `R`. Do not change several build knobs in one experiment.
6.  **Size Reader memory.** Sweep the single total memory budget after graph parameters stabilize. The Reader repartitions it automatically; use search statistics and read rounds to identify whether a larger working set has material benefit.
7.  **Run production-scale validation.** Include cold/warm cache, concurrent queries, selective and broad filters, corrupt/truncated input, and restart/rollover behavior.

## Capacity model {#capacity}

| Resource | Approximation | Controlled by |
|----|----|----|
| Raw-vector file payload | Compact: exactly `E × dimension × N`, with `E=4` for F32 and `E=2` for F16. Interleaved records add adjacency-page tail padding. | Dimension, vector count, encoding, and layout. |
| PQ-code resident/file payload | `pq.m × N` for 8-bit; `ceil(pq.m / 2) × N` for 4-bit | `pq.m` and `pq.bits`. |
| Row IDs | File: `32 + ceil(N × w / 8)`; raw fallback `32 + 8 × N`. Resident storage omits the 32-byte header. | Vector count and global row-ID span; `w` is the minimum width in `0..63`. |
| Lazy row-ID order | `4 × N` | Sparse filters and resident budget. |
| Resident adjacency locators | `4 × N + 8 × ceil(N / 16)` (about 4.5 bytes per node) | Vector count. |
| Hot adjacency prefix | Automatically up to 16 MiB for memory/local latency, 32 MiB for remote latency, or 64 MiB for object-store latency, clipped to the adjacency section and remaining total budget | Measured or hinted latency, adjacency size, required resident state, and `memory_budget_bytes`. |
| Cold-adjacency LRU | Memory/local latency tiers cap it at 64 MiB. Remote/object-store latency tiers may use all remaining budget needed by the cold adjacency section after the raw-vector split; all cases are clipped to section size and total budget. | Shared with retained batch workers; split across 16 shards for production-sized budgets. |
| Adaptive adjacency payload | At most `4 × total_edges`, usually less after BFS-remapped delta-varint encoding, plus page-tail padding | Actual graph degrees, ID locality, and `diskann.max-degree`. |
| PQ codebook | `4 × 2^pq.bits × dimension` | Dimension and `pq.bits`; independent of `pq.m`. |
| Build peak | Raw vectors + IDs + row-ID encoding scratch + codes + codebook + row-ID order + adjacency locators + max(graph-build, in-place graph-remap). PQ training keeps a deterministic reservoir of at most 50,000 rows, reduced automatically so the retained sample, optional cosine-normalized copy, codebook, and parallel KMeans scratch fit the build budget. The low-level Rust `DiskAnnIndex::train` returns an error when even the minimum one-worker plan cannot fit. If the parallel graph estimate exceeds the budget, overlapping coarse shards are built and merged one at a time; the final compact adjacency remains approximately `4 × N × R + 2 × N`. | `N`, `dimension`, `pq.m`, `pq.bits`, `R`, `Lbuild`, memory budget, and Rayon worker count. |
| Query-local adjacency | At most 8 MiB of retained cold-window allocation capacity/references after each beam round, plus the active beam | Internal safety bound; stats report peak bytes and evictions. |
| Raw-vector cache and query buffers | One shared immutable raw-vector LRU, plus up to 8 MiB of reusable/local-reference capacity per active worker under a 64 MiB aggregate worker allowance. Memory/local latency tiers cap the shared LRU at 64 MiB; remote/object-store latency tiers may use the remaining Reader budget up to the vector-section size. | The shared LRU is clipped to the total Reader budget. Session scratch and borrowed references persist for reuse across batch calls without duplicating payload bytes. |
| Graph candidate heaps | `L` retained candidates plus at most `2 × L` live frontier entries; contiguous capacity is reused between queries | `l_search` and graph worker count; evicted lazy entries are periodically compacted. |
| Graph visited scratch | Minimum of dense `ceil(N / 8)` bytes or a reusable deterministic open-addressed table sized from `L × R` | Vector count, `l_search`, graph degree, and graph worker count; scan/rerank-only paths allocate neither form. |
| Filtered PQ query tile | Up to four distance tables and bounded candidate heaps, with distance-table bytes capped at 2 MiB per active tile | `pq.m`, `pq.bits`, and Rayon worker count; unusually large tables automatically reduce the tile width. |
| Parallel exact-rerank references | One borrowed record reference per candidate in the active read chunk. In unfiltered graph search, references may exceed the seed count because every later candidate in an already selected raw-vector window is included; the unique-window count remains seed-bounded. | Allocated only when `candidate references × dimension ≥ 16384`; bounded by the 1024-query, 64 MiB, and 1024-range chunk limits. Benchmark stats expose both candidate references and unique windows. |

These formulas omit headers, alignment, allocator overhead, and per-query caches. Use measured RSS and serialized bytes as the final authority.

## Current implementation boundaries {#boundaries}

- v1 supports squared L2, negative-dot-product inner product, and `1 - cosine`, with dimensions up to 1024 and 4/8-bit PQ.
- The index is static. FreshDiskANN-style insert, delete, consolidation, and graph repair are not implemented.
- Serialization builds the complete graph in memory. Raw vectors are stored as explicit F32 or F16 scalar records; there is no PQ-only result mode.
- Recall is sensitive to both PQ capacity and graph/search budgets. On the public SIFT1M and GIST1M runs, the default ratio resolved to `m=32` and `m=240`. DiskANN Recall@10 was 0.995 and 0.973 at `l_search=200`, while GIST rose to 0.991 at `l_search=500`; use the public-corpus procedure rather than synthetic vectors when validating a new ratio.
- Remote graph traversal has dependent I/O rounds. It improved high-recall single-query latency over raw IVF scans at 2 ms RTT in the measured workload, but compact IVF-PQ won batch throughput when its lower recall was acceptable.
- Direct remote reads rely on the caller's range-read callback for actual concurrency.
- The file has strict structural validation but no embedded checksum; integrity belongs to the outer Paimon file/manifest layer.
- Generated correctness and I/O acceptance workloads pass, but production readiness still requires SIFT1M or representative full-scale recall, soak, concurrency, and failure testing.

> **Recommended maturity label: preview** — The implementation has cross-language tests, corruption checks, memory budgets, and local/remote acceptance instrumentation. Do not label it generally available until representative large-scale benchmarks and operational validation meet the deployment's SLOs.
