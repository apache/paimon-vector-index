---
title: "IVF-SQ"
description: "Build and tune a scalar-quantized IVF index for compact high-throughput search."
---

<!--
  Licensed to the Apache Software Foundation (ASF) under one or more
  contributor license agreements. See the NOTICE file distributed with
  this work for additional information regarding copyright ownership.
  The ASF licenses this file to you under the Apache License, Version 2.0.
-->

# IVF-SQ

Store one unsigned byte per residual dimension and scan the selected IVF lists directly. IVF-SQ targets the gap between raw IVF-FLAT and aggressively compressed IVF-PQ/RQ without paying for a graph in every list.

## Position {#position}

IVF-SQ preserves the IVF partitioning model and replaces every residual `f32` component with an 8-bit scalar code. It usually occupies about one quarter of IVF-FLAT's vector payload. Unlike IVF-PQ, each dimension is quantized independently, so there is no subquantizer-count parameter or codebook lookup table.

> **Use it when** — IVF-FLAT recall is good, its raw-vector payload or scan bandwidth is too large, and a one-byte-per-dimension representation fits the budget. Prefer IVF-PQ for much smaller codes; prefer DiskANN when high-recall large-scale local-SSD search needs page-granular graph traversal.

## Build and search {#algorithm}

1.  Train the IVF coarse centroids and assign training vectors to lists.
2.  Subtract each list centroid and learn per-dimension minimum/maximum residual bounds for that list.
3.  Encode every residual coordinate to an unsigned byte. Empty training lists use the global residual bounds.
4.  At query time, select `nprobe` lists, load their sorted row IDs and codes in one multi-range read, scan the codes with SIMD L2 or inner-product kernels, and merge the top K.

Cosine input is normalized through the shared metric preprocessing path. Filters are checked while scanning, so excluded rows do not enter the top-K heap.

## Configuration {#usage}

```text title="Options API"
index.type = ivf_sq
dimension = 128
nlist = 1024
metric = l2
ivf.coarse-assignment = auto
```

```rust title="Rust"
let config = VectorIndexConfig::IvfSq {
    dimension: 128,
    nlist: 1024,
    metric: MetricType::L2,
    use_approximate_coarse_assignment: true,
};
let params = VectorSearchParams::new(10, 16);
```

## Parameters {#parameters}

| Parameter | Requirement | Effect |
|----|----|----|
| `dimension` | Inferred by Java/Python one-shot training; otherwise \> 0 | Each vector uses exactly `d` SQ-code bytes. |
| `nlist` | Auto from `expected-vector-count`, or explicit \> 0 and no larger than training count | More lists shorten scans but enlarge centroid and per-list-bound metadata. |
| `metric` | Required: L2, inner product, or cosine | Selects preprocessing and the distance kernel. |
| `ivf.coarse-assignment` | `auto` by default; optional `exact` | `auto` uses Vamana when `dimension × nlist ≥ 1,000,000`, trading build speed for possible low-`nprobe` recall loss and graph startup cost; `exact` disables it. |
| `nprobe` | Automatic by default; explicit 1 to `nlist` | Auto accounts for K, average list size, and filter selectivity; explicit values provide a measured override. |

The scalar code width is fixed at 8 bits in v1. There is deliberately no `sq.bits`, graph-width, or search-width option.

## Stable v1 storage {#storage}

**64 B header**IVSQ v1**Global bounds**`2 × d × f32`**Per-list bounds**`2 × nlist × d × f32`**IVF centers**`nlist × d × f32`**Offset table**`nlist × 16 B`**Lists**blocked SQ codes + delta IDs

Every non-empty list is sorted by signed row ID before writing. Codes come first and are transposed within up-to-32-row blocks, with dimension before row lane, so SIMD evaluates multiple candidates together and the reader scans directly from the list payload allocation. The trailing IDs use the shared delta-varint encoding and remain aligned with code lanes. The normative byte layout and golden fixture are in the [storage-format specification](https://github.com/apache/paimon-vector-index/blob/main/core/STORAGE_FORMAT.md#ivf-sq-v1).

## Open-source comparison {#comparison}

[Faiss IVF-SQ](https://github.com/facebookresearch/faiss/blob/main/faiss/IndexScalarQuantizer.cpp) provides residual SQ4/SQ6/SQ8/F16 encodings, parallel add, and query-parallel scanning over generic inverted lists. [Milvus Knowhere](https://github.com/zilliztech/knowhere/blob/main/thirdparty/faiss/faiss/cppcontrib/knowhere/IndexIVFScalarQuantizerCC.cpp) builds on the same scanner model and adds concurrent inverted-list mutation. This implementation deliberately fixes the first immutable format at SQ8, uses per-list per-dimension residual bounds, stores compressed sorted IDs instead of fixed eight-byte IDs, and transposes each 32-row code block for its CPU SIMD kernels.

The comparison did produce two build changes: non-cosine inputs are borrowed instead of copied, and assigned lists are encoded in parallel with one reusable residual vector per worker. It did not justify changing the persisted layout. SQ4/SQ6 would overlap IVF-PQ/RQ, quantile clipping would introduce another corpus-sensitive accuracy parameter, and compressing the relatively small resident bounds would save little beside the `N × d` code payload. The existing codes-first payload already allows one bounded multi-range operation and reuse of the read allocation as the scan buffer.

## I/O and batching {#io}

Open reads the fixed header and contiguous resident metadata in two positional operations; the outer type dispatcher adds one small magic read. A query submits selected list ranges through the abstract positional-read interface in capability- and 64 MiB-bounded multi-range batches. SIFT1M and GloVe-100 use one payload round per query at `nprobe=64`; 960-dimensional GIST1M averages 1.9. Batch search first deduplicates the lists selected across queries, loads each unique list once across the bounded rounds, and then scans queries in parallel. This is especially useful when a remote adapter executes the supplied ranges concurrently.

The add path retains the caller's L2/IP slice without copying it, partitions assigned row positions by list, and encodes those lists in parallel. Each active list task reuses one `d`-component residual buffer and one code buffer; the add path never materializes an `N × d` residual matrix. The writer retains only each list's row-order permutation and encoded IDs. It generates one list's blocked codes directly from the in-memory row-major codes, writes them, and releases the temporary buffer before processing the next list.

During scanning, a candidate whose distance cannot improve a full Top-K heap is rejected before row-ID hashing. Batch remains query-parallel after a measured list-major experiment regressed SIFT/GIST throughput; list payloads are still deduplicated and read once.

IVF-SQ still reads complete selected lists. At high `nprobe`, scan bytes grow linearly; DiskANN is the better fit when the workload requires small page-granular reads from a large local-SSD index.

## Public benchmarks {#benchmarks}

On the documented Apple M4 Pro run with one million-scale public vectors, `nlist=1024`, `nprobe=64`, `k=10`, and 12 Rayon workers:

| Dataset   | Recall@10 | Build / peak RSS  | Warm P95 / batch QPS | Read/query |
|-----------|-----------|-------------------|----------------------|------------|
| SIFT1M    | 0.8627    | 3.93 s / 0.79 GiB | 0.79 ms / 11,082     | 8.38 MiB   |
| GIST1M    | 0.8577    | 22.7 s / 5.09 GiB | 3.56 ms / 1,502      | 70.95 MiB  |
| GloVe-100 | 0.8036    | 3.86 s / 0.71 GiB | 0.71 ms / 12,962     | 6.99 MiB   |

Compared with the immediately preceding implementation on the same files, peak RSS dropped by 56–61%, local P95 improved by 3–8%, and batch throughput improved by about 8–61%, depending on dimension and cache behavior. File bytes and read bytes are unchanged.
