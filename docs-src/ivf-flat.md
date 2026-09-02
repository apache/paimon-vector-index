---
title: "IVF-FLAT"
description: "Build and tune an IVF-FLAT index with exact scans inside selected lists."
---

<!--
  Licensed to the Apache Software Foundation (ASF) under one or more
  contributor license agreements. See the NOTICE file distributed with
  this work for additional information regarding copyright ownership.
  The ASF licenses this file to you under the Apache License, Version 2.0.
-->

# IVF-FLAT

Use IVF coarse partitions to reduce the candidate set, then retain and scan full `f32` vectors inside each selected list. IVF-FLAT is the clearest baseline for separating partition recall loss from quantization loss.

## Positioning and trade-offs {#position}

|                  |                                   |
|------------------|-----------------------------------|
| In-list accuracy | Exact raw-vector distance         |
| Main storage     | `4 × d × N`                       |
| Training         | IVF K-Means only                  |
| Query work       | Scan every vector in probed lists |

### Good fit

- Establishing recall and latency baselines.
- Moderate vector counts or dimensions.
- Accuracy matters more than index size.
- A clear, low-risk implementation is preferred.

### Poor fit

- High-dimensional collections with strict space budgets.
- Large lists where scans dominate tail latency.
- Very small remote-read budgets.

> **“Exact” has a boundary** — IVF-FLAT computes exact distances only inside the `nprobe` selected lists. Whenever `nprobe < nlist`, the global search remains approximate.

## Design and data flow {#design}

### Build

1.  ### Preprocess training vectors

    Cosine mode applies L2 normalization. L2 and inner product keep the input representation.

2.  ### Train coarse centroids

    K-Means produces `nlist × d` `f32` centroid values.

3.  ### Assign vectors

    Large centroid matrices use automatic Vamana assignment by default; exact mode uses the nearest coarse list. The complete processed vector and row ID enter the selected list.

4.  ### Sort and serialize

    The writer retains compact sort permutations and delta-varint IDs, then materializes and writes one sorted raw-vector list at a time instead of duplicating every list in memory.

### Search

1.  Apply the same preprocessing used during construction.
2.  Measure the query against every IVF centroid and select the closest `nprobe` lists.
3.  Read those list payloads through the offset table in bounded concurrent multi-range calls.
4.  Decode row IDs and compute the true metric against every raw vector. A Roaring filter skips disallowed IDs.
5.  For at least 1,048,576 distance components, scan independent lists on Rayon workers. Batch search keeps list-major locality and scans every loaded list for all queries that selected it.
6.  Merge list-local results in the original list order. Missing entries are padded with `-1 / f32::MAX`.

## Public-corpus measurements {#benchmarks}

Apple M4 Pro, 12 Rayon workers, one million SIFT/GIST vectors or 1,183,514 GloVe vectors, 1,000 published queries, `nlist=1024`, `nprobe=64`, Top-10, and warm APFS pages. Times are release-build measurements from 25 July 2026.

| Dataset         | Recall@10 | Local P95 | Local batch QPS | Read / query |
|-----------------|-----------|-----------|-----------------|--------------|
| SIFT1M, 128d    | 0.9937    | 1.88 ms   | 8,510           | 33.19 MiB    |
| GIST1M, 960d    | 0.9549    | 11.38 ms  | 875             | 283.40 MiB   |
| GloVe-100, 100d | 0.8832    | 1.40 ms   | 9,502           | 27.57 MiB    |

> **Latest complete same-file rerun** — The Reader receives each list directly into an `f32`-aligned allocation and scans the raw-vector suffix without allocating a second decoded payload. An internal prefix of at most three bytes compensates for the variable-length row-ID prefix; the persisted bytes do not change. Together with the strict partial-L2 cutoff, this changed SIFT/GIST/GloVe batch throughput from 6,570 / 559 / 6,345 to 8,510 / 875 / 9,502 QPS and P95 from 5.31 / 47.04 / 4.76 ms to 1.88 / 11.38 / 1.40 ms. Recall, file bytes, and read bytes stayed identical. The table reports the complete rerun rather than a best-of result.

The remote model groups every query's selected ranges into calls capped at 64 MiB. SIFT/GloVe average one round; GIST averages five because its raw-vector payload is much larger. Fixed 2/20 ms latency therefore produced P95 7.31/21.80 ms on SIFT, 29.73/130.49 ms on GIST, and 6.60/21.22 ms on GloVe. Batch throughput was 6,763/3,076 QPS on SIFT, 846/349 on GIST, and 7,394/2,948 on GloVe. These modeled numbers do not charge bandwidth, so they should not be read as evidence that transferring tens or hundreds of MiB directly from an object store is cheap.

## Usage {#usage}

The trained state is one-shot: finish training, pass the state to a Writer, add production vectors in batches, and serialize one index file. Readers discover the type from the header.

```java title="Java · build"
Map<String, String> options = new HashMap<>();
options.put("index.type", "ivf_flat");
options.put("dimension", "128");
options.put("nlist", "1024");
options.put("metric", "l2");
// options.put("ivf.coarse-assignment", "exact"); // optional; default auto

try (VectorIndexTraining training =
             VectorIndexTrainer.train(options, trainingVectors, trainingCount);
     VectorIndexWriter writer = new VectorIndexWriter(training)) {
    writer.addVectors(rowIds, vectors, vectorCount);
    writer.writeIndex(vectorIndexOutput);
}
```

```java title="Java · search"
try (VectorIndexReader reader = new VectorIndexReader(vectorIndexInput)) {
    reader.optimizeForSearch();
    VectorSearchParams params = new VectorSearchParams(10, 16);
    VectorSearchResult result = reader.search(query, params);
}
```

```rust title="Rust · configuration"
let config = VectorIndexConfig::IvfFlat {
    dimension: 128,
    nlist: 1024,
    metric: MetricType::L2,
    use_approximate_coarse_assignment: true,
};
let params = VectorSearchParams::new(10, 16);
```

## Parameters {#parameters}

| Parameter | Requirement | Purpose | Effect when increased |
|----|----|----|----|
| `dimension` | Inferred by Java/Python one-shot training; otherwise required and \> 0 | Input dimension | Linearly increases compute and vector payload |
| `nlist` | Auto from `expected-vector-count`, or explicit \> 0 | IVF partition count | Shorter average lists and a larger centroid table; automatic `nprobe` follows the resolved value |
| `metric` | Required: `l2`, `inner_product`, or `cosine` | Training, assignment, and search distance | Semantic, not inferred; it must match ground truth |
| `ivf.coarse-assignment` | `auto` by default; optional `exact` | Controls build-time list assignment | `auto` uses Vamana when `dimension × nlist ≥ 1,000,000`, trading build speed for possible low-`nprobe` recall loss and graph startup cost; `exact` disables it |
| `top_k` | Query-time, \> 0 | Requested results | Increases heap and output work |
| `nprobe` | Automatic by default; explicit 1 to `nlist` | Lists to probe | Auto accounts for K, average list size, and filter selectivity; explicit values remain available for measured overrides |

## v1 storage layout {#storage}

The file is little-endian and has no outer container. The magic constant is `IVFL / 0x4956464C`; raw file bytes appear in reverse ASCII order because the integer is serialized little-endian.

**64 B header**Type, dimension, count, flags**Coarse centroids**`nlist × d × f32`**Offset table**`nlist × 16 B`**Lists 0..N**IDs + raw vectors

### Fixed 64-byte header

| Offset | Size | Field                | Description                             |
|--------|------|----------------------|-----------------------------------------|
| 0      | 4    | `magic: u32`         | `IVFL`                                  |
| 4      | 4    | `version: u32`       | Currently 1                             |
| 8      | 4    | `dimension: i32`     | `d`                                     |
| 12     | 4    | `nlist: i32`         | IVF list count                          |
| 16     | 4    | `metric: u32`        | 0=L2, 1=IP, 2=Cosine                    |
| 20     | 8    | `total_vectors: i64` | Total vector count                      |
| 28     | 4    | `flags: u32`         | Bit 0: delta-varint IDs; required in v1 |
| 32     | 32   | Reserved             | Must be all zero                        |

### Offset table and list payloads

Each offset entry is `(offset: i64, count: i32, id_bytes_len: i32)`, or 16 bytes. The Reader validates that list counts match the header before serving a query. A non-empty list contains:

| Field | Type / size | Description |
|----|----|----|
| `base_id` | `i64` | First sorted row ID |
| `id_bytes_len` | `i32` | Encoded ID stream length |
| `id_bytes` | Variable | One unsigned LEB128 delta per ID; the first delta is zero |
| `vectors` | `count × d × f32` | Raw vectors in sorted-ID order |

## Capacity estimate {#sizing}

> **Approximate file size** — `64 + 4 × nlist × d + 16 × nlist + N × 4 × d + encoded_ids`

For `N=1,000,000` and `d=128`, vector payloads alone are about 488 MiB. Sorted delta-varint ID size depends on ID continuity and cannot be estimated as a fixed eight bytes. With `nlist=1024`, coarse centroids add about 0.5 MiB.

## Tuning order {#tuning}

1.  **Fix the metric.** Build exact ground truth with the production metric and include zero-vector edge cases for cosine.
2.  **Start automatic.** Supply the final corpus count, inspect the resolved `nlist`, and use automatic query width.
3.  **Calibrate only if needed.** Sweep explicit `nprobe` around the automatic value and record Recall@K, P95/P99, selected lists, and bytes read.
4.  **Measure batch search.** Readers submit bounded multi-range batches, which often matters more than single-query latency on object stores.
5.  **Only then compress.** Try IVF-SQ for a one-byte-per-dimension scan, IVF-PQ for stronger compression, or IVF-RQ for the strongest measured compact-IVF recall.

## Implementation boundaries {#limits}

- Index files do not contain checksums; the outer Paimon file and manifest layer provides integrity.
- Roaring64 filters are query payloads. Negative row IDs cannot match the `RoaringTreemap` domain.
- Readers reject unknown versions, non-zero reserved bytes, unknown flags, negative counts, mismatched total counts, and out-of-bounds sections.
- The search-only list payload owns an aligned `f32` allocation and reads bytes into it directly; the public list-materialization API still returns owned row IDs and vectors.
- `optimizeForSearch()` does not change files or results; for IVF-FLAT it preloads centroids and the offset table in one contiguous read. The unified Reader reuses its type-dispatch header, so open plus metadata initialization takes two read rounds.
