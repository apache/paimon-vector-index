---
title: "IVF-PQ"
description: "Build and tune a compact IVF-PQ index with optional OPQ rotation."
---

<!--
  Licensed to the Apache Software Foundation (ASF) under one or more
  contributor license agreements. See the NOTICE file distributed with
  this work for additional information regarding copyright ownership.
  The ASF licenses this file to you under the Apache License, Version 2.0.
-->

# IVF-PQ

Split each `d`-dimensional vector into `m` subspaces and store only one centroid ID per subspace. The unified configuration currently uses 8-bit codes, reducing the main vector payload from `4d` bytes to about `m` bytes.

## Positioning and trade-offs {#position}

|                |                            |
|----------------|----------------------------|
| Vector payload | `m` bytes                  |
| Sub-codebook   | 256 centroids per subspace |
| Search         | ADC distance lookup        |
| Extra training | PQ; optional OPQ           |

### Good fit

- Million-scale or larger collections where raw vectors are too large.
- A mature and tunable accuracy–space trade-off.
- Cache locality and scan throughput matter more than exact distances.
- Long-lived Readers serve many repeated queries.

### Poor fit

- Quantization-induced rank changes are unacceptable.
- The dimension has no useful divisor for `m`.
- Training samples are sparse or the distribution changes frequently.
- The simplest possible build pipeline is required.

> **Current public configuration** — The unified `VectorIndexConfig::IvfPq` and options API always create 8-bit PQ. The core and v1 format contain a 4-bit path, but no public `pq.nbits` option currently exposes it.

## Design and data flow {#design}

### Product quantization

A vector is divided into `m` contiguous subvectors of dimension `dsub = d / m`. Each subspace trains 256 centroids. Encoding stores the nearest centroid as one byte. Search does not fully reconstruct vectors: it builds a 256-entry distance table for each subspace and sums `m` lookups. This is asymmetric distance computation (ADC).

### Training and writing

1.  ### Preprocess and optionally rotate

    Cosine normalizes first. With OPQ, training and production vectors are projected into the learned rotation space.

2.  ### Train IVF centroids

    Learn `nlist` coarse centroids in the final effective space.

3.  ### Train PQ codebooks

    L2 trains on vector-minus-centroid residuals; IP and cosine currently use non-residual encoding.

4.  ### Encode and distribute

    Assign lists, compute residuals where applicable, and produce `m` code bytes per vector.

5.  ### Transpose on disk

    List codes are written as `codes[sub][vector]` for contiguous scans.

### Search

1.  Normalize and apply the same OPQ rotation, then choose `nprobe` IVF lists.
2.  Build an ADC table for each probed list. L2 residual mode can reuse precomputed terms.
3.  Sum code lookups without loading raw `f32` vectors.
4.  Apply the optional Roaring row-ID filter and merge global top K.

## What OPQ changes {#opq}

Standard PQ splits consecutive dimensions. When information is unevenly distributed, some sub-codebooks become crowded while others are underused. OPQ learns a `d × d` orthogonal rotation that redistributes information before PQ while preserving Euclidean geometry.

### Potential benefit

- Lower reconstruction error for the same `m`.
- Most useful for correlated data with uneven energy across dimensions.

### Definite cost

- Slower training and a matrix projection during search.
- An additional `4d²`-byte rotation matrix in the file.
- Benefit is data-dependent and must be measured.

## Usage {#usage}

```java title="Java · build"
Map<String, String> options = new HashMap<>();
options.put("index.type", "ivf_pq");
options.put("dimension", "128");
options.put("nlist", "1024");
options.put("metric", "l2");
options.put("pq.code-ratio", "0.0625"); // optional; default 0.0625
options.put("use-opq", "false"); // optional; default false
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
    // Builds the reusable table for residual-L2 PQ.
    reader.optimizeForSearch();
    VectorSearchParams params = new VectorSearchParams(10, 16);
    VectorSearchResult result = reader.search(query, params);
}
```

```rust title="Rust · configuration"
let config = VectorIndexConfig::ivf_pq(
    128,
    1024,
    MetricType::L2,
    false,
)?;
```

The default relative code budget resolves this 128-dimensional 8-bit index to `m=32`. Option maps can change the target with `pq.code-ratio` or set `pq.m` explicitly. The resolved concrete `m`, rather than the ratio, is stored in the file.

## Parameters {#parameters}

| Parameter | Requirement / default | Purpose | Tuning meaning |
|----|----|----|----|
| `dimension` | Inferred by Java/Python one-shot training; otherwise \> 0 | Input dimension `d` | The inferred or explicit `pq.m` must divide it |
| `nlist` | Auto from `expected-vector-count`, or explicit \> 0 | IVF partition count | Controls list length, centroid cost, and automatic `nprobe` |
| `pq.code-ratio` | Default 0.0625; finite and in `(0, 0.25]` | Target ratio between PQ-code bytes and raw `f32`-vector bytes | The closest valid divisor is selected; `max-bytes-per-vector` can supply the target instead |
| `pq.m` | Optional expert override; \> 0; `d % m == 0` | Concrete subspace count and 8-bit code bytes | Takes precedence over automatic sizing; use after representative recall measurements |
| `use-opq` | Auto enables when `target-recall ≥ 0.9`; explicit true/false wins | Learn and apply an orthogonal rotation | Trades training and query work for possible recall improvement |
| `ivf.coarse-assignment` | `auto` by default; optional `exact` | Controls build-time list assignment | `auto` uses Vamana when `dimension × nlist ≥ 1,000,000`, trading build speed for possible low-`nprobe` recall loss and graph startup cost; `exact` disables it |
| `nprobe` | Automatic by default; explicit expert override | Lists read | Auto accounts for K, average list size, and filter selectivity |

## v1 storage layout {#storage}

**64 B header**Dimension, m, ksub, flags**OPQ (optional)**`d × d × f32`**IVF centers**`nlist × d × f32`**PQ centers**`m × 256 × dsub × f32`**Offset table**`nlist × 16 B`**Lists**IDs + transposed codes

### Fixed 64-byte header

| Offset | Size | Field           | Description                           |
|--------|------|-----------------|---------------------------------------|
| 0      | 4    | `magic`         | `IVPQ / 0x49565051`                   |
| 4      | 4    | `version`       | 1                                     |
| 8      | 4    | `dimension`     | `d`                                   |
| 12     | 4    | `nlist`         | List count                            |
| 16     | 4    | `m`             | Subquantizer count                    |
| 20     | 4    | `ksub`          | Centroids per subspace; 256 for 8-bit |
| 24     | 4    | `dsub`          | `d / m`                               |
| 28     | 4    | `metric`        | 0=L2, 1=IP, 2=Cosine                  |
| 32     | 8    | `total_vectors` | Total vector count                    |
| 40     | 4    | `flags`         | See below                             |
| 44     | 20   | Reserved        | Must be zero                          |

Flags: bit 0 means OPQ is present; bit 1 means residual PQ; bit 2 means delta-varint IDs (required in v1); bit 3 means transposed codes (required in v1).

### List payload

An offset entry is `(offset: i64, count: i32, id_bytes_len: i32)`. A non-empty list stores `base_id: i64`, `id_bytes_len: i32`, the ID stream, then transposed codes. The 8-bit layout is `codes[sub][vector]`. The format also defines a 4-bit layout with two subquantizers per byte.

## I/O and batching {#io}

The Reader loads the fixed header in one positional read and the contiguous OPQ matrix, IVF centers, PQ codebook, and offset table in one further read. The outer type dispatcher adds one small magic read. A query then supplies all selected non-empty list ranges in one capability-bounded multi-range round instead of issuing one request per field or list.

Batch search deduplicates the union of probed lists, sorts them by persisted offset, reads every unique list once, and scans queries in parallel. A single query uses the same physical-order read plan, which avoids backward seeks for a serial fallback `SeekRead`. Each query reuses its distance-table and candidate-distance scratch across lists. Internally, the Reader receives list bytes directly into a 16-byte-aligned typed allocation and exposes the persisted code suffix as a borrowed slice, avoiding a second code allocation; public list materialization continues to return owned arrays. The common one-byte row-ID delta has a checked fast path, and an 8-bit scan initializes distances from the first code column instead of zero-filling the scratch in a separate pass. The persisted `codes[sub][vector]` layout already matches this table-driven scan, so the improvements remain compatible with IVF-PQ v1.

> **Open-source and same-file cross-check** — Conventional [Faiss IVF-PQ](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes) uses 8-bit subquantizers by default, prefetches selected inverted lists, and offers a list-major large-batch path. [FastScan](https://github.com/facebookresearch/faiss/wiki/Fast-accumulation-of-PQ-and-AQ-codes-%28FastScan%29) obtains register-resident tables by changing to 4-bit codes and 32-row blocks. [Lance PQ storage](https://github.com/lance-format/lance/blob/main/rust/lance-index/src/vector/pq/storage.rs) loads partition ranges and transposes code columns. The current v1 format and physical-order multi-range reads already retain the relevant 8-bit choices. Faiss-style list-major query bucketing, parallel ID decode, one contiguous all-query table, and worker-level scratch retention were tested and removed because each regressed at least one of SIFT1M, GIST1M, and GloVe-100. In three-run same-file medians, the retained one-byte ID fast path changed batch throughput from 7,528 / 987 / 8,693 to 7,649 / 1,026 / 8,943 QPS. The final complete warm-local rerun reached 8,353 / 1,094 / 9,330 QPS at unchanged recall and bytes read. No v2 migration is justified.

The writer keeps only each list's row-order permutation and encoded IDs, performs a 32×32 cache-blocked transpose, and reuses one list-sized output buffer across all lists. It never retains an index-sized sorted-code copy. Same-machine SIFT/GIST serialization changed from 47 / 221 ms to 36 / 150 ms, and the reference transpose test plus v1 golden fixtures verify unchanged output bytes.

## Capacity and memory estimates {#sizing}

> **Approximate 8-bit size without OPQ** — `64 + 4×nlist×d + 4×256×d + 16×nlist + N×m + encoded_ids`

Add `4d²` bytes when OPQ is enabled. For `N=1,000,000, d=128, m=16, nlist=1024`, code payloads are about 15.3 MiB, versus about 488 MiB for IVF-FLAT raw vectors, before IDs and model sections.

`optimize_for_search` creates an in-memory residual-L2 table of about `nlist × m × 256 × 4` bytes. The example above uses about 16 MiB. This table is not written back to the file.

## Tuning order {#tuning}

1.  Measure IVF-FLAT with identical `nlist/nprobe` to establish the no-quantization baseline.
2.  Start with `pq.code-ratio=0.0625`, then sweep nearby ratios; compare resolved `m`, file size, Recall@K, and P99 together.
3.  With the PQ budget fixed, sweep `nprobe` to separate coarse-partition misses from PQ rank error.
4.  Test OPQ only when PQ loss is visible and the training budget allows it.
5.  Call `optimizeForSearch()` once for long-lived Readers so precomputation is outside the hot path.

## Implementation boundaries {#limits}

- The public options API has no `pq.nbits`; unified construction creates 8-bit PQ.
- L2 uses residual PQ. Inner product and cosine currently use `by_residual=false`.
- The OPQ rotation, IVF centroids, and PQ codebooks are all stored in the file; Readers need no external model.
- Training samples should represent production data. Distribution drift can degrade both IVF assignment and PQ codebooks.
