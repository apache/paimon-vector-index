---
title: "IVF-RQ"
description: "Build and tune a rotated residual-quantization index for compact higher-recall search."
---

<!--
  Licensed to the Apache Software Foundation (ASF) under one or more
  contributor license agreements. See the NOTICE file distributed with
  this work for additional information regarding copyright ownership.
  The ASF licenses this file to you under the Apache License, Version 2.0.
-->

# IVF-RQ

Partition vectors with IVF, spread each residual through a deterministic orthogonal transform, and store centered scalar levels as bit planes. A cheap sign-plane estimate rejects weak candidates before the remaining planes are evaluated.

## Positioning and trade-offs {#position}

|                    |                      |
|--------------------|----------------------|
| Default code       | `padded_d / 2` bytes |
| Per-vector factors | 5 × `f32`            |
| Learned model      | IVF centroids only   |
| Dimension          | Any positive value   |

### Good fit

- You need higher recall than IVF-SQ at a smaller serialized size.
- Training time and model complexity should stay close to IVF-FLAT.
- The source supports one concurrent multi-range read for selected lists.
- Approximate in-list ranking is acceptable and measured against real ground truth.

### Poor fit

- Raw-vector or exact reranking accuracy is mandatory.
- Sub-millisecond local latency matters more than 0.90-class recall.
- The collection is highly mutable; this implementation writes immutable files.
- Resident PQ tables and lower recall are acceptable in exchange for a still smaller IVF-PQ index.

> **Why it remains IVF-RQ** — The public index family name is unchanged. The pre-release 1-bit/query-bit experiment was replaced completely: bit width is now a property of persisted data, not a per-query switch.

## Design and distance estimation {#design}

1.  ### Train and assign IVF

    Learn `nlist` coarse centroids and form `r = x − c`.

2.  ### Rotate deterministically

    Pad to a multiple of 64, then apply four rounds of random signs, 64-wide normalized FHT, and seeded permutation. The transform is orthogonal and reconstructed from the header seed.

3.  ### Quantize centered levels

    Choose 1–8 bits and refine one per-vector scale for three rounds. Store levels MSB first as bit planes.

4.  ### Estimate in two stages

    The MSB sign plane and its three factors produce an estimate plus a deterministic reconstruction-error bound. Only candidates whose lower bound can enter Top-K evaluate all planes and the full factors.

5.  ### Scan blocked data

    Codes and factors are transposed within 32-vector blocks. The rotated query and byte LUT are built once and reused across every selected list.

The full estimator is exact for a query equal to the encoded source vector, while ranking other vectors through the scaled reconstruction. L2, inner product, and cosine have metric-specific additive/rescale factors; cosine inputs are normalized before IVF assignment.

## Usage {#usage}

```java title="Java · build and query"
Map<String, String> options = new HashMap<>();
options.put("index.type", "ivf_rq");
options.put("dimension", "128");
options.put("nlist", "1024");
options.put("rq.bits", "4"); // optional; 4 is the default
options.put("metric", "l2");
// options.put("ivf.coarse-assignment", "exact"); // optional; default auto

try (VectorIndexTraining training =
             VectorIndexTrainer.train(options, trainingVectors, trainingCount);
     VectorIndexWriter writer = new VectorIndexWriter(training)) {
    writer.addVectors(rowIds, vectors, vectorCount);
    writer.writeIndex(vectorIndexOutput);
}

try (VectorIndexReader reader = new VectorIndexReader(vectorIndexInput)) {
    VectorSearchResult result =
            reader.search(query, new VectorSearchParams(10, 64));
    int storedBits = reader.metadata().rqBits();
}
```

```rust title="Rust · configuration"
let config = VectorIndexConfig::IvfRq {
    dimension: 128,
    nlist: 1024,
    bits: 4,
    metric: MetricType::L2,
    use_approximate_coarse_assignment: true,
};
let params = VectorSearchParams::new(10, 64);
```

## Parameters {#parameters}

| Parameter | Requirement / default | Purpose | Guidance |
|----|----|----|----|
| `dimension` | Inferred by Java/Python one-shot training; otherwise \> 0 | Logical vector dimension | Storage pads internally to a multiple of 64. |
| `nlist` | Auto from `expected-vector-count`, or explicit \> 0 | IVF partition count | Compare the resolved value with the same IVF-FLAT baseline. |
| `rq.bits` | 1–8; auto from `max-bytes-per-vector`, otherwise 4 | Persisted residual level width | Higher values increase recall, file bytes, I/O, and scan work linearly. |
| `metric` | Required | L2 / inner product / cosine | Semantic, not inferred; fixed in the file. |
| `ivf.coarse-assignment` | `auto` by default; optional `exact` | Controls build-time list assignment | `auto` uses Vamana when `dimension × nlist ≥ 1,000,000`, trading build speed for possible low-`nprobe` recall loss and graph startup cost; `exact` disables it. |
| `nprobe` | Automatic by default; explicit expert override | Lists probed | Auto accounts for K, average list size, and filter selectivity. |

> **No query-side bit width** — The Reader always evaluates the representation stored in the file. Changing `rq.bits` requires rebuilding the index; this keeps one file's accuracy and cost contract stable.

## Public-corpus measurements {#benchmarks}

Apple M4 Pro, 12 Rayon workers, one million base vectors, 1,000 published queries, `nlist=1024`, `nprobe=64`, Top-10, warm APFS pages. Times are release-build measurements from 25 July 2026.

| Dataset | Build | File | Recall@10 | Local P95 | Local batch QPS | Read / query |
|----|----|----|----|----|----|----|
| SIFT1M, 128d | 3.92 s | 86.3 MB | 0.9148 | 1.20 ms | 3,074 | 5.81 MB |
| GIST1M, 960d | 23.5 s | 505.8 MB | 0.9039 | 4.41 ms | 444 | 38.81 MB |
| GloVe-100, 100d | 4.03 s | 102.1 MB | 0.8203 | 1.23 ms | 2,917 | 6.18 MB |

On SIFT1M, a controlled bit sweep produced Recall@10 0.7233 / 0.8365 / 0.9149 / 0.9530 / 0.9731 for 2 / 3 / 4 / 5 / 6 bits. The compact v1 factor layout removes four unused bytes per row, reducing the corresponding files from 58.3 / 74.3 / 90.3 / 106.3 / 122.3 MB to approximately 54.3 / 70.3 / 86.3 / 102.3 / 118.3 MB without changing the distance estimator. Four bits is the first point above 0.90 and is therefore the default.

| GIST1M storage model      | Recall@10 | P95      | Batch QPS | Query rounds |
|---------------------------|-----------|----------|-----------|--------------|
| Warm local storage        | 0.9039    | 4.41 ms  | 444       | 1            |
| Remote cache, fixed 2 ms  | 0.9039    | 7.80 ms  | 434       | 1            |
| Object store, fixed 20 ms | 0.9039    | 24.34 ms | 392       | 1            |

> **What changed in this run** — Rows are assigned to IVF lists once, then independent lists are encoded in parallel with task-local residual, rotation, code, and factor scratch. The unfiltered scan removes the per-lane filter test from coarse-code accumulation, and the final estimator checks the Top-K threshold before entering the duplicate-aware heap. Relative to the immediately preceding same-machine build, add time fell 39% / 45% / 42% on SIFT/GIST/GloVe; a strict GIST same-file query A/B improved batch time by 8.7%. Rebuilt file hashes match the serial baseline exactly, so these are execution-path changes rather than a format or estimator change.

The fixed-latency rows model one concurrent multi-range round and do not model bandwidth, retries, TLS, or throttling. Batch QPS is one 1,000-query call, not independent clients.

## v1 storage layout {#storage}

**64 B header**Shape, bits, transform, layout**IVF centers**`nlist × d × f32`**Offset table**`nlist × 16 B`**Lists**IDs + blocked planes + blocked factors

The header records logical and padded dimensions, metric, required layout flags, persisted `rq.bits`, total vectors, rotation seed/rounds, plane bytes, rotation type 2, and compact factor layout 3. Every non-empty list begins with `base_id`, encoded-ID length, and code length, followed by sorted delta-varint IDs.

Within each 32-vector block, bytes are ordered by plane, byte position, then lane. For more than one bit, the five structure-of-arrays fields are coarse `(f_add, f_rescale, f_error)` followed by full `(f_add, f_rescale)`. The omitted full error was never read: the full estimate is the final ranking stage and does not produce another lower bound. Readers reject the old pre-release layouts rather than guessing their meaning.

> **Open-source cross-check** — [Faiss IVFRaBitQ FastScan](https://github.com/facebookresearch/faiss/blob/main/faiss/IndexIVFRaBitQFastScan.h) also groups database vectors in 32-lane blocks and separates RaBitQ correction factors from packed codes. [Lance's RaBitQ kernels](https://github.com/lancedb/lance/blob/main/rust/lance-index/src/vector/bq/ex_dot.rs) likewise use blocked multi-bit codes and architecture-specific dot products. The v1 plane/byte/lane layout and SoA factors already preserve the important scan locality, while sorted delta-varint row IDs avoid fixed-width ID overhead. A quantized-LUT SIMD port remains future work because its rounding error must be incorporated into the conservative first-stage bound; the current release keeps exact F32 table sums.

> **Why the remaining factors stay F32** — [Faiss additive quantizers](https://github.com/facebookresearch/faiss/wiki/Additive-quantizers) can encode norms with qint8 or qint4. IVF-RQ cannot apply that choice blindly to its coarse factors: their error term makes the first-stage lower bound conservative, and inward rounding could incorrectly prune a true Top-K candidate. v1 takes the lossless 4-byte saving by deleting only the unused final-stage error; lower-precision factor encodings remain gated on a proof-preserving rounding rule and public-corpus recall data.

> **I/O contract** — Open reads the 64-byte header once. Resident initialization reads centroids plus the offset table in one contiguous round. Search groups selected list payloads into at most 64 MiB per round and also honors `SeekReadCapabilities.max_ranges_per_pread`. Each returned payload remains the backing allocation for its blocked codes, so the reader decodes only IDs and factors instead of copying the usually much larger code region.

## Capacity estimate {#sizing}

> **Approximate default size** — `64 + 4×nlist×d + 16×nlist + N×(4×padded_d/8 + 20) + encoded_ids`

For one bit, only the two estimate factors are stored, so the factor term is 8 bytes. For 2–8 bits it is 20 bytes. The formula excludes small per-list headers and delta-varint ID variability.

## Tuning order {#tuning}

1.  Run IVF-FLAT with the target `nlist/nprobe` to establish the partition recall ceiling.
2.  Start IVF-RQ at the default four bits.
3.  If recall is low for both indexes, increase `nprobe`. If only IVF-RQ is low, try five bits before increasing I/O through more lists.
4.  Compare IVF-SQ when simpler/faster scans matter; compare IVF-PQ when minimum size matters.
5.  Validate the final choice on a public or production corpus, including batch and the real storage adapter.

## Implementation boundaries {#limits}

- The file is immutable; updates require a new index file.
- Distances are approximate and raw vectors are not retained for exact reranking.
- Four transform rounds and compact factor layout 3 are fixed for v1; readers reject other values.
- Batch scan parallelism uses Rayon while sharing each loaded list payload across queries; a single query also scans independent lists in parallel once its candidate count reaches 8,192.
- `optimize_for_search()` loads and validates resident metadata; it does not change search results.
