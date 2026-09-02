---
title: "Five indexes. One selection map."
description: "Compare Apache Paimon vector indexes by recall, latency, storage, build cost, and I/O."
---

<!--
  Licensed to the Apache Software Foundation (ASF) under one or more
  contributor license agreements. See the NOTICE file distributed with
  this work for additional information regarding copyright ownership.
  The ASF licenses this file to you under the Apache License, Version 2.0.
-->

# Five indexes. One selection map.

Four implementations use IVF to narrow the search space before scanning raw vectors or compact codes. DiskANN instead uses one global Vamana graph with resident PQ navigation and paged F16/F32 reranking. This guide compares recall, latency, storage, build cost, and object-store I/O.

**The short answer** Measure IVF-FLAT first. For compact indexes, choose IVF-SQ for batch throughput, IVF-RQ for higher recall, or IVF-PQ for the smallest files. Choose DiskANN for an immutable collection when high recall and small local-SSD reads justify a much slower build; validate its recall independently for the production metric.

## IVF's shared two-stage search {#shared-path}

For the four IVF families, `nlist` controls the number of coarse partitions and `nprobe` controls how many partitions a query reads. Only the stored representation and scan kernel change.

> **DiskANN follows a different path** — It traverses one global Vamana graph using resident PQ codes, reads adjacency pages on demand, then reranks persisted F16 or F32 vectors from either the same interleaved pages or a separate compact section. It has no IVF lists; the tagged query API uses `l_search` and rejects an IVF `nprobe` override.

1.  Preprocess vectors, including cosine normalization.
2.  Measure distance to IVF coarse centroids.
3.  Select the nearest `nprobe` lists.
4.  Scan raw vectors or compact codes.
5.  Merge candidates into the global top K.

## Usage and development {#documentation}

Index pages focus on algorithms and formats. Shared lifecycle, language bindings, filtering, benchmarks, and build commands live in dedicated guides so the same information is maintained only once.

### API and language bindings

The Trainer / Writer / Reader lifecycle with complete Rust, C, C++, Java/JNI, and Python examples.

[Open the API guide →](api.md)

### Search and filtering

Understand automatic width, explicit IVF `nprobe`, DiskANN `l_search` calibration, warm-up, and Roaring64 filters.

[Explore the search API →](api.md#params)

### Development and benchmarks

Repository modules, Rust checks, cross-language smoke tests, ANN benchmarks, and storage compatibility.

[Open the development guide →](development.md)

### DiskANN deployment and tuning

Understand the memory, local, remote, and object-store profiles, resident-memory model, build parameters, and production-readiness boundary.

[Open the DiskANN guide →](diskann.md)

### Releases and verification

Download signed source releases, create a release candidate, or independently verify one before voting.

[Open the release guides →](releases.md)

## Core differences {#comparison}

This table describes the implementation in this repository, not a general promise made by similarly named algorithms elsewhere. Storage estimates omit row IDs, model sections, offset tables, alignment, and headers unless noted.

| Index | Representation | Candidate search | Main payload per vector | Accuracy profile | Build cost | Primary controls | Best fit |
|----|----|----|----|----|----|----|----|
| [IVF-PQ](ivf-pq.md) | 8-bit PQ codes; optional OPQ | Distance-table lookup over compact codes | About `m` bytes | PQ reconstruction error; OPQ may improve uneven subspaces | Medium to high | Automatic `nlist`/`nprobe`/`pq.m`; target-based OPQ; explicit overrides | Minimum file and selected-list bytes among IVF when the measured 0.58–0.74 recall band is sufficient |
| [IVF-SQ](ivf-sq.md) | 8-bit scalar residual codes with per-list bounds | SIMD code scan in probed lists | About `d` bytes | Per-coordinate scalar quantization loss | Low | Automatic `nlist`/`nprobe`; explicit overrides | Highest measured compact batch throughput when 0.80–0.86 recall meets the target |
| [IVF-RQ](ivf-rq.md) | Multi-bit rotated residual levels + coarse/full factors | Bounded sign-plane scan, then full bit-plane refinement | Default about `padded_d/2+20` bytes | Measured 0.82–0.91 Recall@10 across GloVe-100, SIFT1M, and GIST1M at four bits | Low to medium | Automatic `nlist`/`nprobe`; budget-based bits; explicit overrides | Best measured compact recall when 3–4.5× lower batch throughput than IVF-SQ is acceptable |
| [DiskANN](diskann.md) | Global Vamana + resident PQ + persisted rerank vectors | PQ-guided graph traversal and F32/F16 rerank | `E·d + pq.m + 4(R+1)` bytes, approximately; `E=4` or `2` | Approximate candidate discovery; F32-exact or F16-quantized distances for reranked candidates | Very high | Build preset + deployment/capacity objectives; calibrated automatic `l_search` | Immutable L2 on local SSD when high recall and sub-MiB query reads repay the long build |
| [IVF-FLAT](ivf-flat.md) | Raw `f32` vectors | Exact distance scan in probed lists | About `4d` bytes | No quantization loss; recall mainly depends on `nprobe` | Low | Automatic `nlist`/`nprobe`; explicit overrides | Recall ceiling, frequent rebuilds, IP/cosine, or production sets whose scan bytes are affordable |

## Measured comparison: public SIFT1M, GIST1M, and GloVe-100 corpora {#public-corpus-check}

This repository's unified benchmark builds all five indexes over standard public vectors, searches the same independent public queries, and scores every result against published exact neighbors. The results below are the homepage's sole performance evidence.

### Benchmark setup

> **This is not the zero-configuration benchmark** — Running `cargo bench -p paimon-vindex-core --bench ann_bench` without public file paths uses a 20k-vector, 64-dimensional generated smoke workload. Reproducing the results below requires the public files, recorded IVF and DiskANN search settings, a fixed worker count, and an output directory on the storage device being measured. File shape, training count, and multi-index process isolation are automatic.

The real-data run uses the public [ANN-Benchmarks](https://github.com/erikbern/ann-benchmarks) SIFT1M, GIST1M, and GloVe-100 files, the first 1,000 independent test queries, and their published Top-100 exact neighbors. Recall@10 compares only the first ten published neighbors. SIFT and GIST contain one million base vectors with 128 and 960 dimensions. GloVe contains 1,183,514 vectors with 100 dimensions and angular ground truth; its base and query vectors are L2-normalized during conversion so the common L2 benchmark produces the same neighbor ordering as cosine. The benchmark supplies 65,536 base vectors to every trainer; DiskANN bounds PQ training memory with a deterministic reservoir of at most 50,000 vectors.

| Source | Parameters | Recorded value | Reproduction rule |
|----|----|----|----|
| Public data | `ANN_BASE_FVECS`, `ANN_QUERY_FVECS`, `ANN_GROUND_TRUTH_IVECS` | SIFT1M, GIST1M, or normalized GloVe-100 converted files | Set all three together; otherwise the benchmark generates synthetic vectors. |
| Dataset shape | `ANN_N`, `ANN_NQ`, `ANN_D` | `1,000,000 / 1,183,514`, `1,000`, `128 / 960 / 100` | Inferred from the public files. An explicit value becomes a shape assertion and fails before the full dataset is loaded if it differs. |
| Training input | `ANN_TRAIN_N` | `65,536` | Inferred as `min(N, max(65,536, 64 × nlist))`. DiskANN deterministically retains at most 50,000 of these vectors for bounded PQ training; the IVF trainers consume all 65,536. |
| IVF search | `ANN_NLIST`, `ANN_NPROBE` | `1,024`, `64` | Set explicitly; benchmark defaults are 64 and 8. |
| DiskANN search | `ANN_DISKANN_L_SEARCH` | `100` | May be omitted; this is the benchmark default and the automatic value for `k=10`. |
| Process and device isolation | `ANN_INDEXES`, `ANN_OUTPUT_DIR`, `RAYON_NUM_THREADS` | All five indexes, target APFS path, 12 threads | Public multi-index runs automatically spawn one child process per index. Set a subset only when needed; set the device path and worker count explicitly. |
| Matching defaults | `ANN_K`, `ANN_PQ_CODE_RATIO`, `ANN_RQ_BITS` | `10`, `0.0625`, `4` | May be omitted; the reproduction command pins them so a future default change cannot silently alter the comparison. |
| DiskANN build defaults | `pq.bits`, `R`, `Lbuild`, `alpha`, memory budget, layout, raw-vector encoding, build distance | `8`, `64`, `100`, `1.2`, 8 GiB, compact, F16, product-quantized | Use the benchmark defaults. Leave `ANN_PQ_M` unset so `pq.code-ratio` resolves the concrete value. |
| Reader/I/O model | Automatic read plan, Reader budget, simulated latency | Latency-derived local/remote/object-store plans, 4 GiB automatically partitioned budget, 0/2/20 ms per read round | Fixed by the current benchmark implementation; `ANN_STORAGE_CASES` selects a focused subset. DiskANN range reads use an I/O pool independent of query workers; all five indexes were refreshed after their current reader, storage, and parallel-scan work. |

> **Equal relative PQ budget** — The default `pq.code-ratio=0.0625` automatically resolves SIFT to `pq.m=32`, GIST to `pq.m=240`, and GloVe to `pq.m=25`. Every code occupies 6.25% as many bytes as its raw `f32` vector and leaves four dimensions per PQ sub-vector. The concrete value is persisted in index metadata; use explicit `pq.m` only as an expert override.

The cross-index run was recorded on 25 July 2026 using an Apple M4 Pro with 12 logical CPUs and 48 GiB RAM, a release build with Rust 1.95, real APFS files with warm operating-system pages, and the automatic 4 GiB DiskANN Reader budget. The IVF-RQ staged A/B and rebased IVF-PQ warm-local refresh were recorded on 30 July on the same host and toolchain. The reproduction command pins Rayon to 12 workers instead of relying on automatic host parallelism. The modeled serving profiles add 2 ms or 20 ms per positional-read round while executing all ranges in that round concurrently. DiskANN's benchmark adapter runs those ranges on a separate 12-worker I/O pool so a full query-worker pool cannot starve nested reads; this models the independent executor required of a production concurrent storage callback. For the 20 ms profile, open/optimization and sequential-query latency are computed as measured CPU/I/O time plus 20 ms per observed round; batch QPS retains literal delay injection so query overlap is measured. IVF multi-range calls are bounded to 64 MiB, so an all-query GIST batch uses 4 IVF-PQ, 15 IVF-SQ, or 59 IVF-FLAT payload rounds instead of submitting hundreds of MiB or several GiB as one unbounded call. Unified IVF Readers now reuse the 64-byte dispatch header, so opening and loading resident metadata takes two positional-read rounds rather than three. Each dataset's three DiskANN profile rows reuse the same built graph. Sequential and batch measurements use separately opened and optimized Readers, so the batch does not inherit query-dependent windows from the sequential sweep. Batch QPS measures one `search_batch` call over all 1,000 public queries; it is not concurrent-client QPS. See the [complete public-data command](development.md#ann).

#### Build, file, and peak process memory

| Index | SIFT build | SIFT file / RSS | GIST build | GIST file / RSS | GloVe build | GloVe file / RSS |
|----|----|----|----|----|----|----|
| IVF-PQ | 8.74 s | 0.032 / 0.88 GiB | 55.4 s | 0.230 / 5.00 GiB | 7.92 s | 0.030 / 0.85 GiB |
| IVF-SQ | 3.93 s | 0.122 / 0.79 GiB | 22.7 s | 0.907 / 5.09 GiB | 3.86 s | 0.113 / 0.71 GiB |
| IVF-RQ | 3.92 s | 0.080 / 0.71 GiB | 23.5 s | 0.471 / 4.65 GiB | 4.03 s | 0.095 / 0.68 GiB |
| DiskANN | 74.0 s | 0.361 / 1.51 GiB | 11 min 26 s | 2.089 / 7.94 GiB | 2 min 33 s | 0.396 / 1.45 GiB |
| IVF-FLAT | 4.05 s | 0.479 / 1.83 GiB | 24.9 s | 3.582 / 12.74 GiB | 4.10 s | 0.443 / 1.60 GiB |

> **IVF-SQ build and scan refresh** — The add path now borrows L2/IP input, assigns rows once, and encodes lists in parallel with one residual scratch vector per active list task instead of materializing an additional `N × d` residual matrix. In the immediately preceding same-machine run, SIFT/GIST/GloVe peak RSS was 1.81 / 12.92 / 1.60 GiB; it is now 0.79 / 5.09 / 0.71 GiB. A Top-K threshold fast path skips hash work for candidates that cannot enter the heap: local P95 is now 0.79 / 3.56 / 0.71 ms and batch throughput is 11,082 / 1,502 / 12,962 QPS. An experimental list-major batch scan was slower on SIFT/GIST and was not retained. The blocked-code format, file size, read bytes, and measured Recall@10 remain unchanged.

> **30 July IVF-PQ batch-table reuse refresh** — The rebased Reader retains the v1 zero-copy/transposed-code, ordered-list, one-byte row-ID, and first-column accumulation fast paths. For large 8-bit residual-L2 batches, the default `Auto` mode now factors each distance table into reusable per-list and per-query components when the reuse heuristic and 64 MiB working-memory guard both pass; small or unsuitable batches keep the direct path, and callers may explicitly select `On` or `Off`. Six same-file runs per mode alternated execution order. SIFT/GIST/GloVe median batch throughput changed from 4,191 / 497 / 4,366 QPS with reuse disabled to 7,899 / 950 / 8,048 QPS with `Auto`, gains of 88.5% / 91.3% / 84.3%. The `Auto` medians used below are 0.72 / 2.47 / 0.64 ms P95 and 1,583 / 467 / 1,802 sequential QPS. File bytes, query bytes, and the v1 format are unchanged. GIST and GloVe Recall@10 are unchanged at four decimals; SIFT moved from 0.7143 to 0.7142 because the stable `f64` factored-table path is numerically close but not bit-identical to direct residual-table accumulation. A removed contiguous all-query table remains distinct from this bounded factorization. Faiss FastScan's 4-bit, 32-row design remains a different accuracy/format choice.

> **Latest IVF-FLAT storage and scan review** — The v1 writer retains only sort permutations and encoded IDs, materializes one sorted raw-vector list at a time, and reproduced all three prior public files byte-for-byte. The Reader now receives list bytes directly into an `f32`-aligned allocation; an internal prefix of at most three bytes keeps the raw-vector suffix aligned despite variable-length row IDs, so search no longer allocates and decodes a second vector payload. Together with the strict partial-L2 cutoff, the complete public rerun changed SIFT/GIST/GloVe local batch throughput from 6,570 / 559 / 6,345 to 8,510 / 875 / 9,502 QPS and P95 from 5.31 / 47.04 / 4.76 ms to 1.88 / 11.38 / 1.40 ms. Recall@10, file version, file bytes, and bytes read are unchanged; these are complete-run results, not best-of measurements.

> **Pre-release storage changes produced measurable savings** — The compact IVF-RQ factor layout removes one unused F32 value per row: SIFT/GIST/GloVe files fell from 90.3 / 509.8 / 106.8 MB to 86.3 / 505.8 / 102.1 MB without changing the estimator; the refreshed Recall@10 values are 0.9148 / 0.9039 / 0.8203. Changing the balanced DiskANN default from F32 to F16 rerank vectors reduced the same three files from 0.599 / 3.877 / 0.617 GiB to 0.361 / 2.089 / 0.396 GiB. The current warm-local Recall@10 values are 0.9915 / 0.9336 / 0.8355. Timing changes also include the intervening reader/cache and graph-build work.

> **30 July IVF-RQ staged scan A/B** — The four changes below were measured against one retained v1 index per corpus with the same 1,000 queries, `nlist=1,024`, `nprobe=64`, four stored RQ bits, 12 Rayon workers, and warm APFS pages. SIFT and GloVe use seven interleaved runs. Because GIST-960 showed thermal drift during long stage sweeps, its reported wall-clock changes use five baseline/final pairs with alternating execution order. Every staged binary returned the same Recall@10. The GIST timing A/B used the public Open VDB mirror and returned 0.9037 rather than the earlier ANN-Benchmarks file's 0.9039, so the cross-index table below retains the original recall row and this section uses timing deltas only.

| IVF-RQ change | SIFT1M result | GIST1M result | GloVe-100 result |
|----|----|----|----|
| Block-aggregated scan statistics | 70.91% of batch candidates reached full bit-plane refinement | 95.88% reached bit-plane refinement, but the complete approximate-distance bound reduced exact coarse reevaluation to 1.45% | 96.03% reached full bit-plane refinement |
| Reuse IVF centroid distance | Removes 8.2 million repeated rotated centroid terms per 1,000-query run | Removes 61.4 million repeated terms; end-to-end movement stayed inside run variance because code scan dominates | Removes 8.2 million repeated padded terms per run |
| 16-entry FastScan LUT + NEON/AVX2 | Deliberately bypassed below padded dimension 256 | Versus the optimized scalar scanner: P95 −8.3%, sequential QPS +6.6%, batch QPS +17.8% in paired medians | Deliberately bypassed below padded dimension 256 |
| 32-vector single-query seed threshold | Refinement work −4.3%; P95 −0.9% versus no seed | Exact final evaluations −33.1%; paired sequential QPS +1.3% while P95 was neutral | Refinement work −0.32%; P95 −2.3% versus no seed |

The first statistics prototype incremented counters inside byte-lookup loops and regressed GIST, so it was not retained. The final implementation derives lookup counts once per block or admitted candidate, keeps per-list statistics thread-local, and merges them after parallel work. Centroid reuse derives the RQ query terms from the distances already produced by IVF probing and stores only centroid norms in the Reader. FastScan quantizes two 16-entry nibble tables, evaluates 32 rows with NEON or AVX2, then uses a conservative complete-distance interval before doing exact coarse reevaluation; the final ranking still uses all persisted RQ bit planes and does not require original vectors. Small dimensions stay on the exact scalar byte-LUT path because their public A/B did not justify SIMD setup.

| Same-file endpoint | SIFT P95 / sequential QPS / batch QPS | GIST P95 / sequential QPS / batch QPS | GloVe P95 / sequential QPS / batch QPS |
|----|----|----|----|
| Pre-change baseline | 1.101 ms / 1,020 / 2,706 | 5.599 ms / 203 / 325 | 1.071 ms / 1,041 / 2,855 |
| Final scanner | 1.081 ms / 1,042 / 2,743 | 5.190 ms / 218 / 399 | 1.046 ms / 1,057 / 2,839 |

Against its paired baseline, the final GIST scanner improved median P95 by 8.7%, sequential QPS by 8.1%, and batch QPS by 20.0%. SIFT improved P95 by 1.8%, sequential QPS by 2.1%, and batch QPS by 1.4%. GloVe improved P95 by 2.3% and sequential QPS by 1.6%; its 0.6% batch-QPS decrease is treated as noise, not an improvement. File bytes, bytes read, storage format, and compressed-domain ranking are unchanged.

> **DiskANN read-path review** — DiskANN converts F16 rerank vectors and accumulates L2 directly in one AArch64 NEON loop; its local profile coalesces 16 KiB windows. The final compact-layout batch rerank groups candidate windows with a hash table, then sorts only the unique windows into deterministic I/O order. Against the immediately preceding ordered-map control, median local batch time changed from 117 / 223 / 162 ms to 111 / 215 / 159 ms on SIFT/GIST/GloVe. Local P95 remains 1.50 / 1.83 / 1.90 ms; representative batch throughput is 9,009 / 4,651 / 6,289 QPS with a separate range-I/O executor. Remote and object-store plans remain 32 / 64 KiB. A broader Vec sort/dedup replacement for graph window planners was not retained because it regressed local SIFT/GIST batch time by 7–13%.

> **Final open-source cross-check and format decision** — [Faiss FastScan](https://github.com/facebookresearch/faiss/wiki/Fast-accumulation-of-PQ-and-AQ-codes-%28FastScan%29) still trades down to 4-bit lookup tables and a 32-row layout, so it is not a transparent replacement for the published 8-bit IVF-PQ v1. Faiss Panorama's additional level-oriented energy data was not needed to keep the existing IVF-FLAT v1 progressive cutoff. Lance's partition prefetch and transposed PQ storage match the current batched Readers; its prepared transposed L2 target is explicitly aimed at small target sets such as PQ codebooks, not large flat lists. Faiss RaBitQ's blocked multi-bit scan remains structurally aligned with IVF-RQ, while a direct-factor RQ payload experiment regressed SIFT batch throughput by 10–15% and was removed. [DiskANN3](https://github.com/microsoft/DiskANN)'s asynchronous provider, beam, and working-set model remains aligned with `SeekRead`, the latency-derived read planner, and the bounded caches. No measured result justified a v2 migration for IVF-PQ or IVF-FLAT, and no byte-layout change was retained for the pre-release IVF-SQ, IVF-RQ, or DiskANN formats.

DiskANN spends almost all build time constructing one global graph: about 18× IVF-FLAT on SIFT, 28× on GIST, and 37× on GloVe. The balanced F16 default makes its files about 42% smaller than IVF-FLAT on GIST and 11% smaller on GloVe, but they remain much larger than the compact IVF encodings because persisted rerank vectors, resident codes, and graph edges are all material. Peak RSS remains below the raw IVF writers because the DiskANN writer does not retain a second full raw-vector organization.

#### Warm local-storage result

| Index / search | SIFT Recall / P95 / batch QPS / read | GIST Recall / P95 / batch QPS / read | GloVe Recall / P95 / batch QPS / read |
|----|----|----|----|
| IVF-PQ | 0.7142 / 0.72 ms / 7,899 / 2.18 MiB | 0.7410 / 2.47 ms / 950 / 17.84 MiB | 0.5819 / 0.64 ms / 8,048 / 1.84 MiB |
| IVF-SQ | 0.8627 / 0.79 ms / 11,082 / 8.38 MiB | 0.8577 / 3.56 ms / 1,502 / 70.95 MiB | 0.8036 / 0.71 ms / 12,962 / 6.99 MiB |
| IVF-RQ | 0.9148 / 1.20 ms / 3,074 / 5.54 MiB | 0.9039 / 4.41 ms / 444 / 37.02 MiB | 0.8203 / 1.23 ms / 2,917 / 5.89 MiB |
| DiskANN | 0.9915 / 1.50 ms / 9,009 / 0.66 MiB | 0.9336 / 1.83 ms / 4,651 / 0.83 MiB | 0.8355 / 1.90 ms / 6,289 / 0.96 MiB |
| IVF-FLAT | 0.9937 / 1.88 ms / 8,510 / 33.19 MiB | 0.9549 / 11.38 ms / 875 / 283.40 MiB | 0.8832 / 1.40 ms / 9,502 / 27.57 MiB |

IVF-SQ is the compact-throughput choice: it leads the compact indexes on all three local batch runs. IVF-RQ uses smaller files and raises recall from 0.8627 / 0.8577 / 0.8036 to 0.9148 / 0.9039 / 0.8203, but batch throughput falls by about 3.4–4.4×. IVF-PQ is smaller and faster than RQ, but its 0.5819–0.7410 recall makes it a capacity-first choice rather than a default accuracy compromise. DiskANN is compelling on SIFT and especially GIST: it reads below 1 MiB per query on both, and on GIST greatly outpaces IVF-FLAT. It is not automatically best on GloVe, where IVF-FLAT has higher recall, lower P95, and higher batch throughput at the recorded settings. Treat `l_search` and `nprobe` as calibration points whenever the displayed recall misses the production gate.

#### Remote cache with 2 ms per I/O round

| Index / search | SIFT Recall / P95 / batch QPS / rounds | GIST Recall / P95 / batch QPS / rounds | GloVe Recall / P95 / batch QPS / rounds |
|----|----|----|----|
| IVF-PQ | 0.7142 / 6.16 ms / 7,282 / 1.0 | 0.7410 / 7.12 ms / 1,162 / 1.0 | 0.5819 / 5.94 ms / 8,243 / 1.0 |
| IVF-SQ | 0.8627 / 6.38 ms / 9,345 / 1.0 | 0.8577 / 11.36 ms / 1,359 / 1.9 | 0.8036 / 6.16 ms / 10,502 / 1.0 |
| IVF-RQ | 0.9148 / 7.05 ms / 2,916 / 1.0 | 0.9039 / 7.80 ms / 434 / 1.0 | 0.8203 / 6.74 ms / 3,080 / 1.0 |
| DiskANN | 0.9808 / 15.97 ms / 3,089 / 1.7 | 0.8482 / 14.22 ms / 2,175 / 1.5 | 0.8029 / 18.18 ms / 2,203 / 2.1 |
| IVF-FLAT | 0.9937 / 7.31 ms / 6,763 / 1.0 | 0.9549 / 29.73 ms / 846 / 5.0 | 0.8832 / 6.60 ms / 7,394 / 1.0 |

IVF-PQ and IVF-RQ load each query's selected compact lists in one concurrent multi-range round. IVF-SQ does the same on SIFT/GloVe; GIST's 960-dimensional payload crosses the 64 MiB per-call guard on 89% of queries and averages 1.9 rounds. IVF-FLAT also uses bounded concurrent multi-range reads: SIFT/GloVe fit in one round, while the 283 MiB GIST payload averages 5.0. IVF-RQ preserves the strongest compact-IVF recall with 7.05 / 7.80 / 6.74 ms P95. Parallel IVF-FLAT has much higher recall and competitive fixed-latency results on SIFT/GloVe, but transfers 28–33 MiB per query; this model does not charge bandwidth. DiskANN's adaptive coalescing and caches reduce the average sequential request count to 1.5–1.7 rounds on SIFT/GIST and 2.1 rounds on GloVe.

#### Object store with 20 ms per I/O round

| Index / search | SIFT Recall / P95 / batch QPS / rounds | GIST Recall / P95 / batch QPS / rounds | GloVe Recall / P95 / batch QPS / rounds |
|----|----|----|----|
| IVF-PQ | 0.7142 / 20.70 ms / 6,831 / 1.0 | 0.7410 / 22.14 ms / 1,054 / 1.0 | 0.5819 / 20.62 ms / 8,299 / 1.0 |
| IVF-SQ | 0.8627 / 20.77 ms / 6,781 / 1.0 | 0.8577 / 43.50 ms / 854 / 1.9 | 0.8036 / 20.68 ms / 7,175 / 1.0 |
| IVF-RQ | 0.9148 / 21.05 ms / 2,688 / 1.0 | 0.9039 / 24.34 ms / 392 / 1.0 | 0.8203 / 21.04 ms / 2,767 / 1.0 |
| DiskANN | 0.9808 / 60.57 ms / 537 / 1.0 | 0.8483 / 41.19 ms / 1,011 / 1.0 | 0.8033 / 80.84 ms / 421 / 1.2 |
| IVF-FLAT | 0.9937 / 21.80 ms / 3,076 / 1.0 | 0.9549 / 130.49 ms / 349 / 5.0 | 0.8832 / 21.22 ms / 2,948 / 1.0 |

At 20 ms per round, IVF-RQ is the strongest measured compact one-round option: it reaches 0.90-class recall on SIFT/GIST and 0.8203 on GloVe. IVF-SQ is faster when its lower recall is enough, and IVF-PQ is smaller when stronger quantization loss is acceptable. IVF-FLAT now looks competitive on one-round SIFT/GloVe in this fixed-latency model, but that result assumes 28–33 MiB transfers have no bandwidth cost; GIST's 283 MiB and five rounds expose the boundary. DiskANN averages about one modeled round after warmup, but dependent graph rounds remain visible in P95. A complete local SSD cache remains its preferred deployment.

> **The automatic read plan affects approximate search** — The latency-derived local tier uses graph beam 4 while remote and object-store tiers use beam 16, so the same `l_search` can return different approximate candidates; this is visible for both GIST and GloVe at `l_search=100`. The tiers use 16 KiB, 32 KiB, and 64 KiB coalescing windows respectively. Storage latency itself does not change ground truth. Compare indexes with the same latency and capability hints when isolating media effects.

> **Remote-model boundary** — The vectors and exact neighbors are public corpus data, but the 2 ms and 20 ms profiles are controlled I/O models rather than measurements from a production cache or object store. They add fixed latency without modeling bandwidth, cache misses, TLS, retries, throttling, request limits, or tail-latency variance.

## Choose by constraint {#decision}

There is no best index independent of data distribution. Narrow the field to one or two candidates, then evaluate Recall@K, P95/P99 latency, file size, build time, and object-store bytes on real queries.

> **Practical default order** — First reject indexes that cannot meet the measured recall target. Build IVF-FLAT to establish the corpus-specific IVF ceiling. If a compact representation is required, choose IVF-SQ for throughput, IVF-RQ for recall, or IVF-PQ for minimum bytes. Evaluate DiskANN separately for immutable data served from local SSD; do not select it only because the collection is large or assume L2 results transfer to another metric.

### Measured recommendation matrix

| Production constraint | Start with | Evidence from this run | Move away when |
|----|----|----|----|
| Establish a recall ceiling or debug ranking quality | [IVF-FLAT](ivf-flat.md) | Highest measured recall on all three corpora: 0.9937 / 0.9549 / 0.8832, with roughly four-second SIFT/GloVe builds. | The 28–283 MiB selected-list reads or raw-vector file size exceed the serving budget. |
| Highest compact batch throughput | [IVF-SQ](ivf-sq.md) | 11,082 / 1,502 / 12,962 local batch QPS at 0.8627 / 0.8577 / 0.8036 recall; files are about one quarter of IVF-FLAT. | The recall gate is above SQ, or one byte per dimension is still too large. |
| Strongest recall in a compact IVF file | [IVF-RQ](ivf-rq.md) | 0.9148 / 0.9039 / 0.8203 recall in files smaller than IVF-SQ, with one sequential read round per query in all three modeled profiles. | Batch throughput is the primary SLO; the four-bit scanner is 3–4.5× slower than SQ in the local run. |
| Minimum index file and compact-IVF scan bytes | [IVF-PQ](ivf-pq.md) | The smallest files—0.032 / 0.230 / 0.030 GiB—and the smallest IVF selected-list reads at 1.84–17.84 MiB, with strong batch throughput. | 0.5819–0.7410 recall is below the gate; increase the PQ budget or choose SQ/RQ instead. |
| High-recall immutable data on local SSD | [DiskANN](diskann.md), checked against IVF-FLAT for the same metric | The recorded L2-equivalent run reached 0.9915 / 0.9336 / 0.8355 recall with 0.66 / 0.83 / 0.96 MiB reads; SIFT/GIST P95 is 1.50 / 1.83 ms. | Metric-specific recall misses the gate, rebuilds are frequent, the file is not locally cached, preview maturity is unacceptable, or the corpus behaves like GloVe at `l_search=100`. |
| Frequent rebuilds or rapidly changing snapshots | [IVF-FLAT](ivf-flat.md), [IVF-SQ](ivf-sq.md), or [IVF-RQ](ivf-rq.md) | These build in about 4 seconds on SIFT/GloVe and 23–25 seconds on GIST; IVF-PQ is about 2× slower and DiskANN is 18–37× slower than IVF-FLAT. | The serving phase dominates lifetime cost enough to justify PQ training or graph construction. |
| Direct 2/20 ms remote or object-store reads | Compact IVF selected by recall: PQ → SQ → RQ | PQ and RQ use one sequential multi-range round here; SQ does so on SIFT/GloVe and averages 1.9 rounds on GIST. Choose successively more recall at greater bytes or CPU cost. | Bandwidth, request limits, or real tail latency invalidate the fixed-latency model; prefer a complete local SSD cache and rerun the benchmark. |
| Inner product or cosine | IVF-FLAT as the recall control; DiskANN as an additional candidate for immutable local-SSD serving | All five implementations support L2, IP, and cosine. DiskANN normalizes cosine internally and uses metric-aware graph construction and exact reranking, but the displayed public-corpus matrix was recorded through the L2-equivalent benchmark path. | The selected configuration misses its metric-specific recall gate—retune `nprobe`, representation width, OPQ, or `l_search` before deployment. |

> **A displayed winner can still be the wrong choice** — These recommendations apply to the recorded `nlist=1024`, `nprobe=64`, PQ ratio, RQ bits, and `l_search=100`. For example, the current GloVe run does not reach 0.90 recall with any index, and current GIST reaches 0.95 only with IVF-FLAT. If a required recall threshold is not present in the table, tune and rebuild rather than choosing the closest result.

### I need a trustworthy baseline

Start with IVF-FLAT. It exposes the IVF partition ceiling without quantization loss and rebuilds quickly.

[Explore IVF-FLAT →](ivf-flat.md)

### I need compact high recall

Choose IVF-RQ when its 0.82–0.91 measured recall matters more than batch throughput; compare every result with the IVF-FLAT ceiling.

[Explore IVF-RQ →](ivf-rq.md)

### I need the smallest index

Choose IVF-PQ when its corpus-specific recall passes the gate. It is the capacity-first option, not the automatic middle ground.

[Explore IVF-PQ →](ivf-pq.md)

### I need compact batch speed

Choose IVF-SQ when one byte per dimension fits and its measured 0.80–0.86 recall is enough; it is the fastest compact scanner here.

[Explore IVF-SQ →](ivf-sq.md)

### Raw vectors exceed RAM but fit local SSD

Evaluate DiskANN for immutable L2, IP, or cosine data when high recall and sub-MiB query reads justify a much slower build; retain IVF-FLAT as the metric-specific accuracy control.

[Explore DiskANN →](diskann.md)

### Data lives in S3, OSS, or HDFS

Prefer durable publication plus a complete local SSD cache. For direct remote reads, start with a compact IVF index when one-round scans meet recall; use DiskANN only after measuring its corpus-dependent coalesced graph rounds.

[Compare deployment modes →](diskann.md#deployment)

## How parameters interact {#parameters}

Build parameters define static structures; query parameters define per-request work. For IVF, changing `nlist` usually changes the useful `nprobe` range. DiskANN instead couples graph build quality with online `l_search`.

> **Automate numeric work, keep semantics explicit** — `index.type` and `metric` remain required because changing either changes persistence and result meaning. Rust callers can use `recommend_index` as an advisory starting point and must explicitly accept its result. For measured offline sweeps, `select_calibrated_candidate` chooses the smallest candidate satisfying supplied recall, byte, and build-time objectives and returns no result when the sample does not meet them.

| Parameter | Stage | Indexes | Typical effect when increased | Constraint / default |
|----|----|----|----|----|
| `index.type` | Build | All | Changes the persisted algorithm and its capability boundary | Required; recommendation is advisory, never silently applied |
| `metric` | Build | All | Changes training, ranking, and ground-truth semantics | Required and never inferred |
| `dimension` | Build | All | Changes representation width and distance work | Inferred by Java/Python one-shot training; required by streaming APIs |
| `nlist` | Build | IVF families | Shorter lists and more coarse centroids; a fixed `nprobe` covers less of the collection | Auto: nearest power of two around √N, with at least 64 rows of training density per list; requires `expected-vector-count` |
| `nprobe` | Query | IVF families | Reads more lists; recall usually rises with latency and I/O | Auto is K-, N-, nlist-, and filter-selectivity-aware; explicit values are expert overrides |
| `pq.code-ratio` | Build | IVF-PQ, DiskANN | Raises or lowers the automatically inferred code bytes and subquantizer count | Default 0.0625; finite and positive |
| `pq.m` | Build | IVF-PQ, DiskANN | Expert override for the inferred subquantizer count; larger values often reduce quantization error but add lookup work | Optional; `d % m == 0` |
| `rq.bits` | Build | IVF-RQ | More persisted bit planes improve reconstruction and usually recall while increasing file bytes, I/O, and scan work | Auto from `max-bytes-per-vector`; otherwise `4` |
| `use-opq` | Build | IVF-PQ | Adds training and matrix cost; may improve PQ quality | Auto enables at `target-recall ≥ 0.9`; explicit true/false wins |
| `target-recall` | Build objective | IVF-PQ, DiskANN | Selects OPQ and a coherent DiskANN build preset | Starting policy only; validate measured recall on held-out queries |
| `max-bytes-per-vector` | Build objective and preflight bound | IVF-PQ, IVF-RQ, DiskANN | Reduces code width and may select 4-bit/F16 DiskANN storage; rejects configurations whose conservative persisted-size estimate exceeds the bound | Includes estimated row bytes and, when `expected-vector-count` is set, amortized fixed data; not an exact final-file-size promise |
| `max-build-seconds` | Offline calibration objective | Measured candidate sets | Rejects candidates whose measured build time exceeds the target | Accepted by `VectorIndexBuildPlan`; direct Trainer creation rejects it because build time cannot be safely guessed from hardware |
| `diskann.build-preset` | Build | DiskANN | Moves together across degree, construction width, encoding, and build distance | `fast_build`, `balanced`, or `high_recall`; inferred from target recall |
| `deployment-profile` | Build | DiskANN | Selects interleaved layout for eligible memory/local serving and compact layout for remote/object serving | Explicit layout/encoding/build-distance overrides always win |
| `estimated_random_read_latency_nanos` | Reader input capability | DiskANN | Selects the internal read window, graph beam, and automatic cache partition without probe I/O | 0 measures the mandatory header read; positive values are useful for known remote/cache latency |
| `l_search` | Query | DiskANN | Larger DiskANN candidate list, usually higher recall and latency | Auto uses calibrated 100/200/400 when available, otherwise `max(100, 2k)` |
| `memory_budget_bytes` | Reader | DiskANN | Controls required resident state plus automatically partitioned adjacency/raw-vector caches | 4 GiB; cache sub-budgets are internal |

## Data-lake storage and I/O {#io}

IVF files begin with a 64-byte v1 header and use model/list sections. DiskANN uses a 256-byte header, page-aligned resident/adjacency data, and either densely packed compact vector records or interleaved page-contained records. The Reader dispatches on the first four-byte magic and uses positional reads for both layouts.

|                |                                             |
|----------------|---------------------------------------------|
| Byte order     | Little-endian                               |
| Type discovery | First 4-byte magic                          |
| Row IDs        | IVF delta varints / DiskANN adaptive packed |
| Integrity      | Outer Paimon file layer                     |

> **Format boundary** — v1 files have no footer, checksum, compression envelope, or schema registry. Roaring filters are query payloads and are not embedded. Readers reject unknown versions, required flags, non-zero reserved bytes, and malformed sections.

## A practical evaluation order {#evaluation}

Fix the dataset and query set, then introduce approximation one layer at a time. This makes it possible to attribute loss to IVF selection, vector quantization, or graph traversal.

1.  ### Build ground truth

    Generate exact top K using the production metric, realistic filters, and edge cases such as zero vectors.

2.  ### Measure IVF-FLAT

    Establish the recall ceiling caused by probing only `nprobe` lists and record bytes read.

3.  ### Compare compression

    Use the same `nlist/nprobe` for IVF-SQ, IVF-PQ, and IVF-RQ to isolate added approximation and storage savings.

4.  ### Evaluate disk-backed search

    When raw data exceeds RAM, compare DiskANN on local SSD and realistic remote storage, including cold/warm caches and read rounds.

5.  ### Apply production budgets

    Set thresholds for Recall@K, P99, file size, build time, RSS, and remote bytes rather than optimizing average latency alone.
