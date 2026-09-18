<!--
  ~ Licensed to the Apache Software Foundation (ASF) under one
  ~ or more contributor license agreements.  See the NOTICE file
  ~ distributed with this work for additional information
  ~ regarding copyright ownership.  The ASF licenses this file
  ~ to you under the Apache License, Version 2.0 (the
  ~ "License"); you may not use this file except in compliance
  ~ with the License.  You may obtain a copy of the License at
  ~
  ~   http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing,
  ~ software distributed under the License is distributed on an
  ~ "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  ~ KIND, either express or implied.  See the License for the
  ~ specific language governing permissions and limitations
  ~ under the License.
-->

# Apache Paimon Vector Index Core

`paimon-vindex-core` contains the Rust implementations and seek-based readers
for IVF-FLAT, IVF-SQ, IVF-PQ, IVF-RQ, and DiskANN.

The Rust reader supports distance range search for IVF-FLAT, IVF-RQ, IVF-SQ, and IVF-PQ with
squared L2, using `DistanceBand`, `VectorRangeSearchParams`, and CSR
`RangeSearchResult` buffers. All four families support single and batch queries,
with or without a serialized Roaring allow-list, and a fixed positive `nprobe`.
IVF-FLAT tests exact distances; IVF-RQ tests its one-bit or full multi-bit
estimated distances; IVF-SQ tests scalar-quantized estimates; IVF-PQ tests
floating-point ADC estimates for 4-bit and
8-bit codes, with optional residual encoding and OPQ. Results are uncapped and
unordered. Probing every list removes the IVF coverage gap, but not the
compressed families' quantization error. The range path does not change top-K
search or the v1 storage format.

PQ range uses direct squared-L2 subvector lookup tables and sums their selected
entries in subquantizer order. It does not use top-K's u8 FastScan tables or
precomputed norm identities; membership is independent of list size, batch
size, and `optimize_for_search`. Finite estimates can therefore differ from
top-K's distances. Every filter-eligible row is fully evaluated, with no early
abandonment. Non-finite consumed estimates, rotated queries, or coarse distances
return `InvalidData`, including overflow and distances to unselected centroids.
Unique non-empty lists are read once per call, with oversized lists streamed
through the existing bounded reader. DiskANN range remains unsupported.

See the [range search guide](../docs/range-search.html) for membership,
validation, filtering, and statistics. C/JNI range bindings are not included.

The DiskANN and Vamana code is an independent Apache-licensed implementation
based on the published algorithms and this project's existing storage
abstractions. It does not incorporate source code from Microsoft's
[MIT-licensed DiskANN repository](https://github.com/microsoft/DiskANN).
The implementation supports L2, inner-product, and cosine search with the same
lower-is-better distance semantics as the IVF indexes.

## IVF-SQ range search

Use `DistanceBand` and `VectorRangeSearchParams` with `range_search`,
`range_search_batch`, or their `*_with_roaring_filter` variants. Bands are
half-open `[lower, upper)` in squared-L2 units; results use `RangeSearchResult`
CSR buffers without sorting, padding, or a top-K cap.

IVF-SQ uses the same blocked SIMD estimator as top-K, including the stored
per-list residual bounds. Raising `nprobe` visits more lists but does not remove
quantization error: even at `nprobe == nlist`, membership can differ from the
original vectors' distances at either boundary. Prefer IVF-FLAT when exact
membership is required. There is no original-vector reranking or top-K fallback.

Batch queries share list reads, reuse the existing partition cache, and evaluate
the Roaring allow-list once per list row using query-local one-bit-per-row masks.
Large lists stream in bounded chunks; scan scratch is reused, and a finite upper cut
allows entire SQ blocks to stop after their partial distances reach that cut.
Result memory still grows with the number of hits. Cache hits are excluded from
`call_stats().list_reads()`. Range support does not extend to other metrics,
DiskANN, or language bindings.

The crate ships its [normative v1 storage-format specification](STORAGE_FORMAT.md)
and byte-exact fixtures. Project documentation, language bindings, and
contribution guidance live in the
[Apache Paimon Vector Index repository](https://github.com/apache/paimon-vector-index).
