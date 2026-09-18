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

The Rust reader supports distance range search for IVF-FLAT, IVF-SQ, IVF-PQ and
IVF-RQ with L2, cosine and inner product, using `DistanceBand`,
`VectorRangeSearchParams`, and CSR `RangeSearchResult` buffers. All four families
support single and batch queries,
with or without a serialized Roaring allow-list, and a fixed positive `nprobe`.
IVF-FLAT tests exact distances; IVF-RQ tests its one-bit or full multi-bit
estimated distances; IVF-SQ tests scalar-quantized estimates; IVF-PQ tests
floating-point ADC estimates for 4-bit and 8-bit codes, with optional residual
encoding and OPQ. Results are uncapped and unordered. Probing every list removes
the IVF coverage gap, but not the compressed families' quantization error.
The range path does not change top-K
search or the v1 storage format.

See the [range search guide](../docs/range-search.html) for membership,
validation, filtering, and statistics. C/JNI range bindings are not included.

The DiskANN and Vamana code is an independent Apache-licensed implementation
based on the published algorithms and this project's existing storage
abstractions. It does not incorporate source code from Microsoft's
[MIT-licensed DiskANN repository](https://github.com/microsoft/DiskANN).
The implementation supports L2, inner-product, and cosine search with the same
lower-is-better distance semantics as the IVF indexes.

## Distance semantics

`DistanceBand::new` takes internal, lower-is-better f32 scores: squared L2,
cosine distance (`1 - cos`, not clamped), or negative inner product.
`MetricType::public_distance` converts a score to its public f64 predicate value:
L2 takes the f32 square root before widening, cosine widens unchanged, and inner
product negates. Returned `distances` always remain in internal units.

Use `DistanceBand::from_endpoints` for public f64 predicates. `Ge`/`Gt` belong on
the lower side and `Le`/`Lt` on the upper side. L2 resolves square-root rounding;
cosine searches all finite f32 values (including negative roundoff and signed
zero); inner product reverses both the side and comparison when negating.
Endpoints are never rounded to f32 first. Missing sides are structurally
unbounded, so an inclusive endpoint at `f32::MAX` does not lose that value.
Out-of-domain cosine/IP predicates become empty or unbounded bands; unrepresentable
L2 cuts retain the existing `Unsupported` response.

`IndexType::supports_range_search(metric)` and `reader.supports_range_search()`
report capability without reading list payloads. DiskANN and C/JNI range APIs
remain unsupported. Bad queries, mismatched metrics, non-finite endpoints,
malformed filters and zero `nprobe` are errors even for an empty band.
Non-finite consumed distances or cosine norms return `InvalidData`, not partial
results. Cosine queries use the existing normalization, including leaving zero
vectors at zero; normalization overflow is an error on this path.

## IVF-SQ and IVF-PQ range search

Use `DistanceBand` and `VectorRangeSearchParams` with `range_search`,
`range_search_batch`, or their `*_with_roaring_filter` variants. Bands are
half-open `[lower, upper)` in internal metric units; results use `RangeSearchResult`
CSR buffers without sorting, padding, or a top-K cap.

IVF-SQ uses the same blocked SIMD estimator as top-K, including the stored
per-list residual bounds. Raising `nprobe` visits more lists but does not remove
quantization error: even at `nprobe == nlist`, membership can differ from the
original vectors' distances at either boundary. Prefer IVF-FLAT when exact
membership is required. There is no original-vector reranking or top-K fallback.

Batch queries share list reads, reuse the existing partition cache, and evaluate
the Roaring allow-list once per list row using query-local one-bit-per-row masks.
Large lists stream in bounded chunks; scan scratch is reused, and under L2 a
finite upper cut allows entire SQ blocks to stop after their partial distances
reach that cut.
Result memory still grows with the number of hits. Cache hits are excluded from
`call_stats().list_reads()`. Cosine and IP never prune partial sums and report zero
`early_abandoned`; L2 pruning is unchanged. Filters exclude negative row IDs in
the Roaring API, while direct `RowIdFilter` implementations may admit them.

IVF-PQ sums full f32 subvector distances in subquantizer order rather than using
top-K's quantized FastScan tables.
L2 uses ADC squared distances; cosine uses half the ADC squared distance after
query normalization; IP uses negative estimated dot product. The cosine score
is a unit-vector surrogate, not a re-normalized exact distance to decoded codes.
Like IVF-RQ's cosine estimator, it can differ from exact cosine even for zero
queries. IVF-FLAT and IVF-SQ define cosine distance involving a zero vector as 1.
PQ and RQ always evaluate complete estimates, without early abandonment.
Shared lists are read once per call, oversized PQ lists stream in bounded chunks,
and the PQ batch filter is evaluated once per list row, not once per query.
Query lookup tables are allocated lazily within an 8 MiB cache cap, with reusable
worker scratch beyond that budget. Residual tables are reused across chunks of
the same list and rebuilt for a different list. Large batches scan queries in
parallel; membership does not depend on worker count, list size, batch size or
`optimize_for_search`. Non-finite transformed queries, coarse distances (even to
unselected lists), or consumed estimates return `InvalidData` without partial results.
PQ range scores need not be bit-identical to top-K scores: the range path does
not use top-K table quantization, L2 table expansion, or its cosine score scale.

The crate ships its [normative v1 storage-format specification](STORAGE_FORMAT.md)
and byte-exact fixtures. Project documentation, language bindings, and
contribution guidance live in the
[Apache Paimon Vector Index repository](https://github.com/apache/paimon-vector-index).
