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

The Rust reader supports distance range search for all five families with L2,
cosine and inner product, using `DistanceBand`, `RangeSearchParams`, and CSR
`RangeSearchResult` buffers. All support single/batch queries and optional
serialized Roaring allow-lists. IVF uses an explicit positive `nprobe`; DiskANN
uses an explicit initial graph width, with its discovery semantics described below.
IVF-FLAT tests exact distances; IVF-RQ tests its one-bit or full multi-bit
estimated distances; IVF-SQ tests scalar-quantized estimates; IVF-PQ tests
floating-point ADC estimates for 4-bit and 8-bit codes, with optional residual
encoding and OPQ. Results are uncapped and unordered. Probing every list removes
the IVF coverage gap, but not the compressed families' quantization error.
The range path does not change top-K
search or the v1 storage format.

See the [range search guide](../docs/range-search.html) for membership,
validation, filtering, statistics, and the C, C++, Java, and Python bindings.

The DiskANN and Vamana code is an independent Apache-licensed implementation
based on the published algorithms and this project's existing storage
abstractions. It does not incorporate source code from Microsoft's
[MIT-licensed DiskANN repository](https://github.com/microsoft/DiskANN).
The implementation supports L2, inner-product, and cosine search with the same
lower-is-better distance semantics as the IVF indexes.

## Distance semantics

`DistanceBand::from_raw` takes internal, lower-is-better f32 scores: squared L2,
cosine distance (`1 - cos`, not clamped), or negative inner product.
`MetricType::public_distance` converts a score to its public f64 predicate value:
L2 takes the f32 square root before widening, cosine widens unchanged, and inner
product negates. Returned `raw_distances` always remain in internal units.
Both `RangeSearchResult::raw_distances()` and `QueryResult::raw_distances` borrow
those raw values without converting or copying them. A public L2 radius of 4
can return raw distance 9; an inner-product lower endpoint of 5 can return -6.
Do not compare raw results directly with public endpoint literals.

Use `DistanceBand::from_endpoints` for public f64 predicates. `Ge`/`Gt` belong on
the lower side and `Le`/`Lt` on the upper side. L2 resolves square-root rounding;
cosine searches all finite f32 values (including negative roundoff and signed
zero); inner product reverses both the side and comparison when negating.
Endpoints are never rounded to f32 first. Missing sides are structurally
unbounded, so an inclusive endpoint at `f32::MAX` does not lose that value.
Out-of-domain cosine/IP predicates become empty or unbounded bands; unrepresentable
L2 cuts retain the existing `Unsupported` response.
The `raw_lower()` and `raw_upper()` accessors expose converted half-open cuts,
not the original endpoints. `admit_raw()` accepts internal values only.

Range API callers must migrate `DistanceBand::new` to `from_raw` (or preferably
`from_endpoints`), `distances` to `raw_distances`, and raw cut/membership access
to the explicit names. This is a source API change, not a change to stored data,
returned values, or top-K APIs.

`IndexType::supports_range_search(metric)` and `reader.supports_range_search()`
report capability for all five index families without reading list payloads.
The unified `range_search`, `range_search_batch` and filtered variants use one
`RangeSearchParams` type, constructed with `RangeSearchParams::ivf(band, nprobe)`
or `RangeSearchParams::diskann(band, starting_l)`. Graph tuning uses
`with_initial_slack`, `with_range_slack`, `with_max_working_bytes` and
`with_exhaustive`; non-default graph tuning on IVF options is an error.
`VectorRangeSearchParams` remains a compatibility type alias and its `new`
constructor still selects IVF. Existing `DiskAnnRangeSearchParams` values also
convert into the common type.

All families apply a separate **64 MiB per-call result budget**, configurable with
`with_max_result_bytes(bytes)` (positive). It includes per-query result metadata,
staged-hit capacity, temporary replacement allocations during growth, and the
simultaneous final CSR arrays. Exhaustion returns an error without a partial CSR;
it never truncates hits. This changes formerly unbounded materialized queries:
raise the explicit budget or stream large outputs. Allocator bookkeeping, resident
data, input filters and copies retained by bindings or consumers are excluded.

`range_search_batch_consume(queries, count, params, batch_size, filter, consumer)`
delivers borrowed `(label, raw_distance)` chunks during scanning and returns a
`RangeSearchSummary`. Each chunk has at most `batch_size` hits from one query;
query order and hit order are unspecified. Its result budget covers the consumer
chunk, bounded producer blocks and per-query metadata, regardless of total hits. Producer
blocks transfer ownership into materialized results without duplicating their payload.
Callbacks serialize
but may run on scan workers; do not reenter the reader. Consumer errors stop the
scan and propagate. Already delivered chunks remain delivered if a later scan or
consumer fails. The consumer must process/discard chunks to keep its own memory bounded.

```rust,ignore
reader.range_search_batch_consume(
    &queries, query_count, params.with_max_result_bytes(1024 * 1024),
    1000, None, &mut |query_index, hits| {
        write_chunk(query_index, hits)?;
        Ok(())
    },
)?;
```
Options must match the reader's family; a mismatch returns `Unsupported`, even
for an empty band. No IVF probe count is reinterpreted as a graph search width.
Bad queries, mismatched metrics, non-finite endpoints,
malformed filters and zero `nprobe` are errors even for an empty band.
Non-finite consumed distances or cosine norms return `InvalidData`, not partial
results. Cosine queries use the existing normalization, including leaving zero
vectors at zero; normalization overflow is an error on this path.

## DiskANN range search

Use `RangeSearchParams::diskann(band, starting_l)` with the unified reader's
`range_search`, `range_search_batch`, or their `*_with_roaring_filter` variants.
The old `diskann_range_search*` names remain compatibility aliases, and
`supports_diskann_range_search()` still reports whether the reader is DiskANN.
Typed `DiskAnnIndexReader` methods keep their existing names.
L2, cosine and inner product, Compact/Interleaved layouts,
PQ4/PQ8 navigation and F32/F16 persisted vectors are supported. Existing IVF
parameters, Top-K APIs and storage bytes are unchanged. The common capability
query now returns true for DiskANN as well as IVF.

The independent two-stage implementation first converges a bounded graph candidate
list using persisted-vector distances, with PQ ordering the neighbor work. PQ
never discards a node before that scoring step. It then optionally expands a
growing graph frontier. `initial_slack` (default
1) is the initial fraction within the raw outer cut needed to continue; neither
the lower cut nor the allow-list participates in this decision. `range_slack`
(default 1) widens the navigation cut to `upper + (slack - 1) * abs(upper)`, also
for negative IP cuts. A zero cut is not widened. Final membership always uses
the requested half-open band and decoded persisted-vector distances, not PQ
scores. F16 therefore means the persisted F16 vectors, not their original F32
inputs. Graph/PQ discovery can miss matching rows; these are approximate results,
not an exhaustive SQL predicate evaluation. The second stage rebuilds visitation
and can traverse excluded or out-of-band nodes without returning them.

Set `exhaustive = true` to score all eligible persisted vectors. Unbounded raw
upper cuts and zero cosine queries automatically select this path. It removes
graph/PQ discovery errors at the cost of scanning; output is still uncapped.
Non-finite queries, consumed scores or cosine arithmetic fail without returning
partial output. Invalid arguments and malformed filters fail even for empty bands.

`max_working_bytes` defaults to 64 MiB per query. It bounds traversal/I/O scratch,
including normalization, but excludes resident index data, decoded input filters,
result buffers, which have the independent per-call `max_result_bytes` budget.
Exhaustion is an error, never silent
truncation. Batches execute queries serially with per-query scratch so this
allowance is not multiplied by a worker count. There is no cross-query I/O union.

Range calls use query-local window buffers and a FIFO window cache of up to 8 MiB
(included in the working budget) to avoid rereading windows during graph search.
Known neighbor-vector windows and second-stage adjacency beams are deduplicated
and read in bounded batches respecting the adapter's maximum range count.
They do not access Top-K LRU caches,
scratch, calibration or last-search counters. `result.diskann_stats()` exposes
per-query graph expansions, PQ and stored-vector evaluations, read calls/bytes,
and second-phase/exhaustive flags. IVF list counters are zero; `rows_scanned`
counts stored-vector evaluations, including graph-routing nodes excluded by a
filter. A large range operation on the same mutable reader still serializes with
ordinary searches; use separate readers/sessions and resource scheduling for
concurrent latency-sensitive traffic. Shared hardware and the OS page cache can
still experience contention.

All bindings use a single preferred parameter type with IVF/DiskANN factories,
matching the Top-K API style. Rust, C++ and Python use `RangeSearchParams` with
`ivf` / `diskann`; Java uses `VectorRangeSearchParams.ivf` / `.diskAnn`.
Automatic Top-K width calibration is not a range-search policy: range widths
remain explicit and results remain uncapped.

C++ returns `RangeSearchResult` for either mode, with an empty `diskann_stats`
vector for IVF. Construct its single `RangeSearchParams` type with `ivf` or
`diskann`; the public `search_width` and `width` fields follow Top-K naming.
C++ callers must migrate to these factories and the common query/result names;
there are no separate DiskANN types/methods or legacy tuple adapters.
Python keeps its original five-field IVF result and returns a
`DiskAnnRangeSearchResult` subclass for graph options. Existing Rust, Java and
Python DiskANN parameter constructors/method names remain compatibility adapters.

C uses the original four `paimon_vindex_reader_range_search*` names with one
`const PaimonVindexRangeSearchParams*` parameter. Construct it with
`paimon_vindex_range_search_params_ivf(band, nprobe)` or
`paimon_vindex_range_search_params_diskann(band, starting_l)`. The structure has
`struct_size`, `band`, `search_width`, `width` and graph-tuning fields. Factories
initialize every field and set the initialized prefix through `exhaustive`,
excluding tail padding. Unknown trailing fields are ignored. There is no
`with_options` entry point or separate DiskANN C parameter/search API.

This is range C ABI **2**, reported by `paimon_vindex_range_search_api_version()`.
C callers and bindings must rebuild against the matching header/native library;
old by-value range calls are incompatible. Python checks the version before
configuring signatures and asks for an upgrade when an older range ABI is loaded,
while ordinary Top-K calls remain usable. The persisted index format is unchanged.

The algorithm follows the approach in Microsoft's [range-search reference](https://github.com/microsoft/DiskANN/blob/a3e8963c204c631ad09cac155bf6785ad4e5ba8e/diskann/src/graph/search/range_search.rs),
with independently implemented paged I/O and persisted-vector membership.

## IVF-SQ and IVF-PQ range search

Use `DistanceBand` and `RangeSearchParams::ivf` with `range_search`,
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
Materialized result memory grows with the number of hits up to `max_result_bytes`;
use the consumer API for larger outputs. Cache hits are excluded from
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
worker scratch beyond that budget. Cached residual IP base tables are reused
across lists, applying each list's coarse offset to the first subtable without
changing floating-point addition order. Residual L2/cosine tables are reused
across chunks of the same list and rebuilt for a different list. Large batches
scan queries in parallel; membership does not depend on worker count, list size,
batch size or `optimize_for_search`. Non-finite transformed queries, coarse
distances (even to unselected lists), or consumed estimates return `InvalidData`
without partial results.
PQ range scores need not be bit-identical to top-K scores: the range path does
not use top-K table quantization, L2 table expansion, or its cosine score scale.

The crate ships its [normative v1 storage-format specification](STORAGE_FORMAT.md)
and byte-exact fixtures. Project documentation, language bindings, and
contribution guidance live in the
[Apache Paimon Vector Index repository](https://github.com/apache/paimon-vector-index).

## Top-K statistics

`VectorIndexReader::search_stats()` returns `SearchStats` for every index family
and all single, batch, and Roaring-filtered top-K entry points. Batch counters
include every query, progressive round, and storage worker; `initial_width` and
`effective_width` are maxima across queries. `search_width` identifies IVF nprobe
(1) or DiskANN search-list size (2). Exhaustive DiskANN has zero graph width.
`candidate_count` counts IVF rows before filtering (and repeated single-query
work), or DiskANN PQ evaluations. `strategy` records the algorithm actually run;
`fallback_reason` records insufficient-result expansion/fallback.

`read_calls`, `read_ranges`, and `read_bytes` count adapter requests, including
lazy initialization. Cache units are IVF-FLAT/SQ payload entries or DiskANN
adjacency/raw-vector windows; IVF-PQ/RQ report zero. A core top-K failure clears
the snapshot. Range search, optimization, warmup and calibration preserve it.
Existing algorithm-specific diagnostics remain available. C exposes a bounded,
appendable statistics prefix; C++, Java and Python expose the same snapshot.
