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

# Python vector search

Top-K `search` and `search_batch` accept the same `SearchParams`, including an
optional initial filter expansion cap for automatic IVF search:

```python
from paimon_vindex import SearchParams

params = SearchParams.automatic(top_k=10, max_initial_filter_expansion_factor=4)
ids, distances = reader.search(query, params, filter_bytes=roaring_allowlist)
stats = reader.search_stats()
print(stats.initial_width, stats.effective_width, stats.candidate_count)
print(stats.strategy, stats.fallback_reason, stats.read_calls, stats.read_bytes)
```

The cap is a positive integer or `None` (uncapped). It applies to the initial
filter-selectivity expansion; progressive search can exceed it when results are
short. Explicit IVF/DiskANN widths reject the cap. Smaller caps may reduce recall.
The same parameters work with `search_batch(queries, params, filter_bytes=...)`.

`SearchStats` is an immutable snapshot of the last core top-K call. A batch sums
all queries and retries; widths are batch maxima. IVF candidates count list rows
before filtering, including repeated single-query work; DiskANN candidates count
PQ evaluations. `read_calls`, `read_ranges`, and `read_bytes` count requests to the
input adapter, including lazy loading and parallel workers. `cache_hits` and
`cache_misses` count IVF-FLAT/SQ payload entries or DiskANN adjacency/raw-vector
windows; IVF-PQ/RQ report zero. A failed core top-K call clears the snapshot;
range search, warmup, optimization and calibration preserve it. Read the snapshot
before another thread searches through the same reader. Older native libraries
keep existing top-K behavior; unavailable cap/statistics features raise an upgrade
error when requested.

## Distance range search

`VectorIndexReader.range_search` and `range_search_batch` add variable-length
results alongside the unchanged top-K `search` and `search_batch` APIs. IVF-Flat,
IVF-PQ, IVF-RQ, and IVF-SQ support L2, inner product, and cosine. Query
`reader.supports_range_search()` for any index family, including DiskANN.
Use one `RangeSearchParams` type, created by `.ivf(band, nprobe)` or
`.diskann(band, starting_l, ...)`, with the same `range_search` and
`range_search_batch` entry points. Parameters must
match the index family; IVF tuning is never reinterpreted as graph tuning.
With an older native library lacking range exports, existing imports and top-K
calls still work: capability returns `False`, and range calls/endpoint conversion
raise a clear error asking for a native-library upgrade. A partially available
range ABI is treated as unavailable rather than risking an unfreeable result.

```python
from paimon_vindex import (
    DistanceBand,
    DistanceEndpoint,
    DistanceEndpointOp,
    RangeSearchParams,
)

band = DistanceBand.from_endpoints(
    "l2", upper=DistanceEndpoint(4.0, DistanceEndpointOp.LE)
)
result = reader.range_search_batch(queries, RangeSearchParams.ivf(band, nprobe=8))
query_result = result.query(0)
labels = query_result.labels
raw_distances = query_result.raw_distances
stats = result.stats[0]
```

## DiskANN distance range search

```python
from paimon_vindex import RangeSearchParams, DistanceBand

if reader.supports_range_search():  # reader holds a DiskANN index
    band = DistanceBand.from_raw("l2", raw_upper=4.0)  # squared L2 < 4
    result = reader.range_search_batch(
        queries, RangeSearchParams.diskann(band, starting_l=64))
    labels, raw_distances = result.query(0)
    graph_stats = result.diskann_stats[0]
```

Single/batch methods both accept `roaring_filter=`. All three metrics, both disk
layouts, PQ4/PQ8 and persisted F32/F16 vectors are supported. Graph search is
approximate; final membership uses persisted-vector distances. Set `exhaustive=True`
to remove graph/PQ discovery errors; unbounded upper bands automatically scan.
`initial_slack` and `range_slack` default to 1. The latter widens only navigation
to `upper + (slack - 1) * abs(upper)`, never the result band. Filtering and the
lower cut do not block navigation. Output remains uncapped and unordered.

`max_working_bytes` defaults to 64 MiB per query, excluding resident index data,
decoded filters, result metadata and uncapped output. Exceeding it raises an error.
Batch queries execute serially. Result arrays and immutable `diskann_stats` remain
valid after the reader closes. Range calls do not write Top-K caches or counters;
the same Python reader still serializes native calls. Use separate readers and
appropriate resource scheduling for concurrent traffic. The original
`diskann_range_search*` methods remain compatibility aliases, and
`supports_diskann_range_search()` still identifies DiskANN range capability.
The old `RangeSearchParams(band, nprobe)` and `DiskAnnRangeSearchParams(...)`
constructors remain compatibility adapters. Range search requires native C ABI 2;
the package checks `paimon_vindex_range_search_api_version()` before configuring
ctypes signatures. Older native libraries keep supporting Top-K and raise an
upgrade error for range calls. Rebuild/update the native library and bindings
together when migrating from the old by-value C range ABI.

Calls with DiskANN options return `DiskAnnRangeSearchResult`, a `RangeSearchResult` subclass
with graph diagnostics. The original five-field IVF result and its dataclass
tuple/dictionary serialization shape remain unchanged.

See the [core contract](../core/README.md#diskann-range-search) for normalization,
F16 semantics, input validation and graph-discovery limitations.

## Distance contracts

- Use `DistanceBand.from_endpoints` by default. It accepts optional
  `DistanceEndpoint(value, op)`
  objects: lower uses `GE`/`GT`, upper uses `LE`/`LT`. Endpoint values are public
  distances (L2 square root, inner product, or cosine distance). Double literals
  pass directly to core, which handles rounding, operator strictness, and inner
  product direction reversal. Python performs no endpoint conversion math.
- Advanced callers can use
  `DistanceBand.from_raw(metric, raw_lower=None, raw_upper=None)` for an explicitly
  **raw-distance** interval `[raw_lower, raw_upper)`. Metric names are `l2`,
  `inner_product`, and `cosine`. Raw distances are squared L2, negative inner
  product, and cosine distance, respectively. Both factories produce immutable
  bands with fields `metric`, `raw_lower`, and `raw_upper`.
- `None` means structurally unbounded, not a finite sentinel. Equal finite raw
  cuts describe a valid empty band. Core validates finite, ordered,
  metric-compatible raw cuts during search; invalid bands raise `RuntimeError`.
- Plain `DistanceBand(...)` construction raises `TypeError` directing callers to
  the factories. There are no ambiguous `lower`/`upper` band fields or legacy
  result `.distances` aliases.
- Results expose `raw_distances`, even when the band uses public endpoints.
  An L2 radius of 4 can return a vector at distance 3 with raw distance **9**;
  an inner-product lower endpoint `GE 5` can return a product of 6 with raw
  distance **-6**. Cosine results are cosine distances, not similarities.
  Quantized IVF variants return their own raw distance estimates, not an
  additional exact-vector reranking. Top-K return values are unchanged.

## Queries, filters, and results

Single queries have shape `(dimension,)`; batches have shape
`(query_count, dimension)`. Inputs are converted to aligned, contiguous float32 after
shape and buffer-size checks. Core rejects zero-query batches. `nprobe` is a
positive, platform-sized integer and is independent of top-K parameters.

Both methods accept `roaring_filter=` containing serialized **RoaringTreemap**
bytes, bytearray, or memoryview. `None` selects the unfiltered API; a serialized
empty treemap selects no rows. Empty bytes are not a serialized empty treemap
and are passed to core for validation.

`RangeSearchResult` contains owned NumPy arrays `lims` (`uintp`), `labels`
(`int64`), and `raw_distances` (`float32`), plus an immutable tuple `stats` and the
call-level `list_reads`. `query_count`, `hit_count`, and `query(index)` expose
the CSR shape. The query accessor returns a `RangeSearchQueryResult` named tuple
with `labels` and `raw_distances` fields, preserving unpacking as
`labels, raw_distances = result.query(index)`. It rejects negative or out-of-range
indices. The result's fields are frozen, but its arrays remain mutable. Query
slices share those arrays without copying payloads: writes through either view
are visible in the other. Arrays and statistics remain valid after reader
closure and native result destruction; query views keep the owned arrays alive
even after the `RangeSearchResult` itself is released.

Each `RangeSearchStats` contains `lists_probed`, `rows_scanned`,
`rows_committed`, and `early_abandoned`. These are core's logical counters;
`list_reads` is call-level and should not be summed from per-query counters.
The native result handle is destroyed in `finally`, including conversion and
allocation failures, while the reader's callback-aware lock remains held.

## Verification

```sh
PYTHONPATH=python PAIMON_VINDEX_LIB_PATH=/path/to/native/library \
  python3 -m pytest python/tests
```

The standalone matrix tests run without shared fixtures. To additionally run
the exact cross-language oracle, set `PVI_RANGE_FIXTURES` to the output directory
from `cargo run -p paimon-vindex-ffi --example range_search_fixture -- DIR`.
Oracle tests open a fresh reader for each manifest entry and compare per-query
`(label, float32 bits)` multisets, CSR limits, and every statistics field exactly.
Core does not guarantee row order across calls; the Python wrapper preserves
the order it receives without sorting.
