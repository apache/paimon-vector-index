# IVF-RQ Search Performance Design

## Problem Statement

The initial IVF-RQ implementation has avoidable work in the default
`query_bits=0` scan path, result selection, coarse-list selection, and disk
batch reads. These costs are paid for every search and do not require an index
format or public API change to remove.

## Chosen Approach

Keep the index format and search results unchanged while replacing only the
internal hot-path algorithms:

- Construct each 256-entry byte LUT from the preceding subset pattern.
- Maintain the final result set in an indexed max heap.
- Partially select coarse centroids before sorting only the selected prefix.
- Read all unique RQ lists for a batch through one multi-range `pread` call.

## Design Details

### Byte LUT

For one byte chunk, pattern zero is the negated sum of the chunk. Every other
pattern differs from `pattern & (pattern - 1)` by a single set bit, so its value
is the prior value plus twice the corresponding query dimension. The existing
64-entry cutoff remains appropriate after this eightfold reduction in LUT
construction work.

### Top-K and coarse selection

The result heap stores the worst distance at the root and maintains the ID to
position map as items move. Coarse routing uses `select_nth_unstable_by` and
sorts only the prefix required by `nprobe`.

### Reader I/O

`IVFRQIndexReader` exposes an internal multi-list reader modelled on IVFPQ:
metadata and payload buffers are prepared first, then passed to one
`SeekRead::pread` invocation per bounded 16 MiB payload batch. Results preserve
the requested list order without retaining every probed list in memory.

## Compatibility

No serialized bytes, public API, score formula, or result order contract is
changed. Tests compare optimized paths with the existing scalar semantics and
use counting readers to verify batched I/O.
