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

# Vector Index Storage Format

This document describes the v1 on-disk formats written by the
`paimon-vindex-core` crate starting with version 0.3.0. The 0.3.0 release
intentionally resets the pre-1.0 storage contract: it does not read the
experimental IVF-HNSW-FLAT (`IHFL`) or IVF-HNSW-SQ (`IHSQ`) layouts published
by 0.2.x. Rebuild those indexes when upgrading, and do not rely on 0.2.x
readers as a rollback path for files written by 0.3.0.

## Compatibility Policy

- All multi-byte integers and `f32` values are little-endian.
- The unified reader dispatches by the first 4-byte magic value.
- Magic names below show the `u32` constants in human-readable big-endian form.
  Because the fields are little-endian, the raw file bytes for those constants
  appear in reverse ASCII order.
- Readers reject unknown magic values, unknown versions, unknown required flags,
  non-zero reserved bytes, invalid section sizes, negative counts, and malformed
  list payload metadata.
- Incompatible on-disk changes require a new format version. Version 1 readers
  do not attempt to read future versions.
- Reserved bytes are written as zero and must be read back as zero. They cannot
  acquire meaning within v1. A new field, flag meaning, or reserved-byte use
  requires a new format version unless that format explicitly defined the flag
  as optional from its first v1 release.
- Index files have no outer container, footer, checksum, compression envelope,
  or schema registry. The complete file starts at byte offset 0 with one of the
  headers below.
- File integrity, including length and checksum validation, is guaranteed by
  the outer Paimon file/manifest layer rather than by an embedded index footer.
- Roaring row-id filters are a query-time API payload. They are not embedded in
  any index file format.

## Common Encodings

### Delta-Varint IDs

IVF-PQ, IVF-FLAT, IVF-RQ, and IVF-SQ v1 sort each non-empty list by signed row
id before writing. The first id is stored as `base_id: i64`. The id stream then
stores one unsigned LEB128 varint per id, including the first id's zero delta.
Each delta is computed with wrapping unsigned subtraction from the previous
signed id. Readers reject a decoded sequence that is not monotonically
non-decreasing in signed order.

## DiskANN v1

Raw magic bytes: `DANN` (`u32` value `0x4E4E4144`). Version: `1`. Header size:
256 bytes. DiskANN v1 is little-endian, supports L2, inner product, and cosine,
uses dense `u32` internal node IDs, and applies one BFS locality permutation
consistently to every section.

| Offset | Size | Type | Field |
| ---: | ---: | --- | --- |
| 0 | 4 | `u32` | magic |
| 4 | 4 | `u32` | version |
| 8 | 4 | `u32` | header size, `256` |
| 12 | 4 | `u32` | flags |
| 16 | 4 | `u32` | dimension |
| 20 | 4 | `u32` | metric (`0=L2`, `1=InnerProduct`, `2=Cosine`) |
| 24 | 8 | `u64` | vector count |
| 32 | 4 | `u32` | entry node |
| 36 | 4 | `u32` | maximum graph degree `R` |
| 40 | 4 | `u32` | build search-list size `Lbuild` |
| 44 | 4 | `f32` | robust-prune alpha |
| 48 | 8 | `u64` | build seed |
| 56 | 4 | `u32` | PQ subquantizer count `m` |
| 60 | 4 | `u32` | PQ bits, required `4` or `8` |
| 64 | 4 | `u32` | logical page size, required `4096` |
| 68 | 4 | `u32` | per-node adjacency locator payload size, required `4` |
| 72 | 4 | `u32` | adjacency locator encoding, required `3` |
| 76 | 4 | `u32` | raw-vector encoding: `1=f32`, `2=IEEE 754 binary16` |
| 80 | 4 | `u32` | raw-vector record size, `dimension * element_size` |
| 84 | 4 | `u32` | section count, required `7` |
| 88 | 8 | `u64` | exact total file length |
| 96 | 112 | seven `(offset: u64, length: u64)` pairs | sections in the order below |
| 208 | 48 | bytes | reserved, required zero |

Bits 0, 2, 3, and 4 are required: BFS layout, adaptive adjacency encoding, PQ
codes, and row-ID order. Exactly one storage-layout bit is required: bit 1 for
separate adjacency/vector sections or bit 5 for interleaved vector/adjacency
records. Unknown flags are rejected. Consequently, adding any DiskANN flag that
a v1 writer may emit requires version 2; it is not a compatible v1 extension.

The following header invariants are part of v1 and are enforced symmetrically by
the writer and reader:

- `1 <= dimension <= 1024`;
- `metric` is `0`, `1`, or `2`;
- `1 <= vector_count <= u32::MAX` and `entry_node < vector_count`;
- `1 <= pq_m <= dimension`, and `pq_bits` is 4 or 8;
- `1 <= R <= 1023`, `Lbuild >= R`, and `Lbuild <= u32::MAX`;
- `alpha` is finite and at least 1;
- `raw_vector_encoding` is `1` or `2`, and `vector_record_size` is exactly
  `dimension * 4` or `dimension * 2`, respectively; and
- every stored PQ centroid and decoded raw-vector component is finite. A writer
  using binary16 additionally rejects finite `f32` inputs whose conversion
  would overflow the finite binary16 range.

Inner-product results use negative dot product so lower values remain better.
Inner-product Vamana construction uses the metric-specific occluding prune rule
rather than the L2 triangle-inequality rule. Cosine training and indexed vectors
are normalized before PQ encoding and persistence; queries are normalized
before graph traversal, and final distances retain the public `1 - cosine`
semantics. Zero vectors remain zero and have cosine distance `1`.

Let `E` be the raw-vector element size (`4` for `f32`, `2` for binary16). The
interleaved writer requires `E * dimension + 4 * R <= 4096`. This
content-independent bound guarantees that even a raw-`u32` maximum-degree
adjacency list fits beside its vector without changing layout based on
compression results.

The seven sections are:

1. Self-describing PQ codebook at absolute offset 4096, described below.
2. Row IDs, adaptively encoded in dense node order as described below.
3. PQ codes, `m` bytes per node for 8-bit or `ceil(m / 2)` bytes per node for 4-bit.
   In 4-bit mode, each byte stores the earlier subquantizer in its low nibble
   and the next subquantizer in its high nibble. When `m` is odd, the unused
   high nibble of every final byte is required to be zero.
4. Row-ID order, one `u32` node ID per node, sorted by `(row_id, node_id)`.
5. Block-compressed adjacency index described below.
6. 4096-byte-aligned adaptively encoded adjacency pages, optionally containing
   the interleaved raw-vector records described below.
7. Dense raw-vector records for the separate layout; a zero-length section
   whose offset equals the file length for the interleaved layout.

The PQ-codebook section starts with this 32-byte header:

| Offset | Size | Type | Field |
| ---: | ---: | --- | --- |
| 0 | 4 | `u32` | magic `DPQ1` (`0x31515044`) |
| 4 | 4 | `u32` | codebook version, required `1` |
| 8 | 4 | `u32` | dimension, equal to the file header |
| 12 | 4 | `u32` | subquantizer count `m`, equal to the file header |
| 16 | 4 | `u32` | PQ bits, equal to the file header |
| 20 | 4 | `u32` | centroid count per chunk, `1 << pq_bits` |
| 24 | 4 | `u32` | chunk-offset count, `m + 1` |
| 28 | 4 | bytes | reserved, required zero |

It is followed by exactly `m + 1` little-endian `u32` component offsets. They
start at zero, end at `dimension`, and are strictly increasing. The current
writer creates balanced contiguous chunks: the first `dimension % m` chunks
have `floor(dimension / m) + 1` components and the rest have
`floor(dimension / m)`. Readers use the persisted offsets rather than
re-deriving that policy.

The remaining payload contains exactly `dimension * (1 << pq_bits)` finite
little-endian `f32` centroid components. Its order is
`centroid[chunk][code][component-within-chunk]`: components are contiguous,
followed by code, followed by chunk. Chunk `s` begins at centroid-component
offset `chunk_offsets[s] * (1 << pq_bits)`. The section has no trailing bytes.

The row-ID section starts with this 32-byte header:

| Offset | Size | Type | Field |
| ---: | ---: | --- | --- |
| 0 | 4 | `u32` | encoding: `0=raw i64`, `1=global FOR bit-pack` |
| 4 | 4 | `u32` | bit width: raw requires `64`; FOR requires `0..63` |
| 8 | 8 | `u64` | row-ID count, equal to vector count |
| 16 | 8 | `i64` | base: raw requires `0`; FOR stores the minimum row ID |
| 24 | 8 | bytes | reserved, required zero |

Raw encoding appends exactly `8 * N` little-endian bytes. FOR appends exactly
`ceil(N * bit_width / 8)` bytes containing unsigned `row_id - base` deltas in
dense node order, least-significant bit first. Unused high bits in the final
byte are zero. Width zero has no payload and maps every node to `base`. The
writer selects the minimum width and falls back to raw only for a 64-bit span.
The reader retains the packed payload and performs O(1) random row-ID lookup;
it validates the exact length, metadata, tail bits, and decoded `i64` range.

For `B = ceil(N / 16)`, the adjacency index contains three contiguous arrays:

1. `B` little-endian `u64` block base offsets relative to the start of the
   adjacency-page section;
2. `N` little-endian `u16` byte offsets relative to the corresponding
   16-node block base; and
3. `N` little-endian `u16` values whose bit 15 selects raw `u32` adjacency
   encoding and whose bits 0–14 store degree.

The section length is exactly `8 * B + 4 * N` bytes. Node `i` resolves to
`block_base[i / 16] + relative_offset[i]`; division and remainder by 4096
produce its page index and byte offset. The first relative offset in every
block is zero. A 16-node block can advance at most 15 pages because every list
is page-contained, so the largest valid relative offset is `65535`.

The codebook, encoded row IDs, PQ codes, row-ID order, and adjacency index are
contiguous. The reader probes the declared file tail, loads required resident
sections directly into their final representations, and loads row-ID order
lazily for sparse filtered queries. The adjacency section begins at the next
4096-byte boundary. Writers fill the bytes from the 256-byte header to the
codebook, and the alignment gap before adjacency, with zero. These alignment
bytes carry no v1 semantics and readers ignore them; they cannot be repurposed
without a new format version.
Neighbor IDs are strictly increasing and packed by actual degree. Each list
uses canonical unsigned delta LEB128 when that is strictly smaller than raw
little-endian `u32`; otherwise bit 15 in its locator selects raw encoding. The
first varint is the first absolute neighbor ID (a delta from zero), and later
varints are positive deltas from the preceding ID. Empty lists use delta mode
and no payload bytes. This adaptive choice guarantees that a list and the
complete adjacency payload never exceed the fixed-`u32` representation.

A list never crosses a logical-page boundary. In the separate layout, adjacent
locator ranges are contiguous. In the interleaved layout, each page record is
`[dimension * E raw-vector bytes][encoded adjacency bytes]`, and the locator
points to the first adjacency byte; adjacent records are contiguous.
Raw-vector bytes use the header encoding and little-endian scalar
representation. The remaining adjacency-page tail is zero. The resident block
bases, relative offsets, and degree/encoding values are structurally validated
before graph search. `optimize_for_search` validates every preloaded page in
parallel before publishing the hot prefix; a cold page is validated by shared
single-flight work on first access. Payload validation decodes each list to
establish its exact end and checks raw-vector finiteness, canonical varints, the
uniquely minimal adaptive mode, neighbor IDs, contiguity, and the zero
adjacency-page tail.

In the separate layout, raw-vector record `i` begins at
`vectors.offset + i * vector_record_size`; the section length is exactly
`vector_count * vector_record_size`, with no record or page padding. A record
contains `dimension` little-endian `f32` values when the encoding is `1`, or
`dimension` little-endian IEEE 754 binary16 bit patterns when the encoding is
`2`. Runtime readers group
`max(1, floor(profile_window_bytes / vector_record_size))` complete consecutive
records into one read window and clip the final window to the section end. A
record may therefore cross a 4096-byte address boundary, but never crosses its
runtime read window. Readers validate every consumed component before distance
evaluation. The exact derived lengths, ordering, file length, locator bounds,
degrees, neighbor IDs, duplicate/self edges, finite codebook/vector values,
row-ID encoding, and permutation are validated by the reader.

DiskANN v1 intentionally has no embedded checksum. The enclosing Paimon
file/manifest contract owns object length and checksum validation; the DiskANN
reader owns all structural and semantic checks described above.

The `Memory`, `LocalStorage`, `RemoteStorage`, and `ObjectStore` profiles do not
change this physical format. They group 4096-byte adjacency pages into runtime
read windows of 4096, 16384, 32768, or 65536 bytes respectively. Separate raw
vectors use the complete-record grouping described above with the same profile
window-byte targets.

## IVF-PQ v1

Magic: `IVPQ` (`0x49565051`). Version: `1`. Header size: 64 bytes.

| Offset | Size | Type | Field |
| ---: | ---: | --- | --- |
| 0 | 4 | `u32` | magic |
| 4 | 4 | `u32` | version |
| 8 | 4 | `i32` | dimension `d` |
| 12 | 4 | `i32` | IVF list count `nlist` |
| 16 | 4 | `i32` | PQ subquantizer count `m` |
| 20 | 4 | `i32` | centroid count per subquantizer `ksub` |
| 24 | 4 | `i32` | subvector dimension `dsub` |
| 28 | 4 | `u32` | metric (`0=L2`, `1=InnerProduct`, `2=Cosine`) |
| 32 | 8 | `i64` | total vector count |
| 40 | 4 | `u32` | flags |
| 44 | 20 | bytes | reserved |

Flags:

| Bit | Meaning |
| ---: | --- |
| 0 | OPQ rotation matrix is present |
| 1 | PQ codes are trained/stored by residual |
| 2 | delta-varint ids are used; required in v1 |
| 3 | PQ codes are transposed by subquantizer; required in v1 |

Sections after the header:

1. Optional OPQ rotation matrix: `d * d` `f32` values when flag bit 0 is set.
2. IVF coarse centroids: `nlist * d` `f32` values.
3. PQ centroids: `m * ksub * dsub` `f32` values.
4. Offset table: `nlist` entries of `(offset: i64, count: i32, id_bytes_len: i32)`.
5. List payloads.

For each non-empty list payload:

| Field | Type | Notes |
| --- | --- | --- |
| `base_id` | `i64` | first sorted row id |
| `id_bytes_len` | `i32` | byte length of encoded id stream |
| `id_bytes` | bytes | delta-varint ids |
| `codes` | bytes | transposed PQ codes |

For 8-bit PQ, each vector has `m` code bytes and the stored code layout is
`codes[sub][vector]`. For 4-bit PQ, each byte stores two subquantizers and the
stored layout is `codes[pair][vector]`.

## IVF-FLAT v1

Magic: `IVFL` (`0x4956464C`). Version: `1`. Header size: 64 bytes.

| Offset | Size | Type | Field |
| ---: | ---: | --- | --- |
| 0 | 4 | `u32` | magic |
| 4 | 4 | `u32` | version |
| 8 | 4 | `i32` | dimension `d` |
| 12 | 4 | `i32` | IVF list count `nlist` |
| 16 | 4 | `u32` | metric (`0=L2`, `1=InnerProduct`, `2=Cosine`) |
| 20 | 8 | `i64` | total vector count |
| 28 | 4 | `u32` | flags |
| 32 | 32 | bytes | reserved |

Flags:

| Bit | Meaning |
| ---: | --- |
| 0 | delta-varint ids are used; required in v1 |

Sections after the header:

1. IVF coarse centroids: `nlist * d` `f32` values.
2. Offset table: `nlist` entries of `(offset: i64, count: i32, id_bytes_len: i32)`.
3. List payloads.

For each non-empty list payload:

| Field | Type | Notes |
| --- | --- | --- |
| `base_id` | `i64` | first sorted row id |
| `id_bytes_len` | `i32` | byte length of encoded id stream |
| `id_bytes` | bytes | delta-varint ids |
| `vectors` | `count * d` `f32` | raw stored vectors |

## IVF-RQ v1

Magic: `IVRQ` (`0x49565251`). Version: `1`. Header size: 64 bytes.

| Offset | Size | Type | Field |
| ---: | ---: | --- | --- |
| 0 | 4 | `u32` | magic |
| 4 | 4 | `u32` | version |
| 8 | 4 | `i32` | logical dimension `d` |
| 12 | 4 | `i32` | `padded_d`, the next multiple of 64 |
| 16 | 4 | `i32` | IVF list count `nlist` |
| 20 | 4 | `u32` | metric (`0=L2`, `1=InnerProduct`, `2=Cosine`) |
| 24 | 4 | `u32` | required layout flags |
| 28 | 4 | `u32` | persisted RQ bit width, in `1..=8` |
| 32 | 8 | `i64` | total vector count |
| 40 | 8 | `u64` | deterministic rotation seed |
| 48 | 4 | `u32` | deterministic rotation rounds; `4` |
| 52 | 4 | `i32` | bytes per bit plane, `padded_d / 8` |
| 56 | 4 | `u32` | `rotation_type`; `2` for sign + 64-wide normalized FHT + permutation |
| 60 | 4 | `u32` | `factor_layout`; `3` for compact incremental coarse/full factors |

Flags:

| Bit | Meaning |
| ---: | --- |
| 0 | delta-varint ids are used; required in v1 |
| 1 | codes are transposed within 32-vector blocks; required in v1 |
| 2 | factors use structure-of-arrays layout within each block; required in v1 |

Sections after the header:

1. IVF coarse centroids: `nlist * d` `f32` values.
2. Offset table: `nlist` entries of `(offset: i64, count: i32, id_bytes_len: i32)`.
3. List payloads.

For each non-empty list payload:

| Field | Type | Notes |
| --- | --- | --- |
| `base_id` | `i64` | first sorted row id |
| `id_bytes_len` | `i32` | byte length of encoded id stream |
| `code_bytes_len` | `i32` | exact blocked-code byte length |
| `id_bytes` | bytes | delta-varint ids |
| `codes` | `count * bits * (padded_d / 8)` bytes | MSB-first bit planes; within every up-to-32-vector block the order is plane, byte position, lane |
| `factors` | `count * fields` `f32` | block-SoA fields; 2 coarse fields for one bit, otherwise 3 coarse plus 2 full fields |

The coarse factor fields are `(f_add, f_rescale, f_error)` when multiple bit
planes require a deterministic reconstruction-error lower bound. A one-bit
file stores only `(f_add, f_rescale)` because it has no later refinement stage.
For multi-bit files, candidates that can still enter Top-K are refined with
every plane and the full `(f_add, f_rescale)` estimate. The full
reconstruction-error factor is intentionally omitted because the final stage
does not compute another lower bound.

The orthogonal transform is reconstructed from `(d, rotation_seed,
rotation_rounds)`. It pads with zeros to `padded_d` and applies four rounds of
random signs, normalized 64-wide FHT, and permutation. The Reader rotates each
query once and reuses its byte LUT across every selected list.

The pre-release one-bit/Kac/factor-layout-1 representation used the same magic
and version but was never published. v1 Readers intentionally reject it through
the required padded dimension, flags, rotation type, and factor layout checks.
There is no query-side bit-width parameter; the file fixes the representation.

## IVF-SQ v1

Magic: `IVSQ` (`0x49565351`). Version: `1`. Header size: 64 bytes. IVF-SQ uses
one unsigned 8-bit code per residual dimension and scans every code in each
selected IVF list.

| Offset | Size | Type | Field |
| ---: | ---: | --- | --- |
| 0 | 4 | `u32` | magic |
| 4 | 4 | `u32` | version |
| 8 | 4 | `i32` | dimension `d` |
| 12 | 4 | `i32` | IVF list count `nlist` |
| 16 | 4 | `u32` | metric (`0=L2`, `1=InnerProduct`, `2=Cosine`) |
| 20 | 8 | `i64` | total vector count |
| 28 | 4 | `u32` | SQ bits, required `8` |
| 32 | 4 | `u32` | flags |
| 36 | 4 | `f32` | global minimum SQ bound summary |
| 40 | 4 | `f32` | global maximum SQ bound summary |
| 44 | 20 | bytes | reserved |

Flags:

| Bit | Meaning |
| ---: | --- |
| 0 | sorted delta-varint ids are stored; required in v1 |
| 1 | codes use 32-row blocked dimension-major layout; required in v1 |

Sections after the header:

1. Global SQ min bounds: `d` `f32` values.
2. Global SQ max bounds: `d` `f32` values.
3. Per-list SQ bounds: for each list, `d` min `f32` values followed by `d`
   max `f32` values.
4. IVF coarse centroids: `nlist * d` `f32` values.
5. Offset table: `nlist` entries of
   `(offset: i64, count: i32, id_bytes_len: i32)`.
6. List payloads.

For each non-empty list payload:

| Field | Type | Notes |
| --- | --- | --- |
| `codes` | bytes | `count * d` scalar codes; within every up-to-32-row block the order is dimension, then row lane |
| `base_id` | `i64` | first sorted row id |
| `id_bytes_len` | `i32` | byte length of encoded id stream |
| `id_bytes` | bytes | delta-varint ids |

The global bounds provide a fallback for empty training lists. Non-empty lists
normally use their own per-dimension residual bounds. A reader validates all
offsets, counts, encoded-ID sizes, and SQ bounds before exposing the index.
Putting blocked codes first lets the reader retain the list payload allocation
as the scan buffer after decoding and truncating the trailing ID section.
