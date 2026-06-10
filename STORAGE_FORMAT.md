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

This document describes the v1 on-disk formats written by the Rust core
library. Version 1 is the first release format. Pre-release layouts are not part
of the compatibility contract.

## Compatibility Policy

- All multi-byte integers and `f32` values are little-endian.
- The unified reader dispatches by the first 4-byte magic value.
- Magic names below show the `u32` constants in human-readable big-endian form.
  Because the fields are little-endian, the raw file bytes for those constants
  appear in reverse ASCII order.
- Readers reject unknown magic values, unknown versions, unknown required flags,
  invalid section sizes, negative counts, and malformed list payload metadata.
- Incompatible on-disk changes require a new format version. Version 1 readers
  do not attempt to read future versions.
- Reserved bytes are written as zero. Readers currently skip reserved bytes
  unless a format explicitly assigns them meaning in a later version.
- Index files have no outer container, footer, checksum, compression envelope,
  or schema registry. The complete file starts at byte offset 0 with one of the
  headers below.
- Roaring row-id filters are a query-time API payload. They are not embedded in
  any index file format.

## Common Encodings

### Delta-Varint IDs

IVF-PQ and IVF-FLAT v1 sort each non-empty list by signed row id before writing.
The first id is stored as `base_id: i64`. The id stream then stores one unsigned
LEB128 varint per id, including the first id's zero delta. Each delta is computed
with wrapping unsigned subtraction from the previous signed id. Readers reject a
decoded sequence that is not monotonically non-decreasing in signed order.

### HNSW Graph Section

IVF-HNSW-FLAT and IVF-HNSW-SQ store one graph section per non-empty list. The
section is a contiguous sequence of little-endian `u32` values:

| Field | Count |
| --- | --- |
| `graph_count` | 1 |
| `entry_point` | 1 |
| `max_observed_level` | 1 |
| `level[node]` | `graph_count` |
| `degree[node][level]` followed by neighbor ids | one group for each node level |

Each node has levels `0..=level[node]`. A level-0 node may have at most `2 * m`
neighbors, and higher levels may have at most `m` neighbors.

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

## IVF-HNSW-FLAT v1

Magic: `IHFL` (`0x4948464C`). Version: `1`. Header size: 64 bytes.

| Offset | Size | Type | Field |
| ---: | ---: | --- | --- |
| 0 | 4 | `u32` | magic |
| 4 | 4 | `u32` | version |
| 8 | 4 | `i32` | dimension `d` |
| 12 | 4 | `i32` | IVF list count `nlist` |
| 16 | 4 | `u32` | metric (`0=L2`, `1=InnerProduct`, `2=Cosine`) |
| 20 | 8 | `i64` | total vector count |
| 28 | 4 | `i32` | HNSW `m` |
| 32 | 4 | `i32` | HNSW `ef_construction` |
| 36 | 4 | `i32` | HNSW `max_level` |
| 40 | 24 | bytes | reserved |

Sections after the header:

1. IVF coarse centroids: `nlist * d` `f32` values.
2. Offset table: `nlist` entries of
   `(offset: i64, count: i32, graph_bytes_len: i32, reserved: i64)`.
3. List payloads.

For each non-empty list payload:

| Field | Type | Notes |
| --- | --- | --- |
| `ids` | `count` `i64` | row ids in list order |
| `vectors` | `count * d` `f32` | raw stored vectors |
| `graph` | bytes | HNSW graph section |

## IVF-HNSW-SQ v1

Magic: `IHSQ` (`0x49485351`). Version: `1`. Header size: 64 bytes.

| Offset | Size | Type | Field |
| ---: | ---: | --- | --- |
| 0 | 4 | `u32` | magic |
| 4 | 4 | `u32` | version |
| 8 | 4 | `i32` | dimension `d` |
| 12 | 4 | `i32` | IVF list count `nlist` |
| 16 | 4 | `u32` | metric (`0=L2`, `1=InnerProduct`, `2=Cosine`) |
| 20 | 8 | `i64` | total vector count |
| 28 | 4 | `i32` | HNSW `m` |
| 32 | 4 | `i32` | HNSW `ef_construction` |
| 36 | 4 | `i32` | HNSW `max_level` |
| 40 | 4 | `f32` | global minimum SQ bound summary |
| 44 | 4 | `f32` | global maximum SQ bound summary |
| 48 | 16 | bytes | reserved |

Sections after the header:

1. Global SQ min bounds: `d` `f32` values.
2. Global SQ max bounds: `d` `f32` values.
3. Per-list SQ bounds: for each list, `d` min `f32` values followed by `d`
   max `f32` values.
4. IVF coarse centroids: `nlist * d` `f32` values.
5. Offset table: `nlist` entries of
   `(offset: i64, count: i32, graph_bytes_len: i32, reserved: i64)`.
6. List payloads.

For each non-empty list payload:

| Field | Type | Notes |
| --- | --- | --- |
| `ids` | `count` `i64` | row ids in list order |
| `codes` | bytes | scalar quantized residual codes, `count * d` bytes |
| `graph` | bytes | HNSW graph section over decoded vectors |
