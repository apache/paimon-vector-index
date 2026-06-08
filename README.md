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

# Apache Paimon Vector Index &emsp; [![Build Status]][actions]

[Build Status]: https://img.shields.io/github/actions/workflow/status/apache/paimon-vector-index/ci.yml
[actions]: https://github.com/apache/paimon-vector-index/actions?query=branch%3Amain

Pure Rust IVF-PQ implementation for Apache Paimon. Designed for data lake (S3/HDFS/OSS) with seek-based I/O, supporting both 8-bit and 4-bit PQ with SIMD acceleration.

## Metadata Filter Pushdown

The vector index accepts a serialized 64-bit Roaring bitmap of allowed row IDs during reader search. This lets the Paimon query layer evaluate metadata predicates with table/scalar indexes first, then pass the matching row-id set into IVF-PQ as an ANN prefilter.

Bindings expose the same wire format:

- Rust core: `search_with_reader_roaring_filter` and `search_batch_reader_roaring_filter`
- Java/JNI: `IVFPQReader.search(..., byte[])` and `IVFPQReader.searchBatch(..., byte[])`
- Python: `IVFPQReader.search(..., filter_bytes=...)` and `IVFPQReader.search_batch(..., filter_bytes=...)`

Row IDs must be non-negative to map directly into `RoaringTreemap`'s `u64` domain.

## Language Bindings

The Java binding provides small lifecycle-safe facades over the JNI symbols:
`IVFPQWriter` builds and writes an index, `IVFPQReader` opens an index and runs
single-query or batch search, and result containers expose defensive copies of
IDs and distances.

The Python binding mirrors that flow with `IVFPQWriter` and `IVFPQReader`.
`search` returns one-dimensional NumPy arrays for a single query, while
`search_batch` accepts a two-dimensional query array and returns two-dimensional
NumPy arrays shaped as `(query_count, top_k)`.

## Contributing

Apache Paimon Vector Index is an exciting project currently under active development. Whether you're looking to use it in your projects or contribute to its growth, there are several ways you can get involved:

- Follow the [Contributing Guide](CONTRIBUTING.md) to contribute.
- Create new [Issue](https://github.com/apache/paimon-vector-index/issues/new) for bug report or feature request.
- Start discussion thread at [dev mailing list](mailto:dev@paimon.apache.org) ([subscribe](<mailto:dev-subscribe@paimon.apache.org?subject=(send%20this%20email%20to%20subscribe)>) / [unsubscribe](<mailto:dev-unsubscribe@paimon.apache.org?subject=(send%20this%20email%20to%20unsubscribe)>) / [archives](https://lists.apache.org/list.html?dev@paimon.apache.org))
- Talk to community directly at [Slack #paimon channel](https://join.slack.com/t/the-asf/shared_invite/zt-2l9rns8pz-H8PE2Xnz6KraVd2Ap40z4g).

## Getting Help

Submit [issues](https://github.com/apache/paimon-vector-index/issues/new/choose) for bug report or asking questions in [discussion](https://github.com/apache/paimon-vector-index/discussions/new?category=q-a).

## License

Licensed under <a href="./LICENSE">Apache License, Version 2.0</a>.
