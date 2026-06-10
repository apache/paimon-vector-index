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

Pure Rust vector index implementation for Apache Paimon. Designed for data lake
(S3/HDFS/OSS) with seek-based I/O, supporting IVF-FLAT, IVF-PQ,
IVF-HNSW-FLAT, and IVF-HNSW-SQ indexes.

## Metadata Filter Pushdown

The vector index accepts a serialized 64-bit Roaring bitmap of allowed row IDs
during reader search. This lets the Paimon query layer evaluate metadata
predicates with table/scalar indexes first, then pass the matching row-id set
into vector search as an ANN prefilter.

Bindings expose the same wire format:

- Rust core: `VectorIndexReader::search_with_roaring_filter` and
  `VectorIndexReader::search_batch_with_roaring_filter`
- Java/JNI: `VectorIndexReader.search(..., byte[])` and
  `VectorIndexReader.searchBatch(..., byte[])`
- Python: `VectorIndexReader.search(..., filter_bytes=...)` and
  `VectorIndexReader.search_batch(..., filter_bytes=...)`

Row IDs must be non-negative to map directly into `RoaringTreemap`'s `u64` domain.

## ANN Benchmark

The core crate includes an ANN-style benchmark for comparing Paimon's IVF-PQ,
IVF-HNSW-FLAT, and IVF-HNSW-SQ indexes. It reports build time, reader open/load
time, first-query latency, batch query throughput, and serialized index size:

```bash
cargo bench -p paimon-vindex-core --bench ann_bench -- --nocapture
```

The benchmark is configured with environment variables:

```bash
ANN_N=100000 ANN_NQ=1000 ANN_D=128 ANN_K=10 ANN_NLIST=256 ANN_NPROBE=16 \
ANN_PQ_M=16 ANN_HNSW_EF_CONSTRUCTION=150 ANN_HNSW_EF_SEARCH=80 \
cargo bench -p paimon-vindex-core --bench ann_bench -- --nocapture
```

Benchmark rows report `disk_scope=index_bytes`, which is the serialized vector
index file.

## Unified API

Rust, Java, and Python expose one writer and one reader API. Writers are created
from typed configs, while readers detect the index type from the file header.

Rust:

```rust
use paimon_vindex_core::distance::MetricType;
use paimon_vindex_core::index::{VectorIndexConfig, VectorIndexWriter};

let config = VectorIndexConfig::IvfPq {
    dimension: 128,
    nlist: 1024,
    m: 16,
    metric: MetricType::L2,
    use_opq: false,
};
let mut writer = VectorIndexWriter::new(config)?;
```

Java:

```java
VectorIndexConfig config =
        VectorIndexConfig.ivfHnswFlat(128, 1024, Metric.L2, HnswConfig.DEFAULT);
VectorIndexWriter writer = new VectorIndexWriter(config);
VectorIndexReader reader = new VectorIndexReader(vectorIndexInput);
```

Python:

```python
class VectorIndexInput:
    def __init__(self, data: bytes):
        self.data = data

    def pread_many(self, ranges):
        return [self.data[pos : pos + length] for pos, length in ranges]

writer = VectorIndexWriter(IvfPqConfig(128, 1024, 16, metric="l2"))
reader = VectorIndexReader(VectorIndexInput(index_bytes))
ids, distances = reader.search(query, top_k=10, nprobe=16)
```

Python `search` returns one-dimensional NumPy arrays for a single query, while
`search_batch` accepts a two-dimensional query array and returns arrays shaped
as `(query_count, top_k)`.

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
