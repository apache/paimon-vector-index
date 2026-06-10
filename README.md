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

Apache Paimon Vector Index is a pure Rust vector indexing library designed for
Apache Paimon and data lake storage such as S3, HDFS, and OSS. Index readers use
seek-based positional I/O so query execution can read only the parts of an index
file needed by the selected IVF lists.

The project is no longer limited to IVF-PQ. The unified writer and reader APIs
support multiple index families across Rust, Java/JNI, and Python:

| Index type | Summary | Best fit |
| --- | --- | --- |
| `IVF_FLAT` | IVF partitioning with uncompressed vectors. | Baseline recall and simple storage. |
| `IVF_PQ` | IVF with product quantization and optional OPQ rotation. | Compact indexes with fast approximate scans. |
| `IVF_HNSW_FLAT` | IVF partitioning with an HNSW graph inside each list over raw vectors. | Higher recall within probed IVF lists. |
| `IVF_HNSW_SQ` | IVF partitioning with per-list HNSW and scalar-quantized vectors. | HNSW-style search with smaller vector storage. |

All index types share:

- `L2`, inner product, and cosine metrics.
- Training, vector add, serialization, metadata, single-query search, and batch
  search APIs.
- Reader-side index type detection from the file header.
- Optional row-id prefiltering with serialized 64-bit Roaring bitmaps.

## Workspace

The repository contains three public integration layers:

- `core`: Rust implementation and benchmark suite.
- `jni`: Java classes plus JNI bindings backed by the Rust core.
- `python`: Python extension module backed by the Rust core.

The top-level Cargo workspace includes `core` and `jni`. The Python extension is
kept as a separate Cargo package under `python`.

## Unified API

Writers are created from typed configs, while readers detect the index type from
the serialized file header. The same search parameters are used across index
types:

- `top_k`: number of nearest neighbors to return.
- `nprobe`: number of IVF lists to probe.
- `ef_search`: optional HNSW search breadth for `IVF_HNSW_FLAT` and
  `IVF_HNSW_SQ`. A value of `0` uses the default.

### Rust

```rust
use std::fs::File;

use paimon_vindex_core::distance::MetricType;
use paimon_vindex_core::hnsw::HnswBuildParams;
use paimon_vindex_core::index::{
    VectorIndexConfig, VectorIndexReader, VectorIndexWriter, VectorSearchParams,
};
use paimon_vindex_core::io::PosWriter;

let config = VectorIndexConfig::IvfHnswSq {
    dimension: 128,
    nlist: 1024,
    metric: MetricType::L2,
    hnsw: HnswBuildParams::default(),
};

let mut writer = VectorIndexWriter::new(config)?;
writer.train(&training_vectors, training_count)?;
writer.add_vectors(&row_ids, &vectors, vector_count)?;

let mut file = File::create("vectors.pvindex")?;
let mut out = PosWriter::new(&mut file);
writer.write(&mut out)?;

let file = File::open("vectors.pvindex")?;
let mut reader = VectorIndexReader::open(file)?;
let params = VectorSearchParams::with_ef_search(10, 16, 80);
let (ids, distances) = reader.search(&query, params)?;
```

Other Rust configs follow the same shape:

```rust
VectorIndexConfig::IvfFlat {
    dimension: 128,
    nlist: 1024,
    metric: MetricType::L2,
};

VectorIndexConfig::IvfPq {
    dimension: 128,
    nlist: 1024,
    m: 16,
    metric: MetricType::L2,
    use_opq: false,
};

VectorIndexConfig::IvfHnswFlat {
    dimension: 128,
    nlist: 1024,
    metric: MetricType::L2,
    hnsw: HnswBuildParams::default(),
};
```

### Java/JNI

```java
import java.util.HashMap;
import java.util.Map;

import org.apache.paimon.index.ivfpq.VectorIndexInput;
import org.apache.paimon.index.ivfpq.VectorIndexMetadata;
import org.apache.paimon.index.ivfpq.VectorIndexReader;
import org.apache.paimon.index.ivfpq.VectorSearchResult;
import org.apache.paimon.index.ivfpq.VectorIndexWriter;

Map<String, String> options = new HashMap<>();
options.put("index.type", "ivf_hnsw_sq");
options.put("dimension", "128");
options.put("nlist", "1024");
options.put("metric", "l2");
options.put("hnsw.m", "20");
options.put("hnsw.ef-construction", "150");
options.put("hnsw.max-level", "7");

try (VectorIndexWriter writer = new VectorIndexWriter(options)) {
    writer.train(trainingVectors, trainingCount);
    writer.addVectors(rowIds, vectors, vectorCount);
    writer.writeIndex(vectorIndexOutput);
}

try (VectorIndexReader reader = new VectorIndexReader(vectorIndexInput)) {
    VectorIndexMetadata metadata = reader.metadata();
    VectorSearchResult result = reader.search(query, 10, 16, 80);
}
```

The Java package currently remains `org.apache.paimon.index.ivfpq`, but the API
surface uses string options so it maps directly to Paimon table/index
properties. Rust parses and validates the options when the writer is created.

### Python

```python
from paimon_vindex import VectorIndexReader, VectorIndexWriter


class VectorIndexInput:
    def __init__(self, data: bytes):
        self.data = data

    def pread_many(self, ranges):
        return [self.data[pos : pos + length] for pos, length in ranges]


options = {
    "index.type": "ivf_hnsw_sq",
    "dimension": "128",
    "nlist": "1024",
    "metric": "l2",
    "hnsw.m": "20",
    "hnsw.ef-construction": "150",
    "hnsw.max-level": "7",
}
writer = VectorIndexWriter(options)
writer.train(training_vectors)
writer.add_vectors(row_ids, vectors)
writer.write(output)

reader = VectorIndexReader(VectorIndexInput(index_bytes))
ids, distances = reader.search(query, top_k=10, nprobe=16, ef_search=80)
```

`search` returns one-dimensional NumPy arrays for a single query, while
`search_batch` accepts a two-dimensional query array and returns arrays shaped
as `(query_count, top_k)`.

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

The core crate includes an ANN-style benchmark for comparing Paimon's
`IVF_PQ`, `IVF_HNSW_FLAT`, and `IVF_HNSW_SQ` implementations. It reports build
time, reader open/load time, first-query latency, batch query throughput, and
serialized index size:

```bash
cargo bench -p paimon-vindex-core --bench ann_bench -- --nocapture
```

The benchmark is configured with environment variables:

```bash
ANN_N=100000 ANN_NQ=1000 ANN_D=128 ANN_K=10 ANN_NLIST=256 ANN_NPROBE=16 \
ANN_PQ_M=16 ANN_HNSW_M=20 ANN_HNSW_EF_CONSTRUCTION=150 ANN_HNSW_EF_SEARCH=80 \
cargo bench -p paimon-vindex-core --bench ann_bench -- --nocapture
```

Benchmark rows report `disk_scope=index_bytes`, which is the serialized vector
index file.

## Development

Common Rust commands:

```bash
cargo fmt --all
cargo test --workspace
cargo clippy --workspace --all-targets
```

Java API tests are run from the JNI Java module:

```bash
mvn -f java/pom.xml test
```

Python extension tests are run from the `python` package:

```bash
cd python
cargo test
```

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
