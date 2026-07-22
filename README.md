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
Apache Paimon and data lake storage such as S3, HDFS, and OSS. Its seek-based
readers load only the IVF lists selected by a query.

The library supports IVF-FLAT, IVF-PQ, IVF-RQ, IVF-HNSW-FLAT, and
IVF-HNSW-SQ through shared Rust, C, C++, Java/JNI, and Python APIs.

## Documentation

- [Index selection and architecture](docs/index.html): compare all index
  families and open the detailed page for each implementation.
- [API and language bindings](docs/api.html): lifecycle, query parameters,
  warm-up, Rust, C, C++, Java, Python, and metadata filter pushdown.
- [Development and benchmarks](docs/development.html): workspace layout,
  build and test commands, ANN benchmarks, and storage compatibility checks.
- [Storage format specification](STORAGE_FORMAT.md): normative v1 binary layout
  and compatibility policy.

GitHub shows committed HTML files as source. To view the styled documentation,
clone the repository and serve `docs/` from the repository root:

```shell
python3 -m http.server --directory docs 8000
```

Then open <http://localhost:8000/>. All documentation assets are local, so no
separate build step is required.

## Source Layout

Public implementations live in [`core`](core), [`ffi`](ffi), [`include`](include),
[`jni`](jni), [`java`](java), and [`python`](python). See the
[development guide](docs/development.html#workspace) for responsibilities and
verification commands.

## Contributing

- Read the [Contributing Guide](CONTRIBUTING.md).
- Create an [issue](https://github.com/apache/paimon-vector-index/issues/new)
  for a bug report or feature request.
- Join the [dev mailing list](mailto:dev@paimon.apache.org)
  ([subscribe](<mailto:dev-subscribe@paimon.apache.org?subject=(send%20this%20email%20to%20subscribe)>) /
  [unsubscribe](<mailto:dev-unsubscribe@paimon.apache.org?subject=(send%20this%20email%20to%20unsubscribe)>) /
  [archives](https://lists.apache.org/list.html?dev@paimon.apache.org)).
- Talk to the community in the
  [Slack #paimon channel](https://join.slack.com/t/the-asf/shared_invite/zt-2l9rns8pz-H8PE2Xnz6KraVd2Ap40z4g).

## Getting Help

Submit an [issue](https://github.com/apache/paimon-vector-index/issues/new/choose)
or ask a question in
[GitHub Discussions](https://github.com/apache/paimon-vector-index/discussions/new?category=q-a).

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
