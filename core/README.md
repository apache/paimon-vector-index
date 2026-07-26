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

The DiskANN and Vamana code is an independent Apache-licensed implementation
based on the published algorithms and this project's existing storage
abstractions. It does not incorporate source code from Microsoft's
[MIT-licensed DiskANN repository](https://github.com/microsoft/DiskANN).
The implementation supports L2, inner-product, and cosine search with the same
lower-is-better distance semantics as the IVF indexes.

The crate ships its [normative v1 storage-format specification](STORAGE_FORMAT.md)
and byte-exact fixtures. Project documentation, language bindings, and
contribution guidance live in the
[Apache Paimon Vector Index repository](https://github.com/apache/paimon-vector-index).
