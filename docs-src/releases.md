---
title: "Releases"
description: "Find Apache Paimon Vector Index source releases and convenience artifacts."
---

<!--
  Licensed to the Apache Software Foundation (ASF) under one or more
  contributor license agreements. See the NOTICE file distributed with
  this work for additional information regarding copyright ownership.
  The ASF licenses this file to you under the Apache License, Version 2.0.
-->

# Releases

Download official Apache source releases, find the corresponding Rust, Java, and Python packages, or follow the release-manager and release-candidate verification guides.

## Distribution channels {#channels}

> **The source archive is the Apache release** — Registry packages and native wheels are convenience binaries. The signed source archive approved by the community vote is the authoritative release.

| Component | Location | Coordinates |
|----|----|----|
| Source | [Apache mirrors](https://www.apache.org/dyn/closer.lua/paimon/) | `apache-paimon-vector-index-VERSION-src.tgz` |
| Rust | [crates.io](https://crates.io/crates/paimon-vindex-core) | `paimon-vindex-core` |
| Java | [Maven Central](https://repo.maven.apache.org/maven2/org/apache/paimon/paimon-vector-index-java/) | `org.apache.paimon:paimon-vector-index-java` |
| Python | [PyPI](https://pypi.org/project/paimon-vindex/) | `paimon-vindex` |
| C and C++ | Official source archive | Build `paimon-vindex-ffi` and use `include/` |

Release signatures are checked against the Paimon [KEYS](https://downloads.apache.org/paimon/KEYS) file. Superseded source releases remain available from the [Apache archive](https://archive.apache.org/dist/paimon/).

## Upcoming: 0.5.0 {#upcoming}

The repository is currently developing the 0.5.0 line. Until an ASF vote passes and the signed source archive appears under Apache downloads, code and packages from this line are development artifacts rather than an Apache release.

> **Rust IVF API migration** — Version 0.5.0 makes `quantizer_centroids` private on `IVFFlatIndex`, `IVFPQIndex`, `IVFSQIndex`, and `IVFRQIndex`. Replace direct reads with `quantizer_centroids()` and direct assignments with `set_quantizer_centroids(...)`. The setter validates the centroid shape, rejects replacement after vectors are added, and refreshes cached derived state. IVF variants of `VectorIndexConfig` also require `use_approximate_coarse_assignment`; set it to `true` for the automatic 0.5.0 behavior or `false` for exact nearest-centroid assignment. Option-map callers can select the same policy with `ivf.coarse-assignment=auto|exact`. The policy is fixed when the writer is created; direct IVF indexes do not expose a post-training policy switch. These are source-level changes; the stored index format is unchanged.

## 0.4.0 {#release-040}

Released 21 August 2026. This release improved filtered and batched IVF scans, stabilized IVF-PQ training, added extensible search parameters across language bindings, bridged native diagnostics to Java logging, and tightened serialization allocation bounds.

[Source release](https://www.apache.org/dyn/closer.lua/paimon/paimon-vector-index-0.4.0/apache-paimon-vector-index-0.4.0-src.tgz?action=download) · [Release notes](https://github.com/apache/paimon-vector-index/releases/tag/v0.4.0) · [Rust crate](https://crates.io/crates/paimon-vindex-core/0.4.0) · [Java artifacts](https://repo.maven.apache.org/maven2/org/apache/paimon/paimon-vector-index-java/0.4.0/) · [Python package](https://pypi.org/project/paimon-vindex/0.4.0/)

## 0.3.0 {#release-030}

Released 30 July 2026. This release added native DiskANN and adaptive IVF indexes, IVF-RQ query-width controls and search optimizations, batched positional reads across C and Python, expanded architecture and API documentation, and reproducible source and multi-platform convenience-artifact release verification.

> **Stored-index upgrade note** — Version 0.3.0 intentionally does not read the experimental IVF-HNSW-FLAT (`IHFL`) and IVF-HNSW-SQ (`IHSQ`) files written by 0.2.x. Rebuild those indexes with a 0.3.0-supported index type during upgrade. Index files written by 0.3.0 are likewise not a safe rollback boundary for 0.2.x; retain source vectors or a 0.2-compatible index copy until the upgrade is accepted.

[Source release](https://www.apache.org/dyn/closer.lua/paimon/paimon-vector-index-0.3.0/apache-paimon-vector-index-0.3.0-src.tgz?action=download) · [Release notes](https://github.com/apache/paimon-vector-index/releases/tag/v0.3.0) · [Rust crate](https://crates.io/crates/paimon-vindex-core/0.3.0) · [Java artifacts](https://repo.maven.apache.org/maven2/org/apache/paimon/paimon-vector-index-java/0.3.0/) · [Python package](https://pypi.org/project/paimon-vindex/0.3.0/)

## 0.2.0 {#release-020}

Released 7 July 2026. This release added batched vector-index training and hardened the multi-language release pipeline, Java staging, source JAR contents, and native wheel builds.

[Source release](https://www.apache.org/dyn/closer.lua/paimon/paimon-vector-index-0.2.0/apache-paimon-vector-index-0.2.0-src.tgz?action=download) · [Release notes](https://github.com/apache/paimon-vector-index/releases/tag/v0.2.0) · [Rust crate](https://crates.io/crates/paimon-vindex-core/0.2.0) · [Java artifacts](https://repo.maven.apache.org/maven2/org/apache/paimon/paimon-vector-index-java/0.2.0/) · [Python package](https://pypi.org/project/paimon-vindex/0.2.0/)

## 0.1.0 {#release-010}

Released 22 June 2026. The first release established the Rust vector-index core, seek-based persisted readers, C/C++, Java/JNI, and Python APIs, bitmap filtering, batch search, storage-format validation, and the initial ANN benchmark suite.

[Source release](https://www.apache.org/dyn/closer.lua/paimon/paimon-vector-index-0.1.0/apache-paimon-vector-index-0.1.0-src.tgz?action=download) · [Release notes](https://github.com/apache/paimon-vector-index/releases/tag/v0.1.0) · [Rust crate](https://crates.io/crates/paimon-vindex-core/0.1.0) · [Java artifacts](https://repo.maven.apache.org/maven2/org/apache/paimon/paimon-vector-index-java/0.1.0/) · [Python package](https://pypi.org/project/paimon-vindex/0.1.0/)

## Release guides {#guides}

### Creating a release

Prepare versions and dependency manifests, build and stage an RC, call the vote, and promote an approved release.

[Release Manager guide →](creating-a-release.md)

### Verifying a release candidate

Check provenance, signatures, checksums, archive contents, builds, tests, and staged language artifacts before voting.

[Verification guide →](verifying-a-release-candidate.md)
