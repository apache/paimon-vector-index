---
title: "Verifying a Release Candidate"
description: "Verify the source, signatures, builds, and artifacts of a Vector Index release candidate."
---

<!--
  Licensed to the Apache Software Foundation (ASF) under one or more
  contributor license agreements. See the NOTICE file distributed with
  this work for additional information regarding copyright ownership.
  The ASF licenses this file to you under the Apache License, Version 2.0.
-->

# Verifying a Release Candidate

Confirm the candidate's provenance and integrity, inspect its legal and source-only contents, build it from the archive, test the language surfaces, and report exactly what you checked with your vote.

## Collect the candidate inputs {#inputs}

Use the URLs and commit SHA from the vote email. Substitute the actual version and RC number; download all files from the ASF dist-dev candidate directory rather than from an untrusted mirror.

```bash title="Shell · example"
RELEASE_VERSION="0.3.0"
RC_NUM="1"
RC_TAG="v${RELEASE_VERSION}-rc${RC_NUM}"
RC_DIR="paimon-vector-index-${RELEASE_VERSION}-rc${RC_NUM}"
ARCHIVE="apache-paimon-vector-index-${RELEASE_VERSION}-src.tgz"

curl -O "https://dist.apache.org/repos/dist/dev/paimon/${RC_DIR}/${ARCHIVE}"
curl -O "https://dist.apache.org/repos/dist/dev/paimon/${RC_DIR}/${ARCHIVE}.asc"
curl -O "https://dist.apache.org/repos/dist/dev/paimon/${RC_DIR}/${ARCHIVE}.sha512"
curl -O https://downloads.apache.org/paimon/KEYS
```

> **Keep the evidence linked** — The source archive, detached signature, checksum, signed Git tag, commit SHA, Java staging repository, and TestPyPI version must all identify the same release version and candidate.

## Verify signature, checksum, and Git provenance {#integrity}

```bash title="Shell"
gpg --import KEYS
gpg --verify "${ARCHIVE}.asc" "${ARCHIVE}"

# Linux
sha512sum -c "${ARCHIVE}.sha512"

# macOS
shasum -a 512 -c "${ARCHIVE}.sha512"
```

A good cryptographic signature is necessary but not sufficient: confirm that its fingerprint belongs to the Release Manager in the downloaded Paimon KEYS file.

```bash title="Shell · signed tag and commit"
git clone https://github.com/apache/paimon-vector-index.git candidate-git
git -C candidate-git fetch --tags
git -C candidate-git tag -v "${RC_TAG}"
git -C candidate-git rev-parse "${RC_TAG}^{commit}"
```

The last SHA must equal the commit announced in the vote email. Inspect the changes since the previous release and check that the tag is attached to the intended release branch.

## Inspect the source archive {#contents}

```bash title="Shell"
tar tzf "${ARCHIVE}" | sed -n '1,80p'
tar tzf "${ARCHIVE}" | grep -E \
  '(^|/)(\.git|\.github|\.idea|target|build|__pycache__)(/|$)' && exit 1 || true

tar xzf "${ARCHIVE}"
cd "paimon-vector-index-${RELEASE_VERSION}"
```

Review the extracted tree rather than a Git checkout. At minimum verify:

- There is exactly one top-level directory named `paimon-vector-index-VERSION`.
- `LICENSE`, `NOTICE`, source headers, README files, dependency declarations, and required build files are present and correct.
- No Git metadata, CI configuration, editor state, build output, generated native libraries, credentials, or unrelated binary files are included.
- All bundled third-party material is compatible with the Apache License 2.0 and recorded where required.
- The release contains source sufficient to build the Rust core and the C/C++, Java, and Python bindings.

```bash title="Shell · repository checks included in the archive"
python3 tools/check_license_headers.py
python3 tools/dependencies.py check
cargo deny check licenses
```

## Build and test from the archive {#build}

Record the operating system, architecture, Rust, Python, JDK, Maven, CMake, and compiler versions used. Run as much of the following matrix as your environment supports.

### Rust workspace

```bash title="Shell"
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo build --workspace --release
cargo publish -p paimon-vindex-core --dry-run
```

### C and C++

```bash title="Shell · Linux"
cargo build --release -p paimon-vindex-ffi

cmake -S c -B c/build \
  -DPAIMON_VINDEX_FFI_LIB="$PWD/target/release/libpaimon_vindex_ffi.so"
cmake --build c/build
LD_LIBRARY_PATH="$PWD/target/release" c/build/test_vindex

cmake -S cpp -B cpp/build \
  -DPAIMON_VINDEX_FFI_LIB="$PWD/target/release/libpaimon_vindex_ffi.so"
cmake --build cpp/build
LD_LIBRARY_PATH="$PWD/target/release" cpp/build/test_vindex_cpp
```

Use `libpaimon_vindex_ffi.dylib` on macOS or `paimon_vindex_ffi.dll` on Windows and the platform's equivalent library-loader configuration.

### Java / JNI

```bash title="Shell · Linux"
mvn -f java/pom.xml test
cargo build --release -p paimon-vindex-jni

java -cp java/target/test-classes:java/target/classes \
  org.apache.paimon.index.vector.VectorIndexNativeValidationTest \
  "$PWD/target/release/libpaimon_vindex_jni.so"
java -cp java/target/test-classes:java/target/classes \
  org.apache.paimon.index.vector.VectorIndexNativePanicBoundaryTest \
  "$PWD/target/release/libpaimon_vindex_jni.so"
java -cp java/target/test-classes:java/target/classes \
  org.apache.paimon.index.vector.VectorIndexNativeHandleSafetyTest \
  "$PWD/target/release/libpaimon_vindex_jni.so"
```

### Python

```bash title="Shell"
cargo build --release -p paimon-vindex-ffi
python3 -m venv .venv
. .venv/bin/activate
cd python
PAIMON_VINDEX_LIB_PATH=../target/release pip install -e ".[test]"
PAIMON_VINDEX_LIB_PATH=../target/release pytest -v
```

Also inspect test failures and warnings. A successful compiler exit alone does not establish that the candidate is suitable for release.

## Test the staged convenience artifacts {#artifacts}

> **Do not substitute binary testing for source verification** — These checks establish that the convenience artifacts correspond to a usable release. The community vote still approves the signed source archive.

### Rust

RC tags intentionally do not publish to crates.io. The source-tree `cargo publish --dry-run` above must pass, and a temporary consumer can point to the signed RC tag:

```text title="Cargo.toml"
[dependencies]
paimon-vindex-core = {
  git = "https://github.com/apache/paimon-vector-index",
  tag = "v${RELEASE_VERSION}-rc${RC_NUM}"
}
```

### Java Nexus staging

Use the exact `orgapachepaimon-XXXX` repository from the vote email. Resolve `org.apache.paimon:paimon-vector-index-java:VERSION` from that URL, run a small API smoke test, and inspect the binary JAR:

```bash title="Shell"
jar tf paimon-vector-index-java-${RELEASE_VERSION}.jar | sort

# Expected native entries:
# native/linux/x86_64/libpaimon_vindex_jni.so
# native/linux/aarch64/libpaimon_vindex_jni.so
# native/macos/aarch64/libpaimon_vindex_jni.dylib
# native/windows/x86_64/paimon_vindex_jni.dll
# META-INF/LICENSE
# META-INF/NOTICE
# META-INF/LICENSE-binary
# org/apache/paimon/index/vector/NativeLibraryLoader.class
```

Run the smoke program with the staged main JAR first on the classpath and without passing or preloading a native library. Verify that it creates a native Trainer successfully. Also verify the main JAR plus source and Javadoc JARs, POM, signatures, and checksums. Confirm that the source JAR does not embed native libraries and that the POM version has no `-SNAPSHOT` suffix.

### Python TestPyPI

```bash title="Shell · isolated environment"
python3 -m venv rc-wheel
. rc-wheel/bin/activate
python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  "paimon-vindex==${RELEASE_VERSION}rc${RC_NUM}"
python -c "import paimon_vindex; print('OK')"
```

Check that TestPyPI provides Linux x86-64, Linux aarch64, macOS arm64, and Windows x86-64 wheels. Every wheel must contain `LICENSE`, `NOTICE`, and `LICENSE-binary` alongside the package and in the wheel metadata license directory. Test the wheel matching your platform against a small build/search/read round trip.

## Report your vote {#vote}

Reply to the vote thread with +1, 0, or -1, state whether the vote is binding, and list the checks and platforms you actually completed. Explain any failure precisely enough for the Release Manager to reproduce it.

```text title="Example vote"
+1 (binding/non-binding)

Verified:
- RC tag signature and announced commit
- source signature and SHA-512 checksum
- LICENSE, NOTICE, dependency licenses, and source-only archive contents
- Rust workspace tests and cargo publish dry-run on <OS/ARCH>
- C/C++, Java/JNI, and Python tests on <OS/ARCH>
- Java Nexus and TestPyPI smoke tests
```
