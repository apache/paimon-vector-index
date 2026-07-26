<!--
  ~ Licensed to the Apache Software Foundation (ASF) under one
  ~ or more contributor license agreements.  See the NOTICE file
  ~ distributed with this work for additional information
  ~ regarding copyright ownership.  The ASF licenses this file
  ~ to you under the Apache License, Version 2.0 (the
  ~ "License"); you may not use this file except in compliance
  ~ with the License.  You may obtain a copy of the License at
  ~
  ~ http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing,
  ~ software distributed under the License is distributed on an
  ~ "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  ~ KIND, either express or implied.  See the License for the
  ~ specific language governing permissions and limitations
  ~ under the License.
  -->

# Release tools

This directory contains helper scripts used by release managers and committers.

## ANN-Benchmarks dataset conversion

`convert_ann_benchmarks.py` converts a dense
[ANN-Benchmarks](https://github.com/erikbern/ann-benchmarks) HDF5 file into
base/query `fvecs` files and the published-neighbor `ivecs` file accepted by
`core/benches/ann_bench.rs`.

The script requires `h5py` and `numpy`. For example:

```bash
python3 tools/convert_ann_benchmarks.py \
  gist-960-euclidean.hdf5 /data/gist1m \
  --prefix gist1m --query-limit 1000
```

The optional query limit applies to both queries and ground truth. It does not
truncate the indexed base vectors.

Angular/cosine datasets can be converted for the benchmark's common L2 path by
normalizing base and query vectors:

```bash
python3 tools/convert_ann_benchmarks.py \
  glove-100-angular.hdf5 /data/glove100 \
  --prefix glove100 --query-limit 1000 --normalize-l2
```

For non-zero vectors, squared L2 distance after unit normalization has the same
neighbor ordering as cosine distance. Published neighbor IDs are copied
unchanged. Conversion fails if a vector is zero-length or has a non-finite
norm.

## Java staging deploy

`deploy_java_staging.sh` deploys the Java release candidate artifacts to Apache
Nexus staging from a committer/RM machine.

GitHub Actions does **not** sign or deploy the Java staging artifacts. The
release workflow only:

1. builds the four JNI native libraries;
2. packages the multi-platform JAR and smoke-tests the final JAR without an
   external native path on all four supported platform/architecture pairs; and
3. uploads the native libraries plus the verified `java-package` JARs as
   workflow artifacts.

The committer then runs this script locally. The script checks that the release
workflow run succeeded for the current RC tag, downloads the native libraries,
and `java-package`, verifies their platform formats and legal files, places the
native libraries into the Java resource tree, and runs Maven locally. Both the
CI-generated JAR and the locally staged JAR must pass the bundled-native loader
smoke test before deployment.

### Required local setup

- `gh` GitHub CLI, authenticated with access to `apache/paimon-vector-index`;
- JDK and Maven;
- local GPG setup for the release signing key;
- Maven credentials for server id `apache.releases.https`.

Maven credentials can be supplied by one of these methods:

- configure `~/.m2/settings.xml`;
- pass `--maven-settings /path/to/settings.xml`;
- set `NEXUS_STAGE_DEPLOYER_USER` and `NEXUS_STAGE_DEPLOYER_PW` so the script can
  create a temporary Maven settings file.

### Pre-flight checks

Run these checks before the first dry-run:

```bash
gh auth status
gpg --list-secret-keys --keyid-format LONG
mvn --version
```

Confirm the signing key's public key is already published in Paimon KEYS:

```text
https://downloads.apache.org/paimon/KEYS
```

Confirm Maven can use server id `apache.releases.https`. A typical
`~/.m2/settings.xml` entry is:

```xml
<settings>
  <servers>
    <server>
      <id>apache.releases.https</id>
      <username>YOUR_NEXUS_TOKEN_USER</username>
      <password>YOUR_NEXUS_TOKEN_PASSWORD</password>
    </server>
  </servers>
</settings>
```

The Nexus token is from:

```text
https://repository.apache.org/ -> Profile -> User Token
```

### Find the run id

After pushing the RC tag, open the GitHub Actions run for that RC tag. Use the
`Release` workflow run triggered by the tag, for example `v0.3.0-rc1`.

The run id is the number in the workflow run URL:

```text
https://github.com/apache/paimon-vector-index/actions/runs/12345678901
```

The run id is:

```text
12345678901
```

Do not use the job id, artifact id, PR number, or commit SHA. The script checks
that this run completed successfully and that the run's commit matches the RC tag
checked out locally.

### Parameters

Required for the normal release flow:

- `--release-version 0.3.0`: Java artifact version in `java/pom.xml`. This does
  not include the RC suffix.
- `--rc 1`: RC number. Together with `--release-version`, this derives the tag
  `v0.3.0-rc1`.
- `--run-id 12345678901`: GitHub Actions run id from the RC tag's `Release`
  workflow URL. The script uses it to download the four `native-*` artifacts.

Common options:

- `--dry-run`: verify locally without signing or deploying to Nexus.
- `--maven-settings FILE`: use a specific Maven `settings.xml` containing server
  id `apache.releases.https`.
- `--staging-description TEXT`: override the Nexus staging description.
- `--no-skip-tests`: run Maven tests during dry-run or deploy.

Less common options:

- `--tag TAG`: use an explicit RC tag instead of deriving `vVERSION-rcN`.
- `--repo OWNER/REPO`: GitHub repository for `gh`; defaults to
  `apache/paimon-vector-index`.
- `--no-cleanup`: keep `java/src/main/resources/native` after the script exits.
- `--skip-native-file-check`: skip native binary format checks.

The last option is an escape hatch. Avoid it for normal releases.

### Dry-run before publishing

Always run a dry-run first with the real RC workflow artifacts:

```bash
./tools/deploy_java_staging.sh \
  --release-version 0.3.0 \
  --rc 1 \
  --run-id 12345678901 \
  --dry-run
```

Dry-run mode validates the GitHub Actions run id, downloads the native
libraries and `java-package`, and runs:

```bash
mvn clean verify -Prelease -Dgpg.skip=true -DskipTests
```

It does not sign and does not deploy to Nexus. It verifies:

- `java/pom.xml` version matches `--release-version`;
- current checkout matches the RC tag, such as `v0.3.0-rc1`;
- Java package inputs have no local changes;
- the GitHub Actions run is a successful tag-push `Release` workflow run and its
  commit matches the RC tag;
- all four native libraries are present;
- native library file formats match their target platforms;
- both the CI-generated and locally packaged JARs contain binary legal files,
  the four native libraries, and `NativeLibraryLoader`;
- both JARs load the current platform library without an external native path;
- the Java jar, sources jar, and javadoc jar are produced;
- the Java jar contains all four native library entries.

### Deploy to Nexus staging

After the dry-run succeeds, run the same command without `--dry-run`:

```bash
./tools/deploy_java_staging.sh \
  --release-version 0.3.0 \
  --rc 1 \
  --run-id 12345678901
```

The script repeats the local preflight before creating any remote staging
artifacts:

```bash
mvn clean verify -Prelease -Dgpg.skip=true -DskipTests
```

After that passes, it runs the local Nexus staging deploy:

```bash
mvn deploy -Prelease -DskipTests \
  -DstagingDescription="Apache Paimon Vector Index, version 0.3.0, release candidate 1"
```

The Maven output contains the Nexus staging repository id, for example:

```text
orgapachepaimon-XXXX
```

Use that id in the release vote email.
