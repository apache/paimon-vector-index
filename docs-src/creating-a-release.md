---
title: "Creating a release"
description: "Prepare, vote on, and publish an Apache Paimon Vector Index release candidate."
---

<!--
  Licensed to the Apache Software Foundation (ASF) under one or more
  contributor license agreements. See the NOTICE file distributed with
  this work for additional information regarding copyright ownership.
  The ASF licenses this file to you under the Apache License, Version 2.0.
-->

# Creating a release

Prepare a reproducible source candidate, stage the Java and Python convenience artifacts, coordinate the community vote, and publish only the candidate that was approved.

## Overview {#overview}

> **Release the voted commit, unchanged** — The final `vVERSION` tag must point to the same commit as the successful `vVERSION-rcN` tag. Never rebuild or modify the source archive after the vote.

1.  Agree on the release and select a Release Manager.
2.  Prepare dependency declarations, versions, and the release branch.
3.  Sign an RC tag, run the release workflow, create the source archive, and stage Java.
4.  Upload the source candidate and call a vote lasting at least 72 hours.
5.  Fix issues and create a new RC when required.
6.  After a successful vote, promote the exact RC and release staged artifacts.
7.  Publish release notes, update this page, and announce the release.

| Component | RC tag behavior | Final tag behavior |
|----|----|----|
| Source | RM creates, signs, and uploads the archive to ASF dist dev | The approved files move unchanged to ASF dist release |
| Rust | `cargo publish --dry-run` | Publishes `paimon-vindex-core` to crates.io |
| Java | CI builds four JNI libraries; the RM signs and deploys the JAR to Nexus staging locally | The RM closes and releases the approved Nexus repository |
| Python | Builds four wheels and publishes `VERSIONrcN` to TestPyPI after Rust, Java, and wheel jobs pass | Publishes `VERSION` to PyPI |

This procedure follows the [ASF Release Policy](https://www.apache.org/legal/release-policy.html). The source archive is the release; registry packages are convenience artifacts.

## One-time Release Manager setup {#setup}

### Signing key and ASF distribution repositories

1.  Create a long-lived GPG signing key associated with your `@apache.org` identity and publish it to a public key server.
2.  Append the public key and signatures to the Paimon [KEYS](https://downloads.apache.org/paimon/KEYS) file through the ASF distribution SVN repository.
3.  Configure Git to use the same key and confirm that `git tag -s` and `gpg --verify` work.

```bash title="Shell · inspect signing setup"
gpg --list-secret-keys --keyid-format LONG
git config user.signingkey
svn --version
```

### Publishing credentials

The GitHub repository needs `CARGO_REGISTRY_TOKEN`, `TEST_PYPI_API_TOKEN`, and `PYPI_API_TOKEN`. Java is deliberately deployed from the RM machine: configure GPG plus Maven server ID `apache.releases.https`, and authenticate `gh` for access to the RC workflow artifacts. See [tools/README.md](https://github.com/apache/paimon-vector-index/blob/main/tools/README.md) for Nexus credential options.

## Prepare the release {#prepare}

Work from a fresh clone and substitute the intended values. A release branch represents a minor release line; RC attempts are tags on that branch.

```bash title="Shell · example values"
RELEASE_VERSION="0.3.0"
SHORT_RELEASE_VERSION="0.3"
NEXT_VERSION="0.4.0"
RC_NUM="1"
RELEASE_BRANCH="release-${SHORT_RELEASE_VERSION}"
RC_TAG="v${RELEASE_VERSION}-rc${RC_NUM}"
RELEASE_TAG="v${RELEASE_VERSION}"
```

### Refresh dependency declarations on main

Install `cargo-deny`, check every workspace crate, generate the dependency TSV files, review the changes, and commit them before cutting the branch.

```bash title="Shell"
git checkout main
git pull --ff-only origin main
cargo install cargo-deny
python3 tools/dependencies.py check
python3 tools/dependencies.py generate
git add DEPENDENCIES.rust.tsv \
  core/DEPENDENCIES.rust.tsv ffi/DEPENDENCIES.rust.tsv jni/DEPENDENCIES.rust.tsv
git commit -m "chore: update dependency list for release ${RELEASE_VERSION}"
git push origin main
```

### Cut the release branch and advance main

```bash title="Shell"
git checkout -b "${RELEASE_BRANCH}" origin/main
git push origin "${RELEASE_BRANCH}"

git checkout main
cd tools
OLD_VERSION="${RELEASE_VERSION}" \
NEW_VERSION="${NEXT_VERSION}-SNAPSHOT" \
  ./update_branch_version.sh
cd ..
git push origin main
```

If the current Java version already carries `-SNAPSHOT`, the helper accepts either the clean or snapshot old version. Cargo and Python use the clean SemVer value.

## Build and stage a release candidate {#candidate}

### Set release versions and tag the candidate

```bash title="Shell"
git checkout "${RELEASE_BRANCH}"
git pull --ff-only origin "${RELEASE_BRANCH}"
cd tools
OLD_VERSION="${RELEASE_VERSION}-SNAPSHOT" \
NEW_VERSION="${RELEASE_VERSION}" \
  ./update_branch_version.sh
cd ..
git push origin "${RELEASE_BRANCH}"

git tag -s "${RC_TAG}" -m "Apache Paimon Vector Index ${RELEASE_VERSION} RC${RC_NUM}"
git tag -v "${RC_TAG}"
git push origin "${RC_TAG}"
```

Wait for the tag-triggered [Release workflow](https://github.com/apache/paimon-vector-index/actions/workflows/release.yml). Rust must pass its publish dry-run; Java must build the Linux x86-64, Linux aarch64, macOS aarch64, and Windows x86-64 JNI libraries; Python must build the same platform set and publish the RC wheels to TestPyPI.

### Create and upload the source candidate

```bash title="Shell"
git checkout "${RC_TAG}"
cd tools
RELEASE_VERSION="${RELEASE_VERSION}" ./create_source_release.sh
cd ..

svn checkout https://dist.apache.org/repos/dist/dev/paimon/ \
  paimon-dist-dev --depth=immediates
mkdir "paimon-dist-dev/paimon-vector-index-${RELEASE_VERSION}-rc${RC_NUM}"
cp tools/release/apache-paimon-vector-index-${RELEASE_VERSION}-src.* \
  "paimon-dist-dev/paimon-vector-index-${RELEASE_VERSION}-rc${RC_NUM}/"
svn add "paimon-dist-dev/paimon-vector-index-${RELEASE_VERSION}-rc${RC_NUM}"
svn commit -m "Add Paimon Vector Index ${RELEASE_VERSION} RC${RC_NUM}" \
  paimon-dist-dev
```

The script archives the current commit with its stable commit timestamp, excludes development-only files, writes a reproducible gzip stream, creates an armored detached signature, and records a SHA-512 checksum. Re-running it for the same commit and release version must produce the same `.tgz` bytes.

### Stage the Java convenience artifact

Find the numeric run ID of the successful RC-tag `Release` workflow, then validate the exact CI native artifacts before any remote deploy.

```bash title="Shell · dry-run, then deploy"
./tools/deploy_java_staging.sh \
  --release-version "${RELEASE_VERSION}" \
  --rc "${RC_NUM}" \
  --run-id 12345678901 \
  --dry-run

./tools/deploy_java_staging.sh \
  --release-version "${RELEASE_VERSION}" \
  --rc "${RC_NUM}" \
  --run-id 12345678901
```

Record the resulting `orgapachepaimon-XXXX` Nexus repository ID. Leave the repository staged until the vote finishes.

## Call the vote {#vote}

Send a plain-text message to `dev@paimon.apache.org`. Following the [ASF voting process](https://www.apache.org/foundation/voting.html), the vote must remain open for at least 72 hours and pass with at least three binding +1 votes and more binding +1 than -1 votes.

```text title="Vote template"
Subject: [VOTE] Release Apache Paimon Vector Index ${RELEASE_VERSION} (RC${RC_NUM})

Please review and vote on Apache Paimon Vector Index
${RELEASE_VERSION}, release candidate ${RC_NUM}.

[ ] +1 Approve
[ ]  0 No opinion
[ ] -1 Do not approve (please explain)

Source candidate:
https://dist.apache.org/repos/dist/dev/paimon/paimon-vector-index-${RELEASE_VERSION}-rc${RC_NUM}/

Git tag and commit:
https://github.com/apache/paimon-vector-index/releases/tag/${RC_TAG}
<RC_COMMIT_SHA>

KEYS:
https://downloads.apache.org/paimon/KEYS

Java staging repository:
https://repository.apache.org/content/repositories/orgapachepaimon-XXXX/

Python RC:
https://test.pypi.org/project/paimon-vindex/${RELEASE_VERSION}rc${RC_NUM}/

Verification guide:
https://github.com/apache/paimon-vector-index/blob/${RC_TAG}/docs-src/verifying-a-release-candidate.md

The vote will remain open for at least 72 hours.
```

After the deadline, tally binding and non-binding votes separately and send `[RESULT][VOTE] Release Apache Paimon Vector Index VERSION (RCN)`.

## Replace a rejected candidate {#retry}

1.  Apply fixes to the release branch through normal review.
2.  Drop the old Nexus staging repository.
3.  Remove or retain the superseded dist-dev directory according to the vote thread, but never overwrite its files.
4.  Increment `RC_NUM`, create a new signed RC tag, rebuild every candidate artifact, and start a new vote.

## Finalize an approved release {#finalize}

### Tag and promote the exact RC

```bash title="Shell"
git checkout "${RC_TAG}"
git tag -s "${RELEASE_TAG}" \
  -m "Release Apache Paimon Vector Index ${RELEASE_VERSION}"
test "$(git rev-parse "${RC_TAG}^{commit}")" = \
     "$(git rev-parse "${RELEASE_TAG}^{commit}")"
git tag -v "${RELEASE_TAG}"
git push origin "${RELEASE_TAG}"

svn mv -m "Release Paimon Vector Index ${RELEASE_VERSION}" \
  "https://dist.apache.org/repos/dist/dev/paimon/paimon-vector-index-${RELEASE_VERSION}-rc${RC_NUM}" \
  "https://dist.apache.org/repos/dist/release/paimon/paimon-vector-index-${RELEASE_VERSION}"
```

The final tag publishes Rust to crates.io and Python to PyPI. In Nexus, close the previously recorded Java staging repository, inspect all validation rules, then release it to Maven Central. Do not stage a new Java build.

### Verify and publish release notes

- The source archive, signature, and checksum are visible through Apache mirrors.
- `paimon-vindex-core VERSION` is on crates.io.
- `org.apache.paimon:paimon-vector-index-java:VERSION` is on Maven Central.
- `paimon-vindex VERSION` is on PyPI for all supported platforms.
- A GitHub release exists for `vVERSION`, with reviewed generated notes.

## Update and announce {#announce}

Wait at least one hour after source publication, then update [the releases page](releases.md) with the mirrored source-download, notes, and registry links. After mirrors and registries have converged, announce the release from an `@apache.org` address to `dev@paimon.apache.org` and `announce@apache.org`. Remove superseded releases from the live distribution area when required by ASF policy; they remain in the Apache archive.
