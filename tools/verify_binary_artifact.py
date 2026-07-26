#!/usr/bin/env python3

#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Verify legal files and native resources in binary convenience artifacts."""

import argparse
import glob
import json
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
LEGAL_FILES = ("LICENSE", "NOTICE", "LICENSE-binary")
WHEEL_LEGAL_SOURCES = {
    "LICENSE": "LICENSE",
    "NOTICE": "NOTICE",
    "LICENSE-binary": "LICENSE-binary-ffi",
}
JAR_LEGAL_SOURCES = {
    "LICENSE-binary": "LICENSE-binary",
}
JAR_NATIVE_FILES = (
    "native/linux/x86_64/libpaimon_vindex_jni.so",
    "native/linux/aarch64/libpaimon_vindex_jni.so",
    "native/macos/aarch64/libpaimon_vindex_jni.dylib",
    "native/windows/x86_64/paimon_vindex_jni.dll",
)


def resolve_artifacts(patterns):
    artifacts = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise ValueError(f"artifact pattern matched no files: {pattern}")
        artifacts.extend(Path(match) for match in matches)
    return artifacts


def require_entries(archive, artifact, entries):
    names = set(archive.namelist())
    missing = [entry for entry in entries if entry not in names]
    if missing:
        raise ValueError(f"{artifact} is missing entries: {', '.join(missing)}")


def require_canonical_content(archive, artifact, archive_prefix, sources):
    for name, source in sources.items():
        entry = f"{archive_prefix}/{name}"
        expected = (ROOT / source).read_bytes()
        actual = archive.read(entry)
        if actual != expected:
            raise ValueError(f"{artifact}!/{entry} does not match repository {source}")


def verify_wheel(artifact):
    with zipfile.ZipFile(artifact) as archive:
        names = archive.namelist()
        dist_info_metadata = [
            name for name in names if name.endswith(".dist-info/METADATA")
        ]
        if len(dist_info_metadata) != 1:
            raise ValueError(
                f"{artifact} must contain exactly one .dist-info/METADATA entry"
            )
        dist_info = dist_info_metadata[0][: -len("METADATA")]
        legal_entries = [f"{dist_info}licenses/{name}" for name in LEGAL_FILES]
        package_legal_entries = [f"paimon_vindex/{name}" for name in LEGAL_FILES]
        require_entries(archive, artifact, legal_entries + package_legal_entries)
        require_canonical_content(
            archive, artifact, "paimon_vindex", WHEEL_LEGAL_SOURCES
        )
        require_canonical_content(
            archive,
            artifact,
            f"{dist_info}licenses".rstrip("/"),
            WHEEL_LEGAL_SOURCES,
        )

        native_names = [
            name
            for name in names
            if name.startswith("paimon_vindex/")
            and name.endswith((".so", ".dylib", ".dll"))
        ]
        if len(native_names) != 1:
            raise ValueError(
                f"{artifact} must contain exactly one platform native library, found "
                f"{len(native_names)}"
            )

    print(f"Verified Python wheel: {artifact}")


def verify_jar(artifact, native_files):
    with zipfile.ZipFile(artifact) as archive:
        required = (
            "META-INF/LICENSE",
            "META-INF/NOTICE",
            "META-INF/LICENSE-binary",
            "org/apache/paimon/index/vector/NativeLibraryLoader.class",
        ) + tuple(native_files)
        require_entries(archive, artifact, required)
        require_canonical_content(
            archive,
            artifact,
            "META-INF",
            JAR_LEGAL_SOURCES,
        )

    print(f"Verified Java JAR: {artifact}")


def verify_source_legal_inventory():
    metadata = json.loads(
        subprocess.check_output(
            ["cargo", "metadata", "--format-version", "1", "--locked"],
            cwd=ROOT,
            text=True,
        )
    )
    packages = {package["id"]: package for package in metadata["packages"]}
    nodes = {node["id"]: node for node in metadata["resolve"]["nodes"]}
    for root_name, license_name in (
        ("paimon-vindex-ffi", "LICENSE-binary-ffi"),
        ("paimon-vindex-jni", "LICENSE-binary"),
    ):
        root_id = next(
            package["id"]
            for package in metadata["packages"]
            if package["source"] is None and package["name"] == root_name
        )
        resolved = set()
        pending = [root_id]
        while pending:
            package_id = pending.pop()
            if package_id in resolved:
                continue
            resolved.add(package_id)
            for dependency in nodes[package_id]["deps"]:
                if any(
                    kind["kind"] in (None, "normal")
                    for kind in dependency["dep_kinds"]
                ):
                    pending.append(dependency["pkg"])

        binary_license = (ROOT / license_name).read_text(encoding="utf-8")
        missing = []
        for package_id in resolved:
            package = packages[package_id]
            if package["source"] is None:
                continue
            expression = package.get("license") or ""
            needs_additional_text = (
                "Apache-2.0" not in expression or " AND " in expression
            )
            marker = f"{package['name']} {package['version']}"
            if needs_additional_text and marker not in binary_license:
                missing.append(f"{marker} ({expression})")

        if missing:
            raise ValueError(
                f"{license_name} does not account for {root_name} runtime dependencies: "
                + ", ".join(sorted(missing))
            )

    for name in (
        "LICENSE",
        "NOTICE",
        "LICENSE-binary",
        "LICENSE-binary-ffi",
        "ffi/DEPENDENCIES.rust.tsv",
        "jni/DEPENDENCIES.rust.tsv",
    ):
        if not (ROOT / name).is_file():
            raise ValueError(f"repository legal file is missing: {name}")

    print("Verified binary legal inventory against native Rust dependencies")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", nargs="+", default=[], help="Wheel path or glob")
    parser.add_argument("--jar", nargs="+", default=[], help="JAR path or glob")
    parser.add_argument(
        "--jar-native",
        action="append",
        choices=JAR_NATIVE_FILES,
        help=(
            "Native entry required in a JAR. Repeat for a partial-platform CI build; "
            "the default requires every release platform."
        ),
    )
    parser.add_argument(
        "--source",
        action="store_true",
        help="Verify LICENSE-binary against native Rust runtime dependencies",
    )
    args = parser.parse_args()

    if not args.wheel and not args.jar and not args.source:
        parser.error("at least one --source, --wheel, or --jar is required")

    try:
        if args.source:
            verify_source_legal_inventory()
        for artifact in resolve_artifacts(args.wheel):
            verify_wheel(artifact)
        for artifact in resolve_artifacts(args.jar):
            verify_jar(artifact, args.jar_native or JAR_NATIVE_FILES)
    except (OSError, ValueError, zipfile.BadZipFile, KeyError) as error:
        print(f"Binary artifact verification failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
