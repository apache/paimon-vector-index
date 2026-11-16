# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Fail release jobs before publishing if their source versions do not match."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET


def check_versions(root, ref):
    match = re.fullmatch(
        r"refs/tags/v((?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*))(?:-rc([1-9][0-9]*))?",
        ref,
    )
    if not match:
        raise ValueError("release requires a vX.Y.Z or vX.Y.Z-rcN tag")
    version = match.group(1)
    # RC tags vote on the exact source later tagged as final. Cargo, Maven,
    # and the source pyproject retain X.Y.Z; wheel builds inject X.Y.ZrcN.
    result = subprocess.run(
        ["cargo", "metadata", "--offline", "--no-deps", "--format-version", "1"],
        cwd=root, text=True, capture_output=True, check=True,
    )
    metadata = json.loads(result.stdout)
    members = set(metadata["workspace_members"])
    packages = [p for p in metadata["packages"] if p["id"] in members]
    if not packages:
        raise ValueError("Cargo workspace has no packages")
    names = {p["name"] for p in packages}
    versions = {"Cargo " + p["name"]: p["version"] for p in packages}
    for package in packages:
        for dependency in package["dependencies"]:
            if dependency["name"] in names and dependency.get("path"):
                versions[package["name"] + " -> " + dependency["name"]] = dependency["req"].removeprefix("^")
    text = (root / "python/pyproject.toml").read_text()
    project = re.search(r"(?ms)^\[project\]\s*\n(.*?)(?=^\[|\Z)", text)
    python_versions = re.findall(r'^version\s*=\s*"([^"]+)"\s*$', project.group(1) if project else "", re.M)
    if len(python_versions) != 1:
        raise ValueError("expected one [project].version in python/pyproject.toml")
    versions["Python"] = python_versions[0]
    pom = ET.parse(root / "java/pom.xml").getroot()
    versions["Maven"] = pom.findtext("{*}version")
    mismatches = [f"{name}: {actual!r} (expected {version})" for name, actual in versions.items() if actual != version]
    if mismatches:
        raise ValueError("release version mismatch:\n" + "\n".join(mismatches))
    return version


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--ref", default=os.environ.get("GITHUB_REF", ""))
    args = parser.parse_args()
    try:
        version = check_versions(args.root.resolve(), args.ref)
    except (ValueError, OSError, subprocess.CalledProcessError, ET.ParseError) as error:
        print(f"Release preflight failed: {error}", file=sys.stderr)
        return 1
    print(f"Release preflight passed: {args.ref}, source version {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
