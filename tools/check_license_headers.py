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

"""Validate ASF license headers on source text files."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ASF_HEADER_TOKEN = "Licensed to the Apache Software Foundation"

EXEMPT_FILES = {
    "Cargo.lock",
    # Generated dependency license reports; adding comments changes the TSV format.
    "DEPENDENCIES.rust.tsv",
    "core/DEPENDENCIES.rust.tsv",
    "ffi/DEPENDENCIES.rust.tsv",
    "jni/DEPENDENCIES.rust.tsv",
    "LICENSE",
    "LICENSE-binary",
    "LICENSE-binary-ffi",
    "NOTICE",
    "core/LICENSE",
    "core/NOTICE",
    # Golden test data; adding comments changes the fixture format.
    "core/tests/fixtures/ivf_flat_v1.hex",
    "core/tests/fixtures/ivf_sq_v1.hex",
    "core/tests/fixtures/ivf_pq_4bit_v1.hex",
    "core/tests/fixtures/ivf_pq_v1.hex",
    "core/tests/fixtures/ivf_rq_v1.hex",
    "core/tests/fixtures/diskann_v1.hex",
    "core/tests/fixtures/diskann_compact_multipage_v1.hex",
    "core/tests/fixtures/diskann_interleaved_4bit_v1.hex",
    "core/tests/fixtures/diskann_raw_row_ids_v1.hex",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def source_files(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and Path(result.stdout.strip()).resolve() == root:
        output = subprocess.check_output(["git", "-C", str(root), "ls-files"], text=True)
        return output.splitlines()

    return sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    )


def is_text_file(path: Path) -> bool:
    return b"\0" not in path.read_bytes()


def has_asf_header(path: Path) -> bool:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return ASF_HEADER_TOKEN in "\n".join(lines[:80])


def main() -> int:
    root = repo_root()
    missing = []

    for file_name in source_files(root):
        if file_name in EXEMPT_FILES:
            continue

        path = root / file_name
        if not path.is_file() or not is_text_file(path):
            continue

        if not has_asf_header(path):
            missing.append(file_name)

    if missing:
        print("Files missing ASF license headers:", file=sys.stderr)
        for file_name in missing:
            print(f"  {file_name}", file=sys.stderr)
        return 1

    print("All source text files have ASF license headers or are explicitly exempt.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
