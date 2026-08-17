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
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class ReleaseScriptTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp_dir.name)
        (self.repo / "tools").mkdir()
        (self.repo / "core").mkdir()
        (self.repo / "ffi").mkdir()
        (self.repo / "jni").mkdir()
        (self.repo / "python").mkdir()
        (self.repo / "java").mkdir()

        shutil.copy(
            REPO_ROOT / "tools" / "update_branch_version.sh",
            self.repo / "tools" / "update_branch_version.sh",
        )
        shutil.copy(
            REPO_ROOT / "tools" / "create_release_branch.sh",
            self.repo / "tools" / "create_release_branch.sh",
        )

        self._write(
            "Cargo.toml",
            """
[workspace]
members = ["core", "ffi", "jni"]
resolver = "2"
""",
        )
        self._write(".gitignore", "/target/\n")
        self._write(
            "core/Cargo.toml",
            """
[package]
name = "paimon-vindex-core"
version = "0.4.0"
edition = "2021"
""",
        )
        self._write("core/src/lib.rs", "")
        self._write(
            "ffi/Cargo.toml",
            """
[package]
name = "paimon-vindex-ffi"
version = "0.4.0"
edition = "2021"

[dependencies]
paimon-vindex-core = { path = "../core", version = "0.4.0" }
""",
        )
        self._write("ffi/src/lib.rs", "")
        self._write(
            "jni/Cargo.toml",
            """
[package]
name = "paimon-vindex-jni"
version = "0.4.0"
edition = "2021"

[dependencies]
paimon-vindex-core = { path = "../core", version = "0.4.0" }
""",
        )
        self._write("jni/src/lib.rs", "")
        self._write(
            "python/pyproject.toml",
            """
[project]
name = "paimon-vindex"
version = "0.4.0"
""",
        )
        self._write(
            "java/pom.xml",
            """
<project>
  <modelVersion>4.0.0</modelVersion>
  <groupId>org.apache.paimon</groupId>
  <artifactId>paimon-vector-index-java</artifactId>
  <version>0.4.0-SNAPSHOT</version>
</project>
""",
        )

        self._run(["git", "init", "-q"])
        self._run(["git", "config", "user.name", "Release Script Test"])
        self._run(["git", "config", "user.email", "release-script@example.com"])
        self._run(["cargo", "generate-lockfile"])
        self._run(["git", "add", "."])
        self._run(["git", "commit", "-q", "-m", "initial"])

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_update_branch_version_keeps_workspace_versions_consistent(self):
        env = os.environ.copy()
        env.update(
            {
                "OLD_VERSION": "0.4.0",
                "NEW_VERSION": "0.5.0-SNAPSHOT",
            }
        )
        self._run(
            ["bash", "update_branch_version.sh"],
            cwd=self.repo / "tools",
            env=env,
        )

        self.assertFalse(
            (self.repo / "target").exists(),
            "version updates should not compile the workspace",
        )
        self.assertIn(
            'version = "0.5.0"',
            (self.repo / "core" / "Cargo.toml").read_text(),
        )
        for crate in ("ffi", "jni"):
            manifest = (self.repo / crate / "Cargo.toml").read_text()
            self.assertIn('version = "0.5.0"', manifest)
            self.assertIn(
                'paimon-vindex-core = { path = "../core", version = "0.5.0" }',
                manifest,
            )
        self.assertIn(
            'version = "0.5.0"',
            (self.repo / "python" / "pyproject.toml").read_text(),
        )
        self.assertIn(
            "<version>0.5.0-SNAPSHOT</version>",
            (self.repo / "java" / "pom.xml").read_text(),
        )

        lockfile = (self.repo / "Cargo.lock").read_text()
        for package in (
            "paimon-vindex-core",
            "paimon-vindex-ffi",
            "paimon-vindex-jni",
        ):
            self.assertIn(
                f'name = "{package}"\nversion = "0.5.0"',
                lockfile,
            )

        subject = self._run(
            ["git", "log", "-1", "--format=%s"],
            capture_output=True,
        ).stdout.strip()
        self.assertEqual("Update version to 0.5.0-SNAPSHOT", subject)
        self.assertEqual(
            "",
            self._run(
                ["git", "status", "--short"],
                capture_output=True,
            ).stdout.strip(),
        )

    def test_create_release_branch_uses_minor_release_line(self):
        env = os.environ.copy()
        env["RELEASE_VERSION"] = "0.4.0"
        self._run(
            ["bash", "create_release_branch.sh"],
            cwd=self.repo / "tools",
            env=env,
        )

        branch = self._run(
            ["git", "branch", "--show-current"],
            capture_output=True,
        ).stdout.strip()
        self.assertEqual("release-0.4", branch)

    def test_update_branch_version_reports_missing_old_version(self):
        env = os.environ.copy()
        env.pop("OLD_VERSION", None)
        env["NEW_VERSION"] = "0.5.0-SNAPSHOT"
        result = self._run(
            ["bash", "update_branch_version.sh"],
            cwd=self.repo / "tools",
            env=env,
            check=False,
            capture_output=True,
        )

        self.assertNotEqual(0, result.returncode)
        self.assertIn(
            "OLD_VERSION is unset",
            result.stdout + result.stderr,
        )

    def test_create_release_branch_reports_missing_release_version(self):
        env = os.environ.copy()
        env.pop("RELEASE_VERSION", None)
        result = self._run(
            ["bash", "create_release_branch.sh"],
            cwd=self.repo / "tools",
            env=env,
            check=False,
            capture_output=True,
        )

        self.assertNotEqual(0, result.returncode)
        self.assertIn(
            "RELEASE_VERSION is unset",
            result.stdout + result.stderr,
        )

    def _write(self, relative_path, content):
        path = self.repo / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content.lstrip())

    def _run(self, command, cwd=None, env=None, check=True, capture_output=False):
        return subprocess.run(
            command,
            cwd=cwd or self.repo,
            env=env,
            check=check,
            text=True,
            capture_output=capture_output,
        )


if __name__ == "__main__":
    unittest.main()
