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

import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location("benchmark_pq_full", TOOLS / "benchmark_pq_full.py")
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


class PqSuiteTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.output = self.root / "out"
        self.output.mkdir()
        self.dataset = self.root / "fashion"
        self.dataset.mkdir()
        for name in ["base.fvecs", "query.fvecs", "ground_truth.ivecs"]:
            (self.dataset / ("fashion_" + name)).write_bytes(b"dataset-v1")
        self.checkouts = {}
        self.executables = {}
        for side in ["base", "candidate"]:
            checkout = self.root / side
            self.checkouts[side] = checkout
            (checkout / "core/src").mkdir(parents=True)
            (checkout / "core/src/lib.rs").write_text("// source\n")
            (checkout / "core/benches").mkdir()
            (checkout / "core/benches/ivfpq_filter_scan_bench.rs").write_text("// driver\n")
            (checkout / "Cargo.toml").write_text("[profile.release]\nopt-level = 3\n")
            executable = self.root / (side + "-bench")
            executable.write_bytes(b"binary-v1")
            self.executables[side] = executable
            (self.output / f"build-{side}.jsonl").write_text(json.dumps(dict(
                reason="compiler-artifact", executable=str(executable),
                target=dict(kind=["bench"], name="ivfpq_filter_scan_bench"))) + "\n")
        self.plan = [dict(case="filter-b4", bench="ivfpq_filter_scan_bench", threads=1,
                          round=i, side=side, options={"IVFPQ_BENCH_BITS": "4"}, args=[])
                     for i in [1, 2] for side in ["base", "candidate"]]
        self.calls = []
        self.argv = ["benchmark_pq_full.py", "--base", str(self.checkouts["base"]),
                     "--candidate", str(self.checkouts["candidate"]),
                     "--output", str(self.output), "--dataset-dir", str(self.dataset)]

    @staticmethod
    def stdout():
        return "density_percent,matches_per_list,p50_ms,ids_signature,distances_signature\n" + "".join(
            f"{density},1,10.0,abcdef,123456\n"
            for density in ["0.10", "1.00", "6.25", "12.50", "18.75", "25.00", "50.00", "100.00"])

    def process(self, command, **kwargs):
        self.calls.append(command)
        kwargs["stdout"].write(self.stdout())
        kwargs["stderr"].write("1 maximum resident set size\n")

    def run_suite(self, process=None):
        with patch.object(sys, "argv", self.argv), patch.object(bench, "jobs", return_value=self.plan), \
                patch.object(bench, "run_checked", side_effect=process or self.process):
            bench.main()

    def metadata(self):
        return json.loads((self.output / "suite.json").read_text())

    def test_identical_inputs_resume_without_reexecuting_and_preserve_samples(self):
        self.run_suite()
        self.run_suite()
        self.assertEqual(len(self.calls), 4)
        summary = json.loads((self.output / "measurements.json").read_text())
        self.assertEqual(summary["completed"], summary["planned"])
        timing = next(c for c in summary["comparisons"] if c["metric"] == "p50_ms")
        self.assertEqual(timing["base_samples"], [10.0, 10.0])
        self.assertEqual(timing["candidate_samples"], [10.0, 10.0])

    def test_resume_reexecutes_only_failed_process_and_replaces_partial_output(self):
        def process(command, **kwargs):
            if not self.calls:
                self.calls.append(command)
                kwargs["stdout"].write("incomplete measurement\n")
                raise RuntimeError("process failed")
            self.process(command, **kwargs)
        with self.assertRaises(RuntimeError):
            self.run_suite(process)
        self.assertEqual(len(self.calls), 4)
        self.assertFalse((self.output / "measurements.json").exists())
        self.run_suite()
        self.assertEqual(len(self.calls), 5)
        self.assertEqual(self.calls[0], self.calls[-1])
        self.assertEqual(self.metadata()["failures"], [])
        result = json.loads((self.output / "measurements.json").read_text())
        self.assertEqual(result["completed"], 4)

    def test_resume_rejects_changed_config_binary_dataset_or_base_driver(self):
        self.run_suite()
        for path in [self.checkouts["candidate"] / "Cargo.toml", self.executables["candidate"],
                     self.dataset / "fashion_base.fvecs",
                     self.checkouts["base"] / "core/benches/ivfpq_filter_scan_bench.rs"]:
            original = path.read_bytes()
            try:
                with self.subTest(path=path):
                    path.write_bytes(original + b"changed")
                    with self.assertRaises(ValueError):
                        self.run_suite()
                    self.assertEqual(len(self.calls), 4)
            finally:
                path.write_bytes(original)

    def test_resume_rejects_changed_runtime_controls(self):
        with patch.dict(os.environ, {"VECLIB_MAXIMUM_THREADS": "1"}):
            self.run_suite()
        with patch.dict(os.environ, {"VECLIB_MAXIMUM_THREADS": "4"}):
            with self.assertRaises(ValueError):
                self.run_suite()
        with patch.dict(os.environ, {"VECLIB_MAXIMUM_THREADS": "1"}), \
                patch.object(bench.platform, "node", return_value="another-machine"):
            with self.assertRaises(ValueError):
                self.run_suite()
        self.assertEqual(len(self.calls), 4)

    def test_detects_dataset_changes_during_measurement(self):
        def process(command, **kwargs):
            self.process(command, **kwargs)
            (self.dataset / "fashion_query.fvecs").write_bytes(b"different-data")
        with self.assertRaises(RuntimeError):
            self.run_suite(process)
        self.assertFalse((self.output / "measurements.json").exists())

    def test_summarize_rejects_missing_failed_duplicate_or_mismatched_runs(self):
        self.run_suite()
        original = self.metadata()
        for kind in ["missing", "failed", "duplicate", "job"]:
            data = copy.deepcopy(original)
            if kind == "missing":
                data["runs"].pop()
            elif kind == "failed":
                data["runs"][0]["status"] = "failed"
            elif kind == "duplicate":
                data["runs"].append(data["runs"][0])
            else:
                data["runs"][0]["job"]["threads"] = 7
            with self.subTest(kind=kind):
                (self.output / "suite.json").write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    bench.summarize(self.output)

    def test_summarize_rejects_empty_or_partial_measurements_and_nonfinite_times(self):
        self.run_suite()
        paths = list((self.output / "suite-raw").glob("*.stdout"))
        for kind in ["empty", "missing-density", "duplicate-density", "missing-time", "nonfinite"]:
            for path in paths:
                path.write_text(self.stdout())
            if kind == "empty":
                for path in paths:
                    path.write_text("Benchmark exited without measurements\n")
            else:
                text = paths[0].read_text()
                if kind == "missing-density":
                    text = "\n".join(text.splitlines()[:-1]) + "\n"
                elif kind == "duplicate-density":
                    text += text.splitlines()[-1] + "\n"
                elif kind == "missing-time":
                    text = text.replace("p50_ms", "unknown_column")
                else:
                    text = text.replace("10.0", "nan")
                paths[0].write_text(text)
            with self.subTest(kind=kind):
                with self.assertRaises(ValueError):
                    bench.summarize(self.output)


if __name__ == "__main__":
    unittest.main()
