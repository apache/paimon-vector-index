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
import csv
import importlib.util
from pathlib import Path
import tempfile
import subprocess
import sys
import time
import unittest


SPEC = importlib.util.spec_from_file_location(
    "benchmark_pr", Path(__file__).resolve().parents[1] / "benchmark_pr.py"
)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def row(index="IVF_FLAT"):
    values = dict.fromkeys(bench.CASE_FIELDS, "1")
    values.update({field: "100" for field, _, _ in bench.METRICS})
    values.update(index=index, nq="10", recall_at_10="0.95",
                  steady_sequential_ms="100", steady_batch_ms="100",
                  steady_sequential_queries="100", steady_batch_queries="100")
    return values


def samples():
    return {index: {side: [row(index), row(index)] for side in ("base", "candidate")}
            for index in bench.INDEXES}


class BenchmarkComparisonTest(unittest.TestCase):
    def test_medians_units_and_recall_percentage_points(self):
        values = samples()
        values["IVF_FLAT"]["base"][1]["steady_sequential_qps"] = "300"
        values["IVF_FLAT"]["candidate"][0]["steady_sequential_qps"] = "200"
        values["IVF_FLAT"]["candidate"][1]["steady_sequential_qps"] = "400"
        for value in values["IVF_FLAT"]["candidate"]:
            value["recall_at_10"] = "0.90"
        result = bench.summarize(values, 2)["IVF_FLAT"]
        self.assertEqual(result["steady_sequential_qps"]["base"]["median"], 200)
        self.assertEqual(result["steady_sequential_qps"]["delta"], "+50.0%")
        self.assertEqual(result["recall_at_10"]["delta"], "-5.00 pp")
        self.assertAlmostEqual(result["file_bytes"]["base"]["median"], 100 / (1024 * 1024))

    def test_incomplete_or_mismatched_workloads_fail(self):
        for change in ("missing", "shape", "parameters"):
            with self.subTest(change=change):
                values = samples()
                if change == "missing":
                    values["IVF_FLAT"]["candidate"].pop()
                else:
                    field = "nq" if change == "shape" else "nprobe"
                    values["IVF_FLAT"]["candidate"][0][field] = "999"
                with self.assertRaises(ValueError):
                    bench.summarize(values, 2)

    def test_csv_rejects_missing_duplicate_and_invalid_results(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            good = row()
            bad_values = []
            for field, value in (("steady_batch_qps", "nan"), ("build_ms", "-1"),
                                 ("recall_at_10", "1.1"), ("nq", "0"),
                                 ("steady_batch_qps", "0"), ("index", "DISKANN")):
                bad = copy.copy(good)
                bad[field] = value
                bad_values.append([bad])
            for rows in ([], [good, good], *bad_values):
                with self.subTest(rows=rows):
                    with path.open("w", newline="") as file:
                        writer = csv.DictWriter(file, fieldnames=good)
                        writer.writeheader()
                        writer.writerows(rows)
                    with self.assertRaises(ValueError):
                        bench.read_sample(path, "IVF_FLAT")
            with path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=good)
                writer.writeheader()
                writer.writerow(good)
            self.assertEqual(bench.read_sample(path, "IVF_FLAT"), good)

    def test_parse_errors_identify_file_field_and_value(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.csv"
            for field, value in (("steady_batch_qps", None), ("steady_batch_qps", ""),
                                 ("steady_batch_qps", "oops"), ("nq", "2.5")):
                with self.subTest(field=field, value=value):
                    sample = row()
                    if value is None:
                        del sample[field]
                    else:
                        sample[field] = value
                    with path.open("w", newline="") as file:
                        writer = csv.DictWriter(file, fieldnames=sample)
                        writer.writeheader()
                        writer.writerow(sample)
                    with self.assertRaises(ValueError) as caught:
                        bench.read_sample(path, "IVF_FLAT")
                    self.assertIn(str(path), str(caught.exception))
                    self.assertIn(field, str(caught.exception))
                    if value:
                        self.assertIn(value, str(caught.exception))

    def test_total_budget_stops_before_start_and_limits_running_process(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "must-not-exist"
            with self.assertRaisesRegex(TimeoutError, "total time budget"):
                bench.run_checked([sys.executable, "-c",
                                   "from pathlib import Path; Path(__import__('sys').argv[1]).touch()",
                                   str(marker)], deadline=time.monotonic() - 1, timeout=10)
            self.assertFalse(marker.exists())
        started = time.monotonic()
        with self.assertRaisesRegex(TimeoutError, "remaining total budget"):
            bench.run_checked([sys.executable, "-c", "import time; time.sleep(10)"],
                              deadline=started + 0.1, timeout=10,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.assertLess(time.monotonic() - started, 3)

    def test_short_or_partial_steady_measurements_fail(self):
        for field, value in (("steady_batch_ms", "0"), ("steady_sequential_queries", "11")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                sample = row()
                sample[field] = value
                path = Path(directory) / "incomplete.csv"
                with path.open("w", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=sample)
                    writer.writeheader()
                    writer.writerow(sample)
                with self.assertRaisesRegex(ValueError, "incomplete steady_"):
                    bench.read_sample(path, "IVF_FLAT")

    def test_zero_baseline_is_not_a_false_percentage(self):
        self.assertEqual(bench.delta_text(0, 1, "lower"), "n/a (base=0)")
        self.assertEqual(bench.delta_text(0, 0, "lower"), "0.0%")

    def test_loose_alert_boundaries(self):
        for candidate, expected in ((110, 0), (90, 0), (89, 1), (80, 1), (79, 2)):
            entry = dict(direction="higher", base={"median": 100}, candidate={"median": candidate})
            self.assertEqual(bench.query_alert_level(entry), expected)
        for candidate, expected in ((0.94, 0), (0.939, 1), (0.92, 1), (0.919, 2)):
            entry = dict(direction="recall", base={"median": 0.95}, candidate={"median": candidate})
            self.assertEqual(bench.query_alert_level(entry), expected)

        with self.assertRaises(ValueError):
            bench.query_alert_level(dict(direction="lower", base={"median": 100}, candidate={"median": 120}))


if __name__ == "__main__":
    unittest.main()
