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
#   Unless required by applicable law or agreed to in writing,
#   software distributed under the License is distributed on an
#   "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#   KIND, either express or implied.  See the License for the
#   specific language governing permissions and limitations
#   under the License.

import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("tune_ivfrq", Path(__file__).parents[1] / "tune_ivfrq.py")
tune = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tune)


def candidate(bits, recall, qps, size=1000):
    return dict(rq_bits=bits, recall_at_10=recall, steady_batch_qps=qps,
                file_bytes=size, n=100, nprobe=32)


class TuneIvfRqTests(unittest.TestCase):
    def test_fastest_configuration_must_meet_both_targets(self):
        low_recall = candidate(3, 0.8999, 900)
        oversized = candidate(6, 0.99, 800, size=2000)
        valid = candidate(4, 0.9, 700)
        self.assertEqual(tune.select_candidate([low_recall, oversized, valid], 0.9, 10), valid)
        with self.assertRaisesRegex(ValueError, "no measured configuration"):
            tune.select_candidate([low_recall, oversized], 0.9, 10)

    def test_nonfinite_or_zero_throughput_cannot_win(self):
        valid = candidate(4, 0.91, 100)
        invalid = [candidate(3, 0.99, float("nan")), candidate(3, 0.99, 0),
                   candidate(3, float("nan"), 1000)]
        self.assertEqual(tune.select_candidate(invalid + [valid], 0.9), valid)

    def test_query_splits_are_disjoint_and_do_not_modify_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "queries.fvecs"
            records = [struct.pack("<if", 1, float(row)) for row in range(7)]
            source.write_bytes(b"".join(records))
            calibration, validation = root / "calibration.fvecs", root / "validation.fvecs"
            tune.copy_records(source, calibration, 0, 2)
            tune.copy_records(source, validation, 2, 5)
            self.assertEqual(calibration.read_bytes(), b"".join(records[:2]))
            self.assertEqual(validation.read_bytes(), b"".join(records[2:]))
            with self.assertRaises(ValueError):
                tune.copy_records(source, source, 0, 2)
            self.assertEqual(source.read_bytes(), b"".join(records))

    def test_validation_keeps_default_when_calibration_gain_does_not_repeat(self):
        default = candidate(4, 0.91, 100)
        slower = candidate(5, 0.92, 99, size=1200)
        selected, verification = tune.select_verified_configuration(default, slower, 0.9)
        self.assertIs(selected, default)
        self.assertTrue(verification["candidate_recall_validated"])
        self.assertFalse(verification["performance_validated"])
        faster = candidate(5, 0.92, 110, size=1200)
        self.assertIs(tune.select_verified_configuration(default, faster, 0.9)[0], faster)

    def test_validation_requires_a_configuration_that_meets_the_targets(self):
        default = candidate(4, 0.88, 100)
        adequate = candidate(5, 0.93, 80)
        self.assertIs(tune.select_verified_configuration(default, adequate, 0.9)[0], adequate)
        self.assertIsNone(tune.select_verified_configuration(default, candidate(3, 0.8, 200), 0.9)[0])

    def test_same_configuration_is_aa_noise_and_cannot_claim_a_gain(self):
        default = candidate(4, 0.91, 100)
        same = candidate(4, 0.91, 120)
        selected, verification = tune.select_verified_configuration(default, same, 0.9)
        self.assertIs(selected, default)
        self.assertTrue(verification["same_configuration"])
        self.assertFalse(verification["performance_validated"])

    def test_failed_rerun_replaces_the_previous_successful_recommendation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vectors = root / "vectors.fvecs"
            vectors.write_bytes(struct.pack("<if", 1, 1.0) * 4)
            benchmark = root / "fake_benchmark"
            benchmark.write_text(f"#!{sys.executable}\n" + '''
import csv, os, pathlib, sys
if "ANN_REUSE_INDEX_PATH" not in os.environ:
    path = pathlib.Path(os.environ["ANN_OUTPUT_DIR"]) / str(os.getpid())
    path.mkdir()
    (path / "ivf_rq.index").write_bytes(b"fixture")
row = dict(n=4, rq_bits=int(os.environ["ANN_RQ_BITS"]), nprobe=int(os.environ["ANN_NPROBE"]),
           file_bytes=64, recall_at_10=0.5, steady_batch_qps=100,
           steady_sequential_qps=90, steady_sequential_p95_us=10)
writer = csv.DictWriter(sys.stdout, fieldnames=row.keys())
writer.writeheader()
writer.writerow(row)
''')
            benchmark.chmod(0o755)
            output = root / "results"
            command = [sys.executable, str(Path(tune.__file__)),
                "--benchmark-executable", str(benchmark), "--base-fvecs", str(vectors),
                "--query-fvecs", str(vectors), "--ground-truth-ivecs", str(vectors),
                "--output-dir", str(output), "--bits", "3", "--nprobes", "16,32",
                "--nlist", "64", "--calibration-queries", "2", "--repetitions", "1"]
            successful = subprocess.run(command + ["--target-recall", "0.4"], capture_output=True, text=True)
            self.assertEqual(successful.returncode, 0, successful.stderr)
            self.assertTrue(json.loads((output / "recommendation.json").read_text())["validated"])
            failed = subprocess.run(command + ["--target-recall", "0.9"], capture_output=True, text=True)
            self.assertEqual(failed.returncode, 2, failed.stderr)
            report = json.loads((output / "recommendation.json").read_text())
            self.assertFalse(report["validated"])
            self.assertIsNone(report["options"])
            self.assertEqual(report["status"], "no_calibration_candidate")


if __name__ == "__main__":
    unittest.main()
