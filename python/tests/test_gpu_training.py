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

import ctypes
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest

from paimon_vindex import SearchParams, VectorIndexReader, VectorIndexTrainer, VectorIndexWriter
from paimon_vindex._ffi import lib


class BytesInput:
    def __init__(self, data):
        self.data = data

    def pread_many(self, ranges):
        return [self.data[pos:pos+length] for pos, length in ranges]


def make_trainer(data, metric="l2", **extra):
    options = {"index.type": "ivf_sq", "dimension": str(data.shape[1]),
               "nlist": "4", "metric": metric, **extra}
    return VectorIndexTrainer.create(options).add_training_vectors(data)


def index_bytes(training, data):
    output = io.BytesIO()
    with VectorIndexWriter(training) as writer:
        assert writer.ivf_sq_partition_sizes().sum() == 0
        writer.add_vectors(np.arange(len(data), dtype=np.int64), data)
        assert writer.ivf_sq_partition_sizes().sum() == len(data)
        writer.write(output)
    return output.getvalue()


@pytest.mark.parametrize("metric", ["l2", "cosine", "inner_product"])
def test_prepared_cpu_roundtrip_preserves_bytes_and_cpu_results(metric):
    data = np.random.default_rng(1234).normal(size=(256, 8)).astype(np.float32)
    original = index_bytes(make_trainer(data, metric).finish_training(), data)
    trainer = make_trainer(data, metric)
    with trainer.prepare_training() as prepared:
        with pytest.raises(RuntimeError, match="closed"):
            trainer.add_training_vectors(data)
        sample = prepared.sample
        expected = data / np.linalg.norm(data, axis=1, keepdims=True) if metric == "cosine" else data
        np.testing.assert_allclose(sample, expected, atol=2e-7)
        assert not sample.flags.writeable
        # A caller-owned copy cannot mutate native SQ calibration state.
        sample.flags.writeable = True
        sample[:] = 0
        np.testing.assert_allclose(prepared.sample, expected, atol=2e-7)
        centers = prepared.fit_centroids_cpu()
        training = prepared.finish_with_ivf_centroids(centers)
        with pytest.raises(RuntimeError, match="closed"):
            prepared.fit_centroids_cpu()
    actual = index_bytes(training, data)
    assert actual == original
    with VectorIndexReader(BytesInput(actual)) as reader:
        ids, distances = reader.search(data[0], SearchParams.ivf(5, 4))
        assert len(ids) == 5
        assert np.isfinite(distances).all()
        if metric != "inner_product":
            assert ids[0] == 0


def test_prepared_respects_resolved_nlist_and_both_sample_caps():
    data = np.random.default_rng(0).normal(size=(512, 8)).astype(np.float32)
    with make_trainer(data, nlist="auto", **{"expected-vector-count": "1000000"}).prepare_training() as p:
        assert p.info.nlist == 1024  # Not inferred from the 512-row sample.
    with make_trainer(data, **{"ivf.train.max-points-per-centroid": "2"}).prepare_training() as p:
        assert p.info.sample_count == 8
        assert p.info.calibration_count == 512
        assert p.sample.shape == (8, 8)


def test_prepared_validation_and_consumption():
    data = np.ones((8, 2), dtype=np.float32)
    with make_trainer(data).prepare_training() as p:
        # Transposed shapes have the same element count but are still invalid.
        with pytest.raises(ValueError, match="shape"):
            p.finish_with_ivf_centroids(np.zeros((2, 4), dtype=np.float32))
        with pytest.raises(RuntimeError, match="Lloyd"):
            p.fit_centroids_cpu(initial_centroids=np.ones((4, 2), dtype=np.float32))
        with pytest.raises(RuntimeError, match="finite"):
            p.finish_with_ivf_centroids(np.full((4, 2), np.nan, dtype=np.float32))
        with pytest.raises(RuntimeError, match="closed"):
            p.sample
    with make_trainer(data, **{"index.type": "ivf_flat"}) as trainer:
        with pytest.raises(RuntimeError, match="IVF-SQ"):
            trainer.prepare_training()
        with pytest.raises(RuntimeError, match="closed"):
            trainer.finish_training()


def test_ffi_sample_copy_validates_buffer_before_writing():
    data = np.ones((8, 2), dtype=np.float32)
    with make_trainer(data).prepare_training() as p:
        output = np.full(data.size, 123.0, dtype=np.float32)
        assert lib.paimon_vindex_prepared_training_copy_sample(
            p._handle, output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), output.size - 1,
        ) != 0
        assert (output == 123).all()
        assert lib.paimon_vindex_prepared_training_copy_sample(p._handle, None, output.size) != 0


def test_gpu_module_does_not_import_cuda_for_cpu_users():
    subprocess.run([sys.executable, "-c", "import sys; import paimon_vindex.gpu; "
                    "assert 'cupy' not in sys.modules; assert 'cuvs' not in sys.modules"], check=True)


def test_missing_cuda_dependencies_raise_an_actionable_error(monkeypatch):
    from paimon_vindex.gpu import CuvsKMeans

    monkeypatch.setitem(sys.modules, "cupy", None)
    with pytest.raises(RuntimeError, match="requires compatible CuPy and cuVS"):
        CuvsKMeans()


def test_worker_initialization_failure_releases_resources(monkeypatch):
    from paimon_vindex.gpu import CuvsKMeans

    class Device:
        def __init__(self, device): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def synchronize(self): raise RuntimeError("cleanup synchronization failed")

    def fail_properties(device):
        raise ValueError("device properties failed")

    monkeypatch.setitem(sys.modules, "cupy", types.SimpleNamespace(
        __version__="fake", cuda=types.SimpleNamespace(
            Device=Device, runtime=types.SimpleNamespace(getDeviceProperties=fail_properties))))
    monkeypatch.setitem(sys.modules, "cuvs", types.SimpleNamespace(__version__="fake"))
    monkeypatch.setitem(sys.modules, "cuvs.common", types.SimpleNamespace(Resources=object))
    monkeypatch.setitem(sys.modules, "cuvs.cluster.kmeans", types.SimpleNamespace(KMeansParams=object, fit=None))
    worker = CuvsKMeans.__new__(CuvsKMeans)
    with pytest.raises(ValueError, match="device properties failed"):
        worker.__init__()
    assert worker._resources is None
    worker.close()


@pytest.mark.parametrize("format", ["npy", "fvecs"])
def test_benchmark_builds_and_evaluates_real_input_formats(tmp_path, format):
    rng = np.random.default_rng(42)
    base = rng.normal(size=(256, 8)).astype(np.float32)
    queries = rng.normal(size=(8, 8)).astype(np.float32)
    truth = np.argsort(((queries[:, None, :] - base[None, :, :]) ** 2).sum(axis=2), axis=1)[:, :10].astype(np.int32)
    base_path = tmp_path / f"base.{format}"
    query_path = tmp_path / f"query.{format}"
    truth_path = tmp_path / ("truth.npy" if format == "npy" else "truth.ivecs")
    for path, array, integer in [(base_path, base, False), (query_path, queries, False), (truth_path, truth, True)]:
        if format == "npy":
            np.save(path, array)
        else:
            records = np.empty((len(array), array.shape[1] + 1), dtype="<i4")
            records[:, 0] = array.shape[1]
            records[:, 1:] = array if integer else array.view("<i4")
            records.tofile(path)
    output = tmp_path / "results"
    script = Path(__file__).resolve().parents[2] / "tools" / "benchmark_gpu_training.py"
    command = [sys.executable, str(script), "--preset", "smoke", "--base", str(base_path),
               "--queries", str(query_path), "--neighbors", str(truth_path), "--build-index",
               "--nprobe", "8", "--threads", "2", "--repeats", "1", "--output-dir", str(output)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    runs = [json.loads(line) for line in (output / "runs.jsonl").read_text().splitlines()]
    assert {r["backend"] for r in runs} == {"cpu-auto", "cpu-lloyd"}
    assert len({r["sample_sha256"] for r in runs}) == 1
    for run in runs:
        assert run["query"]["recall_at_k"] >= .9
        assert run["partitions"]["mean"] * 8 == 256
        assert run["build_seconds"] >= run["training_seconds"] > 0
    manifest = (output / "manifest.json").read_bytes()
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert (output / "manifest.json").read_bytes() == manifest


@pytest.mark.skipif(os.environ.get("PAIMON_TEST_CUVS") != "1", reason="set PAIMON_TEST_CUVS=1 on a CUDA host")
@pytest.mark.parametrize("metric", ["l2", "cosine", "inner_product"])
def test_cuvs_training_produces_cpu_readable_index(metric):
    from paimon_vindex.gpu import CuvsKMeans

    rng = np.random.default_rng(1234)
    means = np.eye(4, 8, dtype=np.float32) * 20
    data = means[np.arange(512) % 4] + rng.normal(0, .1, size=(512, 8)).astype(np.float32)
    with CuvsKMeans() as gpu:
        for _ in range(2):  # Reuse the same resources across independent jobs.
            with make_trainer(data, metric).prepare_training() as p:
                centers = gpu.fit(p, initial_centroids=p.sample[:4].copy())
                assert centers.shape == (4, 8)
                assert gpu.last_run["kmeans_seconds"] > 0
                payload = index_bytes(p.finish_with_ivf_centroids(centers), data)
            with VectorIndexReader(BytesInput(payload)) as reader:
                query = data[0]
                scores = ((data - query) ** 2).sum(axis=1)
                if metric == "inner_product":
                    scores = -(data @ query)
                elif metric == "cosine":
                    scores = 1 - (data @ query) / (np.linalg.norm(data, axis=1) * np.linalg.norm(query))
                exact = set(np.argsort(scores)[:10])
                ids, distances = reader.search(query, SearchParams.ivf(10, 4))
                assert len(exact.intersection(ids.tolist())) >= 8
                assert np.isfinite(distances).all()
