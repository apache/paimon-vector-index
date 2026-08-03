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
import threading

import numpy as np
import pytest

from paimon_vindex import (
    IvfPqBatchTableReuseMode,
    SearchParams,
    VectorIndexReader,
    VectorIndexTrainer,
    VectorIndexWriter,
)


class VectorIndexInput:
    def __init__(self, data):
        self.data = data

    def pread_many(self, ranges):
        return [self.data[pos : pos + length] for pos, length in ranges]


def clustered_data(n, d, clusters):
    data = np.zeros((n, d), dtype=np.float32)
    for i in range(n):
        cluster = i % clusters
        for j in range(d):
            data[i, j] = cluster * 20.0 + j * 0.01 + i * 0.0001
    return data


def build_index(options, d, n=512):
    data = clustered_data(n, d, int(options.get("nlist", "4")))
    ids = np.arange(n, dtype=np.int64)
    output = io.BytesIO()
    training = VectorIndexTrainer.train(options, data)
    with VectorIndexWriter(training) as writer:
        writer.add_vectors(ids, data)
        writer.write(output)
    return output.getvalue(), data


def reader_from_bytes(data):
    return VectorIndexReader(VectorIndexInput(data))


def test_python_search_parameters_remain_algorithm_specific():
    params = SearchParams.diskann(top_k=10, l_search=200).to_ffi()

    assert params.search_width == 2
    assert params.width == 200
    automatic = SearchParams.automatic(top_k=10).to_ffi()
    assert automatic.search_width == 0
    assert automatic.width == 0
    assert (
        SearchParams.ivf(10, 16).ivfpq_batch_table_reuse_max_bytes
        == 512 * 1024 * 1024
    )

    batch = SearchParams.ivf(
        top_k=10,
        nprobe=16,
        ivfpq_batch_table_reuse=IvfPqBatchTableReuseMode.ON,
        ivfpq_batch_table_reuse_max_bytes=32 * 1024 * 1024,
    ).to_ffi_v2()
    assert batch.ivfpq_batch_table_reuse == 1
    assert batch.ivfpq_batch_table_reuse_max_bytes == 32 * 1024 * 1024


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SearchParams.automatic(top_k=0),
        lambda: SearchParams.ivf(top_k=5, nprobe=-1),
        lambda: SearchParams.diskann(top_k=5, l_search=-1),
        lambda: SearchParams.ivf(
            top_k=5, nprobe=ctypes.c_size_t(-1).value + 1
        ),
        lambda: SearchParams.ivf(
            top_k=5, nprobe=2, ivfpq_batch_table_reuse=3
        ),
        lambda: SearchParams.ivf(
            top_k=5,
            nprobe=2,
            ivfpq_batch_table_reuse_max_bytes=0,
        ),
    ],
)
def test_python_search_parameters_reject_values_that_ctypes_would_wrap(factory):
    with pytest.raises(ValueError):
        factory()


def test_python_high_level_training_infers_dimension_and_ivf_shape():
    data = clustered_data(512, 16, 8)
    options = {"index.type": "ivf_sq", "metric": "l2"}
    ids = np.arange(data.shape[0], dtype=np.int64)
    output = io.BytesIO()

    training = VectorIndexTrainer.train(options, data)
    with VectorIndexWriter(training) as writer:
        writer.add_vectors(ids, data)
        writer.write(output)

    with reader_from_bytes(output.getvalue()) as reader:
        metadata = reader.metadata()
        assert metadata.dimension == 16
        assert metadata.nlist == 8
        result_ids, _ = reader.search(data[0], SearchParams.automatic(top_k=5))
        assert result_ids[0] == 0


def test_python_high_level_training_preserves_explicit_expected_count():
    data = clustered_data(512, 16, 8)
    options = {
        "index.type": "ivf_sq",
        "metric": "l2",
        "expected-vector-count": "1000000",
    }
    ids = np.arange(data.shape[0], dtype=np.int64)
    output = io.BytesIO()

    training = VectorIndexTrainer.train(options, data)
    with VectorIndexWriter(training) as writer:
        writer.add_vectors(ids, data)
        writer.write(output)

    with reader_from_bytes(output.getvalue()) as reader:
        assert reader.metadata().nlist == 1024


def test_python_read_callback_forwards_ranges_in_one_batch():
    from paimon_vindex import _make_read_ranges_callback
    from paimon_vindex import _ffi

    class RecordingInput(VectorIndexInput):
        def __init__(self, data):
            super().__init__(data)
            self.calls = []

        def pread_many(self, ranges):
            self.calls.append(list(ranges))
            return super().pread_many(ranges)

    source = RecordingInput(bytes(range(32)))
    callback = _make_read_ranges_callback(source)
    first = (ctypes.c_uint8 * 3)()
    second = (ctypes.c_uint8 * 4)()
    requests = (_ffi.PaimonVindexReadRequest * 2)(
        _ffi.PaimonVindexReadRequest(2, first, 3),
        _ffi.PaimonVindexReadRequest(11, second, 4),
    )

    assert callback(None, requests, 2) == 0
    assert source.calls == [[(2, 3), (11, 4)]]
    assert bytes(first) == bytes([2, 3, 4])
    assert bytes(second) == bytes([11, 12, 13, 14])


def test_python_handle_lock_rejects_worker_callback_reentry():
    from paimon_vindex import _NativeHandleLock

    native_handle_lock = _NativeHandleLock()
    rejected = []

    def callback_worker():
        native_handle_lock._enter_callback()
        try:
            with native_handle_lock:
                pass
        except RuntimeError as error:
            rejected.append("reentrant native-handle operation" in str(error))
        finally:
            native_handle_lock._exit_callback()

    with native_handle_lock:
        worker = threading.Thread(target=callback_worker)
        worker.start()
        worker.join(timeout=5)
        assert not worker.is_alive()

    assert rejected == [True]


def test_python_reader_close_waits_for_an_inflight_native_search():
    index_bytes, data = build_index(
        {
            "index.type": "ivf_flat",
            "dimension": "2",
            "nlist": "2",
            "metric": "l2",
        },
        2,
        n=64,
    )

    class BlockingInput(VectorIndexInput):
        def __init__(self, payload):
            super().__init__(payload)
            self.block_reads = False
            self.read_entered = threading.Event()
            self.release_read = threading.Event()

        def pread_many(self, ranges):
            if self.block_reads:
                self.read_entered.set()
                assert self.release_read.wait(timeout=5)
            return super().pread_many(ranges)

    source = BlockingInput(index_bytes)
    reader = VectorIndexReader(source)
    source.block_reads = True
    search_done = threading.Event()
    close_done = threading.Event()
    errors = []

    def search():
        try:
            reader.search(data[0], SearchParams.ivf(top_k=5, nprobe=2))
        except Exception as exc:
            errors.append(exc)
        finally:
            search_done.set()

    def close():
        reader.close()
        close_done.set()

    search_thread = threading.Thread(target=search)
    close_thread = threading.Thread(target=close)
    search_thread.start()
    assert source.read_entered.wait(timeout=5)
    close_thread.start()
    assert not close_done.wait(timeout=0.1)
    source.release_read.set()
    search_thread.join(timeout=5)
    close_thread.join(timeout=5)

    assert search_done.is_set()
    assert close_done.is_set()
    assert errors == []


def test_python_ffi_roundtrips_supported_indexes():
    configs = [
        (
            {
                "index.type": "ivf_flat",
                "dimension": "16",
                "nlist": "4",
                "metric": "l2",
            },
            16,
        ),
        (
            {
                "index.type": "ivf_pq",
                "dimension": "16",
                "nlist": "4",
                "metric": "l2",
                "use-opq": "false",
            },
            16,
        ),
        (
            {
                "index.type": "ivf_rq",
                "dimension": "16",
                "nlist": "4",
                "metric": "l2",
            },
            16,
        ),
        (
            {
                "index.type": "ivf_sq",
                "dimension": "16",
                "nlist": "4",
                "metric": "l2",
            },
            16,
        ),
        (
            {
                "index.type": "diskann",
                "dimension": "16",
                "pq.m": "4",
                "pq.bits": "4",
                "metric": "l2",
                "diskann.max-degree": "8",
                "diskann.build-search-list-size": "16",
            },
            16,
        ),
    ]

    for options, d in configs:
        index_bytes, data = build_index(options, d)
        with reader_from_bytes(index_bytes) as reader:
            metadata = reader.metadata()
            assert reader.index_type == options["index.type"]
            assert metadata.index_type == options["index.type"]
            assert reader.dimension == d
            assert metadata.total_vectors == 512
            if options["index.type"] == "ivf_pq":
                assert metadata.pq_m == 4
                assert metadata.pq_bits == 8
            elif options["index.type"] == "ivf_sq":
                assert metadata.pq_m is None
                assert metadata.pq_bits == 8
            elif options["index.type"] == "diskann":
                assert metadata.pq_m == 4
                assert metadata.pq_bits == 4
                assert metadata.diskann_max_degree == 8
                assert metadata.diskann_build_search_list_size == 16
                assert metadata.diskann_alpha == pytest.approx(1.2)

            params = (
                SearchParams.diskann(top_k=5, l_search=32)
                if options["index.type"] == "diskann"
                else SearchParams.ivf(top_k=5, nprobe=4)
            )
            ids, distances = reader.search(data[0], params)
            reader.optimize_for_search()
            if options["index.type"] == "diskann":
                reader.warmup_queries(np.vstack([data[0], data[1]]), l_search=32)
            optimized_ids, optimized_distances = reader.search(data[0], params)
            assert ids.shape == (5,)
            assert distances.shape == (5,)
            if options["index.type"] == "diskann":
                assert ids[0] >= 0
            else:
                assert ids[0] == 0
            np.testing.assert_array_equal(optimized_ids, ids)
            np.testing.assert_allclose(optimized_distances, distances, rtol=0, atol=1e-4)


def test_python_ffi_batch_search():
    index_bytes, data = build_index(
        {
            "index.type": "ivf_flat",
            "dimension": "2",
            "nlist": "2",
            "metric": "l2",
        },
        2,
        n=64,
    )

    with reader_from_bytes(index_bytes) as reader:
        ids, distances = reader.search_batch(
            np.vstack([data[0], data[1]]),
            SearchParams.ivf(top_k=2, nprobe=2),
        )
        assert ids.shape == (2, 2)
        assert distances.shape == (2, 2)
        assert ids[0, 0] == 0
        assert ids[1, 0] == 1


def test_python_diskann_latency_hint_selects_coalesced_read_plan():
    index_bytes, data = build_index(
        {
            "index.type": "diskann",
            "dimension": "16",
            "pq.m": "4",
            "metric": "l2",
            "diskann.max-degree": "8",
            "diskann.build-search-list-size": "16",
        },
        16,
    )
    source = VectorIndexInput(index_bytes)
    source.estimated_random_read_latency_nanos = 20_000_000

    with VectorIndexReader(source) as reader:
        plan = reader.read_plan()
        assert plan.random_read_latency_nanos == 20_000_000
        assert plan.window_bytes == 64 * 1024
        ids, distances = reader.search(data[0], SearchParams.diskann(top_k=5, l_search=100))

    assert ids.shape == (5,)
    assert distances.shape == (5,)
    assert ids[0] >= 0


def test_python_diskann_automatic_cache_reuses_reads():
    class RecordingInput(VectorIndexInput):
        def __init__(self, data):
            super().__init__(data)
            self.calls = []

        def pread_many(self, ranges):
            self.calls.append(list(ranges))
            return super().pread_many(ranges)

    index_bytes, data = build_index(
        {
            "index.type": "diskann",
            "dimension": "16",
            "pq.m": "4",
            "metric": "l2",
            "diskann.max-degree": "8",
            "diskann.build-search-list-size": "16",
        },
        16,
    )
    source = RecordingInput(index_bytes)

    with VectorIndexReader(source) as reader:
        reader.optimize_for_search()
        reader.search(data[0], SearchParams.diskann(top_k=5, l_search=100))
        first_query_calls = len(source.calls)
        source.calls.clear()
        reader.search(data[0], SearchParams.diskann(top_k=5, l_search=100))

    assert len(source.calls) <= first_query_calls


def test_python_diskann_calibrates_automatic_search_width():
    index_bytes, data = build_index(
        {
            "index.type": "diskann",
            "dimension": "16",
            "pq.m": "4",
            "metric": "l2",
            "diskann.max-degree": "8",
            "diskann.build-search-list-size": "16",
        },
        16,
    )

    with reader_from_bytes(index_bytes) as reader:
        resolved = reader.calibrate_search_width(data[:4], top_k=5)
        assert resolved in {100, 200, 400}
        ids, distances = reader.search(data[0], SearchParams.automatic(top_k=5))
        assert ids.shape == (5,)
        assert distances.shape == (5,)


@pytest.mark.parametrize(
    ("metric", "expected_id", "expected_distance"),
    [
        ("inner_product", 100, -10.0),
        ("cosine", 100, 0.0),
    ],
)
def test_python_diskann_supports_ip_and_cosine(metric, expected_id, expected_distance):
    data = np.asarray(
        [
            [10.0, 0.0],
            [1.0, 1.0],
            [0.0, 8.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
            [-2.0, 1.0],
            [1.0, -2.0],
            [-3.0, -1.0],
            [-1.0, -3.0],
            [-4.0, 0.5],
            [0.5, -4.0],
            [-5.0, -2.0],
            [-2.0, -5.0],
            [-6.0, -1.0],
            [-2.0, 0.0],
        ],
        dtype=np.float32,
    )
    options = {
        "index.type": "diskann",
        "dimension": "2",
        "metric": metric,
        "pq.m": "1",
        "pq.bits": "4",
        "diskann.max-degree": "8",
        "diskann.build-search-list-size": "16",
        "diskann.raw-vector-encoding": "f32",
    }
    output = io.BytesIO()
    training = VectorIndexTrainer.train(options, data)
    with VectorIndexWriter(training) as writer:
        writer.add_vectors(np.arange(100, 116, dtype=np.int64), data)
        writer.write(output)

    with reader_from_bytes(output.getvalue()) as reader:
        assert reader.metadata().metric == metric
        ids, distances = reader.search(
            np.asarray([1.0, 0.0], dtype=np.float32),
            SearchParams.diskann(top_k=1, l_search=16),
        )
        assert ids[0] == expected_id
        assert distances[0] == pytest.approx(expected_distance)
        assert ids[0] >= 0


def test_python_diskann_read_plan_resolves_during_open():
    index_bytes, data = build_index(
        {
            "index.type": "diskann",
            "dimension": "16",
            "pq.m": "4",
            "metric": "l2",
            "diskann.max-degree": "8",
            "diskann.build-search-list-size": "16",
        },
        16,
    )

    with VectorIndexReader(VectorIndexInput(index_bytes)) as reader:
        plan = reader.read_plan()
        assert plan.random_read_latency_nanos > 0
        assert plan.window_bytes > 0
        reader.search(data[0], SearchParams.diskann(top_k=5, l_search=100))


def test_python_reader_rejects_negative_memory_budget():
    with pytest.raises(ValueError, match="memory_budget_bytes"):
        VectorIndexReader(VectorIndexInput(b""), memory_budget_bytes=-1)


def test_python_reader_rejects_negative_latency_hint():
    source = VectorIndexInput(b"")
    source.estimated_random_read_latency_nanos = -1
    with pytest.raises(ValueError, match="estimated_random_read_latency_nanos"):
        VectorIndexReader(source)


def test_python_size_t_arguments_reject_platform_overflow():
    oversized = ctypes.c_size_t(-1).value + 1
    with pytest.raises(ValueError, match="top_k"):
        SearchParams.automatic(oversized)
    with pytest.raises(ValueError, match="nprobe"):
        SearchParams.ivf(5, oversized)
    with pytest.raises(ValueError, match="memory_budget_bytes"):
        VectorIndexReader(VectorIndexInput(b""), memory_budget_bytes=oversized)

    source = VectorIndexInput(b"")
    source.max_ranges_per_read = oversized
    with pytest.raises(ValueError, match="max_ranges_per_read"):
        VectorIndexReader(source)

    source = VectorIndexInput(b"")
    source.estimated_random_read_latency_nanos = ctypes.c_uint64(-1).value + 1
    with pytest.raises(ValueError, match="estimated_random_read_latency_nanos"):
        VectorIndexReader(source)

    index_bytes, data = build_index(
        {
            "index.type": "diskann",
            "dimension": "16",
            "pq.m": "4",
            "metric": "l2",
            "diskann.max-degree": "8",
            "diskann.build-search-list-size": "16",
        },
        16,
    )
    with reader_from_bytes(index_bytes) as reader:
        with pytest.raises(ValueError, match="l_search"):
            reader.warmup_queries(data[:1], l_search=oversized)
        with pytest.raises(ValueError, match="top_k"):
            reader.calibrate_search_width(data[:1], top_k=oversized)


def test_python_reader_rejects_reentrant_callback_operations():
    index_bytes, data = build_index(
        {
            "index.type": "ivf_flat",
            "dimension": "16",
            "nlist": "4",
            "metric": "l2",
        },
        16,
    )

    class ReentrantInput(VectorIndexInput):
        reader = None
        attempted = False

        def pread_many(self, ranges):
            if self.reader is not None and not self.attempted:
                self.attempted = True
                self.reader.close()
            return super().pread_many(ranges)

    source = ReentrantInput(index_bytes)
    reader = VectorIndexReader(source)
    source.reader = reader
    try:
        with pytest.raises(RuntimeError):
            reader.search(data[0], SearchParams.ivf(top_k=5, nprobe=2))
        assert source.attempted
        assert reader.metadata().index_type == "ivf_flat"
    finally:
        reader.close()


def test_python_writer_rejects_reentrant_output_callback_operations():
    data = np.arange(128 * 16, dtype=np.float32).reshape(128, 16)
    training = VectorIndexTrainer.train(
        {
            "index.type": "ivf_flat",
            "dimension": "16",
            "nlist": "4",
            "metric": "l2",
        },
        data,
    )

    class ReentrantOutput(io.BytesIO):
        writer = None
        attempted = False

        def write(self, payload):
            if self.writer is not None and not self.attempted:
                self.attempted = True
                self.writer.close()
            return super().write(payload)

    output = ReentrantOutput()
    writer = VectorIndexWriter(training)
    output.writer = writer
    try:
        with pytest.raises(RuntimeError):
            writer.write(output)
        assert output.attempted
        assert writer.dimension == 16
    finally:
        writer.close()


def test_python_reader_enforces_configured_resident_memory_budget():
    index_bytes, _ = build_index(
        {
            "index.type": "diskann",
            "dimension": "16",
            "pq.m": "4",
            "metric": "l2",
            "diskann.max-degree": "8",
            "diskann.build-search-list-size": "16",
        },
        16,
    )

    with VectorIndexReader(
        VectorIndexInput(index_bytes), memory_budget_bytes=1
    ) as reader:
        with pytest.raises(RuntimeError, match="reader budget"):
            reader.optimize_for_search()


def test_python_ffi_ivfrq_build_bits():
    index_bytes, data = build_index(
        {
            "index.type": "ivf_rq",
            "dimension": "16",
            "nlist": "4",
            "rq.bits": "5",
            "metric": "l2",
        },
        16,
        n=128,
    )

    with reader_from_bytes(index_bytes) as reader:
        assert reader.metadata().rq_bits == 5
        ids, distances = reader.search(data[7], SearchParams.ivf(top_k=5, nprobe=4))
        assert ids.shape == (5,)
        assert distances.shape == (5,)
        assert ids[0] % 4 == 7 % 4

        ids, distances = reader.search_batch(
            np.vstack([data[4], data[7]]), SearchParams.ivf(top_k=5, nprobe=4)
        )
        assert ids[0, 0] % 4 == 4 % 4
        assert ids[1, 0] % 4 == 7 % 4


def test_python_ffi_delegates_validation():
    options = {
        "index.type": "ivf_pq",
        "dimension": "16",
        "nlist": "4",
        "pq.m": "4",
        "metric": "l2",
    }
    with VectorIndexTrainer.create(options) as trainer:
        with pytest.raises(RuntimeError, match="training data length 17"):
            trainer.add_training_vectors(np.zeros((1, 17), dtype=np.float32))

    data = np.zeros((1, 16), dtype=np.float32)
    ids = np.array([1, 2], dtype=np.int64)
    training = VectorIndexTrainer.train(options, data)
    with VectorIndexWriter(training) as writer:
        with pytest.raises(RuntimeError, match="ids length 2 does not match vector count 1"):
            writer.add_vectors(ids, data)

    index_bytes, data = build_index(options, 16)
    with reader_from_bytes(index_bytes) as reader:
        with pytest.raises(RuntimeError, match="query length 15"):
            reader.search(np.zeros(15, dtype=np.float32), SearchParams.ivf(top_k=5, nprobe=2))
        with pytest.raises(ValueError, match="top_k must be"):
            reader.search(data[0], SearchParams.ivf(top_k=0, nprobe=2))
        with pytest.raises(RuntimeError, match="queries length 15"):
            reader.search_batch(
                np.zeros((1, 15), dtype=np.float32), SearchParams.ivf(top_k=5, nprobe=2)
            )
