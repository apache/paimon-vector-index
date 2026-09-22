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

import ast
import ctypes
import io
import os
import struct
import subprocess
import sys
import threading
import textwrap
from collections import Counter
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest


def test_range_public_api_is_exported_without_loading_native_library():
    source = Path(__file__).parents[1] / "paimon_vindex" / "__init__.py"
    module = ast.parse(source.read_text())
    exports = next(
        ast.literal_eval(statement.value)
        for statement in module.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in statement.targets
        )
    )
    assert {
        "DistanceBand",
        "DistanceEndpoint",
        "DistanceEndpointOp",
        "RangeSearchParams",
        "RangeSearchQueryResult",
        "RangeSearchResult",
        "RangeSearchStats",
    } <= set(exports)


@pytest.fixture(scope="module")
def vindex():
    import paimon_vindex

    return paimon_vindex


class BytesInput:
    def __init__(self, data):
        self.data = data

    def pread_many(self, ranges):
        return [self.data[offset : offset + length] for offset, length in ranges]


def make_index(vindex, index_type="ivf_flat", metric="l2"):
    count = 512 if index_type == "ivf_pq" else 128
    data = np.random.default_rng(1729).normal(size=(count, 16)).astype(np.float32)
    labels = np.arange(len(data), dtype=np.int64) + (1 << 40)
    options = {
        "index.type": index_type,
        "dimension": "16",
        "metric": metric,
    }
    if index_type == "diskann":
        options.update({
            "pq.m": "4",
            "pq.bits": "4",
            "diskann.max-degree": "8",
            "diskann.build-search-list-size": "16",
        })
    else:
        options["nlist"] = "4"
        if index_type == "ivf_pq":
            options.update({"pq.m": "4", "use-opq": "false"})
    output = io.BytesIO()
    training = vindex.VectorIndexTrainer.train(options, data)
    with vindex.VectorIndexWriter(training) as writer:
        writer.add_vectors(labels, data)
        writer.write(output)
    return output.getvalue(), data, labels


def roaring_allowlist(labels):
    if not len(labels):
        return struct.pack("<Q", 0)
    return (
        struct.pack("<QI", 1, 256)
        + struct.pack("<IIHHI", 12346, 1, 0, len(labels) - 1, 16)
        + struct.pack(f"<{len(labels)}H", *(int(label) & 65535 for label in labels))
    )


@pytest.fixture(scope="module", params=[
    (index_type, metric)
    for index_type in ("ivf_flat", "ivf_pq", "ivf_rq", "ivf_sq")
    for metric in ("l2", "inner_product", "cosine")
])
def index_case(request, vindex):
    index_type, metric = request.param
    return index_type, metric, make_index(vindex, index_type, metric)


@pytest.fixture(scope="module")
def flat_index(vindex):
    return make_index(vindex)


def test_raw_distance_api_names(vindex):
    assert set(vindex.RangeSearchResult.__dataclass_fields__) == {
        "lims", "labels", "raw_distances", "stats", "list_reads",
    }
    assert set(vindex.DistanceBand.__dataclass_fields__) == {
        "metric", "raw_lower", "raw_upper",
    }
    ffi = vindex._ffi
    assert not hasattr(ffi, "PaimonVindexDistanceBand")
    assert ffi.PaimonVindexRawDistanceBand._fields_ == [
        ("metric", ctypes.c_uint32),
        ("raw_lower_kind", ctypes.c_uint32),
        ("raw_lower", ctypes.c_float),
        ("raw_upper_kind", ctypes.c_uint32),
        ("raw_upper", ctypes.c_float),
    ]
    assert ctypes.sizeof(ffi.PaimonVindexRawDistanceBand) == 20
    assert [
        getattr(ffi.PaimonVindexRawDistanceBand, name).offset
        for name, _ in ffi.PaimonVindexRawDistanceBand._fields_
    ] == [0, 4, 8, 12, 16]
    assert ffi.PaimonVindexRangeSearchParams._fields_[0] == (
        "band", ffi.PaimonVindexRawDistanceBand
    )
    assert [name for name, _ in ffi.PaimonVindexRangeSearchResultView._fields_] == [
        "query_count", "hit_count", "lims", "labels", "raw_distances", "stats",
        "list_reads",
    ]


@pytest.mark.parametrize("args,kwargs", [
    (("l2",), {}),
    (("l2", 0.0, 4.0), {}),
    ((), {"metric": "l2", "lower": 0.0, "upper": 4.0}),
    ((), {"metric": "l2", "raw_lower": 0.0, "raw_upper": 4.0}),
])
def test_ambiguous_band_construction_is_rejected(vindex, args, kwargs):
    with pytest.raises(TypeError, match="from_endpoints.*from_raw"):
        vindex.DistanceBand(*args, **kwargs)


def test_explicit_raw_band_factory(vindex):
    band = vindex.DistanceBand.from_raw("inner_product", raw_lower=-6, raw_upper="2")
    assert band.metric == "inner_product"
    assert band.raw_lower == -6.0 and type(band.raw_lower) is float
    assert band.raw_upper == 2.0 and type(band.raw_upper) is float
    assert not hasattr(band, "lower") and not hasattr(band, "upper")
    assert band == vindex.DistanceBand.from_raw("inner_product", -6.0, 2.0)
    assert band.to_ffi().raw_lower == -6.0
    assert band.to_ffi().raw_upper == 2.0
    unbounded = vindex.DistanceBand.from_raw("l2")
    assert unbounded.raw_lower is None and unbounded.raw_upper is None
    assert unbounded.to_ffi().raw_lower_kind == 0
    assert unbounded.to_ffi().raw_upper_kind == 0
    with pytest.raises(FrozenInstanceError):
        band.raw_lower = 0.0
    with pytest.raises(TypeError):
        vindex.DistanceBand.from_raw("l2", lower=0.0, upper=4.0)
    with pytest.raises(ValueError):
        vindex.DistanceBand.from_raw("l2", raw_lower="invalid")


@pytest.fixture(scope="module", params=["l2", "inner_product", "cosine"])
def endpoint_index(request, vindex):
    metric = request.param
    data = np.zeros((128, 16), dtype=np.float32)
    query = np.zeros(16, dtype=np.float32)
    if metric == "l2":
        data[:, 0] = 5.0
        data[0, 0], data[1, 0] = 3.0, 4.0
        endpoints = {"upper": vindex.DistanceEndpoint(4.0, vindex.DistanceEndpointOp.LE)}
        expected = [9.0, 16.0]
    elif metric == "inner_product":
        query[0] = 2.0
        data[:, 0] = 2.0
        data[0, 0], data[1, 0] = 3.0, 2.5
        endpoints = {"lower": vindex.DistanceEndpoint(5.0, vindex.DistanceEndpointOp.GE)}
        expected = [-6.0, -5.0]
    else:
        query[0] = 1.0
        data[:, 1] = 1.0
        data[0, 0] = 1.0
        data[1] = query
        endpoints = {"upper": vindex.DistanceEndpoint(0.5, vindex.DistanceEndpointOp.LE)}
        expected = [1.0 - 1.0 / np.sqrt(2.0), 0.0]
    labels = np.arange(len(data), dtype=np.int64) + (1 << 40)
    options = {
        "index.type": "ivf_flat", "dimension": "16", "metric": metric,
        "nlist": "1",
    }
    output = io.BytesIO()
    training = vindex.VectorIndexTrainer.train(options, data)
    with vindex.VectorIndexWriter(training) as writer:
        writer.add_vectors(labels, data)
        writer.write(output)
    return metric, output.getvalue(), query, labels, endpoints, expected


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("filter_kind", ["none", "subset", "empty"])
def test_endpoint_search_returns_raw_distances(vindex, endpoint_index, batch, filter_kind):
    metric, payload, query, labels, endpoints, expected = endpoint_index
    band = vindex.DistanceBand.from_endpoints(metric, **endpoints)
    params = vindex.RangeSearchParams(band, 1)
    allowed = {"none": labels, "subset": labels[::3], "empty": labels[:0]}[filter_kind]
    filter_bytes = None if filter_kind == "none" else roaring_allowlist(allowed)
    with vindex.VectorIndexReader(BytesInput(payload)) as reader:
        method = reader.range_search_batch if batch else reader.range_search
        result = method(
            np.stack([query, query]) if batch else query, params,
            roaring_filter=filter_bytes,
        )
    assert not hasattr(result, "distances")
    assert result.raw_distances.dtype == np.dtype(np.float32)
    assert result.raw_distances.flags.owndata and result.raw_distances.flags.writeable
    assert result.query_count == (2 if batch else 1)
    expected_hits = {
        label: raw_distance
        for label, raw_distance in zip(labels[:2], expected)
        if label in allowed
    }
    for query_index in range(result.query_count):
        hits = result.query(query_index)
        assert isinstance(hits, tuple)
        assert hits._fields == ("labels", "raw_distances")
        assert not hasattr(hits, "distances")
        hit_labels, raw_distances = hits
        assert hit_labels is hits.labels and raw_distances is hits.raw_distances
        assert hit_labels.base is result.labels
        assert raw_distances.base is result.raw_distances
        assert set(hit_labels) == set(expected_hits)
        for label, raw_distance in zip(hit_labels, raw_distances):
            assert raw_distance == pytest.approx(expected_hits[label], abs=1e-6)
        if expected_hits:
            assert np.shares_memory(hit_labels, result.labels)
            assert np.shares_memory(raw_distances, result.raw_distances)


@pytest.mark.parametrize("batch", [False, True])
def test_range_accepts_unaligned_contiguous_queries(vindex, flat_index, batch):
    payload, data, _ = flat_index
    source = data[:3] if batch else data[0]
    queries = np.ndarray(
        source.shape, dtype=np.float32,
        buffer=bytearray(source.nbytes + 1), offset=1,
    )
    queries[:] = source
    assert queries.flags.c_contiguous and not queries.flags.aligned
    params = vindex.RangeSearchParams(vindex.DistanceBand.from_raw("l2"), 4)
    with vindex.VectorIndexReader(BytesInput(payload)) as reader:
        method = reader.range_search_batch if batch else reader.range_search
        result = method(queries, params)
        expected = method(queries.copy(), params)
    np.testing.assert_array_equal(result.lims, expected.lims)
    for query in range(result.query_count):
        actual_labels, actual_distances = result.query(query)
        expected_labels, expected_distances = expected.query(query)
        assert Counter(zip(actual_labels, actual_distances.view(np.uint32))) == Counter(
            zip(expected_labels, expected_distances.view(np.uint32))
        )


@pytest.mark.parametrize("missing", ["all", "paimon_vindex_range_search_result_destroy"])
def test_older_native_library_preserves_import_and_topk(missing):
    program = textwrap.dedent('''
        import ctypes
        import runpy
        import sys
        import numpy as np
        import pytest

        missing = sys.argv[1]
        native_loader = ctypes.CDLL
        class OlderLibrary:
            def __init__(self, native):
                self.native = native
            def __getattr__(self, name):
                is_range = (
                    name.startswith("paimon_vindex_reader_range_search")
                    or name.startswith("paimon_vindex_range_search_result")
                    or name in (
                        "paimon_vindex_distance_band_from_endpoints",
                        "paimon_vindex_reader_supports_range_search",
                    )
                )
                if name == missing or (missing == "all" and is_range):
                    raise AttributeError(name)
                return getattr(self.native, name)
        ctypes.CDLL = lambda *args, **kwargs: OlderLibrary(native_loader(*args, **kwargs))
        import paimon_vindex as api
        helpers = runpy.run_path(sys.argv[2])
        payload, data, labels = helpers["make_index"](api)
        with api.VectorIndexReader(helpers["BytesInput"](payload)) as reader:
            params = api.SearchParams.ivf(3, 4)
            expected = reader.search(data[0], params)
            assert expected[0][0] == labels[0]
            assert not reader.supports_range_search()
            range_params = api.RangeSearchParams(api.DistanceBand.from_raw("l2"), 4)
            for call in (
                lambda: reader.range_search(data[0], range_params),
                lambda: reader.range_search_batch(data[:3], range_params),
                lambda: api.DistanceBand.from_endpoints("l2"),
            ):
                with pytest.raises(RuntimeError, match="native library.*range search"):
                    call()
            actual = reader.search(data[0], params)
            np.testing.assert_array_equal(actual[0], expected[0])
            np.testing.assert_array_equal(actual[1], expected[1])
    ''')
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).parents[1])
    result = subprocess.run(
        [sys.executable, "-c", program, missing, str(Path(__file__).resolve())],
        env=environment, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("filter_kind", ["none", "subset", "empty"])
@pytest.mark.parametrize("band_kind", ["unbounded", "bounded", "empty"])
def test_range_matrix(vindex, index_case, batch, filter_kind, band_kind):
    index_type, metric, (payload, data, labels) = index_case
    queries = data[:3] if batch else data[:1]
    if band_kind == "unbounded":
        band = vindex.DistanceBand.from_raw(metric)
    elif band_kind == "empty":
        band = vindex.DistanceBand.from_raw(metric, 0.0, 0.0)
    else:
        lower, upper = {
            "l2": (0.0, 24.0),
            "inner_product": (-4.0, 2.0),
            "cosine": (0.0, 0.95),
        }[metric]
        band = vindex.DistanceBand.from_raw(metric, lower, upper)
    allowed = {
        "none": labels,
        "subset": labels[::3],
        "empty": labels[:0],
    }[filter_kind]
    filter_bytes = None if filter_kind == "none" else roaring_allowlist(allowed)
    params = vindex.RangeSearchParams(band, 4)
    with vindex.VectorIndexReader(BytesInput(payload)) as reader:
        assert reader.supports_range_search() is True
        method = reader.range_search_batch if batch else reader.range_search
        result = method(
            queries if batch else queries[0], params, roaring_filter=filter_bytes
        )
    assert result.query_count == len(queries)
    assert result.hit_count == len(result.labels) == len(result.raw_distances)
    assert result.lims.dtype == np.dtype(np.uintp)
    assert result.labels.dtype == np.dtype(np.int64)
    assert result.raw_distances.dtype == np.dtype(np.float32)
    assert result.lims[0] == 0 and result.lims[-1] == result.hit_count
    assert np.all(result.lims[1:] >= result.lims[:-1])
    assert len(result.stats) == len(queries)
    assert 0 <= result.list_reads <= 4
    for array in (result.lims, result.labels, result.raw_distances):
        assert array.flags.owndata
    for query_index, query in enumerate(queries):
        with vindex.VectorIndexReader(BytesInput(payload)) as reader:
            reference = reader.range_search(query, params, roaring_filter=filter_bytes)
        actual_labels, actual_distances = result.query(query_index)
        actual_order = np.argsort(actual_labels)
        reference_order = np.argsort(reference.labels)
        np.testing.assert_array_equal(
            actual_labels[actual_order], reference.labels[reference_order]
        )
        np.testing.assert_array_equal(
            actual_distances[actual_order].view(np.uint32),
            reference.raw_distances[reference_order].view(np.uint32),
        )
        assert set(actual_labels) <= set(allowed)
        assert result.stats[query_index] == reference.stats[0]
        stats = result.stats[query_index]
        assert stats.rows_committed == len(actual_labels)
        assert stats.rows_scanned >= stats.rows_committed
        assert stats.early_abandoned <= stats.rows_scanned
        if metric != "l2" or index_type in ("ivf_pq", "ivf_rq"):
            assert stats.early_abandoned == 0
        if band_kind == "unbounded":
            assert set(actual_labels) == set(allowed)
        if band_kind == "empty" or filter_kind == "empty":
            assert len(actual_labels) == 0
        if band.raw_lower is not None:
            assert np.all(actual_distances >= np.float32(band.raw_lower))
        if band.raw_upper is not None:
            assert np.all(actual_distances < np.float32(band.raw_upper))


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("filter_bytes", [b"", b"invalid roaring"])
def test_invalid_filter(vindex, flat_index, batch, filter_bytes):
    payload, data, _ = flat_index
    params = vindex.RangeSearchParams(vindex.DistanceBand.from_raw("l2"), 4)
    with vindex.VectorIndexReader(BytesInput(payload)) as reader:
        method = reader.range_search_batch if batch else reader.range_search
        with pytest.raises(RuntimeError, match="[Rr]oaring|filter"):
            method(
                data[:2] if batch else data[0], params, roaring_filter=filter_bytes
            )


@pytest.mark.parametrize("lower, upper", [
    (float("nan"), None), (None, float("inf")), (-1.0, None), (2.0, 1.0),
])
def test_invalid_bands_are_rejected_by_core(vindex, flat_index, lower, upper):
    payload, data, _ = flat_index
    params = vindex.RangeSearchParams(vindex.DistanceBand.from_raw("l2", lower, upper), 4)
    with vindex.VectorIndexReader(BytesInput(payload)) as reader:
        with pytest.raises(RuntimeError):
            reader.range_search(data[0], params)
        with pytest.raises(RuntimeError):
            reader.range_search_batch(data[:0], params)


@pytest.mark.parametrize("nprobe", [-1, 0, 1.5, ctypes.c_size_t(-1).value + 1])
def test_nprobe_rejects_invalid_and_wrapping_values(vindex, nprobe):
    with pytest.raises(ValueError, match="nprobe"):
        vindex.RangeSearchParams(vindex.DistanceBand.from_raw("l2"), nprobe)


@pytest.mark.parametrize("metric", ["l2", "inner_product", "cosine"])
@pytest.mark.parametrize("lower_op", [0, 1])
@pytest.mark.parametrize("upper_op", [2, 3])
def test_endpoints_match_direct_core_conversion(vindex, metric, lower_op, upper_op):
    ffi = vindex._ffi
    lower = vindex.DistanceEndpoint(0.10000000000000002, lower_op)
    upper = vindex.DistanceEndpoint(1.1000000000000003, upper_op)
    band = vindex.DistanceBand.from_endpoints(metric, lower, upper)
    raw_lower = ffi.PaimonVindexDistanceEndpoint(lower.value, lower_op)
    raw_upper = ffi.PaimonVindexDistanceEndpoint(upper.value, upper_op)
    expected = ffi.PaimonVindexRawDistanceBand()
    assert ffi.lib.paimon_vindex_distance_band_from_endpoints(
        {"l2": 0, "inner_product": 1, "cosine": 2}[metric],
        ctypes.byref(raw_lower), ctypes.byref(raw_upper), ctypes.byref(expected),
    ) == 0
    actual = band.to_ffi()
    assert bytes(actual) == bytes(expected)


def test_empty_batch_and_query_accessor(vindex, flat_index):
    payload, data, _ = flat_index
    with vindex.VectorIndexReader(BytesInput(payload)) as reader:
        params = vindex.RangeSearchParams(vindex.DistanceBand.from_raw("l2"), 4)
        with pytest.raises(RuntimeError, match="query count must be greater than 0"):
            reader.range_search_batch(data[:0], params)
        result = reader.range_search(data[0], params)
    for index in (-1, 1, ctypes.c_size_t(-1).value + 1):
        with pytest.raises(IndexError):
            result.query(index)
    with pytest.raises(TypeError):
        result.query(0.5)


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("filter_kind", ["none", "subset", "empty"])
def test_query_access_shares_owned_payload(vindex, flat_index, batch, filter_kind):
    payload, data, labels = flat_index
    params = vindex.RangeSearchParams(vindex.DistanceBand.from_raw("l2"), 4)
    filter_bytes = {
        "none": None,
        "subset": roaring_allowlist(labels[::3]),
        "empty": roaring_allowlist(labels[:0]),
    }[filter_kind]
    with vindex.VectorIndexReader(BytesInput(payload)) as reader:
        method = reader.range_search_batch if batch else reader.range_search
        result = method(
            data[:3] if batch else data[0], params, roaring_filter=filter_bytes
        )
    retained_views = []
    for query_index in range(result.query_count):
        start, end = int(result.lims[query_index]), int(result.lims[query_index + 1])
        hits = result.query(query_index)
        repeated_hits = result.query(query_index)
        assert hits._fields == ("labels", "raw_distances")
        for owned, view, repeated in zip(
            (result.labels, result.raw_distances),
            hits,
            repeated_hits,
        ):
            assert view.base is owned and repeated.base is owned
            assert view.flags.writeable and repeated.flags.writeable
            np.testing.assert_array_equal(view, owned[start:end])
            if len(view):
                assert np.shares_memory(view, repeated)
                view[0] = -1
                assert owned[start] == repeated[0] == -1
                owned[end - 1] = -2
                assert view[-1] == repeated[-1] == -2
            retained_views.append(view)
    del result, hits, repeated_hits, owned, view, repeated
    for view in retained_views:
        if len(view):
            assert view[-1] == -2


@pytest.mark.parametrize("batch,shape,error", [
    (False, (), ValueError), (False, (1, 16), ValueError),
    (False, (15,), RuntimeError), (False, (0,), RuntimeError),
    (True, (16,), ValueError), (True, (1, 1, 16), ValueError),
    (True, (1, 15), RuntimeError), (True, (0, 15), RuntimeError),
])
def test_query_shapes(vindex, flat_index, batch, shape, error):
    payload, _, _ = flat_index
    params = vindex.RangeSearchParams(vindex.DistanceBand.from_raw("l2"), 4)
    with vindex.VectorIndexReader(BytesInput(payload)) as reader:
        method = reader.range_search_batch if batch else reader.range_search
        with pytest.raises(error):
            method(np.zeros(shape), params)


def test_strided_queries_and_filter_buffer_types(vindex, flat_index):
    payload, data, labels = flat_index
    queries = np.asfortranarray(data[:4].astype(np.float64))[::2]
    params = vindex.RangeSearchParams(vindex.DistanceBand.from_raw("l2"), 4)
    serialized = roaring_allowlist(labels[::3])
    for filter_bytes in (bytearray(serialized), memoryview(serialized)):
        with vindex.VectorIndexReader(BytesInput(payload)) as reader:
            result = reader.range_search_batch(
                queries, params, roaring_filter=filter_bytes
            )
        assert result.query_count == 2
        for query_index in range(2):
            assert set(result.query(query_index)[0]) == set(labels[::3])


def test_query_buffer_overflow_before_copy(vindex, flat_index):
    payload, _, _ = flat_index
    params = vindex.RangeSearchParams(vindex.DistanceBand.from_raw("l2"), 4)
    huge = np.lib.stride_tricks.as_strided(
        np.zeros(1, dtype=np.uint8),
        shape=(np.iinfo(np.intp).max // 32, 16), strides=(0, 0),
    )
    with vindex.VectorIndexReader(BytesInput(payload)) as reader:
        with pytest.raises(ValueError, match="overflow|too large"):
            reader.range_search_batch(huge, params)


@pytest.mark.parametrize("operation", ["single", "batch", "capability"])
def test_closed_reader_and_reentry(vindex, flat_index, operation):
    payload, data, _ = flat_index
    reader = vindex.VectorIndexReader(BytesInput(payload))
    params = vindex.RangeSearchParams(vindex.DistanceBand.from_raw("l2"), 4)
    invoke = {
        "single": lambda: reader.range_search(data[0], params),
        "batch": lambda: reader.range_search_batch(data[:2], params),
        "capability": reader.supports_range_search,
    }[operation]
    with reader._native_handle_lock:
        with pytest.raises(RuntimeError, match="reentrant"):
            invoke()
    reader.close()
    with pytest.raises(RuntimeError, match="closed"):
        invoke()


def test_diskann_is_unsupported(vindex):
    payload, data, _ = make_index(vindex, "diskann")
    with vindex.VectorIndexReader(BytesInput(payload)) as reader:
        assert reader.supports_range_search() is False
        params = vindex.RangeSearchParams(vindex.DistanceBand.from_raw("l2"), 4)
        for query in (data[0], data[:2]):
            method = (
                reader.range_search if query.ndim == 1 else reader.range_search_batch
            )
            with pytest.raises(
                RuntimeError, match="[Uu]nsupported|[Dd]isk[Aa][Nn][Nn]|range"
            ):
                method(query, params)


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("filtered", [False, True])
@pytest.mark.parametrize("failure", [
    "view", "copy_lims", "copy_labels", "copy_raw_distances", "stats", "result",
    "search", "null", "null_raw_distances",
    "null_stats", "null_lims", "length", "overflow", "lims",
])
def test_result_destroyed_on_failure(
    vindex, flat_index, monkeypatch, failure, batch, filtered
):
    payload, data, labels = flat_index
    params = vindex.RangeSearchParams(vindex.DistanceBand.from_raw("l2"), 4)
    ffi = vindex._ffi
    destroyed = []
    destroy = ffi.lib.paimon_vindex_range_search_result_destroy
    view_result = ffi.lib.paimon_vindex_range_search_result_view
    search_name = "paimon_vindex_reader_range_search"
    if batch:
        search_name += "_batch"
    if filtered:
        search_name += "_with_roaring_filter"
    search = getattr(ffi.lib, search_name)
    as_array = np.ctypeslib.as_array
    array_count = 0
    copy_failure_at = {"copy_lims": 1, "copy_labels": 2, "copy_raw_distances": 3}.get(
        failure
    )

    def track_destroy(handle):
        destroyed.append(handle.value)
        destroy(handle)

    def alter_view(handle, output):
        status = view_result(handle, output)
        raw = ctypes.cast(
            output, ctypes.POINTER(ffi.PaimonVindexRangeSearchResultView)
        ).contents
        if failure == "view":
            return -1
        if failure == "null":
            raw.labels = ctypes.POINTER(ctypes.c_int64)()
        if failure == "null_raw_distances":
            raw.raw_distances = ctypes.POINTER(ctypes.c_float)()
        if failure == "null_stats":
            raw.stats = ctypes.POINTER(ffi.PaimonVindexRangeSearchStats)()
        if failure == "null_lims":
            raw.lims = ctypes.POINTER(ctypes.c_size_t)()
        if failure == "length":
            raw.query_count += 1
        if failure == "overflow":
            raw.hit_count = ctypes.c_size_t(-1).value
            raw.lims[raw.query_count] = raw.hit_count
        if failure == "lims":
            raw.lims[0] = 1
        return status

    def failed_search(*args):
        assert search(*args) == 0
        return -1

    def fail_copy(*args, **kwargs):
        raise MemoryError("copy failed")

    def fail_array(*args, **kwargs):
        nonlocal array_count
        array_count += 1
        if array_count == copy_failure_at:
            raise MemoryError("copy failed")
        return as_array(*args, **kwargs)

    monkeypatch.setattr(
        ffi.lib, "paimon_vindex_range_search_result_destroy", track_destroy
    )
    monkeypatch.setattr(ffi.lib, "paimon_vindex_range_search_result_view", alter_view)
    if failure == "search":
        monkeypatch.setattr(ffi.lib, search_name, failed_search)
    if copy_failure_at is not None:
        monkeypatch.setattr(np.ctypeslib, "as_array", fail_array)
    if failure == "stats":
        monkeypatch.setattr(vindex, "RangeSearchStats", fail_copy)
    if failure == "result":
        monkeypatch.setattr(vindex, "RangeSearchResult", fail_copy)
    with vindex.VectorIndexReader(BytesInput(payload)) as reader:
        method = reader.range_search_batch if batch else reader.range_search
        with pytest.raises((MemoryError, RuntimeError, ValueError)):
            method(
                data[:2] if batch else data[0], params,
                roaring_filter=roaring_allowlist(labels[::3]) if filtered else None,
            )
        assert len(destroyed) == 1 and destroyed[0]
        assert reader.supports_range_search()


def test_result_copies_survive_native_destruction(vindex, flat_index, monkeypatch):
    payload, data, _ = flat_index
    params = vindex.RangeSearchParams(vindex.DistanceBand.from_raw("l2"), 4)
    ffi = vindex._ffi
    destroy = ffi.lib.paimon_vindex_range_search_result_destroy
    destroyed = []

    def poison_destroy(handle):
        view = ffi.PaimonVindexRangeSearchResultView()
        assert ffi.lib.paimon_vindex_range_search_result_view(
            handle, ctypes.byref(view)
        ) == 0
        ctypes.memset(
            view.lims, 255, (view.query_count + 1) * ctypes.sizeof(ctypes.c_size_t)
        )
        ctypes.memset(view.labels, 255, view.hit_count * ctypes.sizeof(ctypes.c_int64))
        ctypes.memset(
            view.raw_distances, 255, view.hit_count * ctypes.sizeof(ctypes.c_float)
        )
        ctypes.memset(
            view.stats, 255,
            view.query_count * ctypes.sizeof(ffi.PaimonVindexRangeSearchStats),
        )
        destroyed.append(handle.value)
        destroy(handle)

    monkeypatch.setattr(
        ffi.lib, "paimon_vindex_range_search_result_destroy", poison_destroy
    )
    with vindex.VectorIndexReader(BytesInput(payload)) as reader:
        result = reader.range_search_batch(data[:2], params)
    assert len(destroyed) == 1
    np.testing.assert_array_equal(result.lims, [0, 128, 256])
    assert np.all(result.labels >= (1 << 40))
    assert np.all(np.isfinite(result.raw_distances))
    assert [stats.rows_committed for stats in result.stats] == [128, 128]


@pytest.mark.parametrize("metric", ["l2", "inner_product", "cosine"])
@pytest.mark.parametrize("side", ["lower", "upper"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_endpoints_are_core_errors(vindex, metric, side, value):
    endpoint = vindex.DistanceEndpoint(value, 0 if side == "lower" else 3)
    with pytest.raises(RuntimeError, match="finite"):
        vindex.DistanceBand.from_endpoints(metric, **{side: endpoint})


@pytest.mark.parametrize("side,op", [
    ("lower", 2), ("lower", 3), ("upper", 0), ("upper", 1),
])
def test_wrong_side_endpoint_operators_are_core_errors(vindex, side, op):
    with pytest.raises(RuntimeError, match="endpoint"):
        vindex.DistanceBand.from_endpoints(
            "l2", **{side: vindex.DistanceEndpoint(1.0, op)}
        )


@pytest.mark.parametrize("op", [-1, 4, 1 << 32])
def test_endpoint_operator_cannot_wrap(vindex, op):
    with pytest.raises(ValueError, match="operator"):
        vindex.DistanceEndpoint(1.0, op)


def test_nullable_and_extreme_endpoints(vindex):
    for metric in ("l2", "inner_product", "cosine"):
        assert vindex.DistanceBand.from_endpoints(metric) == vindex.DistanceBand.from_raw(metric)
        assert vindex.DistanceBand.from_endpoints(
            metric, lower=vindex.DistanceEndpoint(0.1, vindex.DistanceEndpointOp.GE)
        ).metric == metric
        assert vindex.DistanceBand.from_endpoints(
            metric, upper=vindex.DistanceEndpoint(1.1, vindex.DistanceEndpointOp.LE)
        ).metric == metric
    with pytest.raises(RuntimeError):
        vindex.DistanceBand.from_endpoints(
            "l2", lower=vindex.DistanceEndpoint(1e300, 0)
        )
    with pytest.raises(TypeError, match="DistanceEndpoint"):
        vindex.DistanceBand.from_endpoints("l2", lower=(1.0, 0))


def test_metric_and_parameter_validation(vindex, flat_index):
    for metric in ("unknown", -1, 1 << 32, None):
        with pytest.raises(ValueError, match="metric"):
            vindex.DistanceBand.from_raw(metric)
    with pytest.raises(TypeError, match="band"):
        vindex.RangeSearchParams("l2", 4)
    payload, data, _ = flat_index
    with vindex.VectorIndexReader(BytesInput(payload)) as reader:
        with pytest.raises(TypeError, match="RangeSearchParams"):
            reader.range_search(data[0], vindex.SearchParams.ivf(4, 4))
        with pytest.raises(RuntimeError, match="metric"):
            reader.range_search(
                data[0], vindex.RangeSearchParams(vindex.DistanceBand.from_raw("cosine"), 4)
            )
        with pytest.raises(ValueError, match="bytes"):
            reader.range_search(
                data[0], vindex.RangeSearchParams(vindex.DistanceBand.from_raw("l2"), 4),
                roaring_filter="not bytes",
            )


def test_range_callback_reentry(vindex, flat_index):
    payload, data, _ = flat_index
    params = vindex.RangeSearchParams(vindex.DistanceBand.from_raw("l2"), 4)

    class ReentrantInput(BytesInput):
        operation = None

        def pread_many(self, ranges):
            if self.operation is not None:
                self.operation()
            return super().pread_many(ranges)

    source = ReentrantInput(payload)
    with vindex.VectorIndexReader(source) as reader:
        errors = []

        def reenter():
            for operation in (
                reader.supports_range_search,
                lambda: reader.range_search(data[0], params),
                lambda: reader.range_search_batch(data[:2], params),
                reader.close,
            ):
                with pytest.raises(RuntimeError, match="reentrant"):
                    operation()
                errors.append(True)

        source.operation = reenter
        result = reader.range_search(data[0], params)
        assert errors and result.hit_count == len(data)


def test_range_close_waits_through_result_copy(vindex, flat_index, monkeypatch):
    payload, data, _ = flat_index
    reader = vindex.VectorIndexReader(BytesInput(payload))
    params = vindex.RangeSearchParams(vindex.DistanceBand.from_raw("l2"), 4)
    copy_entered = threading.Event()
    release_copy = threading.Event()
    close_entered = threading.Event()
    close_done = threading.Event()
    errors = []
    result_view = vindex._ffi.lib.paimon_vindex_range_search_result_view

    def blocking_view(*args):
        status = result_view(*args)
        copy_entered.set()
        assert release_copy.wait(timeout=5)
        return status

    def search():
        try:
            reader.range_search(data[0], params)
        except Exception as error:
            errors.append(error)

    def close():
        close_entered.set()
        reader.close()
        close_done.set()

    monkeypatch.setattr(
        vindex._ffi.lib, "paimon_vindex_range_search_result_view", blocking_view
    )
    search_thread = threading.Thread(target=search)
    close_thread = threading.Thread(target=close)
    search_thread.start()
    try:
        assert copy_entered.wait(timeout=5)
        close_thread.start()
        assert close_entered.wait(timeout=5)
        assert not close_done.wait(timeout=0.05)
    finally:
        release_copy.set()
        search_thread.join(timeout=5)
        if close_thread.ident is not None:
            close_thread.join(timeout=5)
        reader.close()
    assert not search_thread.is_alive() and not close_thread.is_alive()
    assert close_done.is_set() and not errors


def oracle_cases():
    root = os.environ.get("PVI_RANGE_FIXTURES")
    if not root:
        return []
    directory = Path(root)
    return [
        (directory, *line.split())
        for line in (directory / "manifest.txt").read_text().splitlines()
        if line.strip()
    ]


@pytest.mark.parametrize(
    "directory,case_name,index_filename", oracle_cases(), ids=str
)
def test_core_oracle(vindex, directory, case_name, index_filename):
    tokens = iter((directory / f"{case_name}.expected").read_text().split())

    def integers(count):
        return [int(next(tokens)) for _ in range(count)]

    (
        dimension, metric, query_count, nprobe, lower_kind, lower_bits,
        upper_kind, upper_bits, filter_len, hit_count, list_reads,
    ) = integers(11)
    queries = (
        np.array(integers(dimension * query_count), dtype=np.uint32)
        .view(np.float32)
        .reshape(query_count, dimension)
    )
    filter_bytes = bytes(integers(filter_len)) if filter_len else None
    expected_lims = np.array(integers(query_count + 1), dtype=np.uintp)
    expected_labels = np.array(integers(hit_count), dtype=np.int64)
    expected_distances = np.array(integers(hit_count), dtype=np.uint32)
    expected_stats = [tuple(integers(4)) for _ in range(query_count)]
    assert next(tokens, None) is None
    assert lower_kind in (0, 1) and upper_kind in (0, 1)
    lower = (
        struct.unpack("<f", struct.pack("<I", lower_bits))[0] if lower_kind else None
    )
    upper = (
        struct.unpack("<f", struct.pack("<I", upper_bits))[0] if upper_kind else None
    )
    band = vindex.DistanceBand.from_raw(
        {0: "l2", 1: "inner_product", 2: "cosine"}[metric], lower, upper
    )
    source = BytesInput((directory / index_filename).read_bytes())
    with vindex.VectorIndexReader(source) as reader:
        params = vindex.RangeSearchParams(band, nprobe)
        method = reader.range_search if query_count == 1 else reader.range_search_batch
        result = method(
            queries[0] if query_count == 1 else queries, params,
            roaring_filter=filter_bytes,
        )
    np.testing.assert_array_equal(result.lims, expected_lims)
    assert result.hit_count == hit_count
    for query_index in range(query_count):
        start = int(expected_lims[query_index])
        end = int(expected_lims[query_index + 1])
        actual_labels, actual_distances = result.query(query_index)
        actual = Counter(
            zip(actual_labels.tolist(), actual_distances.view(np.uint32).tolist())
        )
        expected = Counter(
            zip(
                expected_labels[start:end].tolist(),
                expected_distances[start:end].tolist(),
            )
        )
        assert actual == expected
    assert result.list_reads == list_reads
    assert [
        (
            stats.lists_probed, stats.rows_scanned,
            stats.rows_committed, stats.early_abandoned,
        )
        for stats in result.stats
    ] == expected_stats
