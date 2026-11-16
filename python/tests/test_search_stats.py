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
import struct

import numpy as np
import pytest

from paimon_vindex import SearchFallbackReason, SearchParams, SearchStrategy, SearchWidth, VectorIndexReader
from test_vindex import VectorIndexInput, build_index


def allowlist(labels):
    labels = sorted(labels)
    return (struct.pack("<QI", 1, 0)
            + struct.pack("<IIHHI", 12346, 1, 0, len(labels) - 1, 16)
            + struct.pack(f"<{len(labels)}H", *labels))


class CountingInput(VectorIndexInput):
    def __init__(self, data):
        super().__init__(data)
        self.reads = [0, 0, 0]

    def pread_many(self, ranges):
        self.reads[0] += 1
        self.reads[1] += len(ranges)
        self.reads[2] += sum(length for _, length in ranges)
        return super().pread_many(ranges)


@pytest.mark.parametrize("family", ["ivf_flat", "ivf_sq", "ivf_pq", "ivf_rq", "diskann"])
def test_search_stats_match_single_batch_filter_and_io(family):
    options = {"index.type": family, "dimension": "16", "metric": "l2"}
    if family != "diskann":
        options["nlist"] = "4"
    data_bytes, data = build_index(options, 16)
    source = CountingInput(data_bytes)
    with VectorIndexReader(source) as reader:
        assert reader.search_stats().query_count == 0
        assert reader.search_stats().strategy == SearchStrategy.NOT_EXECUTED
        params = SearchParams.diskann(3, 100) if family == "diskann" else SearchParams.ivf(3, 4)
        for filtered in (False, True):
            for batch in (False, True):
                source.reads[:] = [0, 0, 0]
                search = reader.search_batch if batch else reader.search
                results = search(data[:4] if batch else data[0], params,
                                 filter_bytes=allowlist([0, 4, 8]) if filtered else None)
                stats = reader.search_stats()
                count = 4 if batch else 1
                assert stats.query_count == count
                assert results[0].shape == ((4, 3) if batch else (3,))
                assert [stats.read_calls, stats.read_ranges, stats.read_bytes] == source.reads
                assert stats.candidate_count >= count * 3
                assert stats.fallback_reason == SearchFallbackReason.NONE
                if family == "diskann":
                    assert stats.search_width == SearchWidth.DISKANN_L_SEARCH
                    assert stats.strategy == (SearchStrategy.FILTERED_EXHAUSTIVE if filtered
                                              else SearchStrategy.DISKANN_GRAPH)
                    assert stats.effective_width == (0 if filtered else 100)
                else:
                    assert stats.strategy == SearchStrategy.IVF
                    assert stats.search_width == SearchWidth.IVF_NPROBE
                    assert stats.initial_width == stats.effective_width == 4
                    assert stats.candidate_count == len(data) * count
                    if family in ("ivf_flat", "ivf_sq") and batch:
                        assert stats.cache_hits > 0
                reader.optimize_for_search()
                assert reader.search_stats() == stats


@pytest.mark.parametrize("batch", [False, True])
def test_python_automatic_filter_expansion_cap_reaches_native_search(batch):
    options = {"index.type": "ivf_flat", "dimension": "16", "nlist": "64", "metric": "l2"}
    data_bytes, data = build_index(options, 16)
    with VectorIndexReader(VectorIndexInput(data_bytes)) as reader:
        method = reader.search_batch if batch else reader.search
        queries = data[[63, 63]] if batch else data[63]
        method(queries, SearchParams.automatic(3))
        baseline = reader.search_stats().initial_width
        bitmap = allowlist(range(63, 512, 64))
        method(queries, SearchParams.automatic(3), filter_bytes=bitmap)
        uncapped = reader.search_stats().initial_width
        params = SearchParams.automatic(3, max_initial_filter_expansion_factor=1)
        results = method(queries, params, filter_bytes=bitmap)
        stats = reader.search_stats()
        assert stats.initial_width == baseline < uncapped
        assert stats.query_count == (2 if batch else 1)
        assert np.all(np.isin(results[0], list(range(63, 512, 64))))
        raw = params.to_ffi_ex()
        assert raw.struct_size == (type(raw).ivfpq_batch_table_reuse_max_bytes.offset
                                   + ctypes.sizeof(ctypes.c_size_t))
        assert raw.max_initial_filter_expansion_factor == 1


@pytest.mark.parametrize("factor", [0, -1, 1.5])
def test_python_rejects_invalid_filter_expansion_factor(factor):
    with pytest.raises((TypeError, ValueError)):
        SearchParams.automatic(3, max_initial_filter_expansion_factor=factor)


def test_python_rejects_cap_with_explicit_width():
    with pytest.raises(ValueError, match="automatic IVF"):
        SearchParams(3, SearchWidth.IVF_NPROBE, 4,
                     max_initial_filter_expansion_factor=1)


def test_older_native_topk_remains_usable(monkeypatch):
    from paimon_vindex import _ffi

    options = {"index.type": "ivf_flat", "dimension": "16", "nlist": "4", "metric": "l2"}
    data_bytes, data = build_index(options, 16)
    with VectorIndexReader(VectorIndexInput(data_bytes)) as reader:
        params = SearchParams.ivf(3, 4)
        expected = reader.search(data[0], params)
        monkeypatch.setattr(_ffi, "HAS_SEARCH_PARAMS_EX", False)
        for actual, reference in zip(reader.search(data[0], params), expected):
            np.testing.assert_array_equal(actual, reference)
        batch = reader.search_batch(data[:2], params)
        np.testing.assert_array_equal(batch[0][0], expected[0])
        with pytest.raises(RuntimeError, match="upgrade"):
            reader.search(data[0], SearchParams.automatic(3, 1))
        monkeypatch.setattr(_ffi, "HAS_SEARCH_STATS", False)
        with pytest.raises(RuntimeError, match="upgrade"):
            reader.search_stats()


@pytest.mark.parametrize("family", ["ivf_flat", "ivf_sq", "ivf_pq", "ivf_rq"])
def test_capped_filtered_single_expands_and_batch_counts_only_new_lists(family):
    options = {"index.type": family, "dimension": "16", "nlist": "64", "metric": "l2"}
    encoded, data = build_index(options, 16)
    bitmap = allowlist(range(0, 512, 64))
    params = SearchParams.automatic(8, max_initial_filter_expansion_factor=1)
    with VectorIndexReader(VectorIndexInput(encoded)) as reader:
        for row, expected_width, expected_candidates in [(0, 8, 64), (63, 64, 960)]:
            labels, distances = reader.search(data[row], params, filter_bytes=bitmap)
            assert labels.shape == distances.shape == (8,)
            np.testing.assert_array_equal(np.sort(labels), np.arange(0, 512, 64))
            assert np.all(np.isfinite(distances))
            stats = reader.search_stats()
            assert stats.query_count == 1
            assert stats.initial_width == 8
            assert stats.effective_width == expected_width
            assert stats.candidate_count == expected_candidates
            assert stats.fallback_reason == (SearchFallbackReason.NONE if row == 0
                                             else SearchFallbackReason.INSUFFICIENT_RESULTS)
        for rows, expected_candidates in [([0, 63], 576), ([0] * 32 + [63], 2560)]:
            labels, distances = reader.search_batch(data[rows], params, filter_bytes=bitmap)
            assert labels.shape == distances.shape == (len(rows), 8)
            np.testing.assert_array_equal(np.sort(labels, axis=1),
                                          np.tile(np.arange(0, 512, 64), (len(rows), 1)))
            stats = reader.search_stats()
            assert stats.query_count == len(rows)
            assert stats.initial_width == 8
            assert stats.effective_width == 64
            assert stats.candidate_count == expected_candidates
            assert stats.fallback_reason == SearchFallbackReason.INSUFFICIENT_RESULTS


@pytest.mark.parametrize("automatic", [False, True])
@pytest.mark.parametrize("bitmap, expected_hits", [(b"\0" * 8, 0), (allowlist([0]), 1)])
def test_pq_single_pads_short_filtered_results_like_batch(automatic, bitmap, expected_hits):
    options = {"index.type": "ivf_pq", "dimension": "16", "nlist": "4", "metric": "l2"}
    encoded, data = build_index(options, 16)
    params = SearchParams.automatic(3, 1) if automatic else SearchParams.ivf(3, 4)
    with VectorIndexReader(VectorIndexInput(encoded)) as reader:
        labels, distances = reader.search(data[0], params, filter_bytes=bitmap)
        batch_labels, batch_distances = reader.search_batch(data[:1], params, filter_bytes=bitmap)
        assert labels.shape == distances.shape == (3,)
        np.testing.assert_array_equal(labels, batch_labels[0])
        np.testing.assert_array_equal(distances, batch_distances[0])
        assert np.count_nonzero(distances != np.finfo(np.float32).max) == expected_hits
        np.testing.assert_array_equal(labels[expected_hits:], [-1] * (3 - expected_hits))


def test_pq_padding_keeps_real_negative_ids():
    import io
    from paimon_vindex import VectorIndexTrainer, VectorIndexWriter

    options = {"index.type": "ivf_pq", "dimension": "16", "nlist": "4", "metric": "l2"}
    _, data = build_index(options, 16)
    labels = np.arange(len(data), dtype=np.int64)
    labels[:2] = [-1, -7]
    output = io.BytesIO()
    with VectorIndexTrainer.train(options, data) as training:
        with VectorIndexWriter(training) as writer:
            writer.add_vectors(labels, data)
            writer.write(output)
    params = SearchParams.ivf(len(data) + 3, 4)
    with VectorIndexReader(VectorIndexInput(output.getvalue())) as reader:
        single_labels, single_distances = reader.search(data[0], params)
        batch_labels, batch_distances = reader.search_batch(data[:1], params)
        np.testing.assert_array_equal(single_labels, batch_labels[0])
        np.testing.assert_array_equal(single_distances, batch_distances[0])
        valid = single_distances != np.finfo(np.float32).max
        np.testing.assert_array_equal(np.sort(single_labels[valid]), np.sort(labels))
        np.testing.assert_array_equal(single_labels[~valid], [-1, -1, -1])
