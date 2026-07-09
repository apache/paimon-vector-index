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

import io

import numpy as np
import pytest

from paimon_vindex import SearchParams, VectorIndexReader, VectorIndexTrainer, VectorIndexWriter


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
                "pq.m": "4",
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
                "index.type": "ivf_hnsw_flat",
                "dimension": "16",
                "nlist": "4",
                "metric": "l2",
            },
            16,
        ),
        (
            {
                "index.type": "ivf_hnsw_sq",
                "dimension": "16",
                "nlist": "4",
                "metric": "l2",
                "hnsw.m": "12",
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

            params = SearchParams(top_k=5, nprobe=4, ef_search=32)
            ids, distances = reader.search(data[0], params)
            reader.optimize_for_search()
            optimized_ids, optimized_distances = reader.search(data[0], params)
            assert ids.shape == (5,)
            assert distances.shape == (5,)
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
            SearchParams(top_k=2, nprobe=2),
        )
        assert ids.shape == (2, 2)
        assert distances.shape == (2, 2)
        assert ids[0, 0] == 0
        assert ids[1, 0] == 1


def test_python_ffi_ivfrq_query_bits():
    index_bytes, data = build_index(
        {
            "index.type": "ivf_rq",
            "dimension": "16",
            "nlist": "4",
            "metric": "l2",
        },
        16,
        n=128,
    )

    with reader_from_bytes(index_bytes) as reader:
        for query_bits in (4, 8):
            ids, distances = reader.search(
                data[7], SearchParams(top_k=5, nprobe=4, query_bits=query_bits)
            )
            assert ids.shape == (5,)
            assert distances.shape == (5,)
            assert ids[0] % 4 == 7 % 4

        ids, distances = reader.search_batch(
            np.vstack([data[4], data[7]]), SearchParams(top_k=5, nprobe=4, query_bits=4)
        )
        assert ids[0, 0] % 4 == 4 % 4
        assert ids[1, 0] % 4 == 7 % 4

        with pytest.raises(RuntimeError, match="query_bits"):
            reader.search(data[0], SearchParams(top_k=5, nprobe=4, query_bits=7))


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
            reader.search(np.zeros(15, dtype=np.float32), SearchParams(top_k=5, nprobe=2))
        with pytest.raises(RuntimeError, match="k must be greater than 0"):
            reader.search(data[0], SearchParams(top_k=0, nprobe=2))
        with pytest.raises(RuntimeError, match="queries length 15"):
            reader.search_batch(np.zeros((1, 15), dtype=np.float32), SearchParams(top_k=5, nprobe=2))
