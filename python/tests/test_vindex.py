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

from paimon_vindex import VectorIndexReader, VectorIndexWriter


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
    with VectorIndexWriter(options) as writer:
        writer.train(data)
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

            ids, distances = reader.search(data[0], top_k=5, nprobe=4, ef_search=32)
            assert ids.shape == (5,)
            assert distances.shape == (5,)
            assert ids[0] == 0


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
            top_k=2,
            nprobe=2,
        )
        assert ids.shape == (2, 2)
        assert distances.shape == (2, 2)
        assert ids[0, 0] == 0
        assert ids[1, 0] == 1


def test_python_ffi_delegates_validation():
    options = {
        "index.type": "ivf_pq",
        "dimension": "16",
        "nlist": "4",
        "pq.m": "4",
        "metric": "l2",
    }
    writer = VectorIndexWriter(options)
    with pytest.raises(RuntimeError, match="training data length 17"):
        writer.train(np.zeros((1, 17), dtype=np.float32))

    data = np.zeros((1, 16), dtype=np.float32)
    ids = np.array([1, 2], dtype=np.int64)
    with pytest.raises(RuntimeError, match="ids length 2 does not match vector count 1"):
        writer.add_vectors(ids, data)
    writer.close()

    index_bytes, data = build_index(options, 16)
    with reader_from_bytes(index_bytes) as reader:
        with pytest.raises(RuntimeError, match="query length 15"):
            reader.search(np.zeros(15, dtype=np.float32), top_k=5, nprobe=2)
        with pytest.raises(RuntimeError, match="k must be greater than 0"):
            reader.search(data[0], top_k=0, nprobe=2)
        with pytest.raises(RuntimeError, match="queries length 15"):
            reader.search_batch(np.zeros((1, 15), dtype=np.float32), top_k=5, nprobe=2)
