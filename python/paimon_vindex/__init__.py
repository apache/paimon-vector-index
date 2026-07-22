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

from dataclasses import dataclass
from typing import Mapping, Optional

import ctypes
import numpy as np

from . import _ffi
from ._ffi import lib

INDEX_TYPES = {
    0: "ivf_flat",
    1: "ivf_pq",
    2: "ivf_hnsw_flat",
    3: "ivf_hnsw_sq",
    4: "ivf_rq",
}

METRICS = {
    0: "l2",
    1: "inner_product",
    2: "cosine",
}


@dataclass(frozen=True)
class VectorIndexMetadata:
    index_type: str
    dimension: int
    nlist: int
    metric: str
    total_vectors: int
    pq_m: Optional[int] = None
    hnsw_m: Optional[int] = None
    hnsw_ef_construction: Optional[int] = None
    hnsw_max_level: Optional[int] = None


@dataclass(frozen=True)
class SearchParams:
    top_k: int
    nprobe: int
    ef_search: int = 0
    query_bits: int = 0

    def to_ffi(self):
        return _ffi.PaimonVindexSearchParams(
            self.top_k,
            self.nprobe,
            self.ef_search,
            self.query_bits,
        )


def _check_error(message="operation failed"):
    err = lib.paimon_vindex_last_error()
    if err:
        raise RuntimeError(err.decode("utf-8", errors="replace"))
    raise RuntimeError(message)


def _metadata_from_ffi(raw):
    return VectorIndexMetadata(
        index_type=INDEX_TYPES.get(raw.index_type, f"unknown_{raw.index_type}"),
        dimension=raw.dimension,
        nlist=raw.nlist,
        metric=METRICS.get(raw.metric, f"unknown_{raw.metric}"),
        total_vectors=raw.total_vectors,
        pq_m=raw.pq_m or None,
        hnsw_m=raw.hnsw_m or None,
        hnsw_ef_construction=raw.hnsw_ef_construction or None,
        hnsw_max_level=raw.hnsw_max_level or None,
    )


def _float32_matrix(value, name):
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional float32 array")
    return np.ascontiguousarray(array)


def _float32_vector(value, name):
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional float32 array")
    return np.ascontiguousarray(array)


def _int64_vector(value, name):
    array = np.asarray(value, dtype=np.int64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional int64 array")
    return np.ascontiguousarray(array)


def _bytes_buffer(value, name):
    if isinstance(value, memoryview):
        value = value.tobytes()
    if not isinstance(value, (bytes, bytearray)):
        raise ValueError(f"{name} must be bytes")
    data = bytes(value)
    if not data:
        return None, 0, data
    buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
    return buf, len(data), data


def _option_arrays(options: Mapping[str, str]):
    option_items = list(options.items())
    key_bytes = []
    value_bytes = []
    for key, value in option_items:
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("options must be a mapping of str to str")
        key_bytes.append(key.encode("utf-8"))
        value_bytes.append(value.encode("utf-8"))
    keys = (ctypes.c_char_p * len(key_bytes))(*key_bytes)
    values = (ctypes.c_char_p * len(value_bytes))(*value_bytes)
    return option_items, key_bytes, value_bytes, keys, values


def _make_read_ranges_callback(input):
    @_ffi.READ_RANGES_FN
    def read_ranges_callback(ctx, requests, request_count):
        try:
            ranges = [
                (requests[i].offset, requests[i].len)
                for i in range(request_count)
            ]
            chunks = input.pread_many(ranges)
            if len(chunks) != request_count:
                return -1
            for i, chunk in enumerate(chunks):
                data = bytes(chunk)
                if len(data) != requests[i].len:
                    return -1
                ctypes.memmove(requests[i].buf, data, len(data))
            return 0
        except Exception:
            return -1

    return read_ranges_callback


class VectorIndexTraining:
    def __init__(self, handle):
        self._closed = False
        self._handle = handle

    def _require_open(self):
        if self._closed or not self._handle:
            raise RuntimeError("VectorIndexTraining is closed")

    def _take_handle(self):
        self._require_open()
        handle = self._handle
        self._handle = None
        self._closed = True
        return handle

    def close(self):
        if self._handle:
            lib.paimon_vindex_training_free(self._handle)
            self._handle = None
        self._closed = True

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class VectorIndexTrainer:
    def __init__(self, options: Mapping[str, str]):
        self._closed = False
        (
            option_items,
            self._key_bytes,
            self._value_bytes,
            self._keys,
            self._values,
        ) = _option_arrays(options)
        self._handle = lib.paimon_vindex_trainer_open(
            self._keys,
            self._values,
            len(option_items),
        )
        if not self._handle:
            _check_error("failed to open trainer")
        self._dimension = self._read_dimension()

    @classmethod
    def create(cls, options: Mapping[str, str]):
        return cls(options)

    @classmethod
    def train(cls, options: Mapping[str, str], data):
        with cls(options) as trainer:
            return trainer.add_training_vectors(data).finish_training()

    def _require_open(self):
        if self._closed or not self._handle:
            raise RuntimeError("VectorIndexTrainer is closed")

    def _read_dimension(self):
        out = ctypes.c_size_t(0)
        rc = lib.paimon_vindex_trainer_dimension(self._handle, ctypes.byref(out))
        if rc != 0:
            _check_error("trainer dimension failed")
        return out.value

    @property
    def dimension(self):
        self._require_open()
        return self._dimension

    def add_training_vectors(self, data):
        self._require_open()
        data = _float32_matrix(data, "data")
        if data.shape[1] != self._dimension:
            raise RuntimeError(
                f"training data length {data.size} does not match vector count "
                f"* dimension {data.shape[0] * self._dimension}"
            )
        rc = lib.paimon_vindex_trainer_add_training_vectors(
            self._handle,
            data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            data.shape[0],
        )
        if rc != 0:
            _check_error("add training vectors failed")
        return self

    def finish_training(self):
        self._require_open()
        handle = self._handle
        training = lib.paimon_vindex_trainer_finish(handle)
        lib.paimon_vindex_trainer_free(handle)
        self._handle = None
        self._closed = True
        if not training:
            _check_error("finish training failed")
        return VectorIndexTraining(training)

    def close(self):
        if self._handle:
            lib.paimon_vindex_trainer_free(self._handle)
            self._handle = None
        self._closed = True

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class VectorIndexWriter:
    def __init__(self, training: VectorIndexTraining):
        if not isinstance(training, VectorIndexTraining):
            raise TypeError("training must be a VectorIndexTraining")
        self._closed = False
        training_handle = training._take_handle()
        self._handle = lib.paimon_vindex_writer_open(training_handle)
        lib.paimon_vindex_training_free(training_handle)
        if not self._handle:
            _check_error("failed to open writer")
        self._dimension = self._read_dimension()

    def _require_open(self):
        if self._closed or not self._handle:
            raise RuntimeError("VectorIndexWriter is closed")

    def _read_dimension(self):
        out = ctypes.c_size_t(0)
        rc = lib.paimon_vindex_writer_dimension(self._handle, ctypes.byref(out))
        if rc != 0:
            _check_error("writer dimension failed")
        return out.value

    @property
    def dimension(self):
        self._require_open()
        return self._dimension

    def add_vectors(self, ids, data):
        self._require_open()
        data = _float32_matrix(data, "data")
        ids = _int64_vector(ids, "ids")
        if data.shape[1] != self._dimension:
            raise RuntimeError(
                f"vector data length {data.size} does not match vector count "
                f"* dimension {data.shape[0] * self._dimension}"
            )
        if ids.shape[0] != data.shape[0]:
            raise RuntimeError(
                f"ids length {ids.shape[0]} does not match vector count {data.shape[0]}"
            )
        rc = lib.paimon_vindex_writer_add_vectors(
            self._handle,
            ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            data.shape[0],
        )
        if rc != 0:
            _check_error("add_vectors failed")

    def write(self, file):
        self._require_open()
        pos = 0

        @_ffi.WRITE_FN
        def write_callback(ctx, data, length):
            nonlocal pos
            try:
                payload = ctypes.string_at(data, length)
                written = file.write(payload)
                if written is not None and written != length:
                    return -1
                pos += length
                return 0
            except Exception:
                return -1

        @_ffi.FLUSH_FN
        def flush_callback(ctx):
            try:
                flush = getattr(file, "flush", None)
                if flush is not None:
                    flush()
                return 0
            except Exception:
                return -1

        @_ffi.GET_POS_FN
        def pos_callback(ctx):
            return pos

        output = _ffi.PaimonVindexOutputFile()
        output.ctx = None
        output.write_fn = write_callback
        output.flush_fn = flush_callback
        output.get_pos_fn = pos_callback

        rc = lib.paimon_vindex_writer_write_index(self._handle, output)
        if rc != 0:
            _check_error("write index failed")

    def close(self):
        if self._handle:
            lib.paimon_vindex_writer_free(self._handle)
            self._handle = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class VectorIndexReader:
    def __init__(self, input):
        self._input = input
        self._closed = False

        self._read_ranges_callback = _make_read_ranges_callback(self._input)
        input_file = _ffi.PaimonVindexInputFile()
        input_file.ctx = None
        input_file.read_ranges_fn = self._read_ranges_callback
        self._handle = lib.paimon_vindex_reader_open(input_file)
        if not self._handle:
            _check_error("failed to open reader")
        self._metadata = self.metadata()

    def _require_open(self):
        if self._closed or not self._handle:
            raise RuntimeError("VectorIndexReader is closed")

    @property
    def index_type(self):
        return self.metadata().index_type

    @property
    def dimension(self):
        return self.metadata().dimension

    @property
    def nlist(self):
        return self.metadata().nlist

    @property
    def total_vectors(self):
        return self.metadata().total_vectors

    def metadata(self):
        self._require_open()
        raw = _ffi.PaimonVindexMetadata()
        rc = lib.paimon_vindex_reader_metadata(self._handle, ctypes.byref(raw))
        if rc != 0:
            _check_error("metadata failed")
        return _metadata_from_ffi(raw)

    def optimize_for_search(self):
        self._require_open()
        rc = lib.paimon_vindex_reader_optimize_for_search(self._handle)
        if rc != 0:
            _check_error("optimize_for_search failed")

    def _filter_args(self, filter_bytes):
        if filter_bytes is None:
            return None, 0, None
        return _bytes_buffer(filter_bytes, "filter_bytes")

    def search(self, query, params: SearchParams, filter_bytes=None):
        self._require_open()
        query = _float32_vector(query, "query")
        if query.shape[0] != self._metadata.dimension:
            raise RuntimeError(
                f"query length {query.shape[0]} does not match index dimension "
                f"{self._metadata.dimension}"
            )
        ffi_params = params.to_ffi()
        ids = np.empty(params.top_k, dtype=np.int64)
        distances = np.empty(params.top_k, dtype=np.float32)

        if filter_bytes is None:
            rc = lib.paimon_vindex_reader_search(
                self._handle,
                query.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ffi_params,
                ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                distances.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                params.top_k,
            )
        else:
            filter_buf, filter_len, _ = self._filter_args(filter_bytes)
            rc = lib.paimon_vindex_reader_search_with_roaring_filter(
                self._handle,
                query.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ffi_params,
                filter_buf,
                filter_len,
                ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                distances.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                params.top_k,
            )
        if rc != 0:
            _check_error("search failed")
        return ids, distances

    def search_batch(self, queries, params: SearchParams, filter_bytes=None):
        self._require_open()
        queries = _float32_matrix(queries, "queries")
        if queries.shape[1] != self._metadata.dimension:
            raise RuntimeError(
                f"queries length {queries.size} does not match nq * dimension "
                f"{queries.shape[0] * self._metadata.dimension}"
            )
        ffi_params = params.to_ffi()
        result_len = queries.shape[0] * params.top_k
        ids = np.empty((queries.shape[0], params.top_k), dtype=np.int64)
        distances = np.empty((queries.shape[0], params.top_k), dtype=np.float32)

        if filter_bytes is None:
            rc = lib.paimon_vindex_reader_search_batch(
                self._handle,
                queries.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                queries.shape[0],
                ffi_params,
                ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                distances.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                result_len,
            )
        else:
            filter_buf, filter_len, _ = self._filter_args(filter_bytes)
            rc = lib.paimon_vindex_reader_search_batch_with_roaring_filter(
                self._handle,
                queries.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                queries.shape[0],
                ffi_params,
                filter_buf,
                filter_len,
                ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                distances.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                result_len,
            )
        if rc != 0:
            _check_error("batch search failed")
        return ids, distances

    def close(self):
        if self._handle:
            lib.paimon_vindex_reader_free(self._handle)
            self._handle = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


__all__ = [
    "VectorIndexMetadata",
    "VectorIndexReader",
    "VectorIndexTrainer",
    "VectorIndexTraining",
    "VectorIndexWriter",
]
