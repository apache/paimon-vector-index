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
import operator
import threading
from dataclasses import dataclass
from enum import IntEnum
from typing import Mapping, Optional

import numpy as np

from . import _ffi
from ._ffi import lib

_SIZE_T_MAX = ctypes.c_size_t(-1).value
_UINT64_MAX = ctypes.c_uint64(-1).value
_DEFAULT_IVFPQ_BATCH_TABLE_REUSE_MAX_BYTES = 512 * 1024 * 1024


def _size_t(value, name: str, *, allow_zero: bool) -> int:
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    lower_bound = 0 if allow_zero else 1
    if not lower_bound <= value <= _SIZE_T_MAX:
        raise ValueError(
            f"{name} must be in [{lower_bound}, {_SIZE_T_MAX}]"
        )
    return value


def _uint64(value, name: str) -> int:
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not 0 <= value <= _UINT64_MAX:
        raise ValueError(f"{name} must be in [0, {_UINT64_MAX}]")
    return value


class _NativeHandleLock:
    """Serialize a native handle while rejecting reentry from its callbacks."""

    def __init__(self):
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._owner = None
        self._callback_depths = {}

    def __enter__(self):
        current = threading.get_ident()
        with self._state_lock:
            if (
                self._owner == current
                or self._callback_depths.get(current, 0) > 0
            ):
                raise RuntimeError(
                    "reentrant native-handle operation is not allowed"
                )
        self._lock.acquire()
        with self._state_lock:
            self._owner = current
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        with self._state_lock:
            self._owner = None
        self._lock.release()
        return False

    def _enter_callback(self):
        current = threading.get_ident()
        with self._state_lock:
            self._callback_depths[current] = (
                self._callback_depths.get(current, 0) + 1
            )

    def _exit_callback(self):
        current = threading.get_ident()
        with self._state_lock:
            depth = self._callback_depths[current]
            if depth == 1:
                del self._callback_depths[current]
            else:
                self._callback_depths[current] = depth - 1

INDEX_TYPES = {
    0: "ivf_flat",
    1: "ivf_pq",
    4: "ivf_rq",
    5: "diskann",
    6: "ivf_sq",
}

METRICS = {
    0: "l2",
    1: "inner_product",
    2: "cosine",
}


class SearchWidth(IntEnum):
    AUTO = 0
    IVF_NPROBE = 1
    DISKANN_L_SEARCH = 2


class IvfPqBatchTableReuseMode(IntEnum):
    OFF = 0
    ON = 1
    AUTO = 2


@dataclass(frozen=True)
class VectorIndexMetadata:
    index_type: str
    dimension: int
    nlist: int
    metric: str
    total_vectors: int
    pq_m: Optional[int] = None
    pq_bits: Optional[int] = None
    rq_bits: Optional[int] = None
    diskann_max_degree: Optional[int] = None
    diskann_build_search_list_size: Optional[int] = None
    diskann_alpha: Optional[float] = None


@dataclass(frozen=True)
class VectorIndexReadPlan:
    random_read_latency_nanos: int
    window_bytes: int
    max_ranges_per_read: int
    graph_beam_width: int
    filtered_graph_beam_width: int
    adjacency_preload_bytes: int
    adjacency_cache_bytes: int
    raw_vector_cache_bytes: int
    memory_budget_bytes: int


@dataclass(frozen=True)
class SearchParams:
    top_k: int
    search_width: SearchWidth = SearchWidth.AUTO
    width: int = 0
    ivfpq_batch_table_reuse: IvfPqBatchTableReuseMode = (
        IvfPqBatchTableReuseMode.AUTO
    )
    ivfpq_batch_table_reuse_max_bytes: int = (
        _DEFAULT_IVFPQ_BATCH_TABLE_REUSE_MAX_BYTES
    )

    def __post_init__(self):
        try:
            search_width = SearchWidth(self.search_width)
        except (TypeError, ValueError) as exc:
            raise ValueError("search_width is invalid") from exc
        try:
            reuse_mode = IvfPqBatchTableReuseMode(self.ivfpq_batch_table_reuse)
        except (TypeError, ValueError) as exc:
            raise ValueError("IVF-PQ batch table reuse mode is invalid") from exc
        top_k = _size_t(self.top_k, "top_k", allow_zero=False)
        if search_width == SearchWidth.AUTO:
            width = _size_t(self.width, "automatic search width", allow_zero=True)
            if width:
                raise ValueError("automatic search width must be zero")
        else:
            name = (
                "nprobe"
                if search_width == SearchWidth.IVF_NPROBE
                else "l_search"
            )
            width = _size_t(self.width, name, allow_zero=False)
        reuse_max_bytes = _size_t(
            self.ivfpq_batch_table_reuse_max_bytes,
            "IVF-PQ batch table reuse max bytes",
            allow_zero=False,
        )
        object.__setattr__(self, "top_k", top_k)
        object.__setattr__(self, "search_width", search_width)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "ivfpq_batch_table_reuse", reuse_mode)
        object.__setattr__(
            self, "ivfpq_batch_table_reuse_max_bytes", reuse_max_bytes
        )

    @classmethod
    def automatic(cls, top_k: int):
        return cls(top_k=top_k)

    @classmethod
    def ivf(
        cls,
        top_k: int,
        nprobe: int,
        ivfpq_batch_table_reuse=IvfPqBatchTableReuseMode.AUTO,
        ivfpq_batch_table_reuse_max_bytes=_DEFAULT_IVFPQ_BATCH_TABLE_REUSE_MAX_BYTES,
    ):
        return cls(
            top_k=top_k,
            search_width=SearchWidth.IVF_NPROBE,
            width=nprobe,
            ivfpq_batch_table_reuse=ivfpq_batch_table_reuse,
            ivfpq_batch_table_reuse_max_bytes=ivfpq_batch_table_reuse_max_bytes,
        )

    @classmethod
    def diskann(cls, top_k: int, l_search: int):
        return cls(
            top_k=top_k,
            search_width=SearchWidth.DISKANN_L_SEARCH,
            width=l_search,
        )

    def to_ffi(self):
        return _ffi.PaimonVindexSearchParams(
            self.top_k,
            int(self.search_width),
            self.width,
        )

    def to_ffi_v2(self):
        return _ffi.PaimonVindexSearchParamsV2(
            self.top_k,
            int(self.search_width),
            self.width,
            int(self.ivfpq_batch_table_reuse),
            self.ivfpq_batch_table_reuse_max_bytes,
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
        pq_bits=raw.pq_bits or None,
        rq_bits=raw.rq_bits or None,
        diskann_max_degree=raw.diskann_max_degree or None,
        diskann_build_search_list_size=raw.diskann_build_search_list_size or None,
        diskann_alpha=raw.diskann_alpha or None,
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


def _make_read_ranges_callback(input, native_handle_lock=None):
    @_ffi.READ_RANGES_FN
    def read_ranges_callback(ctx, requests, request_count):
        if native_handle_lock is not None:
            native_handle_lock._enter_callback()
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
        finally:
            if native_handle_lock is not None:
                native_handle_lock._exit_callback()

    return read_ranges_callback


class VectorIndexTraining:
    def __init__(self, handle):
        self._native_handle_lock = _NativeHandleLock()
        self._closed = False
        self._handle = handle

    def _require_open(self):
        if self._closed or not self._handle:
            raise RuntimeError("VectorIndexTraining is closed")

    def _take_handle(self):
        with self._native_handle_lock:
            self._require_open()
            handle = self._handle
            self._handle = None
            self._closed = True
            return handle

    def close(self):
        with self._native_handle_lock:
            if self._handle:
                lib.paimon_vindex_training_free(self._handle)
                self._handle = None
            self._closed = True

    def __enter__(self):
        with self._native_handle_lock:
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


@dataclass(frozen=True)
class PreparedTrainingInfo:
    dimension: int
    nlist: int
    sample_count: int
    calibration_count: int
    vectors_seen: int
    iterations: int
    restarts: int
    seed: int
    metric: str


class PreparedIvfSqTraining:
    """Owned IVF-SQ sample for external centroid training.

    Samples are already preprocessed (normalized for cosine). Fit centers with
    squared L2 even for IP/cosine indexes. Finishing calibrates residual SQ on
    CPU and consumes the native state, including on native validation failure.
    """

    def __init__(self, handle):
        self._native_handle_lock = _NativeHandleLock()
        self._handle = handle
        try:
            info = _ffi.PaimonVindexPreparedTrainingInfo()
            if lib.paimon_vindex_prepared_training_info(handle, ctypes.byref(info)) != 0:
                _check_error("prepared training info failed")
            fields = {name: getattr(info, name) for name, _ in info._fields_}
            fields["metric"] = METRICS[info.metric]
            self._info = PreparedTrainingInfo(**fields)
        except Exception:
            self.close()
            raise

    def _require_open(self):
        if not self._handle:
            raise RuntimeError("PreparedIvfSqTraining is closed")

    @property
    def info(self):
        return self._info

    @property
    def sample(self):
        """Return an independent, read-only float32 copy of the effective sample."""
        with self._native_handle_lock:
            self._require_open()
            sample = np.empty((self.info.sample_count, self.info.dimension), dtype=np.float32)
            rc = lib.paimon_vindex_prepared_training_copy_sample(
                self._handle, sample.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), sample.size,
            )
            if rc != 0:
                _check_error("copy training sample failed")
            sample.flags.writeable = False
            return sample

    def fit_centroids_cpu(self, algorithm="auto", initial_centroids=None):
        """Fit the existing CPU strategy or flat Lloyd for GPU comparisons."""
        if algorithm not in ("auto", "lloyd"):
            raise ValueError("algorithm must be auto or lloyd")
        initial = None
        if initial_centroids is not None:
            initial = _float32_matrix(initial_centroids, "initial_centroids")
            if initial.shape != (self.info.nlist, self.info.dimension):
                raise ValueError("initial centroid shape must be (nlist, dimension)")
        centers = np.empty((self.info.nlist, self.info.dimension), dtype=np.float32)
        with self._native_handle_lock:
            self._require_open()
            rc = lib.paimon_vindex_prepared_training_fit_cpu(
                self._handle, int(algorithm == "lloyd"),
                None if initial is None else initial.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                0 if initial is None else initial.size,
                centers.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), centers.size,
            )
            if rc != 0:
                _check_error("CPU centroid training failed")
        return centers

    def finish_with_ivf_centroids(self, centroids):
        centroids = _float32_matrix(centroids, "centroids")
        if centroids.shape != (self.info.nlist, self.info.dimension):
            raise ValueError("centroid shape must be (nlist, dimension)")
        with self._native_handle_lock:
            self._require_open()
            handle = self._handle
            training = lib.paimon_vindex_prepared_training_finish(
                handle, centroids.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), centroids.size,
            )
            lib.paimon_vindex_prepared_training_free(handle)
            self._handle = None
            if not training:
                _check_error("finish external centroid training failed")
            return VectorIndexTraining(training)

    def close(self):
        with self._native_handle_lock:
            if self._handle:
                lib.paimon_vindex_prepared_training_free(self._handle)
                self._handle = None

    def __enter__(self):
        with self._native_handle_lock:
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
        self._native_handle_lock = _NativeHandleLock()
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
        data = _float32_matrix(data, "data")
        resolved_options = dict(options)
        if resolved_options.get("dimension") in (None, "auto"):
            resolved_options["dimension"] = str(data.shape[1])
        if (
            resolved_options.get("index.type", "").startswith("ivf_")
            and resolved_options.get("nlist") in (None, "auto")
            and "expected-vector-count" not in resolved_options
        ):
            resolved_options["expected-vector-count"] = str(data.shape[0])
        with cls(resolved_options) as trainer:
            return trainer.add_training_vectors(data).finish_training()

    def _require_open(self):
        if self._closed or not self._handle:
            raise RuntimeError("VectorIndexTrainer is closed")

    def _read_dimension(self):
        with self._native_handle_lock:
            self._require_open()
            out = ctypes.c_size_t(0)
            rc = lib.paimon_vindex_trainer_dimension(
                self._handle, ctypes.byref(out)
            )
            if rc != 0:
                _check_error("trainer dimension failed")
            return out.value

    @property
    def dimension(self):
        with self._native_handle_lock:
            self._require_open()
            return self._dimension

    def add_training_vectors(self, data):
        data = _float32_matrix(data, "data")
        with self._native_handle_lock:
            self._require_open()
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

    def prepare_training(self):
        """Consume this IVF-SQ trainer and freeze its sample for external fitting."""
        with self._native_handle_lock:
            self._require_open()
            handle = self._handle
            prepared = lib.paimon_vindex_trainer_prepare(handle)
            lib.paimon_vindex_trainer_free(handle)
            self._handle = None
            self._closed = True
            if not prepared:
                _check_error("prepare training failed")
            return PreparedIvfSqTraining(prepared)

    def finish_training(self):
        with self._native_handle_lock:
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
        with self._native_handle_lock:
            if self._handle:
                lib.paimon_vindex_trainer_free(self._handle)
                self._handle = None
            self._closed = True

    def __enter__(self):
        with self._native_handle_lock:
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


@dataclass(frozen=True)
class IvfSqEncodingModel:
    """Independent model snapshot; arrays have shape (nlist, dimension)."""
    dimension: int
    nlist: int
    metric: str
    exact_assignment: bool
    centroids: np.ndarray
    mins: np.ndarray
    maxs: np.ndarray
    _encoding_vector_width: int


def _partition_ids(value):
    array = np.asarray(value)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError("partition_ids must be a one-dimensional integer array")
    if np.any(array < 0) or np.any(array > np.iinfo(np.uint32).max):
        raise ValueError("partition_ids must fit uint32")
    return np.ascontiguousarray(array, dtype=np.uint32)


class VectorIndexWriter:
    def __init__(self, training: VectorIndexTraining):
        if not isinstance(training, VectorIndexTraining):
            raise TypeError("training must be a VectorIndexTraining")
        self._native_handle_lock = _NativeHandleLock()
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
        with self._native_handle_lock:
            self._require_open()
            out = ctypes.c_size_t(0)
            rc = lib.paimon_vindex_writer_dimension(
                self._handle, ctypes.byref(out)
            )
            if rc != 0:
                _check_error("writer dimension failed")
            return out.value

    @property
    def dimension(self):
        with self._native_handle_lock:
            self._require_open()
            return self._dimension

    def add_vectors(self, ids, data):
        data = _float32_matrix(data, "data")
        ids = _int64_vector(ids, "ids")
        with self._native_handle_lock:
            self._require_open()
            if data.shape[1] != self._dimension:
                raise RuntimeError(
                    f"vector data length {data.size} does not match vector count "
                    f"* dimension {data.shape[0] * self._dimension}"
                )
            if ids.shape[0] != data.shape[0]:
                raise RuntimeError(
                    f"ids length {ids.shape[0]} does not match vector count "
                    f"{data.shape[0]}"
                )
            rc = lib.paimon_vindex_writer_add_vectors(
                self._handle,
                ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                data.shape[0],
            )
            if rc != 0:
                _check_error("add_vectors failed")

    def ivf_sq_partition_sizes(self):
        """Return actual per-partition row counts for an IVF-SQ writer."""
        with self._native_handle_lock:
            self._require_open()
            nlist = ctypes.c_size_t()
            rc = lib.paimon_vindex_writer_ivf_sq_partition_sizes(
                self._handle, None, 0, ctypes.byref(nlist),
            )
            if rc != 0:
                _check_error("partition sizes failed")
            sizes = np.empty(nlist.value, dtype=np.uintp)
            rc = lib.paimon_vindex_writer_ivf_sq_partition_sizes(
                self._handle, sizes.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
                sizes.size, ctypes.byref(nlist),
            )
            if rc != 0:
                _check_error("partition sizes failed")
            return sizes

    def ivf_sq_encoding_model(self):
        """Copy centers and SQ bounds for an external encoder using this writer."""
        with self._native_handle_lock:
            self._require_open()
            info = _ffi.PaimonVindexIvfSqEncodingInfo()
            if lib.paimon_vindex_writer_ivf_sq_encoding_info(self._handle, ctypes.byref(info)) != 0:
                _check_error("encoding model info failed")
            arrays = [np.empty((info.nlist, info.dimension), dtype=np.float32) for _ in range(3)]
            pointers = [a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) for a in arrays]
            if lib.paimon_vindex_writer_ivf_sq_copy_model(self._handle, *pointers, arrays[0].size) != 0:
                _check_error("copy encoding model failed")
            for array in arrays:
                array.flags.writeable = False
            return IvfSqEncodingModel(info.dimension, info.nlist, METRICS[info.metric],
                                      bool(info.exact_assignment), *arrays, info.encoding_vector_width)

    def _preprocess_ivf_sq_vectors(self, data):
        data = _float32_matrix(data, "data")
        if data.shape[1] != self._dimension:
            raise ValueError("data dimension does not match writer")
        result = np.empty(data.shape, dtype=np.float32)
        with self._native_handle_lock:
            self._require_open()
            rc = lib.paimon_vindex_writer_ivf_sq_preprocess(
                self._handle, data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(data),
                result.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), result.size)
            if rc != 0:
                _check_error("preprocess vectors failed")
        return result

    def add_preassigned_vectors(self, ids, data, partition_ids):
        """Add raw vectors with external labels; native code preprocesses and encodes.

        Labels must refer to this writer's centers. Validation failures append no rows.
        """
        data = _float32_matrix(data, "data")
        ids = _int64_vector(ids, "ids")
        lists = _partition_ids(partition_ids)
        if data.shape != (len(ids), self._dimension) or len(lists) != len(ids):
            raise ValueError("data, IDs and partition IDs must have matching shapes")
        with self._native_handle_lock:
            self._require_open()
            rc = lib.paimon_vindex_writer_add_preassigned_vectors(
                self._handle, ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                lists.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)), len(ids))
            if rc != 0:
                _check_error("add preassigned vectors failed")

    def add_encoded_vectors(self, ids, codes, partition_ids):
        """Append row-major uint8 SQ8 codes made with this writer's encoding model.

        Validates shapes and labels; the caller is responsible for the encoding
        algorithm and model. Validation failures append no rows.
        """
        ids = _int64_vector(ids, "ids")
        lists = _partition_ids(partition_ids)
        codes = np.asarray(codes)
        if codes.dtype != np.uint8 or codes.shape != (len(ids), self._dimension) or len(lists) != len(ids):
            raise ValueError("codes must be uint8 (len(ids), dimension), with one partition ID per row")
        codes = np.ascontiguousarray(codes)
        with self._native_handle_lock:
            self._require_open()
            rc = lib.paimon_vindex_writer_add_encoded_vectors(
                self._handle, ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                codes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), codes.size,
                lists.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)), len(ids))
            if rc != 0:
                _check_error("add encoded vectors failed")

    def write(self, file):
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

        with self._native_handle_lock:
            self._require_open()
            rc = lib.paimon_vindex_writer_write_index(self._handle, output)
            if rc != 0:
                _check_error("write index failed")

    def close(self):
        with self._native_handle_lock:
            if self._handle:
                lib.paimon_vindex_writer_free(self._handle)
                self._handle = None
            self._closed = True

    def __enter__(self):
        with self._native_handle_lock:
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


class VectorIndexReader:
    def __init__(
        self,
        input,
        memory_budget_bytes: int = 4 * 1024 * 1024 * 1024,
    ):
        self._native_handle_lock = _NativeHandleLock()
        self._input = input
        self._closed = False

        memory_budget_bytes = _size_t(
            memory_budget_bytes, "memory_budget_bytes", allow_zero=True
        )

        self._read_ranges_callback = _make_read_ranges_callback(
            self._input, self._native_handle_lock
        )
        input_file = _ffi.PaimonVindexInputFile()
        input_file.ctx = None
        input_file.read_ranges_fn = self._read_ranges_callback
        capability_names = (
            "estimated_random_read_latency_nanos",
            "preferred_window_bytes",
            "max_ranges_per_read",
        )
        capabilities = {}
        for name in capability_names:
            value = getattr(self._input, name, 0)
            if name == "estimated_random_read_latency_nanos":
                capabilities[name] = _uint64(value, f"input.{name}")
            else:
                capabilities[name] = _size_t(
                    value, f"input.{name}", allow_zero=True
                )
        input_file.estimated_random_read_latency_nanos = capabilities[
            "estimated_random_read_latency_nanos"
        ]
        input_file.preferred_window_bytes = capabilities["preferred_window_bytes"]
        input_file.max_ranges_per_read = capabilities["max_ranges_per_read"]
        options = _ffi.PaimonVindexReaderOptions(memory_budget_bytes)
        self._handle = lib.paimon_vindex_reader_open_with_options(input_file, options)
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
        with self._native_handle_lock:
            self._require_open()
            raw = _ffi.PaimonVindexMetadata()
            rc = lib.paimon_vindex_reader_metadata(
                self._handle, ctypes.byref(raw)
            )
            if rc != 0:
                _check_error("metadata failed")
            return _metadata_from_ffi(raw)

    def optimize_for_search(self):
        with self._native_handle_lock:
            self._require_open()
            rc = lib.paimon_vindex_reader_optimize_for_search(self._handle)
            if rc != 0:
                _check_error("optimize_for_search failed")

    def warmup_queries(self, queries, l_search: int = 0):
        queries = _float32_matrix(queries, "queries")
        if queries.shape[1] != self._metadata.dimension:
            raise RuntimeError(
                f"queries length {queries.size} does not match nq * dimension "
                f"{queries.shape[0] * self._metadata.dimension}"
            )
        l_search = _size_t(l_search, "l_search", allow_zero=True)
        with self._native_handle_lock:
            self._require_open()
            rc = lib.paimon_vindex_reader_warmup_queries(
                self._handle,
                queries.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                queries.shape[0],
                l_search,
            )
            if rc != 0:
                _check_error("warmup_queries failed")

    def calibrate_search_width(self, queries, top_k: int = 10):
        queries = _float32_matrix(queries, "queries")
        if queries.shape[1] != self._metadata.dimension:
            raise RuntimeError(
                f"queries length {queries.size} does not match nq * dimension "
                f"{queries.shape[0] * self._metadata.dimension}"
            )
        top_k = _size_t(top_k, "top_k", allow_zero=False)
        with self._native_handle_lock:
            self._require_open()
            out = ctypes.c_size_t()
            rc = lib.paimon_vindex_reader_calibrate_search_width(
                self._handle,
                queries.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                queries.shape[0],
                top_k,
                ctypes.byref(out),
            )
            if rc != 0:
                _check_error("calibrate_search_width failed")
            return out.value

    def read_plan(self):
        with self._native_handle_lock:
            self._require_open()
            raw = _ffi.PaimonVindexReadPlan()
            rc = lib.paimon_vindex_reader_read_plan(
                self._handle, ctypes.byref(raw)
            )
            if rc != 0:
                _check_error("read_plan failed")
            return VectorIndexReadPlan(
                random_read_latency_nanos=raw.random_read_latency_nanos,
                window_bytes=raw.window_bytes,
                max_ranges_per_read=raw.max_ranges_per_read,
                graph_beam_width=raw.graph_beam_width,
                filtered_graph_beam_width=raw.filtered_graph_beam_width,
                adjacency_preload_bytes=raw.adjacency_preload_bytes,
                adjacency_cache_bytes=raw.adjacency_cache_bytes,
                raw_vector_cache_bytes=raw.raw_vector_cache_bytes,
                memory_budget_bytes=raw.memory_budget_bytes,
            )

    def _filter_args(self, filter_bytes):
        if filter_bytes is None:
            return None, 0, None
        return _bytes_buffer(filter_bytes, "filter_bytes")

    def search(self, query, params: SearchParams, filter_bytes=None):
        query = _float32_vector(query, "query")
        if query.shape[0] != self._metadata.dimension:
            raise RuntimeError(
                f"query length {query.shape[0]} does not match index dimension "
                f"{self._metadata.dimension}"
            )
        ffi_params = params.to_ffi()
        ids = np.empty(params.top_k, dtype=np.int64)
        distances = np.empty(params.top_k, dtype=np.float32)

        with self._native_handle_lock:
            self._require_open()
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
        queries = _float32_matrix(queries, "queries")
        if queries.shape[1] != self._metadata.dimension:
            raise RuntimeError(
                f"queries length {queries.size} does not match nq * dimension "
                f"{queries.shape[0] * self._metadata.dimension}"
            )
        ffi_params = params.to_ffi_v2()
        result_len = queries.shape[0] * params.top_k
        ids = np.empty((queries.shape[0], params.top_k), dtype=np.int64)
        distances = np.empty((queries.shape[0], params.top_k), dtype=np.float32)

        with self._native_handle_lock:
            self._require_open()
            if filter_bytes is None:
                rc = lib.paimon_vindex_reader_search_batch_v2(
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
                rc = lib.paimon_vindex_reader_search_batch_with_roaring_filter_v2(
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
        with self._native_handle_lock:
            if self._handle:
                lib.paimon_vindex_reader_free(self._handle)
                self._handle = None
            self._closed = True

    def __enter__(self):
        with self._native_handle_lock:
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


__all__ = [
    "IvfSqEncodingModel",
    "PreparedIvfSqTraining",
    "PreparedTrainingInfo",
    "IvfPqBatchTableReuseMode",
    "SearchParams",
    "VectorIndexMetadata",
    "VectorIndexReadPlan",
    "VectorIndexReader",
    "VectorIndexTrainer",
    "VectorIndexTraining",
    "VectorIndexWriter",
]
