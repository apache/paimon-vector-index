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
from typing import Mapping, NamedTuple, Optional, Tuple

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


class DistanceEndpointOp(IntEnum):
    GE = 0
    GT = 1
    LE = 2
    LT = 3


@dataclass(frozen=True)
class DistanceEndpoint:
    """A public-distance predicate literal, preserved as a double for core."""

    value: float
    op: DistanceEndpointOp

    def __post_init__(self):
        try:
            endpoint_op = DistanceEndpointOp(self.op)
        except (TypeError, ValueError) as exc:
            raise ValueError("distance endpoint operator is invalid") from exc
        object.__setattr__(self, "value", float(self.value))
        object.__setattr__(self, "op", endpoint_op)

    def to_ffi(self):
        return _ffi.PaimonVindexDistanceEndpoint(self.value, int(self.op))


def _metric_code(metric):
    for code, name in METRICS.items():
        if metric == name:
            return code
    raise ValueError("metric must be l2, inner_product, or cosine")


@dataclass(frozen=True, init=False)
class DistanceBand:
    """Use from_endpoints for public predicates, from_raw for raw cuts.

    Raw cuts use squared L2, negative inner product, or cosine distance in a
    half-open [raw_lower, raw_upper) band. None denotes an unbounded side.
    Core validates raw bands during search.
    """

    metric: str
    raw_lower: Optional[float] = None
    raw_upper: Optional[float] = None

    def __init__(self, *args, **kwargs):
        raise TypeError(
            "Use DistanceBand.from_endpoints() for public-distance predicates "
            "or DistanceBand.from_raw() for explicit raw-distance cuts"
        )

    @classmethod
    def from_raw(
        cls,
        metric: str,
        raw_lower: Optional[float] = None,
        raw_upper: Optional[float] = None,
    ):
        """Construct raw [raw_lower, raw_upper) cuts without metric conversion.

        None is unbounded. Finite, ordered, metric-compatible cuts are
        validated by core during search, not by this factory.
        """
        _metric_code(metric)
        band = object.__new__(cls)
        object.__setattr__(band, "metric", metric)
        object.__setattr__(
            band, "raw_lower", float(raw_lower) if raw_lower is not None else None
        )
        object.__setattr__(
            band, "raw_upper", float(raw_upper) if raw_upper is not None else None
        )
        return band

    @classmethod
    def from_endpoints(
        cls,
        metric: str,
        lower: Optional[DistanceEndpoint] = None,
        upper: Optional[DistanceEndpoint] = None,
    ):
        """Delegate GE/GT lower and LE/LT upper predicates to core.

        Literals are public distances: sqrtf(squared L2), inner product,
        or cosine distance. No float32 rounding precedes the native call.
        """
        _require_range_search()
        metric_code = _metric_code(metric)
        for endpoint in (lower, upper):
            if endpoint is not None and not isinstance(endpoint, DistanceEndpoint):
                raise TypeError("endpoints must be DistanceEndpoint or None")
        raw_lower = lower.to_ffi() if lower is not None else None
        raw_upper = upper.to_ffi() if upper is not None else None
        band = _ffi.PaimonVindexRawDistanceBand()
        rc = lib.paimon_vindex_distance_band_from_endpoints(
            metric_code,
            ctypes.byref(raw_lower) if raw_lower is not None else None,
            ctypes.byref(raw_upper) if raw_upper is not None else None,
            ctypes.byref(band),
        )
        if rc != 0:
            _check_error("distance endpoint conversion failed")
        return cls.from_raw(
            metric,
            band.raw_lower if band.raw_lower_kind else None,
            band.raw_upper if band.raw_upper_kind else None,
        )

    def to_ffi(self):
        return _ffi.PaimonVindexRawDistanceBand(
            _metric_code(self.metric),
            int(self.raw_lower is not None),
            self.raw_lower if self.raw_lower is not None else 0.0,
            int(self.raw_upper is not None),
            self.raw_upper if self.raw_upper is not None else 0.0,
        )


@dataclass(frozen=True)
class RangeSearchParams:
    band: DistanceBand
    nprobe: int

    def __post_init__(self):
        if not isinstance(self.band, DistanceBand):
            raise TypeError("band must be a DistanceBand")
        object.__setattr__(
            self, "nprobe", _size_t(self.nprobe, "nprobe", allow_zero=False)
        )

    def to_ffi(self):
        return _ffi.PaimonVindexRangeSearchParams(self.band.to_ffi(), self.nprobe)


@dataclass(frozen=True)
class RangeSearchStats:
    """Per-query logical counters copied from the native result."""

    lists_probed: int
    rows_scanned: int
    rows_committed: int
    early_abandoned: int


class RangeSearchQueryResult(NamedTuple):
    """Unpackable label/raw-distance views sharing the owned result arrays."""

    labels: np.ndarray
    raw_distances: np.ndarray


@dataclass(frozen=True)
class RangeSearchResult:
    """Owned CSR arrays and per-query statistics, independent of the reader.

    raw_distances are squared L2, negative inner product, or cosine distance,
    not converted public distances. list_reads counts call-level list reads,
    not the sum of per-query lists_probed.
    """

    lims: np.ndarray
    labels: np.ndarray
    raw_distances: np.ndarray
    stats: Tuple[RangeSearchStats, ...]
    list_reads: int

    @property
    def query_count(self):
        return len(self.lims) - 1

    @property
    def hit_count(self):
        return len(self.labels)

    def query(self, index) -> RangeSearchQueryResult:
        """Return unpackable labels/raw_distances views, without payload copies.

        The mutable views share the result arrays and keep them alive even
        after this result is released. The query index must be nonnegative.
        """
        index = operator.index(index)
        if not 0 <= index < self.query_count:
            raise IndexError("range query index out of bounds")
        start, end = int(self.lims[index]), int(self.lims[index + 1])
        return RangeSearchQueryResult(
            self.labels[start:end], self.raw_distances[start:end]
        )


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


def _require_range_search():
    if not _ffi.RANGE_SEARCH_AVAILABLE:
        raise RuntimeError(
            "loaded native library does not support range search; "
            "rebuild or upgrade the native library"
        )


def _range_buffer_length(length, itemsize, name):
    length = _size_t(length, name, allow_zero=True)
    if length > min(_SIZE_T_MAX, np.iinfo(np.intp).max) // itemsize:
        raise ValueError(f"{name} is too large: buffer length overflow")
    return length


def _range_queries(value, dimension, batch):
    array = np.asarray(value)
    name = "queries" if batch else "query"
    ndim = 2 if batch else 1
    if array.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}-dimensional float32 array")
    if array.shape[-1] != dimension:
        raise RuntimeError(
            f"{name} dimension {array.shape[-1]} does not match "
            f"index dimension {dimension}"
        )
    _range_buffer_length(array.size, ctypes.sizeof(ctypes.c_float), name)
    return np.require(array, dtype=np.float32, requirements=["C", "A"])


def _range_array_copy(pointer, length, dtype, name):
    _range_buffer_length(length, np.dtype(dtype).itemsize, name)
    if not length:
        return np.empty(0, dtype=dtype)
    if not pointer:
        raise RuntimeError(f"range result {name} pointer is null")
    return np.ctypeslib.as_array(pointer, shape=(length,)).copy()


def _range_result_copy(handle, query_count):
    view = _ffi.PaimonVindexRangeSearchResultView()
    if lib.paimon_vindex_range_search_result_view(handle, ctypes.byref(view)) != 0:
        _check_error("failed to view range result")
    if view.query_count != query_count:
        raise RuntimeError("range result query count does not match input")
    _range_buffer_length(
        query_count, ctypes.sizeof(_ffi.PaimonVindexRangeSearchStats), "stats"
    )
    lims = _range_array_copy(view.lims, query_count + 1, np.uintp, "lims")
    if (
        lims[0] != 0
        or lims[-1] != view.hit_count
        or np.any(lims[1:] < lims[:-1])
    ):
        raise RuntimeError("range result has invalid CSR limits")
    labels = _range_array_copy(view.labels, view.hit_count, np.int64, "labels")
    raw_distances = _range_array_copy(
        view.raw_distances, view.hit_count, np.float32, "raw_distances"
    )
    if query_count and not view.stats:
        raise RuntimeError("range result stats pointer is null")
    stats = tuple(
        RangeSearchStats(
            view.stats[index].lists_probed,
            view.stats[index].rows_scanned,
            view.stats[index].rows_committed,
            view.stats[index].early_abandoned,
        )
        for index in range(query_count)
    )
    return RangeSearchResult(lims, labels, raw_distances, stats, view.list_reads)


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

    def supports_range_search(self):
        """Ask core whether this reader supports IVF distance range search."""
        with self._native_handle_lock:
            self._require_open()
            if not _ffi.RANGE_SEARCH_AVAILABLE:
                return False
            supported = ctypes.c_int()
            rc = lib.paimon_vindex_reader_supports_range_search(
                self._handle, ctypes.byref(supported)
            )
            if rc != 0:
                _check_error("range search capability check failed")
            return bool(supported.value)

    def range_search(self, query, params: RangeSearchParams, roaring_filter=None):
        """Search one query, returning owned CSR arrays with raw_distances.

        roaring_filter is optional serialized RoaringTreemap bytes. An empty
        byte string is still a supplied filter and is validated by core.
        """
        return self._range_search(query, params, roaring_filter, batch=False)

    def range_search_batch(
        self, queries, params: RangeSearchParams, roaring_filter=None
    ):
        """Return raw_distances for a query matrix; core rejects empty batches."""
        return self._range_search(queries, params, roaring_filter, batch=True)

    def _range_search(self, value, params, roaring_filter, *, batch):
        _require_range_search()
        if not isinstance(params, RangeSearchParams):
            raise TypeError("params must be RangeSearchParams")
        queries = _range_queries(value, self._metadata.dimension, batch)
        query_count = queries.shape[0] if batch else 1
        ffi_params = params.to_ffi()
        with self._native_handle_lock:
            self._require_open()
            args = [
                self._handle,
                queries.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                queries.size,
            ]
            if batch:
                args.append(query_count)
            args.append(ffi_params)
            if roaring_filter is None:
                search = (
                    lib.paimon_vindex_reader_range_search_batch
                    if batch else lib.paimon_vindex_reader_range_search
                )
            else:
                filter_buf, filter_len, _ = _bytes_buffer(
                    roaring_filter, "roaring_filter"
                )
                _range_buffer_length(filter_len, 1, "roaring_filter")
                args.extend((filter_buf, filter_len))
                search = (
                    lib.paimon_vindex_reader_range_search_batch_with_roaring_filter
                    if batch
                    else lib.paimon_vindex_reader_range_search_with_roaring_filter
                )
            handle = ctypes.c_void_p()
            try:
                rc = search(*args, ctypes.byref(handle))
                if rc != 0:
                    _check_error("range search failed")
                if not handle:
                    raise RuntimeError("range search returned a null result")
                return _range_result_copy(handle, query_count)
            finally:
                if handle:
                    lib.paimon_vindex_range_search_result_destroy(handle)

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
    "DistanceBand",
    "DistanceEndpoint",
    "DistanceEndpointOp",
    "IvfPqBatchTableReuseMode",
    "RangeSearchParams",
    "RangeSearchQueryResult",
    "RangeSearchResult",
    "RangeSearchStats",
    "SearchParams",
    "VectorIndexMetadata",
    "VectorIndexReadPlan",
    "VectorIndexReader",
    "VectorIndexTrainer",
    "VectorIndexTraining",
    "VectorIndexWriter",
]
