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
import os
import platform
from ctypes import (
    CFUNCTYPE,
    POINTER,
    Structure,
    c_char_p,
    c_float,
    c_int,
    c_int64,
    c_size_t,
    c_uint8,
    c_uint32,
    c_uint64,
    c_void_p,
)


def _lib_name():
    system = platform.system()
    if system == "Darwin":
        return "libpaimon_vindex_ffi.dylib"
    if system == "Windows":
        return "paimon_vindex_ffi.dll"
    return "libpaimon_vindex_ffi.so"


def _load_library():
    lib_name = _lib_name()
    search_paths = []

    env_path = os.environ.get("PAIMON_VINDEX_LIB_PATH")
    if env_path:
        if os.path.isfile(env_path):
            return ctypes.CDLL(env_path)
        search_paths.append(env_path)

    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    search_paths.append(pkg_dir)
    search_paths.append(os.path.join(pkg_dir, "..", ".."))

    for rel in [
        os.path.join("..", "..", "target", "release"),
        os.path.join("..", "..", "target", "debug"),
        os.path.join("..", "..", "..", "target", "release"),
        os.path.join("..", "..", "..", "target", "debug"),
    ]:
        search_paths.append(os.path.join(pkg_dir, rel))

    for directory in search_paths:
        candidate = os.path.abspath(os.path.join(directory, lib_name))
        if os.path.isfile(candidate):
            return ctypes.CDLL(candidate)

    try:
        return ctypes.CDLL(lib_name)
    except OSError as exc:
        raise OSError(
            f"Cannot find {lib_name}. Build the native library first with "
            "'cargo build --release -p paimon-vindex-ffi', or set "
            "PAIMON_VINDEX_LIB_PATH to the directory containing it."
        ) from exc


lib = _load_library()

WRITE_FN = CFUNCTYPE(c_int, c_void_p, POINTER(c_uint8), c_size_t)
FLUSH_FN = CFUNCTYPE(c_int, c_void_p)
GET_POS_FN = CFUNCTYPE(c_int64, c_void_p)


class PaimonVindexReadRequest(Structure):
    _fields_ = [
        ("offset", c_uint64),
        ("buf", POINTER(c_uint8)),
        ("len", c_size_t),
    ]


READ_RANGES_FN = CFUNCTYPE(
    c_int,
    c_void_p,
    POINTER(PaimonVindexReadRequest),
    c_size_t,
)


class PaimonVindexOutputFile(Structure):
    _fields_ = [
        ("ctx", c_void_p),
        ("write_fn", WRITE_FN),
        ("flush_fn", FLUSH_FN),
        ("get_pos_fn", GET_POS_FN),
    ]


class PaimonVindexInputFile(Structure):
    _fields_ = [
        ("ctx", c_void_p),
        ("read_ranges_fn", READ_RANGES_FN),
        ("estimated_random_read_latency_nanos", c_uint64),
        ("preferred_alignment_bytes", c_size_t),
        ("preferred_window_bytes", c_size_t),
        ("max_ranges_per_read", c_size_t),
    ]


class PaimonVindexMetadata(Structure):
    _fields_ = [
        ("index_type", c_uint32),
        ("dimension", c_size_t),
        ("nlist", c_size_t),
        ("metric", c_uint32),
        ("total_vectors", c_int64),
        ("pq_m", c_size_t),
        ("pq_bits", c_size_t),
        ("rq_bits", c_size_t),
        ("diskann_max_degree", c_size_t),
        ("diskann_build_search_list_size", c_size_t),
        ("diskann_alpha", c_float),
    ]


class PaimonVindexSearchParams(Structure):
    _fields_ = [
        ("top_k", c_size_t),
        ("search_width", c_uint32),
        ("width", c_size_t),
    ]


class PaimonVindexReaderOptions(Structure):
    _fields_ = [
        ("memory_budget_bytes", c_size_t),
    ]


class PaimonVindexReadPlan(Structure):
    _fields_ = [
        ("random_read_latency_nanos", c_uint64),
        ("preferred_alignment_bytes", c_size_t),
        ("window_bytes", c_size_t),
        ("max_ranges_per_read", c_size_t),
        ("graph_beam_width", c_size_t),
        ("filtered_graph_beam_width", c_size_t),
        ("adjacency_preload_bytes", c_size_t),
        ("adjacency_cache_bytes", c_size_t),
        ("raw_vector_cache_bytes", c_size_t),
        ("memory_budget_bytes", c_size_t),
    ]


lib.paimon_vindex_last_error.argtypes = []
lib.paimon_vindex_last_error.restype = c_char_p

lib.paimon_vindex_trainer_open.argtypes = [
    POINTER(c_char_p),
    POINTER(c_char_p),
    c_size_t,
]
lib.paimon_vindex_trainer_open.restype = c_void_p

lib.paimon_vindex_trainer_free.argtypes = [c_void_p]
lib.paimon_vindex_trainer_free.restype = None

lib.paimon_vindex_trainer_dimension.argtypes = [c_void_p, POINTER(c_size_t)]
lib.paimon_vindex_trainer_dimension.restype = c_int

lib.paimon_vindex_trainer_add_training_vectors.argtypes = [
    c_void_p,
    POINTER(c_float),
    c_size_t,
]
lib.paimon_vindex_trainer_add_training_vectors.restype = c_int

lib.paimon_vindex_trainer_finish.argtypes = [c_void_p]
lib.paimon_vindex_trainer_finish.restype = c_void_p

lib.paimon_vindex_training_free.argtypes = [c_void_p]
lib.paimon_vindex_training_free.restype = None

lib.paimon_vindex_writer_open.argtypes = [c_void_p]
lib.paimon_vindex_writer_open.restype = c_void_p

lib.paimon_vindex_writer_free.argtypes = [c_void_p]
lib.paimon_vindex_writer_free.restype = None

lib.paimon_vindex_writer_dimension.argtypes = [c_void_p, POINTER(c_size_t)]
lib.paimon_vindex_writer_dimension.restype = c_int

lib.paimon_vindex_writer_add_vectors.argtypes = [
    c_void_p,
    POINTER(c_int64),
    POINTER(c_float),
    c_size_t,
]
lib.paimon_vindex_writer_add_vectors.restype = c_int

lib.paimon_vindex_writer_write_index.argtypes = [
    c_void_p,
    PaimonVindexOutputFile,
]
lib.paimon_vindex_writer_write_index.restype = c_int

lib.paimon_vindex_reader_open.argtypes = [PaimonVindexInputFile]
lib.paimon_vindex_reader_open.restype = c_void_p

lib.paimon_vindex_reader_open_with_options.argtypes = [
    PaimonVindexInputFile,
    PaimonVindexReaderOptions,
]
lib.paimon_vindex_reader_open_with_options.restype = c_void_p

lib.paimon_vindex_reader_free.argtypes = [c_void_p]
lib.paimon_vindex_reader_free.restype = None

lib.paimon_vindex_reader_metadata.argtypes = [
    c_void_p,
    POINTER(PaimonVindexMetadata),
]
lib.paimon_vindex_reader_metadata.restype = c_int

lib.paimon_vindex_reader_optimize_for_search.argtypes = [c_void_p]
lib.paimon_vindex_reader_optimize_for_search.restype = c_int
lib.paimon_vindex_reader_warmup_queries.argtypes = [
    c_void_p,
    POINTER(c_float),
    c_size_t,
    c_size_t,
]
lib.paimon_vindex_reader_warmup_queries.restype = c_int
lib.paimon_vindex_reader_calibrate_search_width.argtypes = [
    c_void_p,
    POINTER(c_float),
    c_size_t,
    c_size_t,
    POINTER(c_size_t),
]
lib.paimon_vindex_reader_calibrate_search_width.restype = c_int
lib.paimon_vindex_reader_read_plan.argtypes = [
    c_void_p,
    POINTER(PaimonVindexReadPlan),
]
lib.paimon_vindex_reader_read_plan.restype = c_int

lib.paimon_vindex_reader_search.argtypes = [
    c_void_p,
    POINTER(c_float),
    PaimonVindexSearchParams,
    POINTER(c_int64),
    POINTER(c_float),
    c_size_t,
]
lib.paimon_vindex_reader_search.restype = c_int

lib.paimon_vindex_reader_search_with_roaring_filter.argtypes = [
    c_void_p,
    POINTER(c_float),
    PaimonVindexSearchParams,
    POINTER(c_uint8),
    c_size_t,
    POINTER(c_int64),
    POINTER(c_float),
    c_size_t,
]
lib.paimon_vindex_reader_search_with_roaring_filter.restype = c_int

lib.paimon_vindex_reader_search_batch.argtypes = [
    c_void_p,
    POINTER(c_float),
    c_size_t,
    PaimonVindexSearchParams,
    POINTER(c_int64),
    POINTER(c_float),
    c_size_t,
]
lib.paimon_vindex_reader_search_batch.restype = c_int

lib.paimon_vindex_reader_search_batch_with_roaring_filter.argtypes = [
    c_void_p,
    POINTER(c_float),
    c_size_t,
    PaimonVindexSearchParams,
    POINTER(c_uint8),
    c_size_t,
    POINTER(c_int64),
    POINTER(c_float),
    c_size_t,
]
lib.paimon_vindex_reader_search_batch_with_roaring_filter.restype = c_int
