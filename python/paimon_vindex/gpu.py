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

"""Optional NVIDIA cuVS training and construction of CPU-readable IVF-SQ indexes.

CuPy/cuVS are imported only when CuvsKMeans is constructed. This module never
silently falls back to CPU. Install matching CUDA, CuPy and cuVS packages as
described at https://docs.nvidia.com/cuvs/installation .
"""

import threading
from time import perf_counter

import numpy as np

from . import (PreparedIvfSqTraining, VectorIndexWriter, _float32_matrix,
              _int64_vector, _size_t)


class CuvsKMeans:
    """Reusable, serialized GPU worker for squared-L2 IVF centroid training.

    The GPU runs flat Lloyd iterations. The existing CPU Auto strategy uses
    hierarchical splitting for large nlist, so evaluate final index quality.
    Caller-supplied initial centers allow comparison with CPU Lloyd. Otherwise
    initial rows are chosen without replacement using the prepared state's
    seed. This differs from the CPU Auto strategy's initialization.
    """

    def __init__(self, device=0, *, batch_samples=None, batch_centroids=None):
        started = perf_counter()
        self.device = _size_t(device, "device", allow_zero=True)
        self._tile_options = {}
        for key, value in (("batch_samples", batch_samples), ("batch_centroids", batch_centroids)):
            if value is not None:
                self._tile_options[key] = _size_t(value, key, allow_zero=False)
        self._lock = threading.Lock()
        self.last_run = None
        try:
            import cupy as cp
            import cuvs
            from cuvs.cluster.kmeans import KMeansParams, fit
            from cuvs.common import Resources
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "GPU training requires compatible CuPy and cuVS packages with "
                "cuvs.cluster.kmeans.KMeansParams/fit on an NVIDIA CUDA host. "
                "See https://docs.nvidia.com/cuvs/installation . "
                "CPU training remains available via fit_centroids_cpu()."
            ) from exc
        self._cp = cp
        self._params_type = KMeansParams
        self._fit = fit
        for key in self._tile_options:
            if not hasattr(KMeansParams, key):
                raise RuntimeError(f"installed cuVS does not expose {key}; upgrade cuVS or use its default tiling")
        self.cuvs_version = getattr(cuvs, "__version__", "unknown")
        self.cupy_version = cp.__version__
        self._resources = None
        try:
            with cp.cuda.Device(self.device):
                self._resources = Resources()
                name = cp.cuda.runtime.getDeviceProperties(self.device)["name"]
                self.device_name = name.decode() if isinstance(name, bytes) else str(name)
                cp.cuda.Device(self.device).synchronize()
        except Exception:
            try:
                self.close()
            except Exception:
                pass  # Preserve the initialization error if cleanup also fails.
            raise
        self.initialization_seconds = perf_counter() - started

    def fit(self, prepared, *, initial_centroids=None):
        """Return host float32 centers. Timings include sample export and copies.

        The input sample stays on the device throughout the Lloyd iterations.
        All timings synchronize the device, including work on cuVS streams.
        `last_run` describes the last successful call; it is cleared on failure.
        """
        with self._lock:
            self.last_run = None
            if self._resources is None:
                raise RuntimeError("CuvsKMeans is closed")
            if not isinstance(prepared, PreparedIvfSqTraining):
                raise TypeError("prepared must be a PreparedIvfSqTraining")
            started = perf_counter()
            info = prepared.info
            sample = prepared.sample
            if info.sample_count < info.nlist:
                raise ValueError("cuVS training requires at least nlist sample rows")
            if initial_centroids is None:
                rows = np.random.default_rng(info.seed).choice(
                    info.sample_count, size=info.nlist, replace=False,
                )
                initial = np.ascontiguousarray(sample[rows])
            else:
                initial = _float32_matrix(initial_centroids, "initial_centroids")
            if initial.shape != (info.nlist, info.dimension) or not np.isfinite(initial).all():
                raise ValueError("initial_centroids must be finite with shape (nlist, dimension)")
            cp = self._cp
            with cp.cuda.Device(self.device):
                # Array initialization fixes the initial centers across CPU/GPU
                # references; one supplied initialization implies one run.
                params = self._params_type(
                    metric="sqeuclidean", n_clusters=info.nlist,
                    init_method="Array", max_iter=info.iterations, n_init=1,
                    tol=1e-6, hierarchical=False,
                    **self._tile_options,
                )
                cp.cuda.Device(self.device).synchronize()
                transfer_started = perf_counter()
                device_sample = cp.asarray(sample)
                device_initial = cp.asarray(initial)
                cp.cuda.Device(self.device).synchronize()
                h2d_seconds = perf_counter() - transfer_started
                fit_started = perf_counter()
                device_centers, inertia, n_iter = self._fit(
                    params, device_sample, centroids=device_initial,
                    resources=self._resources,
                )
                cp.cuda.Device(self.device).synchronize()
                kmeans_seconds = perf_counter() - fit_started
                transfer_started = perf_counter()
                centers = np.ascontiguousarray(cp.asnumpy(cp.asarray(device_centers)), dtype=np.float32)
                cp.cuda.Device(self.device).synchronize()
                d2h_seconds = perf_counter() - transfer_started
            if centers.shape != initial.shape or not np.isfinite(centers).all():
                raise RuntimeError("cuVS returned invalid IVF centroids")
            self.last_run = {
                "backend": "cuvs", "algorithm": "lloyd", "device": self.device,
                "device_name": self.device_name, "cuvs_version": self.cuvs_version,
                "cupy_version": self.cupy_version, "dtype": "float32",
                "tile_options": dict(self._tile_options),
                "initialization": "provided" if initial_centroids is not None else "sample_random",
                "seed": info.seed, "sample_bytes": sample.nbytes,
                "centroid_bytes": centers.nbytes, "iterations": int(n_iter),
                "inertia": float(inertia), "h2d_seconds": h2d_seconds,
                "kmeans_seconds": kmeans_seconds, "d2h_seconds": d2h_seconds,
                "total_seconds": perf_counter() - started,
            }
            return centers

    def close(self):
        with self._lock:
            if self._resources is not None:
                try:
                    with self._cp.cuda.Device(self.device):
                        self._cp.cuda.Device(self.device).synchronize()
                finally:
                    # CUDA errors must not leave the worker open or retain its
                    # resources. A poisoned CUDA context may require a new process.
                    self._resources = None

    def __enter__(self):
        with self._lock:
            if self._resources is None:
                raise RuntimeError("CuvsKMeans is closed")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.close()
        except Exception:
            if exc_type is None:
                raise
        return False


class CuvsIvfSqWriter:
    """Batch GPU assignment and optional SQ8 encoding for an existing writer.

    The caller owns the native writer and writes/closes it as usual. Model
    arrays stay on the selected device across batches. Set encode=False to
    keep SQ encoding on CPU. GPU assignment is exact squared L2 for all metrics;
    cosine preprocessing is delegated to Rust. For large centroid matrices,
    train with ivf.coarse-assignment=exact so SQ calibration uses the same policy.
    """

    def __init__(self, writer, device=0, *, encode=True):
        started = perf_counter()
        if not isinstance(writer, VectorIndexWriter):
            raise TypeError("writer must be a VectorIndexWriter")
        if not isinstance(encode, bool):
            raise TypeError("encode must be bool")
        self._lock = threading.Lock()
        self._worker = None
        self._arrays = None
        self._closed = False
        self.last_run = None
        self._writer = writer
        self._encode = encode
        self._model = writer.ivf_sq_encoding_model()
        if not self._model.exact_assignment:
            raise ValueError("GPU building requires SQ calibration with exact assignment; "
                             "train again with ivf.coarse-assignment=exact")
        try:
            self._worker = CuvsKMeans(device)
            from cuvs.cluster.kmeans import predict
            self._predict = predict
            worker = self._worker
            cp = worker._cp
            model = self._model
            with cp.cuda.Device(worker.device):
                self._params = worker._params_type(metric="sqeuclidean", n_clusters=model.nlist,
                                                   hierarchical=False)
                self._arrays = {"centers": cp.asarray(model.centroids)}
                if encode:
                    scales = np.zeros_like(model.mins)
                    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                        np.divide(np.float32(255), model.maxs - model.mins, out=scales,
                                  where=model.mins < model.maxs)
                    self._arrays.update(mins=cp.asarray(model.mins), maxs=cp.asarray(model.maxs),
                                        scales=cp.asarray(scales))
                    self._kernel = cp.ElementwiseKernel(
                        "float32 x, raw I labels, raw float32 centers, raw float32 mins, "
                        "raw float32 maxs, raw float32 scales, int32 d, int32 vector_width",
                        "uint8 code", r'''
                        int dim = i % d;
                        long long at = (long long)labels[i / d] * d + dim;
                        float lo = mins[at], hi = maxs[at];
                        float value = 0.0f;
                        if (paimon_lt(lo, hi)) {
                            float residual = paimon_sub_rn(x, centers[at]);
                            float shifted = paimon_sub_rn(residual, lo);
                            int cutoff = vector_width > 1 ? d / vector_width * vector_width : 0;
                            value = dim < cutoff ? paimon_mul_rn(shifted, scales[at])
                                : paimon_div_rn(paimon_mul_rn(shifted, 255.0f), paimon_sub_rn(hi, lo));
                        }
                        value = fminf(255.0f, fmaxf(0.0f, value));
                        float lower = floorf(value);
                        code = (unsigned char)(lower + (value - lower >= 0.5f ? 1.0f : 0.0f));
                        ''', "paimon_ivfsq_encode_v1", options=("--fmad=false",), preamble=r'''
                        // CuPy appends -ftz=true after user options. Explicit PTX
                        // without .ftz preserves subnormal inputs and results,
                        // including comparisons of tiny nonconstant SQ bounds.
                        __device__ __forceinline__ unsigned int paimon_lt(float a, float b) {
                            unsigned int out;
                            asm("set.lt.u32.f32 %0, %1, %2;" : "=r"(out) : "f"(a), "f"(b));
                            return out;
                        }
                        __device__ __forceinline__ float paimon_sub_rn(float a, float b) {
                            float out;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(out) : "f"(a), "f"(b));
                            return out;
                        }
                        __device__ __forceinline__ float paimon_mul_rn(float a, float b) {
                            float out;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(out) : "f"(a), "f"(b));
                            return out;
                        }
                        __device__ __forceinline__ float paimon_div_rn(float a, float b) {
                            float out;
                            asm("div.rn.f32 %0, %1, %2;" : "=f"(out) : "f"(a), "f"(b));
                            return out;
                        }
                        ''')
                cp.cuda.Device(worker.device).synchronize()
            self.initialization_seconds = perf_counter() - started
        except Exception:
            try:
                self.close()
            except Exception:
                pass  # Preserve the build initialization error.
            raise

    def add_vectors(self, ids, data):
        """Process one batch; last_run includes validation, transfers and native append.

        Initialize once and call repeatedly with bounded batches. A failed batch
        is not appended; prior successful batches stay in the caller-owned writer.
        The first encoding call includes any CUDA kernel compilation cost.
        """
        with self._lock:
            self.last_run = None
            if self._closed:
                raise RuntimeError("CuvsIvfSqWriter is closed")
            started = perf_counter()
            raw = _float32_matrix(data, "data")
            ids = _int64_vector(ids, "ids")
            model = self._model
            if raw.shape != (len(ids), model.dimension) or not len(ids):
                raise ValueError("data must have shape (len(ids), dimension) with at least one row")
            # Check the writer before doing device work; each native operation
            # still locks/checks its handle in case another thread closes it.
            self._writer.dimension
            if model.metric == "cosine":
                processed = self._writer._preprocess_ivf_sq_vectors(raw)
            else:
                if not np.isfinite(raw).all():
                    raise ValueError("data must contain only finite values")
                processed = raw
            input_seconds = perf_counter() - started
            worker = self._worker
            cp = worker._cp
            with cp.cuda.Device(worker.device):
                cp.cuda.Device(worker.device).synchronize()
                tick = perf_counter()
                vectors = cp.asarray(processed)
                cp.cuda.Device(worker.device).synchronize()
                h2d_seconds = perf_counter() - tick
                tick = perf_counter()
                labels, _ = self._predict(self._params, vectors, self._arrays["centers"],
                                          resources=worker._resources)
                cp.cuda.Device(worker.device).synchronize()
                labels = cp.asarray(labels).reshape(-1)
                assignment_seconds = perf_counter() - tick
                tick = perf_counter()
                host_labels = cp.asnumpy(labels)
                cp.cuda.Device(worker.device).synchronize()
                d2h_seconds = perf_counter() - tick
                # Validate before the quantization kernel uses labels as addresses.
                if (host_labels.shape != (len(ids),) or host_labels.dtype.kind not in "iu"
                        or np.any(host_labels < 0) or np.any(host_labels >= model.nlist)):
                    raise RuntimeError("cuVS returned invalid partition IDs")
                host_labels = np.ascontiguousarray(host_labels, dtype=np.uint32)
                encode_seconds = 0.0
                code_bytes = 0
                if self._encode:
                    tick = perf_counter()
                    arrays = self._arrays
                    codes = self._kernel(vectors, labels, arrays["centers"], arrays["mins"],
                                         arrays["maxs"], arrays["scales"], np.int32(model.dimension),
                                         np.int32(model._encoding_vector_width))
                    cp.cuda.Device(worker.device).synchronize()
                    encode_seconds = perf_counter() - tick
                    tick = perf_counter()
                    host_codes = cp.asnumpy(codes)
                    cp.cuda.Device(worker.device).synchronize()
                    d2h_seconds += perf_counter() - tick
                    code_bytes = host_codes.nbytes
            tick = perf_counter()
            if self._encode:
                self._writer.add_encoded_vectors(ids, host_codes, host_labels)
            else:
                self._writer.add_preassigned_vectors(ids, raw, host_labels)
            append_seconds = perf_counter() - tick
            self.last_run = {
                "backend": "cuvs-encode" if self._encode else "cuvs-assign",
                "assignment": "exact", "rows": len(ids), "device": worker.device,
                "device_name": worker.device_name,
                "cuvs_version": worker.cuvs_version, "cupy_version": worker.cupy_version,
                "input_seconds": input_seconds, "h2d_seconds": h2d_seconds,
                "assignment_seconds": assignment_seconds, "encode_seconds": encode_seconds,
                "d2h_seconds": d2h_seconds, "append_seconds": append_seconds,
                "total_seconds": perf_counter() - started,
                "h2d_bytes": processed.nbytes, "d2h_bytes": host_labels.nbytes + code_bytes,
            }

    def close(self):
        with self._lock:
            if self._closed:
                return
            try:
                if self._worker is not None:
                    self._worker.close()
            finally:
                self._arrays = None
                self._worker = None
                self._closed = True

    def __enter__(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("CuvsIvfSqWriter is closed")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.close()
        except Exception:
            if exc_type is None:
                raise
        return False
