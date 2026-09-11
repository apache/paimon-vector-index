<!--
Licensed to the Apache Software Foundation (ASF) under one or more
contributor license agreements. See the NOTICE file distributed with
this work for additional information regarding copyright ownership.
The ASF licenses this file to you under the Apache License, Version 2.0
(the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Experimental GPU training and construction for IVF-SQ

IVF-SQ can train its IVF centers with NVIDIA cuVS, calibrate SQ with Rust,
and optionally assign and encode full batches on GPU. Rust assembles and writes
the existing v1 index. Existing CPU Readers
can read these files. The Rust library and normal Python imports do not require
CUDA. The optional adapters are `paimon_vindex.gpu.CuvsKMeans` and
`paimon_vindex.gpu.CuvsIvfSqWriter`.

This first implementation exposes preparation and external-center completion
in Rust, C and Python. The CUDA adapter is Python-only. It does not add GPU
search or a CUDA runtime to the Java/JNI or C++ wrapper APIs.

GPU execution and speedups must be validated on a CUDA machine. CPU round-trip
tests and benchmark smoke runs do not establish GPU performance.

## Build and run

Build the matching native library from this checkout:

```sh
cargo build --release -p paimon-vindex-ffi
python -m pip install -e 'python[test]'
export PAIMON_VINDEX_LIB_PATH="$PWD/target/release"
```

In the GPU worker environment, install matching CuPy and cuVS packages for its
CUDA version using the [NVIDIA installation guide](https://docs.nvidia.com/cuvs/installation).
The required API is `cuvs.cluster.kmeans.KMeansParams` / `fit`, with array
initialization, and `cuvs.common.Resources`. See the
[official K-means API](https://docs.nvidia.com/cuvs/api-reference/python-api-cluster-kmeans).
Package/driver installation is external to this library; no CUDA dependency is
installed by the normal Python package. The adapter reports installed versions.

For example, install the optional dependencies in a separate virtual environment:

```sh
python -m pip install 'cuvs-cu12==26.8.1' 'cupy-cuda12x[ctk]==14.2.0'
```

The `ctk` extra supplies CUDA user-space libraries. This command does not install
or update the host NVIDIA driver.

```python
import numpy as np
from paimon_vindex import VectorIndexTrainer, VectorIndexWriter
from paimon_vindex.gpu import CuvsKMeans, CuvsIvfSqWriter

vectors = np.load("base.npy", mmap_mode="r")  # shape (N, dimension)
options = {
    "index.type": "ivf_sq",
    "dimension": str(vectors.shape[1]),
    "nlist": "1024",
    "expected-vector-count": str(len(vectors)),
    "metric": "cosine",
    "ivf.coarse-assignment": "exact",
}

# Reuse one worker across successive jobs to amortize GPU initialization.
with CuvsKMeans(device=0) as gpu:
    with VectorIndexTrainer.create(options) as trainer:
        for start in range(0, len(vectors), 8192):
            trainer.add_training_vectors(vectors[start:start + 8192])
        with trainer.prepare_training() as prepared:
            print(prepared.info)
            centers = gpu.fit(prepared)
            print(gpu.last_run)
            training = prepared.finish_with_ivf_centroids(centers)

    with VectorIndexWriter(training) as writer:
        with CuvsIvfSqWriter(writer, device=0) as builder:
            for start in range(0, len(vectors), 8192):
                batch = vectors[start:start + 8192]
                builder.add_vectors(np.arange(start, start + len(batch), dtype=np.int64), batch)
        print(writer.ivf_sq_partition_sizes())
        with open("vectors.index", "wb") as output:
            writer.write(output)
```

The prepared sample is owned by Rust. `prepared.sample` returns an independent,
read-only float32 copy, usable by another training library. External centers
must have shape `(nlist, dimension)` and finite values. The centers are installed
verbatim; Rust assigns calibration samples and recomputes pooled residual SQ
bounds without retraining them.

`prepare_training()` consumes the Trainer on success or failure. Native
completion consumes the prepared state on success or failure; Python shape/type
checks that fail before calling native code leave it open. Both classes support
context managers. Sample copies remain valid after the native state closes.

## Full-vector construction

`CuvsIvfSqWriter` snapshots the writer's centers and per-list SQ bounds and
retains them on its selected device. Each `add_vectors(ids, data)` call uploads
one batch, predicts its nearest centers, computes residual SQ8 codes, and
returns partition IDs and uint8 codes. Rust validates their shapes and partition
IDs, appends them, and performs the existing ID sorting and blocked-code
serialization. No second CPU assignment or encoding is performed.

- Set `encode=False` to use GPU assignment with the existing CPU SQ encoder.
  Call `writer.add_vectors` directly to retain the complete CPU add path.
- The adapter borrows the writer: closing the adapter releases its device
  references and resources, while the writer remains usable. CuPy's process-wide
  allocator may cache freed buffers. Do not close the writer during a build.
- Use bounded batches; the adapter does not automatically split an oversized
  call. Batch scratch is bounded by the caller's batch size. The native writer
  still retains the complete encoded index in host memory until serialization.
- Cosine preprocessing uses the native Rust implementation. L2/IP inputs remain
  unchanged. SQ8 encoding follows native clipping, round-half-up behavior and
  f32 operation order, including scalar tail dimensions and constant bounds.
  Its arithmetic preserves subnormal values independently of CuPy's default
  flush-to-zero compilation mode.
- GPU assignment is exact squared L2. Configure `ivf.coarse-assignment=exact`
  before training for consistent SQ calibration. If the default policy would
  use approximate Vamana, the adapter rejects that model with an actionable error.
- GPU and CPU distance calculations can choose different centers near ties.
  Check recall and partition distribution; bitwise identity of independently
  assigned indexes is not guaranteed.
- Invalid input or backend results are rejected before append. A failed batch
  leaves prior successful batches in the writer. Input validation errors can be
  corrected and retried. CUDA failures may require recreating the worker or its
  process. The adapter never substitutes a CPU build after a GPU failure.
- Closing releases the adapter's resource references even if CUDA synchronization
  fails. Explicit `close()` reports that failure; context-manager cleanup preserves
  an exception already raised by the operation. A closed adapter cannot be reused.

`builder.last_run` reports the last successful batch's validation/preprocessing,
H2D, assignment, encoding, D2H and native append times. Timings synchronize the
device. Constructor setup is `initialization_seconds`; CUDA kernel compilation,
when needed, is included in the first encoding call. Imports of the module do
not load CUDA. GPU construction additionally requires CuPy's CUDA kernel
compilation support (NVRTC), supplied by the installation above.

The backend-independent native interfaces are `writer.ivf_sq_encoding_model()`,
`writer.add_preassigned_vectors(ids, raw_vectors, partition_ids)` and
`writer.add_encoded_vectors(ids, uint8_codes, partition_ids)`, with corresponding
Rust and C functions. The latter validates representation and partition IDs;
external callers are responsible for using the exact model and preprocessing
that belong to that writer. Encoding model arrays are independent, read-only
copies with shape `(nlist, dimension)`.

## Training contract and comparisons

- `nlist` is resolved before preparation. With automatic nlist, supply the
  **full** `expected-vector-count`, rather than the number of sampled rows.
- The existing deterministic reservoir keeps at most
  `max(65536, 64 * nlist)` rows. The effective K-means sample also honors
  `ivf.train.max-points-per-centroid`. Metadata reports `vectors_seen`,
  `calibration_count` and `sample_count` separately. SQ calibration uses the
  whole reservoir, including when the K-means sample is further capped.
- Cosine inputs are normalized once by Rust. L2 and IP inputs are unchanged.
  All center fitting uses **squared L2**. Do not switch the external trainer
  to inner-product or spherical K-means based only on the final search metric.
- The adapter uses float32 flat Lloyd iterations, at most the prepared state's
  iteration budget (currently 25), a `1e-6` tolerance and one initial center set.
  By default it samples initial rows with the prepared seed; callers may supply
  `initial_centroids`. This is a different initialization from CPU Auto.
- CPU Auto retains the existing hierarchical strategy above 256 centers.
  `prepared.fit_centroids_cpu("lloyd", initial_centroids=...)` offers a closer
  algorithmic comparison. Matching initial centers and iteration budgets does
  not guarantee bitwise-equal GPU results or identical convergence behavior.
- The GPU adapter requires at least `nlist` effective sample rows and raises
  errors for missing CUDA dependencies, device failures or invalid results.
  It never silently falls back to CPU.

cuVS releases differ in exposed tiling parameters. By default the adapter uses
the library's tiling. On versions exposing the corresponding properties, use
`CuvsKMeans(batch_samples=16384, batch_centroids=1024)`. Unsupported explicit
options raise an error. Input, centers, initialization and temporary buffers
all consume device memory; sample bytes alone are not a peak-memory estimate.

## Reproducible benchmark

The harness compares CPU Auto, CPU Lloyd and cuVS. CPU Lloyd and cuVS use the
same initial rows from the same prepared sample. CPU Auto uses the existing
algorithm and initialization. Order rotates between repeats. Each run records
sample/center hashes, resolved parameters, stage timings and center artifacts.

| Preset | Synthetic rows | Dimensions | nlist | Raw sample bytes |
|---|---:|---:|---:|---:|
| smoke | 2,048 | 16 | 8 | 128 KiB |
| baseline | 65,536 | 960 | 1,024 | 240 MiB |
| medium | 262,144 | 1,536 | 4,096 | 1.5 GiB |
| large | 1,048,576 | 1,536 | 16,384 | 6 GiB |

Synthetic presets exercise training loads. They are not production recall
benchmarks, and they do not represent a larger full corpus. With `--base`,
actual rows and dimension come from the file; all rows enter the native
reservoir, and `nlist` comes from the preset. The large CPU Lloyd baseline can
take substantial time. Start with smoke to verify the environment.

CPU-only smoke, including encoding and serialization:

```sh
python tools/benchmark_gpu_training.py --preset smoke --synthetic \
  --backends cpu-auto cpu-lloyd --build-index --output-dir /tmp/ivfsq-cpu-smoke
```

On a CUDA host, include `cuvs` explicitly:

```sh
python tools/benchmark_gpu_training.py --preset baseline --synthetic \
  --backends cpu-auto cpu-lloyd cuvs --threads 8 --repeats 3 \
  --output-dir /tmp/ivfsq-gpu-baseline
```

Repeat with `--preset medium` and `--preset large`, using a new output directory
each time. Existing output directories are rejected. The default measures
training only; `--build-index` also builds and writes every base row per run.

Evaluate real data using `.fvecs`/`.ivecs` or float32/integer `.npy` files:

```sh
python tools/benchmark_gpu_training.py --preset baseline \
  --base /data/gist1m/base.fvecs --queries /data/gist1m/query.fvecs \
  --neighbors /data/gist1m/ground_truth.ivecs --metric l2 \
  --backends cpu-auto cpu-lloyd cuvs --build-index --nprobe 64 \
  --output-dir /tmp/ivfsq-gist-gpu
```

To measure complete GPU construction, add `--build-backend cuvs-encode` and
`--coarse-assignment exact`. Use `--build-backend cuvs-assign` to isolate GPU
assignment while retaining CPU SQ encoding. Keep the same assignment policy
for the CPU reference.

For a comparison with identical centers, reuse a saved center artifact:

```sh
for builder in cpu cuvs-assign cuvs-encode; do
  python tools/benchmark_gpu_training.py --preset baseline \
    --base /data/gist1m/base.fvecs --queries /data/gist1m/query.fvecs \
    --neighbors /data/gist1m/ground_truth.ivecs --metric l2 \
    --backends external --centroids /tmp/ivfsq-gist-gpu/cuvs-0-centroids.npy \
    --build-index --build-backend "$builder" --coarse-assignment exact \
    --threads 8 --repeats 3 --nprobe 64 --output-dir "/tmp/ivfsq-fixed-$builder"
done
```

`external` skips centroid fitting: its totals describe rebuilding with supplied
centers, not end-to-end model training. Use `cpu-auto` and `cuvs` for complete
training/build comparisons. The manifest records the external center file hash,
builder, device and batch size.

Ground truth must match the supplied base, metric and zero-based row IDs.
The harness records actual empty/max/P95 partition sizes, Recall@K, first-batch
QPS/read bytes and sequential P95 after that batch. A fixed `nprobe` reveals
quality changes; tune `nprobe` to the same recall target before claiming a
query-performance advantage. No quality claim is produced without ground truth.

`--coarse-assignment auto` preserves the CPU writer/calibrator's existing policy:
it can switch to approximate Vamana assignment when `dimension * nlist >= 1000000`.
Use `--coarse-assignment exact` to compare against exact nearest-center assignment.
This changes CPU SQ calibration and full-row assignment, not the GPU K-means
algorithm. Report this policy with the timing and recall results. Partition
diagnostics describe the written index, not the GPU trainer's cluster labels.

Interpret timing fields as follows:

- `prepare_seconds`: trainer creation, input ingestion/reservoir sampling and
  preprocessing. Native-to-Python export for CPU Lloyd's initialization is
  included in its `centroid_fit_seconds`.
- `centroid_fit_seconds`: centroid training, including required sample export
  and initialization. GPU input/output copies and synchronization are included.
  GPU `kmeans_seconds`, H2D and D2H are also reported separately.
- `sq_calibration_seconds`: CPU residual parameter training using the new centers.
- `training_seconds`: preparation + centroid fitting + SQ calibration.
- `gpu_setup_seconds`: imports, resource/device initialization for the first
  GPU job. `training_including_setup_seconds` adds this cost. Inspect the first
  run separately; a multi-run median can hide one-time startup costs.
- `add_seconds`: full-row addition using the chosen builder. Includes GPU builder
  initialization/teardown, input validation, transfers, encoding and native
  append. `gpu_build_stages` sums per-batch timings; `gpu_build_setup_seconds`
  is already included in `add_seconds`. These are not extra costs to add again.
  `gpu_build` records the device name and CuPy/cuVS versions even when centroid
  training uses a different backend.
- `build_seconds`: training + full-row add + CPU serialization when requested.
  Writing includes buffered flush, not an `fsync` durability guarantee.

Source mapping/generation, diagnostic hashing, common-initialization diagnostics,
center artifact export and query evaluation are outside those timers. File
pages may be cold on the first run. GPU peak memory is explicitly **not
measured**; profile it on the target hardware before setting capacity limits.

`manifest.json`, incremental `runs.jsonl`, `summary.json` and per-run centers
are written to the output directory. If CUDA fails, the process fails rather
than fabricating a CPU-backed GPU result. Preserve the tested package versions
and compare both training and full-build metrics when deciding whether to
enable GPU workers in production.

## Verification

```sh
cargo test -p paimon-vindex-core --test external_training --test external_build
python -m pytest python/tests/test_gpu_training.py python/tests/test_gpu_build.py
```

The CPU checks cover fixed-center SQ calibration, reservoir bounds, batch
invariance, validation, ownership, and byte-identical CPU reference files.
Actual GPU integration tests are explicit and fail if the requested CUDA
environment is unavailable:

```sh
PAIMON_TEST_CUVS=1 python -m pytest python/tests/test_gpu_training.py python/tests/test_gpu_build.py
```

They train on GPU, reuse the worker, write v1 files with CPU, and check CPU
retrieval against exact neighbors for L2, cosine and inner product.
Construction tests also compare GPU and native SQ8 encoding at rounding/clipping
boundaries and subnormal ranges, exercise tail dimensions, and test held-out queries with partial
partition probing and more than 256 centers. Ordinary CPU CI covers malformed
backend labels, recovery, writer ownership, CUDA cleanup failures and atomic validation failures using
a fake CUDA interface; these checks do not replace actual GPU tests.
