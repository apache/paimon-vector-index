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

"""Install a final wheel in isolation and exercise its bundled native library."""

import argparse
import glob
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import venv


def smoke_installed():
    import io
    import numpy as np
    import paimon_vindex
    from paimon_vindex import _ffi, SearchParams, VectorIndexReader, VectorIndexTrainer, VectorIndexWriter

    package = Path(paimon_vindex.__file__).resolve().parent
    assert sys.prefix != sys.base_prefix, "smoke test requires an isolated environment"
    assert Path(sys.prefix).resolve() in package.parents, package
    native = Path(_ffi.lib._name).resolve()
    assert native.parent == package, f"native library was not loaded from the installed wheel: {native}"

    class Input:
        def __init__(self, data):
            self.data = data

        def pread_many(self, ranges):
            return [self.data[pos:pos + length] for pos, length in ranges]

    data = np.array([[0, 0], [0.1, 0], [10, 10], [10.1, 10]], dtype=np.float32)
    ids = np.array([101, 102, 103, 104], dtype=np.int64)
    options = {"index.type": "ivf_flat", "dimension": "2", "nlist": "2", "metric": "l2"}
    output = io.BytesIO()
    with VectorIndexWriter(VectorIndexTrainer.train(options, data)) as writer:
        writer.add_vectors(ids, data)
        writer.write(output)
    assert output.tell() == len(output.getvalue()) > 64
    with VectorIndexReader(Input(output.getvalue())) as reader:
        reader.optimize_for_search()
        actual_ids, distances = reader.search(data[0], SearchParams.ivf(top_k=4, nprobe=2))
    np.testing.assert_array_equal(actual_ids, ids)
    np.testing.assert_allclose(distances, np.sum(data * data, axis=1), rtol=1e-5, atol=1e-6)
    print(f"Final wheel smoke passed: {package}, native {native.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", help="one final wheel path or glob")
    parser.add_argument("--installed", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.installed:
        smoke_installed()
        return
    wheels = sorted(glob.glob(args.wheel or ""))
    if len(wheels) != 1:
        parser.error(f"expected exactly one final wheel, found {len(wheels)}")
    wheel = str(Path(wheels[0]).resolve())
    script = str(Path(__file__).resolve())
    env = os.environ.copy()
    for key in ("PAIMON_VINDEX_LIB_PATH", "PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
        env.pop(key, None)
    with tempfile.TemporaryDirectory(prefix="paimon-wheel-smoke-") as directory:
        root = Path(directory)
        venv.create(root / "venv", with_pip=True)
        python = root / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run([str(python), "-I", "-m", "pip", "install", "--only-binary=:all:", wheel], cwd=root, env=env, check=True)
        subprocess.run([str(python), "-I", script, "--installed"], cwd=root, env=env, check=True)


if __name__ == "__main__":
    main()
