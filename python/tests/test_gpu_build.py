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
import importlib.util
import io
import json
import os
import sys
import subprocess
import threading
import types
from pathlib import Path

import numpy as np
import pytest

from paimon_vindex import SearchParams, VectorIndexReader, VectorIndexTrainer, VectorIndexWriter
from paimon_vindex._ffi import lib
from paimon_vindex.gpu import CuvsIvfSqWriter


class BytesInput:
    def __init__(self, payload):
        self.payload = payload

    def pread_many(self, ranges):
        return [self.payload[p:p+n] for p,n in ranges]


def make_writer(data, metric="l2", nlist=4, centers=None):
    options = {"index.type":"ivf_sq", "dimension":str(data.shape[1]), "nlist":str(nlist),
               "metric":metric, "ivf.coarse-assignment":"exact"}
    with VectorIndexTrainer.create(options) as trainer:
        trainer.add_training_vectors(data)
        if centers is None:
            training = trainer.finish_training()
        else:
            with trainer.prepare_training() as prepared:
                training = prepared.finish_with_ivf_centroids(centers)
    return VectorIndexWriter(training)


def payload(writer):
    result = io.BytesIO()
    writer.write(result)
    return result.getvalue()


@pytest.mark.parametrize("metric", ["l2", "cosine", "inner_product"])
def test_preassigned_native_writer_and_model(metric):
    data = np.random.default_rng(42).normal(size=(257,9)).astype(np.float32)
    ids = np.arange(len(data),dtype=np.int64)*13-2000
    with make_writer(data, metric) as original, make_writer(data, metric) as assigned:
        model = assigned.ivf_sq_encoding_model()
        assert model.exact_assignment and model.centroids.shape == model.mins.shape == model.maxs.shape == (4,9)
        assert not model.centroids.flags.writeable
        processed = assigned._preprocess_ivf_sq_vectors(data)
        lists = ((processed.astype(np.float64)[:,None,:]-model.centroids.astype(np.float64)[None,:,:])**2).sum(axis=2).argmin(axis=1)
        for start in range(0,len(data),31):
            assigned.add_preassigned_vectors(ids[start:start+31], data[start:start+31], lists[start:start+31])
        original.add_vectors(ids,data)
        assert payload(original) == payload(assigned)
        model.centroids.flags.writeable = True
        model.centroids[:] = 123
        assert not np.all(assigned.ivf_sq_encoding_model().centroids == 123)


def test_invalid_batches_and_ffi_model_buffers_preserve_writer():
    data = np.arange(64,dtype=np.float32).reshape(8,8)
    with make_writer(data) as writer:
        writer.add_vectors([100],data[:1])
        before=payload(writer)
        for labels in [[0,4], [-1,0], [2**32,0], [0.5,1.0]]:
            with pytest.raises((ValueError,RuntimeError)):
                writer.add_preassigned_vectors([1,2],data[:2],labels)
            with pytest.raises((ValueError,RuntimeError)):
                writer.add_encoded_vectors([1,2],np.zeros((2,8),dtype=np.uint8),labels)
        with pytest.raises(ValueError):
            writer.add_encoded_vectors([1],np.zeros((1,8),dtype=np.int32),[0])
        with pytest.raises((ValueError,RuntimeError)):
            writer.add_preassigned_vectors([1],np.full((1,8),np.nan,dtype=np.float32),[0])
        out=np.full((4,8),123,dtype=np.float32)
        ptr=out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        assert lib.paimon_vindex_writer_ivf_sq_copy_model(writer._handle,ptr,None,ptr,out.size) != 0
        assert np.all(out==123)
        assert lib.paimon_vindex_writer_add_encoded_vectors(writer._handle,None,None,0,None,1) != 0
        assert payload(writer)==before
        writer.add_encoded_vectors([1],np.zeros((1,8),dtype=np.uint8),[0])
        assert writer.ivf_sq_partition_sizes().sum()==2


def install_fake_cuda(monkeypatch, predict, *, asarray=np.asarray, synchronize=None):
    from paimon_vindex import gpu
    workers = []

    class Device:
        def __init__(self, device): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def synchronize(self):
            if synchronize is not None:
                synchronize()

    class Worker(gpu.CuvsKMeans):
        def __init__(self, device=0):
            self._lock = threading.Lock()
            self.device=device
            self.device_name="fake-contract-device"
            self.cuvs_version=self.cupy_version="fake-contract-test"
            self._resources=object()
            self._params_type=lambda **kwargs: kwargs
            self._cp=types.SimpleNamespace(cuda=types.SimpleNamespace(Device=Device),
                                           asarray=asarray,asnumpy=np.asarray)
            workers.append(self)

    monkeypatch.setattr(gpu,"CuvsKMeans",Worker)
    monkeypatch.setitem(sys.modules,"cuvs.cluster.kmeans",types.SimpleNamespace(predict=predict))
    return workers


@pytest.mark.parametrize("target", ["worker", "builder"])
@pytest.mark.parametrize("body_error", [False, True])
def test_cuda_cleanup_failure_releases_resources_and_preserves_original_error(monkeypatch, target, body_error):
    install_fake_cuda(monkeypatch, lambda *args, **kwargs: (np.array([0]), 0.))
    data = np.arange(64, dtype=np.float32).reshape(8, 8)
    with make_writer(data) as writer:
        builder = CuvsIvfSqWriter(writer, encode=False)
        worker = builder._worker
        resource = worker if target == "worker" else builder

        def fail_sync(self):
            raise RuntimeError("CUDA synchronization failed")

        monkeypatch.setattr(worker._cp.cuda.Device, "synchronize", fail_sync)
        if body_error:
            with pytest.raises(ValueError, match="original operation failed"):
                with resource:
                    raise ValueError("original operation failed")
        else:
            with pytest.raises(RuntimeError, match="CUDA synchronization failed"):
                resource.close()
        assert worker._resources is None
        resource.close()  # Failed cleanup still makes close idempotent.
        with pytest.raises(RuntimeError, match="closed"):
            with resource:
                pass
        if target == "builder":
            assert builder._arrays is None
        builder.close()
        writer.add_vectors([1], data[:1])
        assert writer.ivf_sq_partition_sizes().sum() == 1


def test_gpu_constructor_preserves_allocation_error_when_cleanup_also_fails(monkeypatch):
    def fail_copy(*args):
        raise MemoryError("model allocation failed")

    def fail_sync():
        raise RuntimeError("cleanup synchronization failed")

    workers = install_fake_cuda(monkeypatch, lambda *args: None,
                                asarray=fail_copy, synchronize=fail_sync)
    data = np.arange(64, dtype=np.float32).reshape(8, 8)
    with make_writer(data) as writer:
        with pytest.raises(MemoryError, match="model allocation failed"):
            CuvsIvfSqWriter(writer, encode=False)
        assert workers[0]._resources is None
        writer.add_vectors([1], data[:1])


def test_invalid_training_input_clears_previous_telemetry(monkeypatch):
    workers = install_fake_cuda(monkeypatch, lambda *args: None)
    data = np.arange(64, dtype=np.float32).reshape(8, 8)
    with make_writer(data) as writer, CuvsIvfSqWriter(writer, encode=False):
        worker = workers[0]
        worker.last_run = {"rows": 123}
        with pytest.raises(TypeError, match="PreparedIvfSqTraining"):
            worker.fit(object())
        assert worker.last_run is None


def test_adapter_rejects_bad_backend_labels_and_recovers(monkeypatch):
    state={"labels":np.array([-1,0],dtype=np.int32)}
    install_fake_cuda(monkeypatch,lambda *args,**kwargs:(state["labels"],0.))
    data=np.arange(64,dtype=np.float32).reshape(8,8)
    with make_writer(data) as writer:
        with CuvsIvfSqWriter(writer,encode=False) as gpu:
            before=payload(writer)
            for labels in [np.array([-1,0]), np.array([0,4]), np.array([0]), np.array([0.,1.])]:
                state["labels"]=labels
                with pytest.raises(RuntimeError,match="invalid partition"):
                    gpu.add_vectors([1,2],data[:2])
                assert gpu.last_run is None and payload(writer)==before
            state["labels"]=np.array([0,1],dtype=np.int32)
            gpu.add_vectors([1,2],data[:2])
            assert gpu.last_run["rows"]==2
            with pytest.raises(ValueError): gpu.add_vectors([3],np.full((1,8),np.inf))
            assert gpu.last_run is None
            assert writer.ivf_sq_partition_sizes().sum()==2
        with pytest.raises(RuntimeError,match="closed"): gpu.add_vectors([3],data[:1])
        writer.add_vectors([3],data[:1])  # adapter never owns the writer
        assert writer.ivf_sq_partition_sizes().sum()==3


def test_adapter_requires_consistent_assignment_before_importing_cuda():
    data=np.zeros((4,1024),dtype=np.float32)
    options={"index.type":"ivf_sq","dimension":"1024","nlist":"1024","metric":"l2"}
    with VectorIndexTrainer.create(options) as trainer:
        trainer.add_training_vectors(data)
        with trainer.prepare_training() as p:
            training=p.finish_with_ivf_centroids(np.zeros((1024,1024),dtype=np.float32))
    with VectorIndexWriter(training) as writer:
        with pytest.raises(ValueError,match="ivf.coarse-assignment=exact"):
            CuvsIvfSqWriter(writer)


CUDA=pytest.mark.skipif(os.environ.get("PAIMON_TEST_CUVS")!="1",reason="set PAIMON_TEST_CUVS=1 on CUDA host")


@pytest.mark.parametrize("build_backend", ["cpu", pytest.param("cuvs-encode", marks=CUDA)])
def test_fixed_model_benchmark_and_portable_reader(tmp_path, monkeypatch, build_backend):
    data=np.random.default_rng(1234).normal(size=(256,9)).astype(np.float32)
    with make_writer(data,nlist=8) as writer:
        model=writer.ivf_sq_encoding_model()
    np.save(tmp_path/"base.npy",data)
    np.save(tmp_path/"centers.npy",model.centroids)
    script=Path(__file__).resolve().parents[2]/"tools"/"benchmark_gpu_training.py"
    subprocess.run([sys.executable,str(script),"--preset","smoke","--base",str(tmp_path/"base.npy"),
                    "--backends","external","--centroids",str(tmp_path/"centers.npy"),"--build-index",
                    "--build-backend",build_backend,
                    "--coarse-assignment","exact","--threads","2","--repeats","1",
                    "--output-dir",str(tmp_path/"result")],check=True,capture_output=True,text=True)
    run = json.loads((tmp_path/"result"/"runs.jsonl").read_text())
    if build_backend != "cpu":
        assert run["gpu_build"]["backend"] == build_backend
        for key in ("device_name", "cuvs_version", "cupy_version"):
            assert run["gpu_build"][key]
    spec=importlib.util.spec_from_file_location("gpu_build_benchmark_test",script)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module,"os",types.SimpleNamespace())  # Windows has no os.pread
    queries=data[:8]
    truth=np.argsort(((queries[:,None,:]-data[None,:,:])**2).sum(axis=2),axis=1)[:,:10]
    result=module.evaluate_index(tmp_path/"result"/"external-0.index",queries,truth,10,8)
    assert result["recall_at_k"]>=.9


@CUDA
@pytest.mark.parametrize("dimension",[1,7,8,9,129])
def test_gpu_encoding_clipping_rounding_and_scalar_tails(dimension):
    train=np.stack([np.full(dimension,-1,np.float32),np.full(dimension,1,np.float32)])
    centers=np.zeros((1,dimension),dtype=np.float32)
    values=np.concatenate([np.array([-3,-1,0,1,3],dtype=np.float32),
                           ((np.arange(255,dtype=np.float32)+.5)/np.float32(127.5)-1)])
    data=np.resize(values,(523,dimension)).astype(np.float32)
    ids=np.arange(len(data),dtype=np.int64)[::-1]
    with make_writer(train,nlist=1,centers=centers) as cpu, make_writer(train,nlist=1,centers=centers) as accelerated:
        cpu.add_vectors(ids,data)
        with CuvsIvfSqWriter(accelerated) as gpu:
            for start in range(0,len(data),127): gpu.add_vectors(ids[start:start+127],data[start:start+127])
        assert payload(cpu)==payload(accelerated)


@CUDA
@pytest.mark.parametrize("metric",["l2","cosine","inner_product"])
def test_gpu_encoding_constant_bounds_and_zero_vectors(metric):
    data=np.zeros((32,9),dtype=np.float32)
    centers=np.zeros((1,9),dtype=np.float32)
    with make_writer(data,metric,1,centers) as cpu, make_writer(data,metric,1,centers) as accelerated:
        ids=np.arange(len(data),dtype=np.int64)
        cpu.add_vectors(ids,data)
        with CuvsIvfSqWriter(accelerated) as gpu:
            gpu.add_vectors(ids,data)
        assert payload(cpu)==payload(accelerated)


@CUDA
@pytest.mark.parametrize("dimension", [1, 8, 9])
def test_gpu_encoding_subnormal_bounds_and_overflowing_scale(dimension):
    # Finite, nonconstant bounds can have an infinite f32 encoding scale.
    # Exercise NaN from 0 * inf at the minimum, saturation, and scalar tails.
    tiny = np.float32(np.finfo(np.float32).tiny / 16)
    train = np.stack([np.full(dimension, -tiny, np.float32),
                      np.full(dimension, tiny, np.float32)])
    centers = np.zeros((1, dimension), dtype=np.float32)
    data = np.repeat((np.arange(-2, 3, dtype=np.float32) * tiny)[:, None], dimension, axis=1)
    ids = np.arange(len(data), dtype=np.int64)
    with make_writer(train, nlist=1, centers=centers) as cpu, make_writer(train, nlist=1, centers=centers) as accelerated:
        cpu.add_vectors(ids, data)
        with CuvsIvfSqWriter(accelerated) as gpu:
            gpu.add_vectors(ids, data)
        assert payload(cpu) == payload(accelerated)


@CUDA
@pytest.mark.parametrize("metric",["l2","cosine","inner_product"])
def test_gpu_build_default_training_and_heldout_retrieval(metric):
    from paimon_vindex.gpu import CuvsKMeans
    rng=np.random.default_rng(42)
    k,d,n=300,16,12000
    means=rng.normal(size=(k,d)).astype(np.float32)*5
    means *= np.float32(20) / np.linalg.norm(means,axis=1)[:,None]
    data=means[np.arange(n)%k]+rng.normal(0,.2,size=(n,d)).astype(np.float32)
    queries=means[rng.integers(k,size=96)]+rng.normal(0,.2,size=(96,d)).astype(np.float32)
    options={"index.type":"ivf_sq","dimension":str(d),"nlist":str(k),"metric":metric,
             "ivf.coarse-assignment":"exact"}
    with VectorIndexTrainer.create(options) as trainer:
        trainer.add_training_vectors(data)
        with trainer.prepare_training() as p, CuvsKMeans() as gpu:
            centers=gpu.fit(p)  # exercise default initialization with nlist > 256
    if metric=="l2":
        scores=((queries[:,None,:]-data[None,:,:])**2).sum(axis=2)
    elif metric=="cosine":
        scores=-(queries@data.T)/(np.linalg.norm(queries,axis=1)[:,None]*np.linalg.norm(data,axis=1)[None,:])
    else:
        scores=-(queries@data.T)
    truth=np.argsort(scores,axis=1)[:,:10]
    payloads=[]
    for mode in ["cpu","cuvs-assign","cuvs-encode"]:
        with make_writer(data,metric,k,centers) as writer:
            if mode=="cpu": writer.add_vectors(np.arange(n,dtype=np.int64),data)
            else:
                with CuvsIvfSqWriter(writer,encode=mode=="cuvs-encode") as gpu:
                    for start in range(0,n,1024):
                        gpu.add_vectors(np.arange(start,min(start+1024,n),dtype=np.int64),data[start:start+1024])
            assert writer.ivf_sq_partition_sizes().sum()==n
            payloads.append(payload(writer))
        with VectorIndexReader(BytesInput(payloads[-1])) as reader:
            ids,distances=reader.search_batch(queries,SearchParams.ivf(10,16))
            recall=sum(len(set(found)&set(expected)) for found,expected in zip(ids,truth))/(len(queries)*10)
            assert recall>=.9, (metric,mode,recall)
            assert np.isfinite(distances).all()
    assert payloads[1]==payloads[2]  # same GPU assignments, native versus GPU encoding
