// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

package org.apache.paimon.index.ivfpq;

public final class VectorIndexWriter implements AutoCloseable {

    private final VectorIndexConfig config;
    private final Object nativeHandleLock = new Object();
    private long nativePtr;
    private Thread nativeHandleOwner;

    public VectorIndexWriter(VectorIndexConfig config) {
        if (config == null) {
            throw new NullPointerException("config");
        }
        this.config = config;
        HnswConfig hnsw = config.hnsw();
        this.nativePtr =
                VectorIndexNative.createWriter(
                        config.indexType().code(),
                        config.dimension(),
                        config.nlist(),
                        config.pqM(),
                        config.metric().code(),
                        config.useOpq(),
                        hnsw.m(),
                        hnsw.efConstruction(),
                        hnsw.maxLevel());
    }

    private VectorIndexWriter(long nativePtr, VectorIndexConfig config) {
        this.nativePtr = nativePtr;
        this.config = config;
    }

    static VectorIndexWriter fromNativePointerForTesting(long nativePtr, VectorIndexConfig config) {
        return new VectorIndexWriter(nativePtr, config);
    }

    public VectorIndexConfig config() {
        return config;
    }

    public int dimension() {
        return config.dimension();
    }

    public void train(float[] data, int vectorCount) {
        validateVectors(data, vectorCount);
        synchronized (nativeHandleLock) {
            enterNativeHandle();
            try {
                VectorIndexNative.train(requireOpen(), data, vectorCount);
            } finally {
                exitNativeHandle();
            }
        }
    }

    public void addVectors(long[] ids, float[] data, int vectorCount) {
        if (ids == null) {
            throw new NullPointerException("ids");
        }
        validateVectors(data, vectorCount);
        if (ids.length < vectorCount) {
            throw new IllegalArgumentException(
                    "ids length " + ids.length + " < vectorCount " + vectorCount);
        }
        synchronized (nativeHandleLock) {
            enterNativeHandle();
            try {
                VectorIndexNative.addVectors(requireOpen(), ids, data, vectorCount);
            } finally {
                exitNativeHandle();
            }
        }
    }

    public void writeIndex(Object output) {
        if (output == null) {
            throw new NullPointerException("output");
        }
        synchronized (nativeHandleLock) {
            enterNativeHandle();
            try {
                VectorIndexNative.writeIndex(requireOpen(), output);
            } finally {
                exitNativeHandle();
            }
        }
    }

    @Override
    public void close() {
        synchronized (nativeHandleLock) {
            enterNativeHandle();
            try {
                long ptr = nativePtr;
                nativePtr = 0L;
                if (ptr != 0L) {
                    VectorIndexNative.freeWriter(ptr);
                }
            } finally {
                exitNativeHandle();
            }
        }
    }

    private void validateVectors(float[] data, int vectorCount) {
        if (data == null) {
            throw new NullPointerException("data");
        }
        VectorIndexConfig.validatePositive(vectorCount, "vectorCount");
        long expected = (long) vectorCount * (long) config.dimension();
        if (expected > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("vectorCount * dimension overflows int");
        }
        if (data.length < expected) {
            throw new IllegalArgumentException(
                    "data length " + data.length + " < vectorCount * dimension " + expected);
        }
    }

    private long requireOpen() {
        if (nativePtr == 0L) {
            throw new IllegalStateException("VectorIndexWriter is closed");
        }
        return nativePtr;
    }

    private void enterNativeHandle() {
        Thread current = Thread.currentThread();
        if (nativeHandleOwner == current) {
            throw new IllegalStateException("VectorIndexWriter native handle is already in use");
        }
        nativeHandleOwner = current;
    }

    private void exitNativeHandle() {
        nativeHandleOwner = null;
    }
}
