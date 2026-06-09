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

public final class IVFFlatWriter implements AutoCloseable {

    private final int dimension;
    private long nativePtr;

    public IVFFlatWriter(int dimension, int nlist, Metric metric) {
        if (metric == null) {
            throw new NullPointerException("metric");
        }
        validatePositive(dimension, "dimension");
        validatePositive(nlist, "nlist");
        this.dimension = dimension;
        this.nativePtr = IVFFlatNative.createWriter(dimension, nlist, metric.code());
    }

    private IVFFlatWriter(long nativePtr, int dimension) {
        this.nativePtr = nativePtr;
        this.dimension = dimension;
    }

    static IVFFlatWriter fromNativePointerForTesting(long nativePtr, int dimension) {
        return new IVFFlatWriter(nativePtr, dimension);
    }

    public int dimension() {
        return dimension;
    }

    public void train(float[] data, int vectorCount) {
        validateVectors(data, vectorCount);
        IVFFlatNative.train(requireOpen(), data, vectorCount);
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
        IVFFlatNative.addVectors(requireOpen(), ids, data, vectorCount);
    }

    public void writeIndex(Object output) {
        if (output == null) {
            throw new NullPointerException("output");
        }
        IVFFlatNative.writeIndex(requireOpen(), output);
    }

    @Override
    public void close() {
        long ptr = nativePtr;
        nativePtr = 0L;
        if (ptr != 0L) {
            IVFFlatNative.freeWriter(ptr);
        }
    }

    private void validateVectors(float[] data, int vectorCount) {
        if (data == null) {
            throw new NullPointerException("data");
        }
        validatePositive(vectorCount, "vectorCount");
        long expected = (long) vectorCount * (long) dimension;
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
            throw new IllegalStateException("IVFFlatWriter is closed");
        }
        return nativePtr;
    }

    private static void validatePositive(int value, String name) {
        if (value <= 0) {
            throw new IllegalArgumentException(name + " must be > 0");
        }
    }
}
