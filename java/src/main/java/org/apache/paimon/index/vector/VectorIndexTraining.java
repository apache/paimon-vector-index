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

package org.apache.paimon.index.vector;

public final class VectorIndexTraining implements AutoCloseable {

    private final Object nativeHandleLock = new Object();
    private long nativePtr;
    private Thread nativeHandleOwner;

    VectorIndexTraining(long nativePtr) {
        this.nativePtr = nativePtr;
    }

    static VectorIndexTraining fromNativePointerForTesting(long nativePtr) {
        return new VectorIndexTraining(nativePtr);
    }

    /** Complete typed training state, independent of any JVM or native handle. */
    public byte[] serialize() {
        synchronized (nativeHandleLock) {
            enterNativeHandle();
            try {
                return VectorIndexNative.serializeTraining(requireOpen());
            } finally {
                exitNativeHandle();
            }
        }
    }

    public static VectorIndexTraining deserialize(byte[] payload) {
        if (payload == null) {
            throw new NullPointerException("payload");
        }
        return new VectorIndexTraining(VectorIndexNative.deserializeTraining(payload));
    }

    public String indexType() {
        synchronized (nativeHandleLock) {
            enterNativeHandle();
            try {
                return VectorIndexNative.trainingIndexType(requireOpen());
            } finally {
                exitNativeHandle();
            }
        }
    }

    public int nlist() {
        synchronized (nativeHandleLock) {
            enterNativeHandle();
            try {
                return VectorIndexNative.trainingNlist(requireOpen());
            } finally {
                exitNativeHandle();
            }
        }
    }

    public int assignCentroid(float[] vector) {
        return findNearestCentroids(vector, 1)[0];
    }

    /** A negative matchingVectors value means no row-ID filter. */
    public int resolveNprobe(VectorSearchParams params, long totalVectors, long matchingVectors) {
        if (params == null) {
            throw new NullPointerException("params");
        }
        if (totalVectors < 0 || matchingVectors < -1) {
            throw new IllegalArgumentException("Invalid vector count");
        }
        synchronized (nativeHandleLock) {
            enterNativeHandle();
            try {
                return VectorIndexNative.trainingNprobe(
                        requireOpen(), params, totalVectors, matchingVectors);
            } finally {
                exitNativeHandle();
            }
        }
    }

    public int[] findNearestCentroids(float[] query, int nprobe) {
        if (query == null) {
            throw new NullPointerException("query");
        }
        if (nprobe <= 0) {
            throw new IllegalArgumentException("nprobe must be positive");
        }
        synchronized (nativeHandleLock) {
            enterNativeHandle();
            try {
                return VectorIndexNative.findNearestCentroids(requireOpen(), query, nprobe);
            } finally {
                exitNativeHandle();
            }
        }
    }

    long createCentroidWriterPointer(int centroid) {
        if (centroid < 0) {
            throw new IllegalArgumentException("centroid must be nonnegative");
        }
        synchronized (nativeHandleLock) {
            enterNativeHandle();
            try {
                return VectorIndexNative.createCentroidWriter(requireOpen(), centroid);
            } finally {
                exitNativeHandle();
            }
        }
    }

    long takeNativePointer() {
        synchronized (nativeHandleLock) {
            enterNativeHandle();
            try {
                long ptr = requireOpen();
                nativePtr = 0L;
                return ptr;
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
                    VectorIndexNative.freeTraining(ptr);
                }
            } finally {
                exitNativeHandle();
            }
        }
    }

    private long requireOpen() {
        if (nativePtr == 0L) {
            throw new IllegalStateException("VectorIndexTraining is closed");
        }
        return nativePtr;
    }

    private void enterNativeHandle() {
        Thread current = Thread.currentThread();
        if (nativeHandleOwner == current) {
            throw new IllegalStateException("VectorIndexTraining native handle is already in use");
        }
        nativeHandleOwner = current;
    }

    private void exitNativeHandle() {
        nativeHandleOwner = null;
    }
}
