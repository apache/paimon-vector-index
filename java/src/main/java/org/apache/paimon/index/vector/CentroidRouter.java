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

/** Portable, read-only coarse centers shared by local IVF and DiskANN indexes. */
public final class CentroidRouter implements AutoCloseable {
    /** Consumes ordinary IVF-FLAT training and takes ownership of its coarse centers. */
    public static CentroidRouter fromTraining(VectorIndexTraining coarse) {
        return new CentroidRouter(VectorIndexNative.routerFromTraining(coarse.takeNativePointer()));
    }

    private long nativePtr;

    CentroidRouter(long nativePtr) {
        this.nativePtr = nativePtr;
    }

    public synchronized byte[] serialize() {
        return VectorIndexNative.serializeRouter(requireOpen());
    }

    /** Upper bound for validating stored payload lengths before reading into memory. */
    public static int maxSerializedBytes() {
        return VectorIndexNative.maxRouterBytes();
    }

    public static CentroidRouter deserialize(byte[] payload) {
        if (payload == null) {
            throw new NullPointerException("payload");
        }
        return new CentroidRouter(VectorIndexNative.deserializeRouter(payload));
    }

    public synchronized int nlist() {
        return VectorIndexNative.routerNlist(requireOpen());
    }

    public synchronized int assignCentroid(float[] vector) {
        if (vector == null) {
            throw new NullPointerException("vector");
        }
        return VectorIndexNative.assignCentroid(requireOpen(), vector);
    }

    public synchronized int[] findNearestCentroids(float[] query, int nprobe) {
        if (query == null) {
            throw new NullPointerException("query");
        }
        if (nprobe <= 0) {
            throw new IllegalArgumentException("nprobe must be positive");
        }
        return VectorIndexNative.findNearestCentroids(requireOpen(), query, nprobe);
    }

    /**
     * Resolves the initial number of shards to probe. Automatic routing starts at three shards for
     * four centers and four shards for eight centers. Top-K and filter selectivity can increase
     * the width. Pass -1 for matchingVectors when there is no row-ID filter.
     */
    public synchronized int resolveNprobe(
            VectorSearchParams params, long totalVectors, long matchingVectors) {
        if (params == null) {
            throw new NullPointerException("params");
        }
        return VectorIndexNative.routerNprobe(requireOpen(), params, totalVectors, matchingVectors);
    }

    @Override
    public synchronized void close() {
        if (nativePtr != 0L) {
            VectorIndexNative.freeRouter(nativePtr);
            nativePtr = 0L;
        }
    }

    private long requireOpen() {
        if (nativePtr == 0L) {
            throw new IllegalStateException("CentroidRouter is closed");
        }
        return nativePtr;
    }
}
