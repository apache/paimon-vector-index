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

public final class IVFFlatReader implements AutoCloseable {

    private long nativePtr;

    public IVFFlatReader(Object input) {
        if (input == null) {
            throw new NullPointerException("input");
        }
        this.nativePtr = IVFFlatNative.openReader(input);
    }

    private IVFFlatReader(long nativePtr) {
        this.nativePtr = nativePtr;
    }

    static IVFFlatReader fromNativePointerForTesting(long nativePtr) {
        return new IVFFlatReader(nativePtr);
    }

    public int dimension() {
        return IVFFlatNative.getDimension(requireOpen());
    }

    public long totalVectors() {
        return IVFFlatNative.getTotalVectors(requireOpen());
    }

    public IVFPQResult search(float[] query, int topK, int nprobe) {
        if (query == null) {
            throw new NullPointerException("query");
        }
        validatePositive(topK, "topK");
        validatePositive(nprobe, "nprobe");
        return IVFFlatNative.search(requireOpen(), query, topK, nprobe);
    }

    public IVFPQResult search(float[] query, int topK, int nprobe, byte[] roaringFilter) {
        if (query == null) {
            throw new NullPointerException("query");
        }
        if (roaringFilter == null) {
            throw new NullPointerException("roaringFilter");
        }
        validatePositive(topK, "topK");
        validatePositive(nprobe, "nprobe");
        return IVFFlatNative.searchWithRoaringFilter(
                requireOpen(), query, topK, nprobe, roaringFilter);
    }

    public IVFPQBatchResult searchBatch(float[] queries, int queryCount, int topK, int nprobe) {
        if (queries == null) {
            throw new NullPointerException("queries");
        }
        validatePositive(queryCount, "queryCount");
        validatePositive(topK, "topK");
        validatePositive(nprobe, "nprobe");
        return IVFFlatNative.searchBatch(requireOpen(), queries, queryCount, topK, nprobe);
    }

    public IVFPQBatchResult searchBatch(
            float[] queries, int queryCount, int topK, int nprobe, byte[] roaringFilter) {
        if (queries == null) {
            throw new NullPointerException("queries");
        }
        if (roaringFilter == null) {
            throw new NullPointerException("roaringFilter");
        }
        validatePositive(queryCount, "queryCount");
        validatePositive(topK, "topK");
        validatePositive(nprobe, "nprobe");
        return IVFFlatNative.searchBatchWithRoaringFilter(
                requireOpen(), queries, queryCount, topK, nprobe, roaringFilter);
    }

    @Override
    public void close() {
        long ptr = nativePtr;
        nativePtr = 0L;
        if (ptr != 0L) {
            IVFFlatNative.freeReader(ptr);
        }
    }

    private long requireOpen() {
        if (nativePtr == 0L) {
            throw new IllegalStateException("IVFFlatReader is closed");
        }
        return nativePtr;
    }

    private static void validatePositive(int value, String name) {
        if (value <= 0) {
            throw new IllegalArgumentException(name + " must be > 0");
        }
    }
}
