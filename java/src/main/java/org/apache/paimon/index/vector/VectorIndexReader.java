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

public final class VectorIndexReader implements AutoCloseable {

    private final Object nativeHandleLock = new Object();
    private long nativePtr;
    private Thread nativeHandleOwner;
    private VectorIndexMetadata metadata;

    public VectorIndexReader(VectorIndexInput input) {
        if (input == null) {
            throw new NullPointerException("input");
        }
        this.nativePtr = VectorIndexNative.openReader(input);
    }

    private VectorIndexReader(long nativePtr) {
        this.nativePtr = nativePtr;
    }

    static VectorIndexReader fromNativePointerForTesting(long nativePtr) {
        return new VectorIndexReader(nativePtr);
    }

    public VectorIndexMetadata metadata() {
        synchronized (nativeHandleLock) {
            enterNativeHandle();
            try {
                requireOpen();
                if (metadata == null) {
                    metadata = VectorIndexNative.metadata(nativePtr);
                }
                return metadata;
            } finally {
                exitNativeHandle();
            }
        }
    }

    public String indexType() {
        return metadata().indexType();
    }

    public int dimension() {
        return metadata().dimension();
    }

    public long totalVectors() {
        return metadata().totalVectors();
    }

    public void optimizeForSearch() {
        synchronized (nativeHandleLock) {
            enterNativeHandle();
            try {
                VectorIndexNative.optimizeForSearch(requireOpen());
            } finally {
                exitNativeHandle();
            }
        }
    }

    public VectorSearchResult search(float[] query, VectorSearchParams params) {
        validateQuery(query);
        validateParams(params);
        synchronized (nativeHandleLock) {
            enterNativeHandle();
            try {
                return VectorIndexNative.search(requireOpen(), query, params);
            } finally {
                exitNativeHandle();
            }
        }
    }

    /**
     * Searches with a required serialized 64-bit Roaring inclusion filter.
     *
     * <p>{@code roaringFilter} must not be {@code null}. Use the unfiltered overload when no
     * filter is needed.
     */
    public VectorSearchResult search(
            float[] query, VectorSearchParams params, byte[] roaringFilter) {
        validateQuery(query);
        validateParams(params);
        if (roaringFilter == null) {
            throw new NullPointerException("roaringFilter");
        }
        synchronized (nativeHandleLock) {
            enterNativeHandle();
            try {
                return VectorIndexNative.searchWithRoaringFilter(
                        requireOpen(), query, params, roaringFilter);
            } finally {
                exitNativeHandle();
            }
        }
    }

    /**
     * Searches with independently optional serialized 64-bit Roaring inclusion and exclusion
     * filters. Exclusion takes precedence.
     *
     * <p>Pass {@code null} when the corresponding filter is absent. The filters may be absent
     * independently; when both are {@code null}, the search is unfiltered. An empty byte array is
     * a present but invalid serialized payload, not an absent filter. A logically empty bitmap
     * must still be serialized in the Roaring format; an empty inclusion bitmap admits no rows,
     * while an empty exclusion bitmap excludes no rows.
     */
    public VectorSearchResult search(
            float[] query,
            VectorSearchParams params,
            byte[] includeRoaringFilter,
            byte[] excludeRoaringFilter) {
        validateQuery(query);
        validateParams(params);
        synchronized (nativeHandleLock) {
            enterNativeHandle();
            try {
                return VectorIndexNative.searchWithRoaringFilterAndExclusions(
                        requireOpen(),
                        query,
                        params,
                        includeRoaringFilter,
                        excludeRoaringFilter);
            } finally {
                exitNativeHandle();
            }
        }
    }

    public VectorSearchBatchResult searchBatch(
            float[] queries, int queryCount, VectorSearchParams params) {
        if (queries == null) {
            throw new NullPointerException("queries");
        }
        validateParams(params);
        synchronized (nativeHandleLock) {
            enterNativeHandle();
            try {
                return VectorIndexNative.searchBatch(requireOpen(), queries, queryCount, params);
            } finally {
                exitNativeHandle();
            }
        }
    }

    /**
     * Batch searches with independently optional serialized 64-bit Roaring inclusion and
     * exclusion filters. Exclusion takes precedence.
     *
     * <p>Pass {@code null} when the corresponding filter is absent. The filters may be absent
     * independently; when both are {@code null}, the search is unfiltered. An empty byte array is
     * a present but invalid serialized payload, not an absent filter. A logically empty bitmap
     * must still be serialized in the Roaring format; an empty inclusion bitmap admits no rows,
     * while an empty exclusion bitmap excludes no rows.
     */
    public VectorSearchBatchResult searchBatch(
            float[] queries,
            int queryCount,
            VectorSearchParams params,
            byte[] includeRoaringFilter,
            byte[] excludeRoaringFilter) {
        if (queries == null) {
            throw new NullPointerException("queries");
        }
        validateParams(params);
        synchronized (nativeHandleLock) {
            enterNativeHandle();
            try {
                return VectorIndexNative.searchBatchWithRoaringFilterAndExclusions(
                        requireOpen(),
                        queries,
                        queryCount,
                        params,
                        includeRoaringFilter,
                        excludeRoaringFilter);
            } finally {
                exitNativeHandle();
            }
        }
    }

    /**
     * Batch searches with a required serialized 64-bit Roaring inclusion filter.
     *
     * <p>{@code roaringFilter} must not be {@code null}. Use the unfiltered overload when no
     * filter is needed.
     */
    public VectorSearchBatchResult searchBatch(
            float[] queries, int queryCount, VectorSearchParams params, byte[] roaringFilter) {
        if (queries == null) {
            throw new NullPointerException("queries");
        }
        validateParams(params);
        if (roaringFilter == null) {
            throw new NullPointerException("roaringFilter");
        }
        synchronized (nativeHandleLock) {
            enterNativeHandle();
            try {
                return VectorIndexNative.searchBatchWithRoaringFilter(
                        requireOpen(), queries, queryCount, params, roaringFilter);
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
                    VectorIndexNative.freeReader(ptr);
                }
            } finally {
                exitNativeHandle();
            }
        }
    }

    private void validateQuery(float[] query) {
        if (query == null) {
            throw new NullPointerException("query");
        }
    }

    private void validateParams(VectorSearchParams params) {
        if (params == null) {
            throw new NullPointerException("params");
        }
    }

    private long requireOpen() {
        if (nativePtr == 0L) {
            throw new IllegalStateException("VectorIndexReader is closed");
        }
        return nativePtr;
    }

    private void enterNativeHandle() {
        Thread current = Thread.currentThread();
        if (nativeHandleOwner == current) {
            throw new IllegalStateException("VectorIndexReader native handle is already in use");
        }
        nativeHandleOwner = current;
    }

    private void exitNativeHandle() {
        nativeHandleOwner = null;
    }
}
