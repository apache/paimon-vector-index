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
    private final NativeCallbackContext nativeCallbackContext = new NativeCallbackContext();
    private long nativePtr;
    private Thread nativeHandleOwner;
    private VectorIndexMetadata metadata;

    public VectorIndexReader(VectorIndexInput input) {
        this(input, 4L * 1024 * 1024 * 1024);
    }

    public VectorIndexReader(VectorIndexInput input, long memoryBudgetBytes) {
        if (input == null) {
            throw new NullPointerException("input");
        }
        if (memoryBudgetBytes < 0) {
            throw new IllegalArgumentException("memoryBudgetBytes must be non-negative");
        }
        this.nativePtr =
                VectorIndexNative.openReaderWithOptions(
                        new CallbackTrackingInput(input, nativeCallbackContext), memoryBudgetBytes);
    }

    private VectorIndexReader(long nativePtr) {
        this.nativePtr = nativePtr;
    }

    static VectorIndexReader fromNativePointerForTesting(long nativePtr) {
        return new VectorIndexReader(nativePtr);
    }

    public VectorIndexMetadata metadata() {
        rejectCallbackReentry();
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
        rejectCallbackReentry();
        synchronized (nativeHandleLock) {
            enterNativeHandle();
            try {
                VectorIndexNative.optimizeForSearch(requireOpen());
            } finally {
                exitNativeHandle();
            }
        }
    }

    public void warmupQueries(float[] queries, int queryCount, int lSearch) {
        if (queries == null) {
            throw new NullPointerException("queries");
        }
        if (queryCount < 0) {
            throw new IllegalArgumentException("queryCount must be non-negative");
        }
        if (lSearch < 0) {
            throw new IllegalArgumentException("lSearch must be non-negative");
        }
        rejectCallbackReentry();
        synchronized (nativeHandleLock) {
            enterNativeHandle();
            try {
                VectorIndexNative.warmupQueries(requireOpen(), queries, queryCount, lSearch);
            } finally {
                exitNativeHandle();
            }
        }
    }

    public int calibrateSearchWidth(float[] queries, int queryCount, int topK) {
        if (queries == null) {
            throw new NullPointerException("queries");
        }
        if (queryCount <= 0 || topK <= 0) {
            throw new IllegalArgumentException("queryCount and topK must be positive");
        }
        rejectCallbackReentry();
        synchronized (nativeHandleLock) {
            enterNativeHandle();
            try {
                return VectorIndexNative.calibrateSearchWidth(
                        requireOpen(), queries, queryCount, topK);
            } finally {
                exitNativeHandle();
            }
        }
    }

    public VectorIndexReadPlan readPlan() {
        rejectCallbackReentry();
        synchronized (nativeHandleLock) {
            enterNativeHandle();
            try {
                return VectorIndexNative.readPlan(requireOpen());
            } finally {
                exitNativeHandle();
            }
        }
    }

    public VectorSearchResult search(float[] query, VectorSearchParams params) {
        validateQuery(query);
        validateParams(params);
        rejectCallbackReentry();
        synchronized (nativeHandleLock) {
            enterNativeHandle();
            try {
                return VectorIndexNative.search(requireOpen(), query, params);
            } finally {
                exitNativeHandle();
            }
        }
    }

    public VectorSearchResult search(
            float[] query, VectorSearchParams params, byte[] roaringFilter) {
        validateQuery(query);
        validateParams(params);
        if (roaringFilter == null) {
            throw new NullPointerException("roaringFilter");
        }
        rejectCallbackReentry();
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

    public VectorSearchBatchResult searchBatch(
            float[] queries, int queryCount, VectorSearchParams params) {
        if (queries == null) {
            throw new NullPointerException("queries");
        }
        validateParams(params);
        rejectCallbackReentry();
        synchronized (nativeHandleLock) {
            enterNativeHandle();
            try {
                return VectorIndexNative.searchBatch(requireOpen(), queries, queryCount, params);
            } finally {
                exitNativeHandle();
            }
        }
    }

    public VectorSearchBatchResult searchBatch(
            float[] queries, int queryCount, VectorSearchParams params, byte[] roaringFilter) {
        if (queries == null) {
            throw new NullPointerException("queries");
        }
        validateParams(params);
        if (roaringFilter == null) {
            throw new NullPointerException("roaringFilter");
        }
        rejectCallbackReentry();
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
        rejectCallbackReentry();
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

    private void rejectCallbackReentry() {
        if (nativeCallbackContext.isActiveOnCurrentThread()) {
            throw new IllegalStateException(
                    "VectorIndexReader native handle is already in use by its input callback");
        }
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

    private static final class NativeCallbackContext {
        private final ThreadLocal<Integer> depth = new ThreadLocal<Integer>();

        private void enter() {
            Integer currentDepth = depth.get();
            depth.set(currentDepth == null ? 1 : currentDepth + 1);
        }

        private void exit() {
            Integer currentDepth = depth.get();
            if (currentDepth == null) {
                throw new IllegalStateException("input callback scope is not active");
            }
            if (currentDepth == 1) {
                depth.remove();
            } else {
                depth.set(currentDepth - 1);
            }
        }

        private boolean isActiveOnCurrentThread() {
            Integer currentDepth = depth.get();
            return currentDepth != null && currentDepth > 0;
        }
    }

    private static final class CallbackTrackingInput implements VectorIndexInput {
        private final VectorIndexInput delegate;
        private final NativeCallbackContext callbackContext;

        private CallbackTrackingInput(
                VectorIndexInput delegate, NativeCallbackContext callbackContext) {
            this.delegate = delegate;
            this.callbackContext = callbackContext;
        }

        @Override
        public void pread(long[] positions, byte[][] buffers) {
            callbackContext.enter();
            try {
                delegate.pread(positions, buffers);
            } finally {
                callbackContext.exit();
            }
        }

        @Override
        public long estimatedRandomReadLatencyNanos() {
            return delegate.estimatedRandomReadLatencyNanos();
        }

        @Override
        public long preferredReadWindowBytes() {
            return delegate.preferredReadWindowBytes();
        }

        @Override
        public long maxRangesPerRead() {
            return delegate.maxRangesPerRead();
        }
    }
}
