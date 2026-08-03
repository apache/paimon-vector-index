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

public final class VectorSearchParams {

    public static final long DEFAULT_IVF_PQ_BATCH_TABLE_REUSE_MAX_BYTES = 512L * 1024 * 1024;

    static final int SEARCH_WIDTH_AUTO = 0;
    static final int SEARCH_WIDTH_IVF_NPROBE = 1;
    static final int SEARCH_WIDTH_DISKANN_L_SEARCH = 2;

    private final int topK;
    private final int searchWidth;
    private final int width;
    private final int maxInitialFilterExpansionFactor;
    private final int ivfPqBatchTableReuseMode;
    private final long ivfPqBatchTableReuseMaxBytes;

    public VectorSearchParams(int topK, int nprobe) {
        this(
                topK,
                SEARCH_WIDTH_IVF_NPROBE,
                nprobe,
                0,
                IvfPqBatchTableReuseMode.AUTO.code(),
                DEFAULT_IVF_PQ_BATCH_TABLE_REUSE_MAX_BYTES);
    }

    private VectorSearchParams(
            int topK,
            int searchWidth,
            int width,
            int maxInitialFilterExpansionFactor,
            int ivfPqBatchTableReuseMode,
            long ivfPqBatchTableReuseMaxBytes) {
        this.topK = topK;
        this.searchWidth = searchWidth;
        this.width = width;
        this.maxInitialFilterExpansionFactor = maxInitialFilterExpansionFactor;
        this.ivfPqBatchTableReuseMode = ivfPqBatchTableReuseMode;
        this.ivfPqBatchTableReuseMaxBytes = ivfPqBatchTableReuseMaxBytes;
    }

    public static VectorSearchParams automatic(int topK) {
        return new VectorSearchParams(
                topK,
                SEARCH_WIDTH_AUTO,
                0,
                0,
                IvfPqBatchTableReuseMode.AUTO.code(),
                DEFAULT_IVF_PQ_BATCH_TABLE_REUSE_MAX_BYTES);
    }

    public static VectorSearchParams ivf(int topK, int nprobe) {
        return new VectorSearchParams(
                topK,
                SEARCH_WIDTH_IVF_NPROBE,
                nprobe,
                0,
                IvfPqBatchTableReuseMode.AUTO.code(),
                DEFAULT_IVF_PQ_BATCH_TABLE_REUSE_MAX_BYTES);
    }

    public static VectorSearchParams diskAnn(int topK, int lSearch) {
        return new VectorSearchParams(
                topK,
                SEARCH_WIDTH_DISKANN_L_SEARCH,
                lSearch,
                0,
                IvfPqBatchTableReuseMode.AUTO.code(),
                DEFAULT_IVF_PQ_BATCH_TABLE_REUSE_MAX_BYTES);
    }

    public int topK() {
        return topK;
    }

    int searchWidth() {
        return searchWidth;
    }

    int width() {
        return width;
    }

    int maxInitialFilterExpansionFactor() {
        return maxInitialFilterExpansionFactor;
    }

    /**
     * Limits filter-driven expansion of the initial automatic IVF nprobe.
     *
     * <p>A factor of 1 keeps the unfiltered automatic width. Lower factors reduce initial search
     * work but may reduce recall compared with uncapped automatic search. Progressive expansion
     * occurs only when fewer than {@code topK} filtered results are found.
     */
    public VectorSearchParams withMaxInitialFilterExpansionFactor(int factor) {
        if (factor <= 0) {
            throw new IllegalArgumentException(
                    "Maximum initial filter expansion factor must be greater than 0");
        }
        if (searchWidth != SEARCH_WIDTH_AUTO) {
            throw new IllegalStateException(
                    "Maximum initial filter expansion factor requires automatic IVF search");
        }
        return new VectorSearchParams(
                topK,
                searchWidth,
                width,
                factor,
                ivfPqBatchTableReuseMode,
                ivfPqBatchTableReuseMaxBytes);
    }

    public IvfPqBatchTableReuseMode ivfPqBatchTableReuse() {
        return IvfPqBatchTableReuseMode.fromCode(ivfPqBatchTableReuseMode);
    }

    int ivfPqBatchTableReuseMode() {
        return ivfPqBatchTableReuseMode;
    }

    public long ivfPqBatchTableReuseMaxBytes() {
        return ivfPqBatchTableReuseMaxBytes;
    }

    public VectorSearchParams withIvfPqBatchTableReuse(IvfPqBatchTableReuseMode mode) {
        if (mode == null) {
            throw new IllegalArgumentException("IVF-PQ batch table reuse mode is null");
        }
        return new VectorSearchParams(
                topK,
                searchWidth,
                width,
                maxInitialFilterExpansionFactor,
                mode.code(),
                ivfPqBatchTableReuseMaxBytes);
    }

    public VectorSearchParams withIvfPqBatchTableReuse(String mode) {
        return withIvfPqBatchTableReuse(IvfPqBatchTableReuseMode.fromString(mode));
    }

    public VectorSearchParams withIvfPqBatchTableReuseMaxBytes(long maxBytes) {
        if (maxBytes <= 0) {
            throw new IllegalArgumentException(
                    "IVF-PQ batch table reuse max bytes must be positive");
        }
        return new VectorSearchParams(
                topK,
                searchWidth,
                width,
                maxInitialFilterExpansionFactor,
                ivfPqBatchTableReuseMode,
                maxBytes);
    }

    public VectorSearchParams withLSearch(int lSearch) {
        return new VectorSearchParams(
                topK,
                SEARCH_WIDTH_DISKANN_L_SEARCH,
                lSearch,
                0,
                ivfPqBatchTableReuseMode,
                ivfPqBatchTableReuseMaxBytes);
    }
}
