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

    static final int SEARCH_WIDTH_AUTO = 0;
    static final int SEARCH_WIDTH_IVF_NPROBE = 1;
    static final int SEARCH_WIDTH_DISKANN_L_SEARCH = 2;

    private final int topK;
    private final int searchWidth;
    private final int width;
    private final int ivfPqBatchTableReuseMode;

    public VectorSearchParams(int topK, int nprobe) {
        this(
                topK,
                SEARCH_WIDTH_IVF_NPROBE,
                nprobe,
                IvfPqBatchTableReuseMode.AUTO.code());
    }

    private VectorSearchParams(
            int topK, int searchWidth, int width, int ivfPqBatchTableReuseMode) {
        this.topK = topK;
        this.searchWidth = searchWidth;
        this.width = width;
        this.ivfPqBatchTableReuseMode = ivfPqBatchTableReuseMode;
    }

    public static VectorSearchParams automatic(int topK) {
        return new VectorSearchParams(
                topK, SEARCH_WIDTH_AUTO, 0, IvfPqBatchTableReuseMode.AUTO.code());
    }

    public static VectorSearchParams ivf(int topK, int nprobe) {
        return new VectorSearchParams(
                topK,
                SEARCH_WIDTH_IVF_NPROBE,
                nprobe,
                IvfPqBatchTableReuseMode.AUTO.code());
    }

    public static VectorSearchParams diskAnn(int topK, int lSearch) {
        return new VectorSearchParams(
                topK,
                SEARCH_WIDTH_DISKANN_L_SEARCH,
                lSearch,
                IvfPqBatchTableReuseMode.AUTO.code());
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

    public IvfPqBatchTableReuseMode ivfPqBatchTableReuse() {
        return IvfPqBatchTableReuseMode.fromCode(ivfPqBatchTableReuseMode);
    }

    int ivfPqBatchTableReuseMode() {
        return ivfPqBatchTableReuseMode;
    }

    public VectorSearchParams withIvfPqBatchTableReuse(IvfPqBatchTableReuseMode mode) {
        if (mode == null) {
            throw new IllegalArgumentException("IVF-PQ batch table reuse mode is null");
        }
        return new VectorSearchParams(topK, searchWidth, width, mode.code());
    }

    public VectorSearchParams withIvfPqBatchTableReuse(String mode) {
        return withIvfPqBatchTableReuse(IvfPqBatchTableReuseMode.fromString(mode));
    }

    public VectorSearchParams withLSearch(int lSearch) {
        return new VectorSearchParams(
                topK, SEARCH_WIDTH_DISKANN_L_SEARCH, lSearch, ivfPqBatchTableReuseMode);
    }
}
