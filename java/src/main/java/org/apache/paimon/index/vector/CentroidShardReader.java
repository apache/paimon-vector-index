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

/** Searches a shard selected by an external router. IVF searches its only populated list. DiskANN searches its local graph. */
public final class CentroidShardReader implements VectorSearchReader {
    private final VectorIndexReader handle;

    public CentroidShardReader(VectorIndexInput input) {
        this(input, VectorIndexReader.DEFAULT_MEMORY_BUDGET);
    }

    public static CentroidShardReader openWithMemoryBudget(VectorIndexInput input, long memoryBudgetBytes) {
        return new CentroidShardReader(input, memoryBudgetBytes);
    }

    private CentroidShardReader(VectorIndexInput input, long memoryBudgetBytes) {
        handle = new VectorIndexReader(input, memoryBudgetBytes, true);
    }

    @Override
    public VectorIndexMetadata metadata() {
        return handle.metadata();
    }

    @Override
    public void optimizeForSearch() {
        handle.optimizeForSearch();
    }

    @Override
    public VectorSearchStats searchStats() {
        return handle.searchStats();
    }

    @Override
    public VectorSearchResult search(float[] query, VectorSearchParams params) {
        return handle.search(query, params);
    }

    @Override
    public VectorSearchResult search(float[] query, VectorSearchParams params, byte[] roaringFilter) {
        return handle.search(query, params, roaringFilter);
    }

    @Override
    public VectorSearchBatchResult searchBatch(float[] queries, int count, VectorSearchParams params) {
        return handle.searchBatch(queries, count, params);
    }

    @Override
    public VectorSearchBatchResult searchBatch(
            float[] queries, int count, VectorSearchParams params, byte[] roaringFilter) {
        return handle.searchBatch(queries, count, params, roaringFilter);
    }

    @Override
    public void close() {
        handle.close();
    }
}
