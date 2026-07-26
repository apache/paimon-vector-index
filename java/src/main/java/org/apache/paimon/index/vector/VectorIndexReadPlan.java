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

/** The concrete DiskANN read policy derived from the input and Reader memory budget. */
public final class VectorIndexReadPlan {

    private final long randomReadLatencyNanos;
    private final long windowBytes;
    private final long maxRangesPerRead;
    private final long graphBeamWidth;
    private final long filteredGraphBeamWidth;
    private final long adjacencyPreloadBytes;
    private final long adjacencyCacheBytes;
    private final long rawVectorCacheBytes;
    private final long memoryBudgetBytes;

    public VectorIndexReadPlan(
            long randomReadLatencyNanos,
            long windowBytes,
            long maxRangesPerRead,
            long graphBeamWidth,
            long filteredGraphBeamWidth,
            long adjacencyPreloadBytes,
            long adjacencyCacheBytes,
            long rawVectorCacheBytes,
            long memoryBudgetBytes) {
        this.randomReadLatencyNanos = randomReadLatencyNanos;
        this.windowBytes = windowBytes;
        this.maxRangesPerRead = maxRangesPerRead;
        this.graphBeamWidth = graphBeamWidth;
        this.filteredGraphBeamWidth = filteredGraphBeamWidth;
        this.adjacencyPreloadBytes = adjacencyPreloadBytes;
        this.adjacencyCacheBytes = adjacencyCacheBytes;
        this.rawVectorCacheBytes = rawVectorCacheBytes;
        this.memoryBudgetBytes = memoryBudgetBytes;
    }

    public long randomReadLatencyNanos() {
        return randomReadLatencyNanos;
    }

    public long windowBytes() {
        return windowBytes;
    }

    public long maxRangesPerRead() {
        return maxRangesPerRead;
    }

    public long graphBeamWidth() {
        return graphBeamWidth;
    }

    public long filteredGraphBeamWidth() {
        return filteredGraphBeamWidth;
    }

    public long adjacencyPreloadBytes() {
        return adjacencyPreloadBytes;
    }

    public long adjacencyCacheBytes() {
        return adjacencyCacheBytes;
    }

    public long rawVectorCacheBytes() {
        return rawVectorCacheBytes;
    }

    public long memoryBudgetBytes() {
        return memoryBudgetBytes;
    }
}
