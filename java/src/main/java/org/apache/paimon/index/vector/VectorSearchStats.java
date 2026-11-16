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

/**
 * Last top-K call. Batch counters sum all work; widths are maxima. IVF candidates
 * are rows before filtering; DiskANN candidates are PQ evaluations. Reads count
 * adapter requests, including lazy initialization. Cache units are IVF payload
 * entries or DiskANN adjacency/raw windows. Other operations preserve the snapshot.
 */
public final class VectorSearchStats {
    public enum Strategy {
        NOT_EXECUTED, IVF, DISKANN_GRAPH, FILTERED_EXHAUSTIVE, GRAPH_THEN_EXHAUSTIVE, MIXED
    }

    public enum FallbackReason {
        NONE, INSUFFICIENT_RESULTS
    }

    private final long[] values;

    VectorSearchStats(long[] values) {
        if (values.length != 12) {
            throw new IllegalArgumentException("invalid search statistics length");
        }
        this.values = values.clone();
    }

    public long queryCount() {
        return values[0];
    }

    /** Width kind: 0 before a call, 1 for IVF nprobe, 2 for DiskANN search-list size. */
    public long searchWidth() {
        return values[1];
    }

    public long initialWidth() {
        return values[2];
    }

    public long effectiveWidth() {
        return values[3];
    }

    public long candidateCount() {
        return values[4];
    }

    public Strategy strategy() {
        return Strategy.values()[(int) values[5]];
    }

    public FallbackReason fallbackReason() {
        return FallbackReason.values()[(int) values[6]];
    }

    public long readCalls() {
        return values[7];
    }

    public long readRanges() {
        return values[8];
    }

    public long readBytes() {
        return values[9];
    }

    public long cacheHits() {
        return values[10];
    }

    public long cacheMisses() {
        return values[11];
    }
}
