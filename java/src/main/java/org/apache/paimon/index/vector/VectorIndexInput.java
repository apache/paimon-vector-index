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

public interface VectorIndexInput {

    /**
     * Reads all requested ranges. DiskANN batch search may call this method concurrently from
     * multiple native worker threads, so implementations must be thread-safe.
     */
    void pread(long[] positions, byte[][] buffers);

    /**
     * Estimated end-to-end latency of one representative random read in nanoseconds, or zero to
     * let DiskANN use the mandatory header read as its measurement.
     */
    default long estimatedRandomReadLatencyNanos() {
        return 0L;
    }

    /** Efficient range-read alignment, or zero when unspecified. */
    default long preferredReadAlignmentBytes() {
        return 0L;
    }

    /** Efficient coalesced random-read window, or zero when unspecified. */
    default long preferredReadWindowBytes() {
        return 0L;
    }

    /** Maximum ranges accepted by one pread call, or zero when unlimited/unspecified. */
    default long maxRangesPerRead() {
        return 0L;
    }
}
