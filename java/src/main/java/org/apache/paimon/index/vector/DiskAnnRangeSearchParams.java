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

import java.util.Objects;

/** DiskANN graph range options. Unbounded upper cuts use exhaustive stored-vector scanning.
 * Graph recall is approximate; membership uses persisted F32/F16 vectors. The working
 * budget excludes resident data and uncapped results. */
public final class DiskAnnRangeSearchParams implements VectorRangeSearchOptions {
    private final VectorDistanceBand band;
    private final int startingL;
    private final float initialSlack;
    private final float rangeSlack;
    private final long maxWorkingBytes;
    private final boolean exhaustive;

    public DiskAnnRangeSearchParams(VectorDistanceBand band, int startingL) {
        this(band, startingL, 1.0f, 1.0f, 64L * 1024 * 1024, false);
    }

    public DiskAnnRangeSearchParams(VectorDistanceBand band, int startingL,
            float initialSlack, float rangeSlack, long maxWorkingBytes, boolean exhaustive) {
        this.band = Objects.requireNonNull(band, "band");
        if (startingL <= 0 || maxWorkingBytes <= 0) {
            throw new IllegalArgumentException("startingL and maxWorkingBytes must be positive");
        }
        if (!Float.isFinite(initialSlack) || initialSlack < 0 || initialSlack > 1) {
            throw new IllegalArgumentException("initialSlack must be finite and in [0, 1]");
        }
        if (!Float.isFinite(rangeSlack) || rangeSlack < 1) {
            throw new IllegalArgumentException("rangeSlack must be finite and at least 1");
        }
        this.startingL = startingL;
        this.initialSlack = initialSlack;
        this.rangeSlack = rangeSlack;
        this.maxWorkingBytes = maxWorkingBytes;
        this.exhaustive = exhaustive;
    }

    public VectorDistanceBand band() { return band; }
    public int startingL() { return startingL; }
    public float initialSlack() { return initialSlack; }
    public float rangeSlack() { return rangeSlack; }
    public long maxWorkingBytes() { return maxWorkingBytes; }
    public boolean exhaustive() { return exhaustive; }
}
