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

/** Unified distance-band options, constructed with {@link #ivf} or {@link #diskAnn}. */
public class VectorRangeSearchParams {
    private final VectorDistanceBand band;
    private final int searchWidth;
    private final int width;
    private final float initialSlack;
    private final float rangeSlack;
    private final long maxWorkingBytes;
    private final long maxResultBytes;
    private final boolean exhaustive;

    /** Compatibility constructor for IVF options. */
    public VectorRangeSearchParams(VectorDistanceBand band, int nprobe) {
        this(band, VectorSearchParams.SEARCH_WIDTH_IVF_NPROBE, nprobe,
                1.0f, 1.0f, 64L * 1024 * 1024, false);
    }

    protected VectorRangeSearchParams(VectorDistanceBand band, int searchWidth, int width,
            float initialSlack, float rangeSlack, long maxWorkingBytes, boolean exhaustive) {
        this(band, searchWidth, width, initialSlack, rangeSlack, maxWorkingBytes, exhaustive, 64L * 1024 * 1024);
    }

    private VectorRangeSearchParams(VectorDistanceBand band, int searchWidth, int width,
            float initialSlack, float rangeSlack, long maxWorkingBytes, boolean exhaustive, long maxResultBytes) {
        if (maxResultBytes <= 0) { throw new IllegalArgumentException("maxResultBytes must be positive"); }
        this.maxResultBytes = maxResultBytes;
        this.band = Objects.requireNonNull(band, "band");
        if (searchWidth != VectorSearchParams.SEARCH_WIDTH_IVF_NPROBE
                && searchWidth != VectorSearchParams.SEARCH_WIDTH_DISKANN_L_SEARCH) {
            throw new IllegalArgumentException("Range search requires explicit IVF or DiskANN width");
        }
        if (width <= 0 || maxWorkingBytes <= 0) {
            throw new IllegalArgumentException("nprobe/startingL and maxWorkingBytes must be positive");
        }
        if (!Float.isFinite(initialSlack) || initialSlack < 0 || initialSlack > 1) {
            throw new IllegalArgumentException("initialSlack must be finite and in [0, 1]");
        }
        if (!Float.isFinite(rangeSlack) || rangeSlack < 1) {
            throw new IllegalArgumentException("rangeSlack must be finite and at least 1");
        }
        if (searchWidth == VectorSearchParams.SEARCH_WIDTH_IVF_NPROBE
                && (initialSlack != 1 || rangeSlack != 1
                    || maxWorkingBytes != 64L * 1024 * 1024 || exhaustive)) {
            throw new IllegalArgumentException("DiskANN range tuning cannot be used with IVF options");
        }
        this.searchWidth = searchWidth;
        this.width = width;
        this.initialSlack = initialSlack;
        this.rangeSlack = rangeSlack;
        this.maxWorkingBytes = maxWorkingBytes;
        this.exhaustive = exhaustive;
    }

    public static VectorRangeSearchParams ivf(VectorDistanceBand band, int nprobe) {
        return new VectorRangeSearchParams(band, nprobe);
    }

    public static VectorRangeSearchParams diskAnn(VectorDistanceBand band, int startingL) {
        return diskAnn(band, startingL, 1.0f, 1.0f, 64L * 1024 * 1024, false);
    }

    public static VectorRangeSearchParams diskAnn(VectorDistanceBand band, int startingL,
            float initialSlack, float rangeSlack, long maxWorkingBytes, boolean exhaustive) {
        return new VectorRangeSearchParams(band, VectorSearchParams.SEARCH_WIDTH_DISKANN_L_SEARCH,
                startingL, initialSlack, rangeSlack, maxWorkingBytes, exhaustive);
    }

    public final VectorRangeSearchParams withInitialSlack(float value) {
        return new VectorRangeSearchParams(band, searchWidth, width, value, rangeSlack, maxWorkingBytes, exhaustive, maxResultBytes);
    }
    public final VectorRangeSearchParams withRangeSlack(float value) {
        return new VectorRangeSearchParams(band, searchWidth, width, initialSlack, value, maxWorkingBytes, exhaustive, maxResultBytes);
    }
    public final VectorRangeSearchParams withMaxWorkingBytes(long value) {
        return new VectorRangeSearchParams(band, searchWidth, width, initialSlack, rangeSlack, value, exhaustive, maxResultBytes);
    }
    public final VectorRangeSearchParams withExhaustive(boolean value) {
        return new VectorRangeSearchParams(band, searchWidth, width, initialSlack, rangeSlack, maxWorkingBytes, value, maxResultBytes);
    }

    /** Bounds live native result buffers per call, including CSR conversion. */
    public final VectorRangeSearchParams withMaxResultBytes(long value) {
        return new VectorRangeSearchParams(band, searchWidth, width, initialSlack, rangeSlack, maxWorkingBytes, exhaustive, value);
    }
    public final long maxResultBytes() { return maxResultBytes; }

    public final VectorDistanceBand band() { return band; }
    public final int searchWidth() { return searchWidth; }
    public final int width() { return width; }
    public final int nprobe() {
        if (searchWidth != VectorSearchParams.SEARCH_WIDTH_IVF_NPROBE) {
            throw new IllegalStateException("nprobe is only available for IVF options");
        }
        return width;
    }
    public final int startingL() {
        if (searchWidth != VectorSearchParams.SEARCH_WIDTH_DISKANN_L_SEARCH) {
            throw new IllegalStateException("startingL is only available for DiskANN options");
        }
        return width;
    }
    public final float initialSlack() { return initialSlack; }
    public final float rangeSlack() { return rangeSlack; }
    public final long maxWorkingBytes() { return maxWorkingBytes; }
    public final boolean exhaustive() { return exhaustive; }
}
