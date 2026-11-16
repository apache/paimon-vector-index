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

/** Runs in a fresh JVM with the standalone legacy/unsupported JNI fixture. */
public class VectorIndexRangeNativeCompatibilityTest {
    private static native int legacyRangeCalls();

    public static void main(String[] args) {
        if (args.length != 1) {
            throw new IllegalArgumentException("expected the legacy JNI fixture path");
        }
        VectorIndexNativeLoaderSmokeTest.configureExternalLibrary(args);
        VectorDistanceBand band = VectorDistanceBand.fromRaw("l2", null, null);
        float[] query = new float[4];
        byte[] filter = new byte[8];
        // The fixture owns this sentinel handle; its range methods would accept
        // calls without reading the new budget, just like the old JNI parser.
        try (VectorIndexReader reader = VectorIndexReader.fromNativePointerForTesting(1)) {
            reject(reader::supportsRangeSearch);
            reject(reader::supportsDiskAnnRangeSearch);
            for (long budget : new long[] {64L * 1024 * 1024, 1}) {
                for (VectorRangeSearchParams params : new VectorRangeSearchParams[] {
                        VectorRangeSearchParams.ivf(band, 1).withMaxResultBytes(budget),
                        VectorRangeSearchParams.diskAnn(band, 1).withMaxResultBytes(budget)}) {
                    reject(() -> reader.rangeSearch(query, params));
                    reject(() -> reader.rangeSearch(query, params, filter));
                    reject(() -> reader.rangeSearchBatch(query, 1, params));
                    reject(() -> reader.rangeSearchBatch(query, 1, params, filter));
                    reject(() -> reader.rangeSearchBatchConsume(query, 1, params, 1, null,
                            (q, labels, distances) -> { throw new AssertionError("consumer called"); }));
                }
            }
            DiskAnnRangeSearchParams params = new DiskAnnRangeSearchParams(band, 1);
            reject(() -> reader.diskAnnRangeSearch(query, params));
            reject(() -> reader.diskAnnRangeSearch(query, params, filter));
            reject(() -> reader.diskAnnRangeSearchBatch(query, 1, params));
            reject(() -> reader.diskAnnRangeSearchBatch(query, 1, params, filter));
            if (legacyRangeCalls() != 0) {
                throw new AssertionError("incompatible native range implementation was invoked");
            }
        }
        System.out.println("Incompatible JNI range libraries rejected before native dispatch");
    }

    private static void reject(Runnable query) {
        try {
            query.run();
        } catch (UnsupportedOperationException error) {
            if (error.getMessage().contains("result API 1")
                    && error.getMessage().contains("upgrade")) {
                return;
            }
            throw new AssertionError("missing native-upgrade diagnostic", error);
        }
        throw new AssertionError("incompatible JNI library accepted a range call");
    }
}
