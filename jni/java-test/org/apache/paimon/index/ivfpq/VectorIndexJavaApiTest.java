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

package org.apache.paimon.index.ivfpq;

import java.util.Arrays;

public class VectorIndexJavaApiTest {

    public static void main(String[] args) {
        testMetricCodes();
        testIndexTypeCodes();
        testSingleResultCopiesArrays();
        testBatchResultCopiesArraysAndSlicesRows();
        testConfigValidation();
        testMetadata();
        testReaderAndWriterApiCompile();
    }

    private static void testMetricCodes() {
        assertEquals(0, Metric.L2.code());
        assertEquals(1, Metric.INNER_PRODUCT.code());
        assertEquals(2, Metric.COSINE.code());
    }

    private static void testIndexTypeCodes() {
        assertEquals(0, IndexType.IVF_FLAT.code());
        assertEquals(1, IndexType.IVF_PQ.code());
        assertEquals(2, IndexType.IVF_HNSW_FLAT.code());
        assertEquals(3, IndexType.IVF_HNSW_SQ.code());
    }

    private static void testSingleResultCopiesArrays() {
        long[] ids = new long[] {11L, 7L};
        float[] distances = new float[] {0.1f, 0.3f};

        VectorSearchResult result = new VectorSearchResult(ids, distances);
        ids[0] = 99L;
        distances[0] = 9.0f;

        assertArrayEquals(new long[] {11L, 7L}, result.ids());
        assertArrayEquals(new float[] {0.1f, 0.3f}, result.distances());

        long[] resultIds = result.ids();
        resultIds[0] = 99L;
        assertArrayEquals(new long[] {11L, 7L}, result.ids());
    }

    private static void testBatchResultCopiesArraysAndSlicesRows() {
        long[] ids = new long[] {1L, 2L, 3L, 4L, 5L, 6L};
        float[] distances = new float[] {0.1f, 0.2f, 0.3f, 1.1f, 1.2f, 1.3f};

        VectorSearchBatchResult result = new VectorSearchBatchResult(ids, distances, 2, 3);
        ids[0] = 99L;
        distances[0] = 9.0f;

        assertEquals(2, result.queryCount());
        assertEquals(3, result.topK());
        assertArrayEquals(new long[] {1L, 2L, 3L, 4L, 5L, 6L}, result.ids());
        assertArrayEquals(new float[] {0.1f, 0.2f, 0.3f, 1.1f, 1.2f, 1.3f}, result.distances());
        assertArrayEquals(new long[] {4L, 5L, 6L}, result.idsForQuery(1));
        assertArrayEquals(new float[] {1.1f, 1.2f, 1.3f}, result.distancesForQuery(1));

        assertThrows(IllegalArgumentException.class, new ThrowingRunnable() {
            @Override
            public void run() {
                new VectorSearchBatchResult(new long[] {1L}, new float[] {1.0f}, 2, 3);
            }
        });
        assertThrows(IndexOutOfBoundsException.class, new ThrowingRunnable() {
            @Override
            public void run() {
                result.idsForQuery(2);
            }
        });
    }

    private static void testConfigValidation() {
        VectorIndexConfig flat = VectorIndexConfig.ivfFlat(16, 4, Metric.L2);
        assertEquals(IndexType.IVF_FLAT, flat.indexType());
        assertEquals(16, flat.dimension());
        assertEquals(4, flat.nlist());

        IvfPqConfig pq = new IvfPqConfig(16, 4, 4, Metric.L2, true);
        assertEquals(4, pq.m());
        assertTrue(pq.useOpq());

        HnswConfig hnsw = new HnswConfig(12, 64, 5);
        assertEquals(12, hnsw.m());
        assertEquals(64, hnsw.efConstruction());
        assertEquals(5, hnsw.maxLevel());

        VectorIndexConfig hnswFlat = VectorIndexConfig.ivfHnswFlat(16, 4, Metric.L2, hnsw);
        assertEquals(IndexType.IVF_HNSW_FLAT, hnswFlat.indexType());
        assertEquals(12, hnswFlat.hnsw().m());

        VectorIndexConfig hnswSq = VectorIndexConfig.ivfHnswSq(16, 4, Metric.L2, hnsw);
        assertEquals(IndexType.IVF_HNSW_SQ, hnswSq.indexType());

        assertThrows(IllegalArgumentException.class, new ThrowingRunnable() {
            @Override
            public void run() {
                new IvfPqConfig(10, 4, 3, Metric.L2, false);
            }
        });
    }

    private static void testMetadata() {
        VectorIndexMetadata metadata =
                new VectorIndexMetadata(
                        IndexType.IVF_HNSW_FLAT.code(),
                        16,
                        4,
                        Metric.COSINE.code(),
                        123L,
                        0,
                        20,
                        150,
                        7);
        assertEquals(IndexType.IVF_HNSW_FLAT, metadata.indexType());
        assertEquals(16, metadata.dimension());
        assertEquals(4, metadata.nlist());
        assertEquals(Metric.COSINE, metadata.metric());
        assertEquals(123L, metadata.totalVectors());
        assertEquals(20, metadata.hnsw().m());
    }

    private static void testReaderAndWriterApiCompile() {
        VectorIndexConfig config = VectorIndexConfig.ivfPq(2, 4, 1, Metric.L2, false);
        VectorIndexReader closedReader = VectorIndexReader.fromNativePointerForTesting(0L);
        closedReader.close();
        closedReader.close();

        VectorIndexWriter closedWriter = VectorIndexWriter.fromNativePointerForTesting(0L, config);
        closedWriter.close();
        closedWriter.close();

        if (System.currentTimeMillis() < 0) {
            VectorIndexReader reader = new VectorIndexReader(new EmptyVectorIndexInput());
            reader.metadata();
            reader.indexType();
            reader.dimension();
            reader.totalVectors();
            reader.search(new float[] {0.0f, 1.0f}, 10, 4);
            reader.search(new float[] {0.0f, 1.0f}, 10, 4, 32);
            reader.search(new float[] {0.0f, 1.0f}, 10, 4, new byte[] {1, 2});
            reader.search(new float[] {0.0f, 1.0f}, 10, 4, 32, new byte[] {1, 2});
            reader.searchBatch(new float[] {0.0f, 1.0f, 2.0f, 3.0f}, 2, 10, 4);
            reader.searchBatch(new float[] {0.0f, 1.0f, 2.0f, 3.0f}, 2, 10, 4, 32);
            reader.searchBatch(new float[] {0.0f, 1.0f, 2.0f, 3.0f}, 2, 10, 4, new byte[] {1, 2});
            reader.searchBatch(
                    new float[] {0.0f, 1.0f, 2.0f, 3.0f}, 2, 10, 4, 32, new byte[] {1, 2});

            VectorIndexWriter writer = new VectorIndexWriter(config);
            writer.train(new float[] {0.0f, 1.0f, 2.0f, 3.0f}, 2);
            writer.addVectors(new long[] {1L, 2L}, new float[] {0.0f, 1.0f, 2.0f, 3.0f}, 2);
            writer.writeIndex(new Object());
        }
    }

    private static void assertEquals(int expected, int actual) {
        if (expected != actual) {
            throw new AssertionError("expected " + expected + " but got " + actual);
        }
    }

    private static void assertEquals(long expected, long actual) {
        if (expected != actual) {
            throw new AssertionError("expected " + expected + " but got " + actual);
        }
    }

    private static void assertEquals(Object expected, Object actual) {
        if (!expected.equals(actual)) {
            throw new AssertionError("expected " + expected + " but got " + actual);
        }
    }

    private static void assertTrue(boolean value) {
        if (!value) {
            throw new AssertionError("expected true");
        }
    }

    private static void assertArrayEquals(long[] expected, long[] actual) {
        if (!Arrays.equals(expected, actual)) {
            throw new AssertionError(
                    "expected " + Arrays.toString(expected) + " but got " + Arrays.toString(actual));
        }
    }

    private static void assertArrayEquals(float[] expected, float[] actual) {
        if (!Arrays.equals(expected, actual)) {
            throw new AssertionError(
                    "expected " + Arrays.toString(expected) + " but got " + Arrays.toString(actual));
        }
    }

    private static void assertThrows(Class<? extends Throwable> expected, ThrowingRunnable runnable) {
        try {
            runnable.run();
        } catch (Throwable t) {
            if (expected.isInstance(t)) {
                return;
            }
            throw new AssertionError(
                    "expected " + expected.getName() + " but got " + t.getClass().getName(), t);
        }
        throw new AssertionError("expected " + expected.getName());
    }

    private interface ThrowingRunnable {
        void run() throws Throwable;
    }

    private static final class EmptyVectorIndexInput implements VectorIndexInput {
        @Override
        public void pread(long[] positions, byte[][] buffers) {}
    }
}
