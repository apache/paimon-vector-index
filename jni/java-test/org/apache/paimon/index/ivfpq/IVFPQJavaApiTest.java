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

public class IVFPQJavaApiTest {

    public static void main(String[] args) {
        testMetricCodes();
        testSingleResultCopiesArrays();
        testBatchResultCopiesArraysAndSlicesRows();
        testReaderAndWriterApiCompile();
        testFlatReaderAndWriterApiCompile();
    }

    private static void testMetricCodes() {
        assertEquals(0, Metric.L2.code());
        assertEquals(1, Metric.INNER_PRODUCT.code());
        assertEquals(2, Metric.COSINE.code());
    }

    private static void testSingleResultCopiesArrays() {
        long[] ids = new long[] {11L, 7L};
        float[] distances = new float[] {0.1f, 0.3f};

        IVFPQResult result = new IVFPQResult(ids, distances);
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

        IVFPQBatchResult result = new IVFPQBatchResult(ids, distances, 2, 3);
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
                new IVFPQBatchResult(new long[] {1L}, new float[] {1.0f}, 2, 3);
            }
        });
        assertThrows(IndexOutOfBoundsException.class, new ThrowingRunnable() {
            @Override
            public void run() {
                result.idsForQuery(2);
            }
        });
    }

    private static void testReaderAndWriterApiCompile() {
        IVFPQReader closedReader = IVFPQReader.fromNativePointerForTesting(0L);
        closedReader.close();
        closedReader.close();

        IVFPQWriter closedWriter = IVFPQWriter.fromNativePointerForTesting(0L, 2);
        closedWriter.close();
        closedWriter.close();

        if (System.currentTimeMillis() < 0) {
            IVFPQReader reader = new IVFPQReader(new Object());
            reader.dimension();
            reader.totalVectors();
            reader.search(new float[] {0.0f, 1.0f}, 10, 4);
            reader.search(new float[] {0.0f, 1.0f}, 10, 4, new byte[] {1, 2});
            reader.searchBatch(new float[] {0.0f, 1.0f, 2.0f, 3.0f}, 2, 10, 4);
            reader.searchBatch(new float[] {0.0f, 1.0f, 2.0f, 3.0f}, 2, 10, 4, new byte[] {1, 2});

            IVFPQWriter writer = new IVFPQWriter(2, 4, 1, Metric.L2, false);
            writer.train(new float[] {0.0f, 1.0f, 2.0f, 3.0f}, 2);
            writer.addVectors(new long[] {1L, 2L}, new float[] {0.0f, 1.0f, 2.0f, 3.0f}, 2);
            writer.writeIndex(new Object());
        }
    }

    private static void testFlatReaderAndWriterApiCompile() {
        IVFFlatReader closedReader = IVFFlatReader.fromNativePointerForTesting(0L);
        closedReader.close();
        closedReader.close();

        IVFFlatWriter closedWriter = IVFFlatWriter.fromNativePointerForTesting(0L, 2);
        closedWriter.close();
        closedWriter.close();

        if (System.currentTimeMillis() < 0) {
            IVFFlatReader reader = new IVFFlatReader(new Object());
            reader.dimension();
            reader.totalVectors();
            reader.search(new float[] {0.0f, 1.0f}, 10, 4);
            reader.search(new float[] {0.0f, 1.0f}, 10, 4, new byte[] {1, 2});
            reader.searchBatch(new float[] {0.0f, 1.0f, 2.0f, 3.0f}, 2, 10, 4);
            reader.searchBatch(new float[] {0.0f, 1.0f, 2.0f, 3.0f}, 2, 10, 4, new byte[] {1, 2});

            IVFFlatWriter writer = new IVFFlatWriter(2, 4, Metric.L2);
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

    private static void assertArrayEquals(long[] expected, long[] actual) {
        if (!Arrays.equals(expected, actual)) {
            throw new AssertionError("expected " + Arrays.toString(expected) + " but got " + Arrays.toString(actual));
        }
    }

    private static void assertArrayEquals(float[] expected, float[] actual) {
        if (!Arrays.equals(expected, actual)) {
            throw new AssertionError("expected " + Arrays.toString(expected) + " but got " + Arrays.toString(actual));
        }
    }

    private static void assertThrows(Class<? extends Throwable> expected, ThrowingRunnable runnable) {
        try {
            runnable.run();
        } catch (Throwable t) {
            if (expected.isInstance(t)) {
                return;
            }
            throw new AssertionError("expected " + expected.getName() + " but got " + t.getClass().getName(), t);
        }
        throw new AssertionError("expected " + expected.getName());
    }

    private interface ThrowingRunnable {
        void run() throws Throwable;
    }
}
