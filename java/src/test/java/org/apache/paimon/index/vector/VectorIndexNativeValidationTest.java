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

import java.io.ByteArrayOutputStream;
import java.util.HashMap;
import java.util.Map;

public class VectorIndexNativeValidationTest {

    private static final int ROUNDTRIP_DIMENSION = 8;
    private static final int ROUNDTRIP_NLIST = 4;
    private static final int ROUNDTRIP_PER_LIST = 128;
    private static final int ROUNDTRIP_VECTOR_COUNT = ROUNDTRIP_NLIST * ROUNDTRIP_PER_LIST;

    public static void main(String[] args) {
        if (args.length != 1) {
            throw new IllegalArgumentException("native library path is required");
        }

        System.load(args[0]);

        testWriterValidationComesFromCore();
        testWriterRejectsNonFiniteValues();
        testStagedTrainingRoundtrip();
        testStagedTrainingStateValidation();
        testReaderValidationComesFromCore();
        testReaderRejectsNonFiniteQueries();
        testSupportedIndexRoundtrips();
    }

    private static void testWriterValidationComesFromCore() {
        final VectorIndexTrainer trainingDataTrainer = VectorIndexTrainer.create(ivfFlatOptions());
        try {
            assertThrowsMessage(
                    RuntimeException.class,
                    "training data length 2 does not match vector count * dimension 1",
                    new ThrowingRunnable() {
                        @Override
                        public void run() {
                            trainingDataTrainer.addTrainingVectors(new float[] {0.0f, 1.0f}, 1);
                        }
                    });
        } finally {
            trainingDataTrainer.close();
        }

        final VectorIndexWriter addVectorWriter =
                newWriter(ivfFlatOptions(), new float[] {0.0f, 1.0f}, 2);
        try {
            assertThrowsMessage(
                    RuntimeException.class,
                    "ids length 2 does not match vector count 1",
                    new ThrowingRunnable() {
                        @Override
                        public void run() {
                            addVectorWriter.addVectors(new long[] {1L, 2L}, new float[] {0.0f}, 1);
                        }
                    });
        } finally {
            addVectorWriter.close();
        }

        final VectorIndexTrainer vectorCountTrainer = VectorIndexTrainer.create(ivfFlatOptions());
        try {
            assertThrowsMessage(
                    RuntimeException.class,
                    "vector count must be greater than 0",
                    new ThrowingRunnable() {
                        @Override
                        public void run() {
                            vectorCountTrainer.addTrainingVectors(new float[0], 0);
                        }
                    });
        } finally {
            vectorCountTrainer.close();
        }
    }

    private static void testReaderValidationComesFromCore() {
        VectorIndexReader reader =
                new VectorIndexReader(new ByteArraySeekableInputStream(buildIndexBytes()));
        try {
            assertThrowsMessage(
                    RuntimeException.class,
                    "query length 2 does not match index dimension 1",
                    new ThrowingRunnable() {
                        @Override
                        public void run() {
                            reader.search(new float[] {0.0f, 1.0f}, 1, 1);
                        }
                    });
            assertThrowsMessage(
                    RuntimeException.class,
                    "k must be greater than 0",
                    new ThrowingRunnable() {
                        @Override
                        public void run() {
                            reader.search(new float[] {0.0f}, 0, 1);
                        }
                    });
            assertThrowsMessage(
                    RuntimeException.class,
                    "queries length 2 does not match nq * dimension 1",
                    new ThrowingRunnable() {
                        @Override
                        public void run() {
                            reader.searchBatch(new float[] {0.0f, 1.0f}, 1, 1, 1);
                        }
                    });
        } finally {
            reader.close();
        }
    }

    private static void testWriterRejectsNonFiniteValues() {
        final VectorIndexTrainer trainingTrainer = VectorIndexTrainer.create(ivfFlatOptions());
        try {
            assertThrowsMessage(
                    RuntimeException.class,
                    "training data contains non-finite value at offset 0: NaN",
                    new ThrowingRunnable() {
                        @Override
                        public void run() {
                            trainingTrainer.addTrainingVectors(new float[] {Float.NaN, 1.0f}, 2);
                        }
                    });
        } finally {
            trainingTrainer.close();
        }

        final VectorIndexWriter vectorWriter =
                newWriter(ivfFlatOptions(), new float[] {0.0f, 1.0f}, 2);
        try {
            assertThrowsMessage(
                    RuntimeException.class,
                    "vector data contains non-finite value at offset 0: inf",
                    new ThrowingRunnable() {
                        @Override
                        public void run() {
                            vectorWriter.addVectors(
                                    new long[] {1L, 2L},
                                    new float[] {Float.POSITIVE_INFINITY, 1.0f},
                                    2);
                        }
                    });
        } finally {
            vectorWriter.close();
        }
    }

    private static void testStagedTrainingRoundtrip() {
        runStagedTrainingRoundtrip(
                "ivf_flat", ivfFlatOptions(ROUNDTRIP_DIMENSION, ROUNDTRIP_NLIST), 0, 0);
        runStagedTrainingRoundtrip(
                "ivf_pq", ivfPqOptions(ROUNDTRIP_DIMENSION, ROUNDTRIP_NLIST, 4), 4, 0);
        runStagedTrainingRoundtrip(
                "ivf_rq", ivfRqOptions(ROUNDTRIP_DIMENSION, ROUNDTRIP_NLIST), 0, 0);
        runStagedTrainingRoundtrip(
                "ivf_hnsw_flat",
                ivfHnswOptions("ivf_hnsw_flat", ROUNDTRIP_DIMENSION, ROUNDTRIP_NLIST),
                0,
                4);
        runStagedTrainingRoundtrip(
                "ivf_hnsw_sq",
                ivfHnswOptions("ivf_hnsw_sq", ROUNDTRIP_DIMENSION, ROUNDTRIP_NLIST),
                0,
                4);
    }

    private static void testStagedTrainingStateValidation() {
        final VectorIndexTrainer emptyTrainer = VectorIndexTrainer.create(ivfFlatOptions());
        try {
            assertThrowsMessage(
                    RuntimeException.class,
                    "no training vectors added",
                    new ThrowingRunnable() {
                        @Override
                        public void run() {
                            emptyTrainer.finishTraining();
                        }
                    });
            assertThrowsMessage(
                    IllegalStateException.class,
                    "VectorIndexTrainer is closed",
                    new ThrowingRunnable() {
                        @Override
                        public void run() {
                            emptyTrainer.addTrainingVectors(new float[] {0.0f}, 1);
                        }
                    });
        } finally {
            emptyTrainer.close();
        }

        final VectorIndexTrainer stagedTrainer = VectorIndexTrainer.create(ivfFlatOptions());
        final VectorIndexTraining stagedTraining;
        try {
            stagedTraining =
                    stagedTrainer.addTrainingVectors(new float[] {0.0f}, 1)
                            .addTrainingVectors(new float[] {1.0f}, 1)
                            .finishTraining();
            assertThrowsMessage(
                    IllegalStateException.class,
                    "VectorIndexTrainer is closed",
                    new ThrowingRunnable() {
                        @Override
                        public void run() {
                            stagedTrainer.addTrainingVectors(new float[] {2.0f}, 1);
                        }
                    });
        } finally {
            stagedTrainer.close();
        }

        final VectorIndexWriter writer = new VectorIndexWriter(stagedTraining);
        try {
            assertThrowsMessage(
                    IllegalStateException.class,
                    "VectorIndexTraining is closed",
                    new ThrowingRunnable() {
                        @Override
                        public void run() {
                            new VectorIndexWriter(stagedTraining);
                        }
                    });
        } finally {
            writer.close();
        }

        final VectorIndexTrainer invalidBatchTrainer = VectorIndexTrainer.create(ivfFlatOptions());
        try {
            assertThrowsMessage(
                    RuntimeException.class,
                    "training data length 2 does not match vector count * dimension 1",
                    new ThrowingRunnable() {
                        @Override
                        public void run() {
                            invalidBatchTrainer.addTrainingVectors(new float[] {0.0f, 1.0f}, 1);
                        }
                    });
            assertThrowsMessage(
                    RuntimeException.class,
                    "vector count must be greater than 0",
                    new ThrowingRunnable() {
                        @Override
                        public void run() {
                            invalidBatchTrainer.addTrainingVectors(new float[0], 0);
                        }
                    });
        } finally {
            invalidBatchTrainer.close();
        }
    }

    private static void testReaderRejectsNonFiniteQueries() {
        VectorIndexReader reader =
                new VectorIndexReader(new ByteArraySeekableInputStream(buildIndexBytes()));
        try {
            assertInvalidInput(
                    new ThrowingRunnable() {
                        @Override
                        public void run() {
                            reader.search(new float[] {Float.NaN}, 1, 1);
                        }
                    },
                    "query contains non-finite value at offset 0: NaN");
            assertInvalidInput(
                    new ThrowingRunnable() {
                        @Override
                        public void run() {
                            reader.searchBatch(new float[] {Float.NEGATIVE_INFINITY}, 1, 1, 1);
                        }
                    },
                    "queries contains non-finite value at offset 0: -inf");
            assertInvalidInput(
                    new ThrowingRunnable() {
                        @Override
                        public void run() {
                            reader.search(new float[] {Float.NaN}, 1, 1, new byte[] {(byte) 0xFF});
                        }
                    },
                    "query contains non-finite value at offset 0: NaN");
        } finally {
            reader.close();
        }
    }

    private static void testSupportedIndexRoundtrips() {
        runRoundtrip("ivf_flat", ivfFlatOptions(ROUNDTRIP_DIMENSION, ROUNDTRIP_NLIST), 0, 0);
        runRoundtrip("ivf_pq", ivfPqOptions(ROUNDTRIP_DIMENSION, ROUNDTRIP_NLIST, 4), 4, 0);
        runRoundtrip("ivf_rq", ivfRqOptions(ROUNDTRIP_DIMENSION, ROUNDTRIP_NLIST), 0, 0);
        runRoundtrip(
                "ivf_hnsw_flat",
                ivfHnswOptions("ivf_hnsw_flat", ROUNDTRIP_DIMENSION, ROUNDTRIP_NLIST),
                0,
                4);
        runRoundtrip(
                "ivf_hnsw_sq",
                ivfHnswOptions("ivf_hnsw_sq", ROUNDTRIP_DIMENSION, ROUNDTRIP_NLIST),
                0,
                4);
    }

    private static void runRoundtrip(
            String indexType, Map<String, String> options, int expectedPqM, int expectedHnswM) {
        byte[] indexBytes =
                buildIndexBytes(
                        options, roundtripData(), roundtripIds(), ROUNDTRIP_VECTOR_COUNT);
        assertRoundtrip(indexType, indexBytes, expectedPqM, expectedHnswM);
    }

    private static void runStagedTrainingRoundtrip(
            String indexType, Map<String, String> options, int expectedPqM, int expectedHnswM) {
        byte[] indexBytes =
                buildStagedIndexBytes(
                        options, roundtripData(), roundtripIds(), ROUNDTRIP_VECTOR_COUNT);
        assertRoundtrip(indexType, indexBytes, expectedPqM, expectedHnswM);
    }

    private static void assertRoundtrip(
            String indexType, byte[] indexBytes, int expectedPqM, int expectedHnswM) {
        VectorIndexReader reader =
                new VectorIndexReader(new ByteArraySeekableInputStream(indexBytes));
        try {
            VectorIndexMetadata metadata = reader.metadata();
            assertEquals(indexType, metadata.indexType());
            assertEquals(ROUNDTRIP_DIMENSION, metadata.dimension());
            assertEquals(ROUNDTRIP_NLIST, metadata.nlist());
            assertEquals("l2", metadata.metric());
            assertEquals((long) ROUNDTRIP_VECTOR_COUNT, metadata.totalVectors());
            assertEquals(expectedPqM, metadata.pqM());
            assertEquals(expectedHnswM, metadata.hnswM());

            reader.optimizeForSearch();

            VectorSearchResult single = reader.search(queryForCenter(0.0f), 2, 4, 16);
            assertIdInCluster(single.ids()[0], 0);
            assertFinite(single.distances()[0], indexType + " single distance");

            VectorSearchBatchResult batch =
                    reader.searchBatch(batchQueries(), 2, 1, 4, 16);
            assertIdInCluster(batch.ids()[0], 0);
            assertIdInCluster(batch.ids()[1], 1);
            assertFinite(batch.distances()[0], indexType + " batch distance 0");
            assertFinite(batch.distances()[1], indexType + " batch distance 1");

            if ("ivf_rq".equals(indexType)) {
                VectorSearchResult queryBitsSingle =
                        reader.search(queryForCenter(0.0f), 2, 4, 16, 4);
                assertIdInCluster(queryBitsSingle.ids()[0], 0);
                assertFinite(queryBitsSingle.distances()[0], "ivf_rq queryBits single distance");

                VectorSearchBatchResult queryBitsBatch =
                        reader.searchBatch(batchQueries(), 2, 1, 4, 16, 8);
                assertIdInCluster(queryBitsBatch.ids()[0], 0);
                assertIdInCluster(queryBitsBatch.ids()[1], 1);

                assertThrowsMessage(
                        RuntimeException.class,
                        "query_bits",
                        new ThrowingRunnable() {
                            @Override
                            public void run() {
                                reader.search(queryForCenter(0.0f), 2, 4, 16, 7);
                            }
                        });
            }
        } finally {
            reader.close();
        }
    }

    private static byte[] buildIndexBytes() {
        return buildIndexBytes(ivfFlatOptions(), new float[] {0.0f, 1.0f}, new long[] {1L, 2L}, 2);
    }

    private static byte[] buildIndexBytes(
            Map<String, String> options, float[] data, long[] ids, int vectorCount) {
        VectorIndexWriter writer = newWriter(options, data, vectorCount);
        ByteArrayPositionOutputStream output = new ByteArrayPositionOutputStream();
        try {
            writer.addVectors(ids, data, vectorCount);
            writer.writeIndex(output);
            return output.toByteArray();
        } finally {
            writer.close();
        }
    }

    private static byte[] buildStagedIndexBytes(
            Map<String, String> options, float[] data, long[] ids, int vectorCount) {
        VectorIndexTrainer trainer = VectorIndexTrainer.create(options);
        VectorIndexWriter writer = null;
        ByteArrayPositionOutputStream output = new ByteArrayPositionOutputStream();
        int dimension = data.length / vectorCount;
        try {
            int offset = 0;
            while (offset < vectorCount) {
                int batchCount = Math.min(ROUNDTRIP_PER_LIST / 2, vectorCount - offset);
                trainer.addTrainingVectors(
                        copyVectors(data, dimension, offset, batchCount), batchCount);
                offset += batchCount;
            }
            writer = new VectorIndexWriter(trainer.finishTraining());
            writer.addVectors(ids, data, vectorCount);
            writer.writeIndex(output);
            return output.toByteArray();
        } finally {
            trainer.close();
            if (writer != null) {
                writer.close();
            }
        }
    }

    private static VectorIndexWriter newWriter(
            Map<String, String> options, float[] data, int vectorCount) {
        return new VectorIndexWriter(VectorIndexTrainer.train(options, data, vectorCount));
    }

    private static float[] copyVectors(float[] data, int dimension, int offset, int count) {
        float[] copy = new float[count * dimension];
        System.arraycopy(data, offset * dimension, copy, 0, copy.length);
        return copy;
    }

    private static Map<String, String> ivfFlatOptions() {
        return ivfFlatOptions(1, 1);
    }

    private static Map<String, String> ivfFlatOptions(int dimension, int nlist) {
        Map<String, String> options = new HashMap<String, String>();
        options.put("index.type", "ivf_flat");
        options.put("dimension", Integer.toString(dimension));
        options.put("nlist", Integer.toString(nlist));
        options.put("metric", "l2");
        return options;
    }

    private static Map<String, String> ivfPqOptions(int dimension, int nlist, int m) {
        Map<String, String> options = ivfFlatOptions(dimension, nlist);
        options.put("index.type", "ivf_pq");
        options.put("pq.m", Integer.toString(m));
        options.put("use-opq", "false");
        return options;
    }

    private static Map<String, String> ivfRqOptions(int dimension, int nlist) {
        Map<String, String> options = ivfFlatOptions(dimension, nlist);
        options.put("index.type", "ivf_rq");
        return options;
    }

    private static Map<String, String> ivfHnswOptions(String indexType, int dimension, int nlist) {
        Map<String, String> options = ivfFlatOptions(dimension, nlist);
        options.put("index.type", indexType);
        options.put("hnsw.m", "4");
        options.put("hnsw.ef-construction", "16");
        options.put("hnsw.max-level", "4");
        return options;
    }

    private static float[] roundtripData() {
        float[] data = new float[ROUNDTRIP_VECTOR_COUNT * ROUNDTRIP_DIMENSION];
        for (int i = 0; i < ROUNDTRIP_VECTOR_COUNT; i++) {
            int cluster = i / ROUNDTRIP_PER_LIST;
            int local = i % ROUNDTRIP_PER_LIST;
            float center = cluster * 20.0f;
            for (int dim = 0; dim < ROUNDTRIP_DIMENSION; dim++) {
                data[i * ROUNDTRIP_DIMENSION + dim] =
                        center + dim * 0.01f + (local % 16) * 0.001f;
            }
        }
        return data;
    }

    private static float[] queryForCenter(float center) {
        float[] query = new float[ROUNDTRIP_DIMENSION];
        for (int dim = 0; dim < ROUNDTRIP_DIMENSION; dim++) {
            query[dim] = center + dim * 0.01f;
        }
        return query;
    }

    private static float[] batchQueries() {
        float[] first = queryForCenter(0.0f);
        float[] second = queryForCenter(20.0f);
        float[] queries = new float[2 * ROUNDTRIP_DIMENSION];
        System.arraycopy(first, 0, queries, 0, ROUNDTRIP_DIMENSION);
        System.arraycopy(second, 0, queries, ROUNDTRIP_DIMENSION, ROUNDTRIP_DIMENSION);
        return queries;
    }

    private static long[] roundtripIds() {
        long[] ids = new long[ROUNDTRIP_VECTOR_COUNT];
        for (int i = 0; i < ROUNDTRIP_VECTOR_COUNT; i++) {
            int cluster = i / ROUNDTRIP_PER_LIST;
            int local = i % ROUNDTRIP_PER_LIST;
            ids[i] = clusterBaseId(cluster) + local;
        }
        return ids;
    }

    private static long clusterBaseId(int cluster) {
        return (cluster + 1L) * 100000L;
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

    private static void assertArrayEquals(long[] expected, long[] actual) {
        if (expected.length != actual.length) {
            throw new AssertionError(
                    "expected length " + expected.length + " but got " + actual.length);
        }
        for (int i = 0; i < expected.length; i++) {
            if (expected[i] != actual[i]) {
                throw new AssertionError(
                        "expected[" + i + "] " + expected[i] + " but got " + actual[i]);
            }
        }
    }

    private static void assertIdInCluster(long id, int cluster) {
        long base = clusterBaseId(cluster);
        if (id < base || id >= base + ROUNDTRIP_PER_LIST) {
            throw new AssertionError("id " + id + " should be in cluster " + cluster);
        }
    }

    private static void assertFinite(float value, String label) {
        if (!Float.isFinite(value)) {
            throw new AssertionError(label + " should be finite but was " + value);
        }
    }

    private static void assertThrowsMessage(
            Class<? extends Throwable> expected, String expectedMessage, ThrowingRunnable runnable) {
        try {
            runnable.run();
        } catch (Throwable t) {
            if (!expected.isInstance(t)) {
                throw new AssertionError(
                        "expected " + expected.getName() + " but got " + t.getClass().getName(), t);
            }
            String message = t.getMessage();
            if (message == null || !message.contains(expectedMessage)) {
                throw new AssertionError("unexpected exception message: " + message, t);
            }
            return;
        }
        throw new AssertionError("expected " + expected.getName());
    }

    private static void assertInvalidInput(ThrowingRunnable runnable, String expectedMessage) {
        try {
            runnable.run();
        } catch (RuntimeException e) {
            String message = e.getMessage();
            if (message == null || !message.contains(expectedMessage)) {
                throw new AssertionError("unexpected exception message: " + message, e);
            }
            if (message.contains("Rust panic in JNI call")) {
                throw new AssertionError("invalid input should not cross the panic boundary", e);
            }
            if (message.contains("invalid RoaringTreemap")) {
                throw new AssertionError("query validation should run before filter decoding", e);
            }
            return;
        } catch (Throwable t) {
            throw new AssertionError("expected RuntimeException but got " + t.getClass().getName(), t);
        }
        throw new AssertionError("expected RuntimeException");
    }

    private interface ThrowingRunnable {
        void run() throws Throwable;
    }

    public static final class ByteArrayPositionOutputStream {
        private final ByteArrayOutputStream out = new ByteArrayOutputStream();

        public void write(byte[] bytes) {
            out.write(bytes, 0, bytes.length);
        }

        public byte[] toByteArray() {
            return out.toByteArray();
        }
    }

    public static final class ByteArraySeekableInputStream implements VectorIndexInput {
        private final byte[] data;

        ByteArraySeekableInputStream(byte[] data) {
            this.data = data.clone();
        }

        @Override
        public void pread(long[] positions, byte[][] buffers) {
            if (positions.length != buffers.length) {
                throw new IllegalArgumentException("positions and buffers length mismatch");
            }
            for (int i = 0; i < positions.length; i++) {
                long readPosition = positions[i];
                byte[] buffer = buffers[i];
                if (readPosition < 0 || readPosition + buffer.length > data.length) {
                    throw new IllegalArgumentException("read out of range: " + readPosition);
                }
                System.arraycopy(data, (int) readPosition, buffer, 0, buffer.length);
            }
        }
    }
}
