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

public class VectorIndexNativePanicBoundaryTest {

    public static void main(String[] args) {
        VectorIndexNativeLoaderSmokeTest.configureExternalLibrary(args);

        testVoidEntrypointErrorBecomesRuntimeException();
        testObjectEntrypointSkipsNanScores();
        testChunkedOutputReusesBuffersAndPreservesJavaException();

        VectorIndexTrainer survivor = VectorIndexTrainer.create(ivfFlatOptions());
        survivor.close();
    }

    private static void testVoidEntrypointErrorBecomesRuntimeException() {
        final VectorIndexTrainer trainer = VectorIndexTrainer.create(ivfFlatOptions());
        try {
            assertThrowsMessage(
                    RuntimeException.class,
                    "training data length 1 does not match vector count * dimension 2",
                    new ThrowingRunnable() {
                        @Override
                        public void run() {
                            trainer.addTrainingVectors(new float[] {1.0f}, 2);
                        }
                    });
        } finally {
            trainer.close();
        }
    }

    private static void testObjectEntrypointSkipsNanScores() {
        ByteArrayPositionOutputStream output = new ByteArrayPositionOutputStream();
        VectorIndexWriter writer =
                new VectorIndexWriter(
                        VectorIndexTrainer.train(ivfFlatOptions(), new float[] {0.0f, 1.0f}, 2));
        try {
            writer.addVectors(new long[] {1L, 2L}, new float[] {0.0f, 1.0f}, 2);
            writer.writeIndex(output);
        } finally {
            writer.close();
        }
        // NaN scores formerly triggered a Rust panic. Both signs must now be
        // skipped without losing valid neighbors or the missing-result padding.
        for (int nanBits : new int[] {0x7fc00000, 0xffc00000}) {
            byte[] indexBytes = output.toByteArray();
            corruptFirstIvfFlatVector(indexBytes, Float.intBitsToFloat(nanBits));

            VectorIndexReader reader =
                    new VectorIndexReader(new ByteArraySeekableInputStream(indexBytes));
            try {
                assertEquals(1, reader.dimension());
                VectorSearchParams params = new VectorSearchParams(2, 1);
                VectorSearchResult single = reader.search(new float[] {0.0f}, params);
                assertNanSkipped(single.ids(), single.distances());

                VectorSearchBatchResult batch =
                        reader.searchBatch(new float[] {0.0f, 0.0f}, 2, params);
                assertEquals(2, batch.queryCount());
                assertEquals(2, batch.topK());
                for (int query = 0; query < batch.queryCount(); query++) {
                    assertNanSkipped(batch.idsForQuery(query), batch.distancesForQuery(query));
                }
                assertEquals(2L, reader.totalVectors());
            } finally {
                reader.close();
            }
        }
    }

    private static void assertNanSkipped(long[] ids, float[] distances) {
        assertEquals(2, ids.length);
        assertEquals(2, distances.length);
        assertEquals(2L, ids[0]);
        assertEquals(-1L, ids[1]);
        assertEquals(Float.floatToRawIntBits(1.0f), Float.floatToRawIntBits(distances[0]));
        assertEquals(Float.floatToRawIntBits(Float.MAX_VALUE), Float.floatToRawIntBits(distances[1]));
    }

    private static void testChunkedOutputReusesBuffersAndPreservesJavaException() {
        Map<String, String> options = ivfFlatOptions();
        options.put("index.type", "ivf_sq");
        options.put("dimension", "16");
        final int rows = 150001;
        float[] data = new float[rows * 16];
        long[] ids = new long[rows];
        for (int row = 0; row < rows; row++) {
            ids[row] = row + 1;
            for (int dim = 0; dim < 16; dim++) {
                data[row * 16 + dim] = (row * 7 + dim * 13) % 256;
            }
        }
        float[] training = java.util.Arrays.copyOf(data, 256 * 16);
        VectorIndexWriter writer =
                new VectorIndexWriter(VectorIndexTrainer.train(options, training, 256));
        try {
            writer.addVectors(ids, data, rows);
            ChunkCheckingOutput output = new ChunkCheckingOutput(false);
            writer.writeIndex(output);
            if (output.fullChunks < 2) {
                throw new AssertionError("SQ list did not exercise repeated full chunks");
            }
            byte[] bytes = output.out.toByteArray();
            // Global SQ bounds, list SQ bounds, then the coarse centroid.
            int table = 64 + 5 * 16 * Float.BYTES;
            int codesOffset = (int) readLongLe(bytes, table);
            // Check the complete blocked SQ byte order, including signed bytes
            // and the short final 32-row block, across both JNI boundaries.
            int offset = codesOffset;
            for (int start = 0; start < rows; start += 32) {
                int lanes = Math.min(32, rows - start);
                for (int dim = 0; dim < 16; dim++) {
                    for (int lane = 0; lane < lanes; lane++) {
                        assertEquals((int) data[(start + lane) * 16 + dim], bytes[offset++] & 255);
                    }
                }
            }
            ChunkCheckingOutput failing = new ChunkCheckingOutput(true);
            try {
                writer.writeIndex(failing);
                throw new AssertionError("expected Java output exception");
            } catch (RuntimeException error) {
                if (error != failing.failure) {
                    throw new AssertionError("JNI replaced the original Java exception", error);
                }
                assertEquals(2, failing.fullChunks);
            }
        } finally {
            writer.close();
        }
    }

    public static final class ChunkCheckingOutput {
        private final ByteArrayOutputStream out = new ByteArrayOutputStream();
        private final RuntimeException failure = new RuntimeException("chunk write failed");
        private final boolean fail;
        private byte[] fullBuffer;
        private int fullChunks;

        ChunkCheckingOutput(boolean fail) {
            this.fail = fail;
        }

        public void write(byte[] bytes) {
            if (bytes.length > 1024 * 1024) {
                throw new AssertionError("JNI write exceeds the chunk limit");
            }
            if (bytes.length == 1024 * 1024) {
                fullChunks++;
                if (fullBuffer != null && fullBuffer != bytes) {
                    throw new AssertionError("JNI did not reuse the full-chunk Java buffer");
                }
                fullBuffer = bytes;
                if (fail && fullChunks == 2) {
                    throw failure;
                }
            }
            out.write(bytes, 0, bytes.length);
        }
    }

    private static Map<String, String> ivfFlatOptions() {
        Map<String, String> options = new HashMap<String, String>();
        options.put("index.type", "ivf_flat");
        options.put("dimension", "1");
        options.put("nlist", "1");
        options.put("metric", "l2");
        return options;
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

    private static void corruptFirstIvfFlatVector(byte[] indexBytes, float value) {
        int dimension = readIntLe(indexBytes, 8);
        int nlist = readIntLe(indexBytes, 12);
        int offsetTable = 64 + dimension * nlist * Float.BYTES;
        int listOffset = (int) readLongLe(indexBytes, offsetTable);
        int idBytesLength = readIntLe(indexBytes, listOffset + Long.BYTES);
        int firstVectorOffset = listOffset + Long.BYTES + Integer.BYTES + idBytesLength;
        writeFloatLe(indexBytes, firstVectorOffset, value);
    }

    private static int readIntLe(byte[] bytes, int offset) {
        return (bytes[offset] & 0xFF)
                | ((bytes[offset + 1] & 0xFF) << 8)
                | ((bytes[offset + 2] & 0xFF) << 16)
                | ((bytes[offset + 3] & 0xFF) << 24);
    }

    private static long readLongLe(byte[] bytes, int offset) {
        long result = 0L;
        for (int i = 0; i < Long.BYTES; i++) {
            result |= (long) (bytes[offset + i] & 0xFF) << (8 * i);
        }
        return result;
    }

    private static void writeFloatLe(byte[] bytes, int offset, float value) {
        int bits = Float.floatToRawIntBits(value);
        bytes[offset] = (byte) bits;
        bytes[offset + 1] = (byte) (bits >>> 8);
        bytes[offset + 2] = (byte) (bits >>> 16);
        bytes[offset + 3] = (byte) (bits >>> 24);
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
