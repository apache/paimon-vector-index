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
        if (args.length != 1) {
            throw new IllegalArgumentException("native library path is required");
        }

        System.load(args[0]);

        testVoidEntrypointErrorBecomesRuntimeException();
        testObjectEntrypointPanicBecomesRuntimeException();

        VectorIndexWriter survivor = new VectorIndexWriter(ivfFlatOptions());
        survivor.close();
    }

    private static void testVoidEntrypointErrorBecomesRuntimeException() {
        final VectorIndexWriter writer = new VectorIndexWriter(ivfFlatOptions());
        try {
            assertThrowsMessage(
                    RuntimeException.class,
                    "cannot add vectors before training is complete",
                    new ThrowingRunnable() {
                        @Override
                        public void run() {
                            writer.addVectors(new long[] {1L}, new float[] {1.0f}, 1);
                        }
                    });
        } finally {
            writer.close();
        }
    }

    private static void testObjectEntrypointPanicBecomesRuntimeException() {
        ByteArrayPositionOutputStream output = new ByteArrayPositionOutputStream();
        VectorIndexWriter writer = new VectorIndexWriter(ivfFlatOptions());
        try {
            writer.train(new float[] {0.0f, 1.0f}, 2);
            writer.addVectors(new long[] {1L, 2L}, new float[] {0.0f, 1.0f}, 2);
            writer.writeIndex(output);
        } finally {
            writer.close();
        }
        byte[] indexBytes = output.toByteArray();
        corruptFirstIvfFlatVector(indexBytes, Float.NaN);

        VectorIndexReader reader =
                new VectorIndexReader(new ByteArraySeekableInputStream(indexBytes));
        try {
            assertEquals(1, reader.dimension());
            assertThrows(RuntimeException.class, new ThrowingRunnable() {
                @Override
                public void run() {
                    reader.search(new float[] {0.0f}, 2, 1);
                }
            });
            assertEquals(2L, reader.totalVectors());
        } finally {
            reader.close();
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

    private static void assertThrows(Class<? extends Throwable> expected, ThrowingRunnable runnable) {
        try {
            runnable.run();
        } catch (Throwable t) {
            if (expected.isInstance(t)) {
                String message = t.getMessage();
                if (message == null || !message.contains("Rust panic in JNI call")) {
                    throw new AssertionError("unexpected exception message: " + message, t);
                }
                return;
            }
            throw new AssertionError("expected " + expected.getName() + " but got " + t.getClass().getName(), t);
        }
        throw new AssertionError("expected " + expected.getName());
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
