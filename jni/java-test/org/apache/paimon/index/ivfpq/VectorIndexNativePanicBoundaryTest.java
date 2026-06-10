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

import java.io.ByteArrayOutputStream;

public class VectorIndexNativePanicBoundaryTest {

    public static void main(String[] args) {
        if (args.length != 1) {
            throw new IllegalArgumentException("native library path is required");
        }

        System.load(args[0]);

        testVoidEntrypointPanicBecomesRuntimeException();
        testObjectEntrypointPanicBecomesRuntimeException();

        VectorIndexWriter survivor = new VectorIndexWriter(VectorIndexConfig.ivfFlat(1, 1, Metric.L2));
        survivor.close();
    }

    private static void testVoidEntrypointPanicBecomesRuntimeException() {
        final VectorIndexWriter writer =
                new VectorIndexWriter(VectorIndexConfig.ivfFlat(1, 1, Metric.L2));
        try {
            assertThrows(RuntimeException.class, new ThrowingRunnable() {
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
        VectorIndexWriter writer =
                new VectorIndexWriter(VectorIndexConfig.ivfFlat(1, 1, Metric.L2));
        try {
            writer.train(new float[] {0.0f, 1.0f}, 2);
            writer.addVectors(new long[] {1L, 2L}, new float[] {Float.NaN, 1.0f}, 2);
            writer.writeIndex(output);
        } finally {
            writer.close();
        }

        VectorIndexReader reader =
                new VectorIndexReader(new ByteArraySeekableInputStream(output.toByteArray()));
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
