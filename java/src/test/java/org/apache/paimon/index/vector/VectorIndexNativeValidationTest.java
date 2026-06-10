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

    public static void main(String[] args) {
        if (args.length != 1) {
            throw new IllegalArgumentException("native library path is required");
        }

        System.load(args[0]);

        testWriterValidationComesFromCore();
        testReaderValidationComesFromCore();
    }

    private static void testWriterValidationComesFromCore() {
        final VectorIndexWriter writer = new VectorIndexWriter(ivfFlatOptions());
        try {
            assertThrowsMessage(
                    RuntimeException.class,
                    "training data length 2 does not match vector count * dimension 1",
                    new ThrowingRunnable() {
                        @Override
                        public void run() {
                            writer.train(new float[] {0.0f, 1.0f}, 1);
                        }
                    });
            assertThrowsMessage(
                    RuntimeException.class,
                    "ids length 2 does not match vector count 1",
                    new ThrowingRunnable() {
                        @Override
                        public void run() {
                            writer.addVectors(new long[] {1L, 2L}, new float[] {0.0f}, 1);
                        }
                    });
            assertThrowsMessage(
                    RuntimeException.class,
                    "vector count must be greater than 0",
                    new ThrowingRunnable() {
                        @Override
                        public void run() {
                            writer.train(new float[0], 0);
                        }
                    });
        } finally {
            writer.close();
        }
    }

    private static void testReaderValidationComesFromCore() {
        VectorIndexReader reader = new VectorIndexReader(new ByteArraySeekableInputStream(buildIndexBytes()));
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

    private static byte[] buildIndexBytes() {
        VectorIndexWriter writer = new VectorIndexWriter(ivfFlatOptions());
        ByteArrayPositionOutputStream output = new ByteArrayPositionOutputStream();
        try {
            writer.train(new float[] {0.0f, 1.0f}, 2);
            writer.addVectors(new long[] {1L, 2L}, new float[] {0.0f, 1.0f}, 2);
            writer.writeIndex(output);
            return output.toByteArray();
        } finally {
            writer.close();
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
