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

public class VectorIndexNativeHandleSafetyTest {

    public static void main(String[] args) {
        if (args.length != 1) {
            throw new IllegalArgumentException("native library path is required");
        }

        System.load(args[0]);

        testWriterRejectsReentrantCloseDuringNativeCall();
        testReaderRejectsReentrantCloseDuringNativeCall();
    }

    private static void testWriterRejectsReentrantCloseDuringNativeCall() {
        VectorIndexWriter writer = newPopulatedWriter();
        SelfClosingOutputStream output = new SelfClosingOutputStream(writer);
        try {
            writer.writeIndex(output);
            assertTrue(output.closeAttempted());
            assertTrue(output.closeRejected());
            assertTrue(output.toByteArray().length > 0);
        } finally {
            writer.close();
        }
    }

    private static void testReaderRejectsReentrantCloseDuringNativeCall() {
        SelfClosingSeekableInputStream input =
                new SelfClosingSeekableInputStream(buildIndexBytes());
        VectorIndexReader reader = new VectorIndexReader(input);
        input.setReader(reader);
        try {
            VectorSearchResult result = reader.search(new float[] {0.0f}, 1, 1);
            assertEquals(1, result.ids().length);
            assertTrue(input.closeAttempted());
            assertTrue(input.closeRejected());
            assertEquals(2L, reader.totalVectors());
        } finally {
            reader.close();
        }
    }

    private static byte[] buildIndexBytes() {
        VectorIndexWriter writer = newPopulatedWriter();
        ByteArrayPositionOutputStream output = new ByteArrayPositionOutputStream();
        try {
            writer.writeIndex(output);
            return output.toByteArray();
        } finally {
            writer.close();
        }
    }

    private static VectorIndexWriter newPopulatedWriter() {
        VectorIndexWriter writer = new VectorIndexWriter(ivfFlatOptions());
        writer.train(new float[] {0.0f, 1.0f}, 2);
        writer.addVectors(new long[] {1L, 2L}, new float[] {0.0f, 1.0f}, 2);
        return writer;
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

    private static void assertTrue(boolean value) {
        if (!value) {
            throw new AssertionError("expected true");
        }
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

    public static final class SelfClosingOutputStream {
        private final VectorIndexWriter writer;
        private final ByteArrayOutputStream out = new ByteArrayOutputStream();
        private boolean closeAttempted;
        private boolean closeRejected;

        SelfClosingOutputStream(VectorIndexWriter writer) {
            this.writer = writer;
        }

        public void write(byte[] bytes) {
            if (!closeAttempted) {
                closeAttempted = true;
                try {
                    writer.close();
                } catch (IllegalStateException expected) {
                    closeRejected = true;
                }
            }
            out.write(bytes, 0, bytes.length);
        }

        boolean closeAttempted() {
            return closeAttempted;
        }

        boolean closeRejected() {
            return closeRejected;
        }

        byte[] toByteArray() {
            return out.toByteArray();
        }
    }

    public static final class SelfClosingSeekableInputStream implements VectorIndexInput {
        private final byte[] data;
        private VectorIndexReader reader;
        private boolean closeAttempted;
        private boolean closeRejected;

        SelfClosingSeekableInputStream(byte[] data) {
            this.data = data.clone();
        }

        void setReader(VectorIndexReader reader) {
            this.reader = reader;
        }

        @Override
        public void pread(long[] positions, byte[][] buffers) {
            tryCloseReaderOnce();
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

        boolean closeAttempted() {
            return closeAttempted;
        }

        boolean closeRejected() {
            return closeRejected;
        }

        private void tryCloseReaderOnce() {
            if (reader == null || closeAttempted) {
                return;
            }
            closeAttempted = true;
            try {
                reader.close();
            } catch (IllegalStateException expected) {
                closeRejected = true;
            }
        }
    }
}
