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
import java.io.PrintStream;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * Standalone check that native IVF-PQ timing diagnostics reach SLF4J (via
 * NativeLogBridge) instead of the raw process stdout.
 *
 * <p>Requires the environment variable PAIMON_VINDEX_LOG_IVFPQ_TIMING to be set
 * before the JVM starts (the Rust gate reads the process environment); prints a
 * skip notice and exits 0 otherwise. Run with slf4j-simple on the classpath:
 *
 * <pre>
 * PAIMON_VINDEX_LOG_IVFPQ_TIMING=1 java -cp ... \
 *   org.apache.paimon.index.vector.NativeLogBridgeSmokeTest [/path/to/libpaimon_vindex_jni.so]
 * </pre>
 */
public class NativeLogBridgeSmokeTest {

    private static final String TIMING_MARKER = "ivfpq_batch_timing";
    private static final String[] REQUIRED_TIMING_FIELDS = {
        "topk",
        "unique_list_rows",
        "query_list_pairs",
        "pq_codes_evaluated",
        "matched_rows",
        "read_calls",
        "requested_bytes",
        "queries_below_k",
        "min_hits_per_query",
        "io_read_ms",
        "decode_ms"
    };

    public static void main(String[] args) {
        if (System.getenv("PAIMON_VINDEX_LOG_IVFPQ_TIMING") == null) {
            if (Boolean.getBoolean("vindex.smoke.require-timing")) {
                throw new AssertionError(
                        "PAIMON_VINDEX_LOG_IVFPQ_TIMING must be set in the process environment "
                                + "when vindex.smoke.require-timing=true");
            }
            System.out.println(
                    "SKIP: PAIMON_VINDEX_LOG_IVFPQ_TIMING is not set in the process environment");
            return;
        }
        VectorIndexNativeLoaderSmokeTest.configureExternalLibrary(args);

        PrintStream originalOut = System.out;
        PrintStream originalErr = System.err;
        ByteArrayOutputStream capturedOut = new ByteArrayOutputStream();
        ByteArrayOutputStream capturedErr = new ByteArrayOutputStream();
        String out;
        String err;
        try {
            // Capture before the first native/SLF4J use: slf4j-simple logs to
            // System.err by default and does not cache the stream.
            System.setOut(new PrintStream(capturedOut, true));
            System.setErr(new PrintStream(capturedErr, true));
            runIvfPqBatchSearch();
        } finally {
            System.setOut(originalOut);
            System.setErr(originalErr);
            out = capturedOut.toString();
            err = capturedErr.toString();
        }

        if (out.contains(TIMING_MARKER)) {
            throw new AssertionError(
                    "timing record leaked to stdout instead of the log bridge:\n" + out);
        }
        if (!err.contains(TIMING_MARKER)) {
            throw new AssertionError(
                    "timing record missing from SLF4J output.\nstderr:\n"
                            + err
                            + "\nstdout:\n"
                            + out);
        }
        for (String field : REQUIRED_TIMING_FIELDS) {
            if (!err.contains(field + "=")) {
                throw new AssertionError("timing field " + field + " missing:\n" + err);
            }
        }
        if (err.contains("read_decode_ms=")) {
            throw new AssertionError("combined read/decode timing was not split:\n" + err);
        }
        if (err.contains("native log bridge disabled")) {
            throw new AssertionError("bridge unexpectedly degraded to stdout:\n" + err);
        }
        System.out.println("OK: " + TIMING_MARKER + " delivered via NativeLogBridge/SLF4J only");
    }

    private static void runIvfPqBatchSearch() {
        int dimension = 8;
        int vectorCount = 512;
        float[] data = new float[vectorCount * dimension];
        long[] ids = new long[vectorCount];
        Random random = new Random(13L);
        for (int row = 0; row < vectorCount; row++) {
            ids[row] = row;
            for (int column = 0; column < dimension; column++) {
                data[row * dimension + column] = (float) random.nextGaussian();
            }
        }

        Map<String, String> options = new HashMap<String, String>();
        options.put("index.type", "ivf_pq");
        options.put("dimension", Integer.toString(dimension));
        options.put("nlist", "2");
        options.put("metric", "l2");
        options.put("use-opq", "false");

        VectorIndexWriter writer =
                new VectorIndexWriter(VectorIndexTrainer.train(options, data, vectorCount));
        byte[] indexBytes;
        try {
            writer.addVectors(ids, data, vectorCount);
            VectorIndexNativeHandleSafetyTest.ByteArrayPositionOutputStream output =
                    new VectorIndexNativeHandleSafetyTest.ByteArrayPositionOutputStream();
            writer.writeIndex(output);
            indexBytes = output.toByteArray();
        } finally {
            writer.close();
        }

        VectorIndexReader reader = new VectorIndexReader(new ByteArrayInput(indexBytes));
        try {
            int queryCount = 4;
            int topK = 3;
            float[] queries = new float[queryCount * dimension];
            for (int offset = 0; offset < queries.length; offset++) {
                queries[offset] = (float) random.nextGaussian();
            }
            VectorSearchBatchResult result =
                    reader.searchBatch(queries, queryCount, new VectorSearchParams(topK, 2));
            if (result.ids().length != queryCount * topK) {
                throw new AssertionError("unexpected batch result size: " + result.ids().length);
            }
        } finally {
            reader.close();
        }
    }

    private static final class ByteArrayInput implements VectorIndexInput {
        private final byte[] data;

        ByteArrayInput(byte[] data) {
            this.data = data;
        }

        @Override
        public void pread(long[] positions, byte[][] buffers) {
            for (int i = 0; i < positions.length; i++) {
                System.arraycopy(data, (int) positions[i], buffers[i], 0, buffers[i].length);
            }
        }
    }
}
