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

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import org.apache.paimon.index.vector.VectorIndexNativeValidationTest.ByteArrayPositionOutputStream;
import org.apache.paimon.index.vector.VectorIndexNativeValidationTest.ByteArraySeekableInputStream;

/** Exercises the actual JNI boundary, including ownership and absolute IDs. */
public class VectorIndexCentroidShardingTest {

    public static void main(String[] args) {
        VectorIndexNativeLoaderSmokeTest.configureExternalLibrary(args);
        run();
    }

    static void run() {
        checkDiskAnnRouterOptions();
        runAllTypes();
        float[] data = new float[256 * 4];
        for (int i = 0; i < 256; i++) {
            data[i * 4] = i * 0.13f;
            data[i * 4 + 1] = i % 17 + 1;
            data[i * 4 + 2] = 2;
            data[i * 4 + 3] = 3;
        }
        Map<String, String> options = new HashMap<>();
        options.put("index.type", "ivf_pq");
        options.put("nlist", "4");
        options.put("metric", "l2");
        byte[] payload;
        try (VectorIndexTraining model = VectorIndexTrainer.train(options, data, 256)) {
            payload = model.serialize();
        }
        VectorIndexWriter[] writers = new VectorIndexWriter[4];
        int[] counts = new int[4];
        try {
            try (VectorIndexTraining model = VectorIndexTraining.deserialize(payload)) {
                check(model.nlist() == 4, "nlist survived serialization");
                for (int c = 0; c < 4; c++) {
                    writers[c] = new VectorIndexWriter(model, c);
                }
                for (int i = 0; i < 256; i++) {
                    float[] vector = Arrays.copyOfRange(data, i * 4, (i + 1) * 4);
                    int c = model.assignCentroid(vector);
                    writers[c].addRoutedIvfCentroidVectors(
                            new long[] {(1L << 35) + 101L * i}, vector, 1);
                    counts[c]++;
                }
                check(
                        Arrays.equals(payload, model.serialize()),
                        "writers must not consume shared training");
                expectFailure(() -> new VectorIndexWriter(model, 4));
                expectFailure(() -> model.assignCentroid(new float[] {Float.NaN, 0, 0, 0}));
            }
            for (int c = 0; c < 4; c++) {
                final int centroid = c;
                ByteArrayPositionOutputStream output = new ByteArrayPositionOutputStream();
                // Writers own their model state and outlive the training handle.
                writers[c].writeIndex(output);
                try (VectorIndexReader reader =
                        new VectorIndexReader(
                                new ByteArraySeekableInputStream(output.toByteArray()))) {
                    float[] query = Arrays.copyOf(data, 4);
                    VectorSearchParams params = VectorSearchParams.ivf(256, 1);
                    VectorSearchResult actual = reader.searchRoutedIvfShard(query, params, c);
                    VectorSearchResult expected =
                            reader.search(query, VectorSearchParams.ivf(256, 4));
                    check(
                            Arrays.equals(actual.ids(), expected.ids()),
                            "routed search scans the fixed list");
                    check(
                            Arrays.equals(actual.distances(), expected.distances()),
                            "routed estimates match normal search");
                    long present = Arrays.stream(actual.ids()).filter(id -> id >= 0).count();
                    check(present == counts[c], "sparse absolute row IDs survived JNI");
                    float[] queries = new float[8];
                    System.arraycopy(query, 0, queries, 0, 4);
                    System.arraycopy(query, 0, queries, 4, 4);
                    VectorSearchBatchResult batch =
                            reader.searchRoutedIvfShardBatch(queries, 2, params, c);
                    check(Arrays.equals(actual.ids(), batch.idsForQuery(0)), "first batch query");
                    check(Arrays.equals(actual.ids(), batch.idsForQuery(1)), "second batch query");
                    expectFailure(() -> reader.searchRoutedIvfShard(query, params, 4));
                    expectFailure(
                            () ->
                                    reader.searchRoutedIvfShard(
                                            query, params, centroid, new byte[] {1}));
                }
            }
            expectFailure(
                    () ->
                            VectorIndexTraining.deserialize(
                                    Arrays.copyOf(payload, payload.length - 1)));
            VectorIndexTraining closed = VectorIndexTraining.deserialize(payload);
            closed.close();
            expectFailure(closed::serialize);
            expectFailure(() -> closed.assignCentroid(new float[4]));
            expectFailure(() -> new VectorIndexWriter(closed, 0));
        } finally {
            for (VectorIndexWriter writer : writers) {
                if (writer != null) {
                    writer.close();
                }
            }
        }
    }

    private static void checkDiskAnnRouterOptions() {
        Map<String, String> options = new HashMap<>();
        options.put(" index.type ", " diskann ");
        options.put("dimension", "2");
        options.put("metric", "cosine");
        for (String count : new String[] {" 2 ", " auto "}) {
            options.put(" nlist ", count);
            try (VectorIndexTrainer trainer = VectorIndexTrainer.createCentroid(options);
                    VectorIndexTraining model =
                            trainer.addTrainingVectors(new float[] {1, 0, 0, 1}, 2)
                                    .finishTraining()) {
                check(model.nlist() == (" 2 ".equals(count) ? 2 : 64), "outer router count");
            }
        }
        options.put("nlist", "2");
        expectFailure(() -> VectorIndexTrainer.createCentroid(options).close());
    }

    private static void runAllTypes() {
        float[] data = new float[256 * 4];
        for (int i = 0; i < 256; i++) {
            data[i * 4] = (i / 64) * 100f + (i % 64) * .13f;
            data[i * 4 + 1] = i % 17;
            data[i * 4 + 2] = 1;
            data[i * 4 + 3] = 3;
        }
        for (String kind : new String[] {"ivf_flat", "ivf_pq", "ivf_sq", "ivf_rq", "diskann"}) {
            Map<String, String> options = new HashMap<>();
            options.put("index.type", kind);
            options.put("dimension", "4");
            options.put("nlist", "4");
            options.put("metric", "l2");
            if ("ivf_pq".equals(kind) || "diskann".equals(kind)) {
                options.put("pq.m", "2");
            }
            if ("diskann".equals(kind)) {
                options.put("diskann.max-degree", "8");
                options.put("diskann.build-search-list-size", "16");
            }
            byte[] payload;
            try (VectorIndexTrainer trainer = VectorIndexTrainer.createCentroid(options)) {
                trainer.addTrainingVectors(data, 256);
                try (VectorIndexTraining model = trainer.finishTraining()) {
                    payload = model.serialize();
                }
            }
            try (VectorIndexTraining model = VectorIndexTraining.deserialize(payload)) {
                check(kind.equals(model.indexType()), "typed JNI model survived");
                check(Arrays.equals(payload, model.serialize()), "typed JNI model roundtrip");
                int[] assigned = new int[256];
                for (int i = 0; i < assigned.length; i++) {
                    float[] vector = Arrays.copyOfRange(data, i * 4, (i + 1) * 4);
                    assigned[i] = model.assignCentroid(vector);
                    check(
                            assigned[i] == model.findNearestCentroids(vector, model.nlist())[0],
                            "scalar assignment must agree with nearest-centroid search for " + kind);
                }
                for (int c = 0; c < model.nlist(); c++) {
                    try (VectorIndexWriter writer = new VectorIndexWriter(model, c)) {
                        int count = 0;
                        for (int i = 0; i < 256; i++) {
                            if (assigned[i] == c) {
                                float[] v = Arrays.copyOfRange(data, i * 4, (i + 1) * 4);
                                writer.addCentroidVectors(new long[] {(1L << 35) + 101L * i}, v, 1);
                                count++;
                            }
                        }
                        if (count == 0 && "diskann".equals(kind)) {
                            continue;
                        }
                        ByteArrayPositionOutputStream output = new ByteArrayPositionOutputStream();
                        writer.writeIndex(output);
                        try (VectorIndexReader reader =
                                new VectorIndexReader(
                                        new ByteArraySeekableInputStream(output.toByteArray()))) {
                            VectorSearchParams params =
                                    "diskann".equals(kind)
                                            ? VectorSearchParams.diskAnn(256, 256)
                                            : VectorSearchParams.ivf(256, 1);
                            VectorSearchResult result =
                                    reader.searchCentroidShard(Arrays.copyOf(data, 4), params, c);
                            check(
                                    Arrays.stream(result.ids()).filter(id -> id >= 0).count()
                                            == count,
                                    "typed centroid count");
                            float[] queries = Arrays.copyOf(data, 8);
                            VectorSearchBatchResult batch =
                                    reader.searchCentroidShardBatch(queries, 2, params, c);
                            check(
                                    Arrays.stream(batch.idsForQuery(1))
                                                    .filter(id -> id >= 0)
                                                    .count()
                                            == count,
                                    "typed centroid batch count");
                        }
                    }
                }
            }
        }
    }

    private static void check(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError(message);
        }
    }

    private static void expectFailure(Runnable action) {
        try {
            action.run();
        } catch (RuntimeException expected) {
            return;
        }
        throw new AssertionError("expected validation failure");
    }
}
