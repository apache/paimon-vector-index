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

/** Exercise the external router and standard shard API through the packaged JNI library. */
public final class VectorIndexCentroidShardingTest {
    public static void main(String[] args) {
        VectorIndexNativeLoaderSmokeTest.configureExternalLibrary(args);
        run();
    }

    static void run() {
        float[] rows = new float[256 * 4];
        for (int i = 0; i < 256; i++) {
            Arrays.fill(rows, i * 4, i * 4 + 4, (i / 64) * 100 + (i % 64) * 0.1f);
        }
        Map<String, String> options = new HashMap<>();
        options.put("dimension", "4");
        options.put("metric", "l2");
        options.put("nlist", "4");
        Map<String, String> localOptions = new HashMap<>(options);
        localOptions.put("index.type", "ivf_sq");
        try (VectorIndexTraining local = VectorIndexTrainer.train(localOptions, rows, 256)) {
            try {
                CentroidRouter.fromTraining(local).close();
                throw new AssertionError("expected IVF-FLAT coarse training");
            } catch (RuntimeException expected) {
                check(expected.getMessage().contains("IVF-FLAT"), "coarse training validation");
            }
        }
        byte[] payload;
        options.put("index.type", "ivf_flat");
        try (VectorIndexTrainer trainer = VectorIndexTrainer.create(options);
                VectorIndexTraining coarse = trainer.addTrainingVectors(rows, 256).finishTraining();
                CentroidRouter router = CentroidRouter.fromTraining(coarse)) {
            payload = router.serialize();
        }
        try (CentroidRouter router = CentroidRouter.deserialize(payload)) {
            check(payload.length < CentroidRouter.maxSerializedBytes(), "router size bound");
            float[] query = Arrays.copyOf(rows, 4);
            int selected = router.assignCentroid(query);
            check(router.findNearestCentroids(query, 1)[0] == selected, "routing consistency");
            check(router.resolveNprobe(VectorSearchParams.ivf(1, 2), 256, -1) == 2, "outer width");
            long[] ids = new long[256];
            float[] vectors = new float[rows.length];
            int count = 0;
            for (int i = 0; i < 256; i++) {
                float[] row = Arrays.copyOfRange(rows, i * 4, i * 4 + 4);
                if (selected == router.assignCentroid(row)) {
                    ids[count] = (1L << 35) + i;
                    System.arraycopy(row, 0, vectors, count * 4, 4);
                    count++;
                }
            }
            check(count > 1 && count < 256, "selected local training set");
            float[] local = Arrays.copyOf(vectors, count * 4);
            for (String kind : new String[] {"ivf_flat", "ivf_sq", "ivf_pq", "ivf_rq", "diskann"}) {
                Map<String, String> shard = new HashMap<>();
                shard.put("index.type", kind);
                shard.put("dimension", "4");
                shard.put("metric", "l2");
                if (!"diskann".equals(kind)) {
                    shard.put("nlist", "2");
                }
                ByteArrayPositionOutputStream output = new ByteArrayPositionOutputStream();
                try (VectorIndexTraining training = VectorIndexTrainer.train(shard, local, count);
                        VectorIndexWriter writer = new VectorIndexWriter(training)) {
                    writer.addVectors(Arrays.copyOf(ids, count), local, count);
                    writer.writeIndex(output);
                }
                try (VectorIndexReader reader =
                        new VectorIndexReader(new ByteArraySeekableInputStream(output.toByteArray()))) {
                    VectorSearchParams params = "diskann".equals(kind)
                            ? VectorSearchParams.diskAnn(count, 256) : VectorSearchParams.ivf(count, 2);
                    long[] hits = reader.search(query, params).ids();
                    Arrays.sort(hits);
                    check(Arrays.equals(hits, Arrays.copyOf(ids, count)), kind + " absolute row IDs");
                }
            }
        }
    }

    private static void check(boolean condition, String message) {
        if (!condition) { throw new AssertionError(message); }
    }
}
