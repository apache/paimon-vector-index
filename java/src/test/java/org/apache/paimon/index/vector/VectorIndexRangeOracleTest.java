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

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

public class VectorIndexRangeOracleTest {

    public static void main(String[] args) {
        VectorIndexNativeLoaderSmokeTest.configureExternalLibrary(args);
        runIfConfigured();
    }

    static void runIfConfigured() {
        String directory = System.getenv("PVI_RANGE_FIXTURES");
        if (directory == null || directory.isEmpty()) {
            return;
        }
        Path root = Paths.get(directory);
        int cases = 0;
        try {
            for (String line :
                    Files.readAllLines(root.resolve("manifest.txt"), StandardCharsets.UTF_8)) {
                if (line.trim().isEmpty()) {
                    continue;
                }
                String[] fields = line.trim().split("\\s+");
                require(fields.length == 2, "manifest entry: " + line);
                try {
                    runCase(root.resolve(fields[0] + ".expected"), root.resolve(fields[1]));
                } catch (Throwable error) {
                    throw new AssertionError("Core range oracle case " + fields[0], error);
                }
                cases++;
            }
        } catch (IOException error) {
            throw new AssertionError("Cannot read core range oracle " + root, error);
        }
        require(cases > 0, "oracle manifest is empty");
        System.out.println("Core range oracle: " + cases + " cases passed");
    }

    private static void runCase(Path expected, Path index) throws IOException {
        try (Scanner values = new Scanner(expected, StandardCharsets.UTF_8.name())) {
            int dimension = values.nextInt();
            int metricCode = values.nextInt();
            int queryCount = values.nextInt();
            int nprobe = values.nextInt();
            Float lower = readBound(values);
            Float upper = readBound(values);
            int filterLength = values.nextInt();
            int hitCount = values.nextInt();
            long listReads = values.nextLong();
            float[] queries = new float[Math.multiplyExact(dimension, queryCount)];
            for (int offset = 0; offset < queries.length; offset++) {
                queries[offset] = Float.intBitsToFloat(readBits(values));
            }
            byte[] filter = new byte[filterLength];
            for (int offset = 0; offset < filter.length; offset++) {
                int value = values.nextInt();
                require(value >= 0 && value <= 255, "filter byte");
                filter[offset] = (byte) value;
            }
            long[] lims = readLongs(values, Math.addExact(queryCount, 1));
            long[] labels = readLongs(values, hitCount);
            int[] distances = new int[hitCount];
            for (int offset = 0; offset < hitCount; offset++) {
                distances[offset] = readBits(values);
            }
            long[][] stats = new long[4][queryCount];
            for (int queryIndex = 0; queryIndex < queryCount; queryIndex++) {
                for (int counter = 0; counter < 4; counter++) {
                    stats[counter][queryIndex] = values.nextLong();
                }
            }
            require(!values.hasNext(), "unexpected trailing oracle data");
            String[] metrics = {"l2", "inner_product", "cosine"};
            require(metricCode >= 0 && metricCode < metrics.length, "metric code");
            VectorRangeSearchParams params =
                    new VectorRangeSearchParams(
                            VectorDistanceBand.fromRaw(metrics[metricCode], lower, upper), nprobe);
            try (VectorIndexReader reader =
                    new VectorIndexReader(
                            new VectorIndexNativeValidationTest.ByteArraySeekableInputStream(
                                    Files.readAllBytes(index)))) {
                require(reader.supportsRangeSearch(), "range support");
                VectorRangeSearchResult actual;
                if (queryCount == 1) {
                    actual =
                            filterLength == 0
                                    ? reader.rangeSearch(queries, params)
                                    : reader.rangeSearch(queries, params, filter);
                } else {
                    actual =
                            filterLength == 0
                                    ? reader.rangeSearchBatch(queries, queryCount, params)
                                    : reader.rangeSearchBatch(queries, queryCount, params, filter);
                }
                require(actual.queryCount() == queryCount, "query count");
                require(actual.hitCount() == hitCount, "hit count");
                for (int queryIndex = 0; queryIndex < queryCount; queryIndex++) {
                    int start = Math.toIntExact(lims[queryIndex]);
                    int end = Math.toIntExact(lims[queryIndex + 1]);
                    require(actual.queryStart(queryIndex) == start, "query start");
                    require(actual.queryEnd(queryIndex) == end, "query end");
                    require(
                            rows(labels, distances, start, end)
                                    .equals(rows(actual, queryIndex)),
                            "label/distance-bit multiset for query " + queryIndex);
                    require(stats[0][queryIndex] == actual.listsProbed(queryIndex), "listsProbed");
                    require(stats[1][queryIndex] == actual.rowsScanned(queryIndex), "rowsScanned");
                    require(stats[2][queryIndex] == actual.rowsCommitted(queryIndex), "rowsCommitted");
                    require(
                            stats[3][queryIndex] == actual.earlyAbandoned(queryIndex),
                            "earlyAbandoned");
                }
                require(listReads == actual.listReads(), "listReads");
            }
        }
    }

    private static Float readBound(Scanner values) {
        int kind = values.nextInt();
        int bits = readBits(values);
        require(kind == 0 || kind == 1, "bound kind");
        return kind == 0 ? null : Float.intBitsToFloat(bits);
    }

    private static Map<Long, List<Integer>> rows(
            long[] labels, int[] distances, int start, int end) {
        Map<Long, List<Integer>> result = new HashMap<Long, List<Integer>>();
        for (int offset = start; offset < end; offset++) {
            result.computeIfAbsent(labels[offset], label -> new ArrayList<Integer>())
                    .add(distances[offset]);
        }
        for (List<Integer> values : result.values()) {
            Collections.sort(values);
        }
        return result;
    }

    private static Map<Long, List<Integer>> rows(VectorRangeSearchResult result, int queryIndex) {
        Map<Long, List<Integer>> rows = new HashMap<Long, List<Integer>>();
        for (int hitIndex = result.queryStart(queryIndex);
                hitIndex < result.queryEnd(queryIndex);
                hitIndex++) {
            rows.computeIfAbsent(result.labelAt(hitIndex), label -> new ArrayList<Integer>())
                    .add(Float.floatToRawIntBits(result.rawDistanceAt(hitIndex)));
        }
        for (List<Integer> values : rows.values()) {
            Collections.sort(values);
        }
        return rows;
    }

    private static int readBits(Scanner values) {
        long value = values.nextLong();
        require(value >= 0 && value <= 0xffff_ffffL, "f32 bits");
        return (int) value;
    }

    private static long[] readLongs(Scanner values, int count) {
        long[] result = new long[count];
        for (int offset = 0; offset < count; offset++) {
            result[offset] = values.nextLong();
        }
        return result;
    }

    private static void require(boolean condition, String description) {
        if (!condition) {
            throw new AssertionError(description);
        }
    }
}
