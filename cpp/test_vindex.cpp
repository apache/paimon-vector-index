/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include "paimon_vindex.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#define ASSERT_EQ(a, b) do { \
    if ((a) != (b)) { \
        fprintf(stderr, "FAIL %s:%d: %s != %s\n", __FILE__, __LINE__, #a, #b); \
        abort(); \
    } \
} while (0)

#define ASSERT_TRUE(x) do { \
    if (!(x)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); \
        abort(); \
    } \
} while (0)

struct MemBuffer {
    std::vector<uint8_t> data;
    size_t pos = 0;
};

static paimon::vindex::OutputFile make_output(MemBuffer& buf) {
    paimon::vindex::OutputFile out;
    out.write_fn = [&buf](const uint8_t* data, size_t len) -> int {
        buf.data.insert(buf.data.end(), data, data + len);
        buf.pos += len;
        return 0;
    };
    out.flush_fn = []() -> int { return 0; };
    out.get_pos_fn = [&buf]() -> int64_t { return static_cast<int64_t>(buf.pos); };
    return out;
}

static paimon::vindex::InputFile make_input(const MemBuffer& buf) {
    paimon::vindex::InputFile in;
    in.read_at_fn = [&buf](uint64_t offset, uint8_t* dst, size_t len) -> int {
        if (offset + len > buf.data.size()) return -1;
        memcpy(dst, buf.data.data() + offset, len);
        return 0;
    };
    return in;
}

static void test_basic_roundtrip() {
    std::vector<std::pair<std::string, std::string>> options = {
        {"index.type", "ivf_flat"},
        {"dimension", "2"},
        {"nlist", "2"},
        {"metric", "l2"},
    };
    paimon::vindex::Writer writer(options);
    ASSERT_EQ(writer.dimension(), 2);

    std::vector<float> data = {
        0.0f, 0.0f,
        1.0f, 0.0f,
        10.0f, 10.0f,
        11.0f, 10.0f,
    };
    std::vector<int64_t> ids = {100, 101, 200, 201};
    writer.train(data.data(), 4);
    writer.add_vectors(ids.data(), data.data(), 4);

    MemBuffer buf;
    writer.write_index(make_output(buf));
    ASSERT_TRUE(!buf.data.empty());

    paimon::vindex::Reader reader(make_input(buf));
    auto metadata = reader.metadata();
    ASSERT_EQ(metadata.index_type, PAIMON_VINDEX_INDEX_TYPE_IVF_FLAT);
    ASSERT_EQ(metadata.dimension, 2);
    ASSERT_EQ(metadata.nlist, 2);
    ASSERT_EQ(metadata.metric, PAIMON_VINDEX_METRIC_L2);
    ASSERT_EQ(metadata.total_vectors, 4);

    reader.optimize_for_search();

    const float query[] = {0.0f, 0.0f};
    auto result = reader.search(query, 2, 2);
    ASSERT_EQ(result.ids.size(), 2);
    ASSERT_EQ(result.ids[0], 100);
    ASSERT_TRUE(std::isfinite(result.distances[0]));

    const float queries[] = {0.0f, 0.0f, 10.0f, 10.0f};
    auto batch = reader.search_batch(queries, 2, 1, 2);
    ASSERT_EQ(batch.ids.size(), 2);
    ASSERT_EQ(batch.ids[0], 100);
    ASSERT_EQ(batch.ids[1], 200);
    printf("PASS test_basic_roundtrip\n");
}

int main() {
    test_basic_roundtrip();
    return 0;
}
