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

constexpr size_t kRoundtripDimension = 2;
constexpr size_t kRoundtripNlist = 4;
constexpr size_t kRoundtripPerList = 128;
constexpr size_t kRoundtripVectorCount = kRoundtripNlist * kRoundtripPerList;

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

static int64_t cluster_base_id(size_t cluster) {
    return static_cast<int64_t>((cluster + 1) * 100000);
}

static std::vector<float> roundtrip_data() {
    std::vector<float> data(kRoundtripVectorCount * kRoundtripDimension);
    for (size_t i = 0; i < kRoundtripVectorCount; i++) {
        size_t cluster = i / kRoundtripPerList;
        size_t local = i % kRoundtripPerList;
        float center = static_cast<float>(cluster) * 20.0f;
        data[i * kRoundtripDimension] = center + static_cast<float>(local % 16) * 0.001f;
        data[i * kRoundtripDimension + 1] = center + static_cast<float>(local / 16) * 0.001f;
    }
    return data;
}

static std::vector<int64_t> roundtrip_ids() {
    std::vector<int64_t> ids(kRoundtripVectorCount);
    for (size_t i = 0; i < kRoundtripVectorCount; i++) {
        size_t cluster = i / kRoundtripPerList;
        size_t local = i % kRoundtripPerList;
        ids[i] = cluster_base_id(cluster) + static_cast<int64_t>(local);
    }
    return ids;
}

static void assert_id_in_cluster(int64_t id, size_t cluster) {
    int64_t base = cluster_base_id(cluster);
    ASSERT_TRUE(id >= base);
    ASSERT_TRUE(id < base + static_cast<int64_t>(kRoundtripPerList));
}

static void run_roundtrip(
        const char* name,
        const std::vector<std::pair<std::string, std::string>>& options,
        uint32_t expected_index_type,
        size_t expected_pq_m,
        size_t expected_hnsw_m) {
    std::vector<float> data = roundtrip_data();
    std::vector<int64_t> ids = roundtrip_ids();
    paimon::vindex::Trainer trainer(options);
    ASSERT_EQ(trainer.dimension(), 2);
    paimon::vindex::Training training =
        trainer.add_training_vectors(data.data(), kRoundtripVectorCount).finish_training();

    paimon::vindex::Writer writer(std::move(training));
    ASSERT_EQ(writer.dimension(), 2);
    writer.add_vectors(ids.data(), data.data(), kRoundtripVectorCount);

    MemBuffer buf;
    writer.write_index(make_output(buf));
    ASSERT_TRUE(!buf.data.empty());

    paimon::vindex::Reader reader(make_input(buf));
    auto metadata = reader.metadata();
    ASSERT_EQ(metadata.index_type, expected_index_type);
    ASSERT_EQ(metadata.dimension, 2);
    ASSERT_EQ(metadata.nlist, 4);
    ASSERT_EQ(metadata.metric, PAIMON_VINDEX_METRIC_L2);
    ASSERT_EQ(metadata.total_vectors, kRoundtripVectorCount);
    ASSERT_EQ(metadata.pq_m, expected_pq_m);
    ASSERT_EQ(metadata.hnsw_m, expected_hnsw_m);

    reader.optimize_for_search();

    const float query[] = {0.0f, 0.0f};
    auto result = reader.search(query, 2, 4, 16);
    ASSERT_EQ(result.ids.size(), 2);
    assert_id_in_cluster(result.ids[0], 0);
    ASSERT_TRUE(std::isfinite(result.distances[0]));

    const float queries[] = {0.0f, 0.0f, 20.0f, 20.0f};
    auto batch = reader.search_batch(queries, 2, 1, 4, 16);
    ASSERT_EQ(batch.ids.size(), 2);
    assert_id_in_cluster(batch.ids[0], 0);
    assert_id_in_cluster(batch.ids[1], 1);
    printf("PASS %s\n", name);
}

static void test_supported_index_roundtrips() {
    run_roundtrip(
        "ivf_flat_roundtrip",
        {
            {"index.type", "ivf_flat"},
            {"dimension", "2"},
            {"nlist", "4"},
            {"metric", "l2"},
        },
        PAIMON_VINDEX_INDEX_TYPE_IVF_FLAT,
        0,
        0);

    run_roundtrip(
        "ivf_pq_roundtrip",
        {
            {"index.type", "ivf_pq"},
            {"dimension", "2"},
            {"nlist", "4"},
            {"metric", "l2"},
            {"pq.m", "1"},
        },
        PAIMON_VINDEX_INDEX_TYPE_IVF_PQ,
        1,
        0);

    run_roundtrip(
        "ivf_hnsw_flat_roundtrip",
        {
            {"index.type", "ivf_hnsw_flat"},
            {"dimension", "2"},
            {"nlist", "4"},
            {"metric", "l2"},
            {"hnsw.m", "4"},
        },
        PAIMON_VINDEX_INDEX_TYPE_IVF_HNSW_FLAT,
        0,
        4);

    run_roundtrip(
        "ivf_hnsw_sq_roundtrip",
        {
            {"index.type", "ivf_hnsw_sq"},
            {"dimension", "2"},
            {"nlist", "4"},
            {"metric", "l2"},
            {"hnsw.m", "4"},
        },
        PAIMON_VINDEX_INDEX_TYPE_IVF_HNSW_SQ,
        0,
        4);
}

int main() {
    test_supported_index_roundtrips();
    return 0;
}
