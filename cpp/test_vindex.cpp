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

constexpr size_t kRoundtripDimension = 8;
constexpr size_t kRoundtripNlist = 4;
constexpr size_t kRoundtripPerList = 128;
constexpr size_t kRoundtripVectorCount = kRoundtripNlist * kRoundtripPerList;

// Portable RoaringTreemap payloads for {100000, 100001} and {100000}.
const std::vector<uint8_t> kRoaringIncludeClusterZero = {
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 58, 48, 0, 0,
    1, 0, 0, 0, 1, 0, 1, 0, 16, 0, 0, 0, 160, 134, 161, 134,
};
const std::vector<uint8_t> kRoaringExcludeClusterZeroFirst = {
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 58, 48, 0,
    0, 1, 0, 0, 0, 1, 0, 0, 0, 16, 0, 0, 0, 160, 134,
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

static int64_t cluster_base_id(size_t cluster) {
    return static_cast<int64_t>((cluster + 1) * 100000);
}

static std::vector<float> roundtrip_data() {
    std::vector<float> data(kRoundtripVectorCount * kRoundtripDimension);
    for (size_t i = 0; i < kRoundtripVectorCount; i++) {
        size_t cluster = i / kRoundtripPerList;
        size_t local = i % kRoundtripPerList;
        float center = static_cast<float>(cluster) * 20.0f;
        for (size_t dim = 0; dim < kRoundtripDimension; dim++) {
            data[i * kRoundtripDimension + dim] =
                center + static_cast<float>(dim) * 0.01f +
                static_cast<float>(local % 16) * 0.001f;
        }
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

static std::vector<float> query_for_center(float center) {
    std::vector<float> query(kRoundtripDimension);
    for (size_t dim = 0; dim < kRoundtripDimension; dim++) {
        query[dim] = center + static_cast<float>(dim) * 0.01f;
    }
    return query;
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
    ASSERT_EQ(trainer.dimension(), kRoundtripDimension);
    paimon::vindex::Training training =
        trainer.add_training_vectors(data.data(), kRoundtripVectorCount).finish_training();

    paimon::vindex::Writer writer(std::move(training));
    ASSERT_EQ(writer.dimension(), kRoundtripDimension);
    writer.add_vectors(ids.data(), data.data(), kRoundtripVectorCount);

    MemBuffer buf;
    writer.write_index(make_output(buf));
    ASSERT_TRUE(!buf.data.empty());

    paimon::vindex::Reader reader(make_input(buf));
    auto metadata = reader.metadata();
    ASSERT_EQ(metadata.index_type, expected_index_type);
    ASSERT_EQ(metadata.dimension, kRoundtripDimension);
    ASSERT_EQ(metadata.nlist, 4);
    ASSERT_EQ(metadata.metric, PAIMON_VINDEX_METRIC_L2);
    ASSERT_EQ(metadata.total_vectors, kRoundtripVectorCount);
    ASSERT_EQ(metadata.pq_m, expected_pq_m);
    ASSERT_EQ(metadata.hnsw_m, expected_hnsw_m);

    reader.optimize_for_search();

    auto query = query_for_center(0.0f);
    auto result = reader.search(query.data(), paimon::vindex::SearchParams{2, 4, 16});
    ASSERT_EQ(result.ids.size(), 2);
    assert_id_in_cluster(result.ids[0], 0);
    ASSERT_TRUE(std::isfinite(result.distances[0]));
    if (expected_index_type == PAIMON_VINDEX_INDEX_TYPE_IVF_RQ) {
        auto query_bits_result =
            reader.search(query.data(), paimon::vindex::SearchParams{2, 4, 16, 4});
        assert_id_in_cluster(query_bits_result.ids[0], 0);
        ASSERT_TRUE(std::isfinite(query_bits_result.distances[0]));
    }

    auto filtered = reader.search_with_roaring_filter_and_exclusions(
        query.data(),
        paimon::vindex::SearchParams{1, 4, 16},
        kRoaringIncludeClusterZero.data(),
        kRoaringIncludeClusterZero.size(),
        kRoaringExcludeClusterZeroFirst.data(),
        kRoaringExcludeClusterZeroFirst.size());
    ASSERT_EQ(filtered.ids[0], 100001);

    auto exclusion_only = reader.search_with_roaring_filter_and_exclusions(
        query.data(),
        paimon::vindex::SearchParams{1, 4, 16},
        nullptr,
        0,
        kRoaringExcludeClusterZeroFirst.data(),
        kRoaringExcludeClusterZeroFirst.size());
    assert_id_in_cluster(exclusion_only.ids[0], 0);
    ASSERT_TRUE(exclusion_only.ids[0] != 100000);

    auto query0 = query_for_center(0.0f);
    auto query1 = query_for_center(20.0f);
    std::vector<float> queries;
    queries.insert(queries.end(), query0.begin(), query0.end());
    queries.insert(queries.end(), query1.begin(), query1.end());
    auto batch = reader.search_batch(queries.data(), 2, paimon::vindex::SearchParams{1, 4, 16});
    ASSERT_EQ(batch.ids.size(), 2);
    assert_id_in_cluster(batch.ids[0], 0);
    assert_id_in_cluster(batch.ids[1], 1);
    if (expected_index_type == PAIMON_VINDEX_INDEX_TYPE_IVF_RQ) {
        auto query_bits_batch =
            reader.search_batch(queries.data(), 2, paimon::vindex::SearchParams{1, 4, 16, 8});
        assert_id_in_cluster(query_bits_batch.ids[0], 0);
        assert_id_in_cluster(query_bits_batch.ids[1], 1);
    }
    auto filtered_batch = reader.search_batch_with_roaring_filter_and_exclusions(
        queries.data(),
        2,
        paimon::vindex::SearchParams{1, 4, 16},
        kRoaringIncludeClusterZero.data(),
        kRoaringIncludeClusterZero.size(),
        kRoaringExcludeClusterZeroFirst.data(),
        kRoaringExcludeClusterZeroFirst.size());
    ASSERT_EQ(filtered_batch.ids[0], 100001);
    ASSERT_EQ(filtered_batch.ids[1], 100001);
    printf("PASS %s\n", name);
}

static void test_supported_index_roundtrips() {
    run_roundtrip(
        "ivf_flat_roundtrip",
        {
            {"index.type", "ivf_flat"},
            {"dimension", "8"},
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
            {"dimension", "8"},
            {"nlist", "4"},
            {"metric", "l2"},
            {"pq.m", "4"},
        },
        PAIMON_VINDEX_INDEX_TYPE_IVF_PQ,
        4,
        0);

    run_roundtrip(
        "ivf_rq_roundtrip",
        {
            {"index.type", "ivf_rq"},
            {"dimension", "8"},
            {"nlist", "4"},
            {"metric", "l2"},
        },
        PAIMON_VINDEX_INDEX_TYPE_IVF_RQ,
        0,
        0);

    run_roundtrip(
        "ivf_hnsw_flat_roundtrip",
        {
            {"index.type", "ivf_hnsw_flat"},
            {"dimension", "8"},
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
            {"dimension", "8"},
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
