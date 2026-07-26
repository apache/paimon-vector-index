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
    mutable size_t max_read_request_count = 0;
};

constexpr size_t kRoundtripDimension = 8;
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
    in.read_ranges_fn = [&buf](
            paimon::vindex::ReadRequest* requests,
            size_t request_count) -> int {
        buf.max_read_request_count = std::max(buf.max_read_request_count, request_count);
        for (size_t i = 0; i < request_count; i++) {
            const auto& request = requests[i];
            if (request.offset + request.len > buf.data.size()) return -1;
            memcpy(request.buf, buf.data.data() + request.offset, request.len);
        }
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
        size_t expected_pq_bits) {
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

    paimon::vindex::Reader reader(
        make_input(buf),
        static_cast<size_t>(4ULL * 1024 * 1024 * 1024));
    auto metadata = reader.metadata();
    ASSERT_EQ(metadata.index_type, expected_index_type);
    ASSERT_EQ(metadata.dimension, kRoundtripDimension);
    ASSERT_EQ(
        metadata.nlist,
        expected_index_type == PAIMON_VINDEX_INDEX_TYPE_DISKANN ? 1 : 4);
    ASSERT_EQ(metadata.metric, PAIMON_VINDEX_METRIC_L2);
    ASSERT_EQ(metadata.total_vectors, kRoundtripVectorCount);
    ASSERT_EQ(metadata.pq_m, expected_pq_m);
    ASSERT_EQ(metadata.pq_bits, expected_pq_bits);
    ASSERT_EQ(
        metadata.rq_bits,
        expected_index_type == PAIMON_VINDEX_INDEX_TYPE_IVF_RQ ? 5 : 0);
    if (expected_index_type == PAIMON_VINDEX_INDEX_TYPE_DISKANN) {
        ASSERT_EQ(metadata.diskann_max_degree, 8);
        ASSERT_EQ(metadata.diskann_build_search_list_size, 16);
        ASSERT_TRUE(std::fabs(metadata.diskann_alpha - 1.2f) < 1e-6f);
        auto read_plan = reader.read_plan();
        ASSERT_EQ(read_plan.memory_budget_bytes, 4ULL * 1024 * 1024 * 1024);
        ASSERT_TRUE(read_plan.window_bytes > 0);
    }

    reader.optimize_for_search();

    auto query = query_for_center(0.0f);
    if (expected_index_type == PAIMON_VINDEX_INDEX_TYPE_DISKANN) {
        auto calibrated_width = reader.calibrate_search_width(query.data(), 1, 2);
        ASSERT_TRUE(
            calibrated_width == 100 ||
            calibrated_width == 200 ||
            calibrated_width == 400);
    }
    auto search_params = expected_index_type == PAIMON_VINDEX_INDEX_TYPE_DISKANN
        ? paimon::vindex::SearchParams::automatic(2)
        : paimon::vindex::SearchParams{2, 4};
    if (expected_index_type == PAIMON_VINDEX_INDEX_TYPE_DISKANN) {
        reader.warmup_queries(query.data(), 1, 32);
    }
    auto result = reader.search(query.data(), search_params);
    ASSERT_EQ(result.ids.size(), 2);
    assert_id_in_cluster(result.ids[0], 0);
    ASSERT_TRUE(std::isfinite(result.distances[0]));
    if (expected_index_type == PAIMON_VINDEX_INDEX_TYPE_IVF_PQ) {
        ASSERT_TRUE(buf.max_read_request_count > 1);
    }
    auto query0 = query_for_center(0.0f);
    auto query1 = query_for_center(20.0f);
    std::vector<float> queries;
    queries.insert(queries.end(), query0.begin(), query0.end());
    queries.insert(queries.end(), query1.begin(), query1.end());
    auto batch_params = expected_index_type == PAIMON_VINDEX_INDEX_TYPE_DISKANN
        ? paimon::vindex::SearchParams::diskann(1, 100)
        : paimon::vindex::SearchParams{1, 4};
    auto batch = reader.search_batch(queries.data(), 2, batch_params);
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
        },
        PAIMON_VINDEX_INDEX_TYPE_IVF_PQ,
        2,
        8);

    run_roundtrip(
        "ivf_rq_roundtrip",
        {
            {"index.type", "ivf_rq"},
            {"dimension", "8"},
            {"nlist", "4"},
            {"rq.bits", "5"},
            {"metric", "l2"},
        },
        PAIMON_VINDEX_INDEX_TYPE_IVF_RQ,
        0,
        0);

    run_roundtrip(
        "ivf_sq_roundtrip",
        {
            {"index.type", "ivf_sq"},
            {"dimension", "8"},
            {"nlist", "4"},
            {"metric", "l2"},
        },
        PAIMON_VINDEX_INDEX_TYPE_IVF_SQ,
        0,
        8);

    run_roundtrip(
        "diskann_roundtrip",
        {
            {"index.type", "diskann"},
            {"dimension", "8"},
            {"metric", "l2"},
            {"pq.m", "4"},
            {"pq.bits", "4"},
            {"diskann.max-degree", "8"},
            {"diskann.build-search-list-size", "16"},
        },
        PAIMON_VINDEX_INDEX_TYPE_DISKANN,
        4,
        4);
}

int main() {
    test_supported_index_roundtrips();
    return 0;
}
