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
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <thread>
#include <type_traits>
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

#include "../c/range_test_support.h"

static_assert(!std::is_aggregate_v<paimon::vindex::DistanceBand>);
static_assert(!std::is_default_constructible_v<paimon::vindex::DistanceBand>);
static_assert(!std::is_constructible_v<paimon::vindex::DistanceBand,
                                     uint32_t, uint32_t, float, uint32_t, float>);
static_assert(!std::is_constructible_v<paimon::vindex::DistanceBand, PaimonVindexRawDistanceBand>);
static_assert(std::is_same_v<decltype(paimon::vindex::RangeSearchResult::raw_distances),
                             std::vector<float>>);
static_assert(std::is_same_v<decltype(paimon::vindex::SearchResult::distances), std::vector<float>>);

template <typename Result, typename = void>
struct HasDistances : std::false_type {};

template <typename Result>
struct HasDistances<Result, std::void_t<decltype(std::declval<Result>().distances)>> : std::true_type {};

static_assert(!HasDistances<paimon::vindex::RangeSearchResult>::value);
static_assert(HasDistances<paimon::vindex::SearchResult>::value);

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

    paimon::vindex::Reader* active_reader = nullptr;
    bool reentrant_attempted = false;
    bool reentrant_rejected = false;
    auto input = make_input(buf);
    if (expected_index_type == PAIMON_VINDEX_INDEX_TYPE_IVF_FLAT) {
        auto base_read = input.read_ranges_fn;
        input.read_ranges_fn =
            [&, base_read](paimon::vindex::ReadRequest* requests, size_t request_count) {
                if (active_reader != nullptr && !reentrant_attempted) {
                    reentrant_attempted = true;
                    try {
                        active_reader->metadata();
                    } catch (const paimon::vindex::Error& error) {
                        reentrant_rejected =
                            std::string(error.what()).find("reentrant native-handle operation") !=
                            std::string::npos;
                    }
                }
                return base_read(requests, request_count);
            };
    }
    paimon::vindex::Reader reader(
        std::move(input),
        static_cast<size_t>(4ULL * 1024 * 1024 * 1024));
    active_reader = &reader;
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
    if (expected_index_type == PAIMON_VINDEX_INDEX_TYPE_IVF_FLAT) {
        ASSERT_TRUE(reentrant_attempted);
        ASSERT_TRUE(reentrant_rejected);
    }

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
    batch_params.ivfpq_batch_table_reuse = PAIMON_VINDEX_IVFPQ_BATCH_TABLE_REUSE_OFF;
    batch_params.ivfpq_batch_table_reuse_max_bytes = 1;
    auto batch = reader.search_batch(queries.data(), 2, batch_params);
    ASSERT_EQ(batch.ids.size(), 2);
    assert_id_in_cluster(batch.ids[0], 0);
    assert_id_in_cluster(batch.ids[1], 1);
    ASSERT_EQ(reader.supports_range_search(), expected_index_type != PAIMON_VINDEX_INDEX_TYPE_DISKANN);
    if (expected_index_type == PAIMON_VINDEX_INDEX_TYPE_DISKANN) {
        bool rejected = false;
        try {
            reader.range_search(query, paimon::vindex::RangeSearchParams{});
        } catch (const paimon::vindex::Error&) {
            rejected = true;
        }
        ASSERT_TRUE(rejected);
    }
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

static void test_worker_callback_reentry_is_rejected() {
    int callback_context = 0;
    paimon::vindex::detail::NativeHandleMutex mutex;
    mutex.set_callback_context(&callback_context);
    std::atomic<bool> rejected(false);

    std::lock_guard<paimon::vindex::detail::NativeHandleMutex> operation(mutex);
    std::thread callback_worker([&]() {
        paimon::vindex::detail::NativeCallbackScope callback_scope(&callback_context);
        try {
            std::lock_guard<paimon::vindex::detail::NativeHandleMutex> reentrant(mutex);
        } catch (const paimon::vindex::Error& error) {
            rejected.store(
                std::string(error.what()).find("reentrant native-handle operation") !=
                    std::string::npos,
                std::memory_order_relaxed);
        }
    });
    callback_worker.join();
    ASSERT_TRUE(rejected.load(std::memory_order_relaxed));
    printf("PASS worker_callback_reentry_is_rejected\n");
}

static void test_extensible_search_params_forward_query_tuning() {
    auto params = paimon::vindex::SearchParams::automatic(10);
    params.max_initial_filter_expansion_factor = 4;
    params.ivfpq_batch_table_reuse = PAIMON_VINDEX_IVFPQ_BATCH_TABLE_REUSE_ON;
    params.ivfpq_batch_table_reuse_max_bytes = 32 * 1024 * 1024;

    auto raw = params.to_ffi_ex();
    ASSERT_EQ(raw.struct_size, PAIMON_VINDEX_SEARCH_PARAMS_EX_V1_SIZE);
    ASSERT_EQ(raw.max_initial_filter_expansion_factor, 4);
    ASSERT_EQ(raw.ivfpq_batch_table_reuse, PAIMON_VINDEX_IVFPQ_BATCH_TABLE_REUSE_ON);
    ASSERT_EQ(raw.ivfpq_batch_table_reuse_max_bytes, 32 * 1024 * 1024);
    printf("PASS extensible_search_params_forward_query_tuning\n");
}

template <typename Operation>
static void assert_range_error(Operation operation) {
    bool rejected = false;
    try {
        operation();
    } catch (const paimon::vindex::Error& error) {
        rejected = !std::string(error.what()).empty();
    }
    ASSERT_TRUE(rejected);
}

static PaimonVindexRangeSearchResultView range_view(
        const paimon::vindex::RangeSearchResult& result) {
    return {result.query_count, result.labels.size(), result.lims.data(),
            result.labels.data(), result.raw_distances.data(), result.stats.data(),
            result.list_reads};
}

static paimon::vindex::RangeSearchParams range_cpp_params(
        PaimonVindexRangeSearchParams raw) {
    return {paimon::vindex::DistanceBand::from_raw(
                raw.band.metric, raw.band.raw_lower_kind, raw.band.raw_lower,
                raw.band.raw_upper_kind, raw.band.raw_upper), raw.nprobe};
}

static void consume_range_fixture(const RangeFixture* fixture) {
    MemBuffer buffer;
    buffer.data.assign(fixture->index_data, fixture->index_data + fixture->index_len);
    paimon::vindex::RangeSearchResult result;
    {
        paimon::vindex::Reader reader(make_input(buffer));
        ASSERT_TRUE(reader.supports_range_search());
        ASSERT_EQ(reader.metadata().dimension, fixture->dimension);
        auto params = range_cpp_params(fixture->params);
        const size_t query_len = fixture->dimension * fixture->query_count;
        if (fixture->query_count == 1) {
            result = fixture->filter_len == 0
                ? reader.range_search(fixture->queries, query_len, params)
                : reader.range_search_with_roaring_filter(
                      fixture->queries, query_len, params, fixture->filter, fixture->filter_len);
        } else {
            result = fixture->filter_len == 0
                ? reader.range_search_batch(fixture->queries, query_len, fixture->query_count, params)
                : reader.range_search_batch_with_roaring_filter(
                      fixture->queries, query_len, fixture->query_count, params,
                      fixture->filter, fixture->filter_len);
        }
    }
    auto view = range_view(result);
    range_fixture_assert(fixture, &view);
}

static void test_range_raw_factory() {
    using namespace paimon::vindex;
    for (uint32_t metric : {PAIMON_VINDEX_METRIC_L2, PAIMON_VINDEX_METRIC_INNER_PRODUCT}) {
        const float raw_lower = metric == PAIMON_VINDEX_METRIC_L2 ? 4.0f : -6.0f;
        const float raw_upper = metric == PAIMON_VINDEX_METRIC_L2 ? 9.0f : -5.0f;
        const auto band = DistanceBand::from_raw(
            metric, PAIMON_VINDEX_BOUND_FINITE, raw_lower, PAIMON_VINDEX_BOUND_FINITE, raw_upper);
        ASSERT_EQ(band.metric(), metric);
        ASSERT_EQ(band.raw_lower_kind(), PAIMON_VINDEX_BOUND_FINITE);
        ASSERT_EQ(band.raw_lower(), raw_lower);
        ASSERT_EQ(band.raw_upper_kind(), PAIMON_VINDEX_BOUND_FINITE);
        ASSERT_EQ(band.raw_upper(), raw_upper);
        auto raw = band.to_ffi();
        ASSERT_EQ(raw.metric, band.metric());
        ASSERT_EQ(raw.raw_lower_kind, band.raw_lower_kind());
        ASSERT_EQ(range_float_bits(raw.raw_lower), range_float_bits(raw_lower));
        ASSERT_EQ(raw.raw_upper_kind, band.raw_upper_kind());
        ASSERT_EQ(range_float_bits(raw.raw_upper), range_float_bits(raw_upper));
    }
    const RangeSearchParams defaults;
    ASSERT_EQ(defaults.band.metric(), PAIMON_VINDEX_METRIC_L2);
    ASSERT_EQ(defaults.band.raw_lower_kind(), PAIMON_VINDEX_BOUND_UNBOUNDED);
    ASSERT_EQ(defaults.band.raw_upper_kind(), PAIMON_VINDEX_BOUND_UNBOUNDED);
    ASSERT_EQ(defaults.nprobe, 1);
    printf("PASS range_raw_factory\n");
}

static void test_range_endpoint_raw_results() {
    using namespace paimon::vindex;
    const char* metrics[] = {"l2", "inner_product"};
    const uint32_t metric_codes[] = {PAIMON_VINDEX_METRIC_L2, PAIMON_VINDEX_METRIC_INNER_PRODUCT};
    const float coordinates[][3] = {{3.0f, 4.0f, 5.0f}, {3.0f, 2.5f, 2.0f}};
    const float expected_raw_distances[][2] = {{9.0f, 16.0f}, {-6.0f, -5.0f}};
    const std::vector<int64_t> labels = {
        INT64_C(1) << 40, (INT64_C(1) << 40) + 1, (INT64_C(1) << 40) + 2};
    for (size_t metric_index = 0; metric_index < 2; ++metric_index) {
        RangeSearchResult result;
        {
            std::vector<float> data(3 * RANGE_DIMENSION);
            std::vector<float> query(RANGE_DIMENSION);
            for (size_t row = 0; row < 3; ++row) {
                data[row * RANGE_DIMENSION] = coordinates[metric_index][row];
            }
            if (metric_codes[metric_index] == PAIMON_VINDEX_METRIC_INNER_PRODUCT) query[0] = 2.0f;
            Trainer trainer({{"index.type", "ivf_flat"}, {"dimension", "8"},
                             {"nlist", "1"}, {"metric", metrics[metric_index]}});
            Writer writer(trainer.add_training_vectors(data.data(), 3).finish_training());
            writer.add_vectors(labels.data(), data.data(), 3);
            MemBuffer buffer;
            writer.write_index(make_output(buffer));
            Reader reader(make_input(buffer));
            const auto band = metric_index == 0
                ? DistanceBand::from_endpoints(metric_codes[metric_index], std::nullopt,
                                              DistanceEndpoint{4.0, PAIMON_VINDEX_CUT_LE})
                : DistanceBand::from_endpoints(metric_codes[metric_index],
                                              DistanceEndpoint{5.0, PAIMON_VINDEX_CUT_GE});
            result = reader.range_search(query, RangeSearchParams{band, 1});
            const auto raw_band = DistanceBand::from_raw(
                band.metric(), band.raw_lower_kind(), band.raw_lower(),
                band.raw_upper_kind(), band.raw_upper());
            auto raw_result = reader.range_search(query, RangeSearchParams{raw_band, 1});
            ASSERT_TRUE(result.labels == raw_result.labels);
            ASSERT_TRUE(result.raw_distances == raw_result.raw_distances);
        }
        const auto view = range_view(result);
        range_assert_shape(&view);
        ASSERT_EQ(result.query_count, 1);
        ASSERT_EQ(result.labels.size(), 2);
        ASSERT_EQ(result.raw_distances.size(), 2);
        for (size_t expected = 0; expected < 2; ++expected) {
            ASSERT_EQ(std::count(result.labels.begin(), result.labels.end(), labels[expected]), 1);
            auto found = std::find(result.labels.begin(), result.labels.end(), labels[expected]);
            size_t hit = static_cast<size_t>(found - result.labels.begin());
            ASSERT_EQ(range_float_bits(result.raw_distances[hit]),
                      range_float_bits(expected_raw_distances[metric_index][expected]));
        }
        printf("PASS range_endpoint_raw_results %s\n", metrics[metric_index]);
    }
}

static void test_range_endpoints() {
    using namespace paimon::vindex;
    for (uint32_t metric : {PAIMON_VINDEX_METRIC_L2, PAIMON_VINDEX_METRIC_COSINE,
                            PAIMON_VINDEX_METRIC_INNER_PRODUCT}) {
        for (uint32_t lower_op : {PAIMON_VINDEX_CUT_GE, PAIMON_VINDEX_CUT_GT}) {
            for (uint32_t upper_op : {PAIMON_VINDEX_CUT_LE, PAIMON_VINDEX_CUT_LT}) {
                auto band = DistanceBand::from_endpoints(
                    metric, DistanceEndpoint{0.5, lower_op}, DistanceEndpoint{1.0, upper_op});
                for (float raw_distance : {-1.0f, -0.5f, -0.0f, 0.0f, 0.25f,
                                       std::nextafter(0.25f, 0.0f), 0.5f, 1.0f,
                                       std::nextafter(1.0f, 2.0f), 4.0f}) {
                    if (metric == PAIMON_VINDEX_METRIC_L2 && raw_distance < 0) continue;
                    double public_value = metric == PAIMON_VINDEX_METRIC_L2
                        ? static_cast<double>(std::sqrt(raw_distance))
                        : metric == PAIMON_VINDEX_METRIC_INNER_PRODUCT
                            ? -static_cast<double>(raw_distance) : static_cast<double>(raw_distance);
                    bool expected = (lower_op == PAIMON_VINDEX_CUT_GE
                        ? public_value >= 0.5 : public_value > 0.5) &&
                        (upper_op == PAIMON_VINDEX_CUT_LE ? public_value <= 1.0 : public_value < 1.0);
                    ASSERT_EQ(expected,
                        (band.raw_lower_kind() == PAIMON_VINDEX_BOUND_UNBOUNDED || raw_distance >= band.raw_lower()) &&
                        (band.raw_upper_kind() == PAIMON_VINDEX_BOUND_UNBOUNDED || raw_distance < band.raw_upper()));
                }
            }
        }
        auto unbounded = DistanceBand::from_endpoints(metric);
        ASSERT_EQ(unbounded.raw_lower_kind(), PAIMON_VINDEX_BOUND_UNBOUNDED);
        ASSERT_EQ(unbounded.raw_upper_kind(), PAIMON_VINDEX_BOUND_UNBOUNDED);
        auto precise = DistanceBand::from_endpoints(
            metric, DistanceEndpoint{std::nextafter(1.0, 2.0), PAIMON_VINDEX_CUT_GE});
        float raw_boundary = metric == PAIMON_VINDEX_METRIC_INNER_PRODUCT ? -1.0f : 1.0f;
        ASSERT_TRUE(!((precise.raw_lower_kind() == PAIMON_VINDEX_BOUND_UNBOUNDED || raw_boundary >= precise.raw_lower()) &&
                      (precise.raw_upper_kind() == PAIMON_VINDEX_BOUND_UNBOUNDED || raw_boundary < precise.raw_upper())));
    }
    assert_range_error([] {
        DistanceBand::from_endpoints(PAIMON_VINDEX_METRIC_L2,
            DistanceEndpoint{1.0, PAIMON_VINDEX_CUT_LT});
    });
    assert_range_error([] {
        DistanceBand::from_endpoints(PAIMON_VINDEX_METRIC_L2, std::nullopt,
            DistanceEndpoint{std::numeric_limits<double>::infinity(), PAIMON_VINDEX_CUT_LE});
    });
    printf("PASS range_endpoints\n");
}

static void test_range_matrix() {
    using namespace paimon::vindex;
    const std::pair<const char*, uint32_t> metrics[] = {
        {"l2", PAIMON_VINDEX_METRIC_L2}, {"cosine", PAIMON_VINDEX_METRIC_COSINE},
        {"inner_product", PAIMON_VINDEX_METRIC_INNER_PRODUCT}};
    std::vector<float> data(RANGE_VECTOR_COUNT * RANGE_DIMENSION);
    std::vector<int64_t> labels(RANGE_VECTOR_COUNT);
    std::vector<float> queries(RANGE_QUERY_COUNT * RANGE_DIMENSION);
    range_fill_data(data.data(), labels.data(), queries.data());
    const std::vector<float> query(queries.begin(), queries.begin() + RANGE_DIMENSION);
    for (const char* index_type : {"ivf_flat", "ivf_sq", "ivf_pq", "ivf_rq"}) {
        for (const auto& metric : metrics) {
            Trainer trainer({{"index.type", index_type}, {"dimension", "8"},
                             {"nlist", "4"}, {"metric", metric.first}});
            Writer writer(trainer.add_training_vectors(data.data(), RANGE_VECTOR_COUNT).finish_training());
            writer.add_vectors(labels.data(), data.data(), RANGE_VECTOR_COUNT);
            MemBuffer buffer;
            writer.write_index(make_output(buffer));
            RangeSearchResult retained;
            {
                Reader reader(make_input(buffer));
                ASSERT_TRUE(reader.supports_range_search());
                auto params = range_cpp_params(range_all_params(metric.second));
                auto batch = reader.range_search_batch(queries, RANGE_QUERY_COUNT, params);
                auto batch_view = range_view(batch);
                range_assert_shape(&batch_view);
                ASSERT_EQ(batch.query_count, RANGE_QUERY_COUNT);
                ASSERT_EQ(batch.labels.size(), RANGE_QUERY_COUNT * RANGE_VECTOR_COUNT);
                ASSERT_TRUE(batch.list_reads > 0);
                for (const auto& stats : batch.stats) {
                    ASSERT_EQ(stats.lists_probed, RANGE_NLIST);
                    ASSERT_EQ(stats.rows_scanned, RANGE_VECTOR_COUNT);
                    ASSERT_EQ(stats.rows_committed, RANGE_VECTOR_COUNT);
                    ASSERT_EQ(stats.early_abandoned, 0);
                }
                retained = reader.range_search(query, params);
                ASSERT_EQ(retained.labels.size(), RANGE_VECTOR_COUNT);
                auto topk = reader.search(query.data(), SearchParams{5, RANGE_NLIST});
                ASSERT_EQ(topk.ids.size(), 5);
                for (int64_t label : topk.ids) {
                    ASSERT_TRUE(std::find(retained.labels.begin(), retained.labels.end(), label) != retained.labels.end());
                }
                auto filtered = reader.range_search_batch_with_roaring_filter(
                    queries, RANGE_QUERY_COUNT, params, range_filter, sizeof(range_filter));
                auto filtered_view = range_view(filtered);
                range_assert_shape(&filtered_view);
                ASSERT_EQ(filtered.labels.size(), RANGE_QUERY_COUNT * 2);
                for (int64_t label : filtered.labels) {
                    ASSERT_TRUE(label == labels[1] || label == labels[3]);
                }
                auto filtered_single = reader.range_search_with_roaring_filter(
                    query, params, range_filter, sizeof(range_filter));
                ASSERT_EQ(filtered_single.labels.size(), 2);
                auto empty_filter = reader.range_search_with_roaring_filter(
                    query, params, range_empty_filter, sizeof(range_empty_filter));
                ASSERT_TRUE(empty_filter.labels.empty());
                ASSERT_TRUE(empty_filter.lims == std::vector<size_t>({0, 0}));
                auto bounded = params;
                auto bounds = std::minmax_element(retained.raw_distances.begin(), retained.raw_distances.end());
                bounded.band = DistanceBand::from_raw(
                    metric.second, PAIMON_VINDEX_BOUND_FINITE,
                    metric.second == PAIMON_VINDEX_METRIC_L2 ? std::max(0.0f, *bounds.first) : *bounds.first,
                    PAIMON_VINDEX_BOUND_FINITE, *bounds.first + (*bounds.second - *bounds.first) / 2.0f);
                auto subset = reader.range_search_batch(queries, RANGE_QUERY_COUNT, bounded);
                auto subset_view = range_view(subset);
                range_assert_shape(&subset_view);
                ASSERT_TRUE(subset.labels.size() > 0 && subset.labels.size() < batch.labels.size());
                for (size_t query_index = 0; query_index < RANGE_QUERY_COUNT; ++query_index) {
                    size_t expected = 0;
                    for (size_t hit = batch.lims[query_index]; hit < batch.lims[query_index + 1]; ++hit) {
                        if (batch.raw_distances[hit] >= bounded.band.raw_lower() &&
                            batch.raw_distances[hit] < bounded.band.raw_upper()) {
                            ++expected;
                            ASSERT_TRUE(std::find(subset.labels.begin() + subset.lims[query_index],
                                subset.labels.begin() + subset.lims[query_index + 1], batch.labels[hit]) !=
                                subset.labels.begin() + subset.lims[query_index + 1]);
                        }
                    }
                    ASSERT_EQ(subset.lims[query_index + 1] - subset.lims[query_index], expected);
                }
                bounded.band = DistanceBand::from_raw(
                    metric.second, PAIMON_VINDEX_BOUND_FINITE, bounded.band.raw_upper(),
                    PAIMON_VINDEX_BOUND_FINITE, bounded.band.raw_upper());
                auto empty = reader.range_search_batch(queries, RANGE_QUERY_COUNT, bounded);
                ASSERT_TRUE(empty.labels.empty() && empty.raw_distances.empty());
                ASSERT_TRUE(empty.lims == std::vector<size_t>({0, 0, 0, 0}));
                ASSERT_EQ(empty.list_reads, 0);
                assert_range_error([&] { reader.range_search(nullptr, RANGE_DIMENSION, params); });
                assert_range_error([&] { reader.range_search(std::vector<float>{}, params); });
                assert_range_error([&] { reader.range_search_batch(queries, 2, params); });
                assert_range_error([&] { reader.range_search_batch(nullptr, 0, 0, params); });
                assert_range_error([&] {
                    reader.range_search_batch(nullptr, 0, SIZE_MAX / RANGE_DIMENSION + 1, params);
                });
                assert_range_error([&] {
                    reader.range_search_with_roaring_filter(
                        nullptr, RANGE_DIMENSION, params, range_filter, sizeof(range_filter));
                });
                assert_range_error([&] {
                    reader.range_search_with_roaring_filter(
                        std::vector<float>{}, params, range_filter, sizeof(range_filter));
                });
                assert_range_error([&] {
                    reader.range_search_with_roaring_filter(
                        query.data(), SIZE_MAX, params, range_filter, sizeof(range_filter));
                });
                assert_range_error([&] {
                    reader.range_search_batch_with_roaring_filter(
                        nullptr, queries.size(), RANGE_QUERY_COUNT, params,
                        range_filter, sizeof(range_filter));
                });
                assert_range_error([&] {
                    reader.range_search_batch_with_roaring_filter(
                        queries, 2, params, range_filter, sizeof(range_filter));
                });
                assert_range_error([&] {
                    reader.range_search_batch_with_roaring_filter(
                        nullptr, 0, 0, params, range_filter, sizeof(range_filter));
                });
                assert_range_error([&] {
                    reader.range_search_batch_with_roaring_filter(
                        queries.data(), 0, SIZE_MAX / RANGE_DIMENSION + 1, params,
                        range_filter, sizeof(range_filter));
                });
                assert_range_error([&] { reader.range_search_with_roaring_filter(query, params, nullptr, 1); });
                assert_range_error([&] { reader.range_search_with_roaring_filter(query, params, range_filter, 1); });
                auto invalid = params;
                invalid.nprobe = 0;
                assert_range_error([&] { reader.range_search(query, invalid); });
                invalid = bounded;
                invalid.nprobe = 0;
                assert_range_error([&] { reader.range_search(query, invalid); });
                invalid = params;
                invalid.band = DistanceBand::from_raw(
                    (metric.second + 1) % 3, params.band.raw_lower_kind(), params.band.raw_lower(),
                    params.band.raw_upper_kind(), params.band.raw_upper());
                assert_range_error([&] { reader.range_search(query, invalid); });
                auto nan_query = query;
                nan_query[0] = std::numeric_limits<float>::quiet_NaN();
                assert_range_error([&] { reader.range_search(nan_query, params); });
                Reader moved(std::move(reader));
                assert_range_error([&] { reader.supports_range_search(); });
                assert_range_error([&] { reader.range_search(query, params); });
                ASSERT_TRUE(moved.supports_range_search());
                ASSERT_EQ(moved.range_search(query, params).labels.size(), retained.labels.size());
            }
            auto retained_view = range_view(retained);
            range_assert_shape(&retained_view);
            ASSERT_EQ(retained.labels.size(), RANGE_VECTOR_COUNT);
            printf("PASS range_matrix %s %s\n", index_type, metric.first);
        }
    }
}

int main() {
    test_supported_index_roundtrips();
    test_worker_callback_reentry_is_rejected();
    test_extensible_search_params_forward_query_tuning();
    test_range_endpoints();
    test_range_raw_factory();
    test_range_endpoint_raw_results();
    test_range_matrix();
    range_fixture_run_all(consume_range_fixture);
    return 0;
}
