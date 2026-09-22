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

#include "paimon_vindex.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ASSERT_TRUE(x) do { \
    if (!(x)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); \
        abort(); \
    } \
} while (0)

#define ASSERT_EQ_I64(a, b) do { \
    int64_t av = (int64_t)(a); \
    int64_t bv = (int64_t)(b); \
    if (av != bv) { \
        fprintf(stderr, "FAIL %s:%d: %s=%lld %s=%lld\n", \
                __FILE__, __LINE__, #a, (long long)av, #b, (long long)bv); \
        abort(); \
    } \
} while (0)

#include "range_test_support.h"

_Static_assert(_Generic(((PaimonVindexRangeSearchParams *)0)->band,
                       PaimonVindexRawDistanceBand: 1, default: 0),
               "range parameters must use an explicitly raw band");
_Static_assert(_Generic(((PaimonVindexRangeSearchResultView *)0)->raw_distances,
                       const float *: 1, default: 0),
               "range results must expose borrowed raw distances");

struct MemBuffer {
    uint8_t *data;
    size_t len;
    size_t cap;
    size_t pos;
    size_t max_read_request_count;
};

enum {
    ROUNDTRIP_DIMENSION = 8,
    ROUNDTRIP_NLIST = 4,
    ROUNDTRIP_PER_LIST = 128,
    ROUNDTRIP_VECTOR_COUNT = ROUNDTRIP_NLIST * ROUNDTRIP_PER_LIST,
};

struct SearchParamsExPrefix {
    uintptr_t struct_size;
    uintptr_t top_k;
    uint32_t search_width;
    uintptr_t width;
};

static void fail_ffi(const char *message) {
    const char *err = paimon_vindex_last_error();
    fprintf(stderr, "%s: %s\n", message, err == NULL ? "(no error)" : err);
    abort();
}

static void assert_last_error_contains(const char *needle) {
    const char *err = paimon_vindex_last_error();
    if (err == NULL || strstr(err, needle) == NULL) {
        fprintf(
            stderr,
            "FAIL %s:%d: last error should contain '%s', got '%s'\n",
            __FILE__,
            __LINE__,
            needle,
            err == NULL ? "(null)" : err);
        abort();
    }
}

static int mem_write(void *ctx, const uint8_t *data, uintptr_t len) {
    struct MemBuffer *buf = (struct MemBuffer *)ctx;
    if (len > SIZE_MAX - buf->len) {
        return -1;
    }
    size_t required = buf->len + (size_t)len;
    if (required > buf->cap) {
        size_t new_cap = buf->cap == 0 ? 1024 : buf->cap;
        while (new_cap < required) {
            if (new_cap > SIZE_MAX / 2) {
                return -1;
            }
            new_cap *= 2;
        }
        uint8_t *next = (uint8_t *)realloc(buf->data, new_cap);
        if (next == NULL) {
            return -1;
        }
        buf->data = next;
        buf->cap = new_cap;
    }
    memcpy(buf->data + buf->len, data, (size_t)len);
    buf->len = required;
    buf->pos += (size_t)len;
    return 0;
}

static int mem_flush(void *ctx) {
    (void)ctx;
    return 0;
}

static int64_t mem_pos(void *ctx) {
    struct MemBuffer *buf = (struct MemBuffer *)ctx;
    return (int64_t)buf->pos;
}

static int mem_read_ranges(
        void *ctx,
        struct PaimonVindexReadRequest *requests,
        uintptr_t request_count) {
    struct MemBuffer *buf = (struct MemBuffer *)ctx;
    if (request_count > buf->max_read_request_count) {
        buf->max_read_request_count = (size_t)request_count;
    }
    for (uintptr_t i = 0; i < request_count; i++) {
        uint64_t offset = requests[i].offset;
        uintptr_t len = requests[i].len;
        if (offset > SIZE_MAX || len > SIZE_MAX) {
            return -1;
        }
        size_t off = (size_t)offset;
        size_t n = (size_t)len;
        if (off > buf->len || n > buf->len - off) {
            return -1;
        }
        memcpy(requests[i].buf, buf->data + off, n);
    }
    return 0;
}

static int failing_write(void *ctx, const uint8_t *data, uintptr_t len) {
    (void)ctx;
    (void)data;
    (void)len;
    return -1;
}

static int failing_flush(void *ctx) {
    (void)ctx;
    return -1;
}

static int failing_read_ranges(
        void *ctx,
        struct PaimonVindexReadRequest *requests,
        uintptr_t request_count) {
    (void)ctx;
    (void)requests;
    (void)request_count;
    return -1;
}

static int64_t cluster_base_id(size_t cluster) {
    return (int64_t)((cluster + 1) * 100000);
}

static void fill_roundtrip_data(float *data, int64_t *ids) {
    for (size_t i = 0; i < ROUNDTRIP_VECTOR_COUNT; i++) {
        size_t cluster = i / ROUNDTRIP_PER_LIST;
        size_t local = i % ROUNDTRIP_PER_LIST;
        float center = (float)cluster * 20.0f;
        for (size_t dim = 0; dim < ROUNDTRIP_DIMENSION; dim++) {
            data[i * ROUNDTRIP_DIMENSION + dim] =
                center + (float)dim * 0.01f + (float)(local % 16) * 0.001f;
        }
        ids[i] = cluster_base_id(cluster) + (int64_t)local;
    }
}

static void fill_query(float *query, float center) {
    for (size_t dim = 0; dim < ROUNDTRIP_DIMENSION; dim++) {
        query[dim] = center + (float)dim * 0.01f;
    }
}

static void assert_id_in_cluster(int64_t id, size_t cluster) {
    int64_t base = cluster_base_id(cluster);
    ASSERT_TRUE(id >= base);
    ASSERT_TRUE(id < base + ROUNDTRIP_PER_LIST);
}

static void run_roundtrip(
        const char *name,
        const char *const *keys,
        const char *const *values,
        uintptr_t num_options,
        uint32_t expected_index_type,
        uintptr_t expected_pq_m,
        uintptr_t expected_pq_bits) {
    PaimonVindexTrainerHandle *trainer =
        paimon_vindex_trainer_open(keys, values, num_options);
    if (trainer == NULL) {
        fail_ffi("trainer open failed");
    }

    uintptr_t dimension = 0;
    if (paimon_vindex_trainer_dimension(trainer, &dimension) != 0) {
        fail_ffi("trainer dimension failed");
    }
    ASSERT_EQ_I64(dimension, ROUNDTRIP_DIMENSION);

    float *data = (float *)malloc(sizeof(float) * ROUNDTRIP_VECTOR_COUNT * ROUNDTRIP_DIMENSION);
    int64_t *ids = (int64_t *)malloc(sizeof(int64_t) * ROUNDTRIP_VECTOR_COUNT);
    ASSERT_TRUE(data != NULL);
    ASSERT_TRUE(ids != NULL);
    fill_roundtrip_data(data, ids);

    if (paimon_vindex_trainer_add_training_vectors(trainer, data, ROUNDTRIP_VECTOR_COUNT) != 0) {
        fail_ffi("trainer add training vectors failed");
    }
    PaimonVindexTrainingHandle *training = paimon_vindex_trainer_finish(trainer);
    if (training == NULL) {
        fail_ffi("trainer finish failed");
    }
    paimon_vindex_trainer_free(trainer);

    PaimonVindexWriterHandle *writer = paimon_vindex_writer_open(training);
    if (writer == NULL) {
        fail_ffi("writer open failed");
    }
    paimon_vindex_training_free(training);
    if (paimon_vindex_writer_add_vectors(writer, ids, data, ROUNDTRIP_VECTOR_COUNT) != 0) {
        fail_ffi("writer add failed");
    }

    struct MemBuffer buf = {0};
    struct PaimonVindexOutputFile output = {
        .ctx = &buf,
        .write_fn = mem_write,
        .flush_fn = mem_flush,
        .get_pos_fn = mem_pos,
    };
    if (paimon_vindex_writer_write_index(writer, output) != 0) {
        fail_ffi("writer write failed");
    }
    paimon_vindex_writer_free(writer);
    ASSERT_TRUE(buf.len > 0);

    struct PaimonVindexInputFile input = {
        .ctx = &buf,
        .read_ranges_fn = mem_read_ranges,
    };
    PaimonVindexReaderHandle *reader = paimon_vindex_reader_open(input);
    if (reader == NULL) {
        fail_ffi("reader open failed");
    }

    struct PaimonVindexMetadata metadata = {0};
    if (paimon_vindex_reader_metadata(reader, &metadata) != 0) {
        fail_ffi("reader metadata failed");
    }
    ASSERT_EQ_I64(metadata.index_type, expected_index_type);
    ASSERT_EQ_I64(metadata.metric, PAIMON_VINDEX_METRIC_L2);
    ASSERT_EQ_I64(metadata.dimension, ROUNDTRIP_DIMENSION);
    ASSERT_EQ_I64(
        metadata.nlist,
        expected_index_type == PAIMON_VINDEX_INDEX_TYPE_DISKANN ? 1 : 4);
    ASSERT_EQ_I64(metadata.total_vectors, ROUNDTRIP_VECTOR_COUNT);
    ASSERT_EQ_I64(metadata.pq_m, expected_pq_m);
    ASSERT_EQ_I64(metadata.pq_bits, expected_pq_bits);
    ASSERT_EQ_I64(
        metadata.rq_bits,
        expected_index_type == PAIMON_VINDEX_INDEX_TYPE_IVF_RQ ? 5 : 0);
    if (expected_index_type == PAIMON_VINDEX_INDEX_TYPE_DISKANN) {
        ASSERT_EQ_I64(metadata.diskann_max_degree, 8);
        ASSERT_EQ_I64(metadata.diskann_build_search_list_size, 16);
        ASSERT_TRUE(fabsf(metadata.diskann_alpha - 1.2f) < 1e-6f);
        struct PaimonVindexReadPlan read_plan = {0};
        if (paimon_vindex_reader_read_plan(reader, &read_plan) != 0) {
            fail_ffi("reader read plan failed");
        }
        ASSERT_TRUE(read_plan.window_bytes > 0);
        ASSERT_EQ_I64(
            read_plan.memory_budget_bytes,
            (uintptr_t) 4 * 1024 * 1024 * 1024);
    }

    if (paimon_vindex_reader_optimize_for_search(reader) != 0) {
        fail_ffi("reader optimize_for_search failed");
    }
    float query[ROUNDTRIP_DIMENSION];
    fill_query(query, 0.0f);
    if (expected_index_type == PAIMON_VINDEX_INDEX_TYPE_DISKANN) {
        uintptr_t calibrated_width = 0;
        if (paimon_vindex_reader_calibrate_search_width(
                reader, query, 1, 2, &calibrated_width) != 0) {
            fail_ffi("reader calibrate_search_width failed");
        }
        ASSERT_TRUE(
            calibrated_width == 100 ||
            calibrated_width == 200 ||
            calibrated_width == 400);
    }
    int64_t result_ids[2] = {0};
    float result_distances[2] = {0};
    struct PaimonVindexSearchParams search_params = {
        .top_k = 2,
        .search_width = expected_index_type == PAIMON_VINDEX_INDEX_TYPE_DISKANN
                            ? PAIMON_VINDEX_SEARCH_WIDTH_AUTO
                            : PAIMON_VINDEX_SEARCH_WIDTH_IVF_NPROBE,
        .width = expected_index_type == PAIMON_VINDEX_INDEX_TYPE_DISKANN ? 0 : 4};
    if (paimon_vindex_reader_search(
            reader, query, search_params, result_ids, result_distances, 2) != 0) {
        fail_ffi("reader search failed");
    }
    if (expected_index_type == PAIMON_VINDEX_INDEX_TYPE_DISKANN &&
        paimon_vindex_reader_warmup_queries(reader, query, 1, 32) != 0) {
        fail_ffi("reader warmup_queries failed");
    }
    assert_id_in_cluster(result_ids[0], 0);
    ASSERT_TRUE(isfinite(result_distances[0]));
    if (expected_index_type == PAIMON_VINDEX_INDEX_TYPE_IVF_PQ) {
        ASSERT_TRUE(buf.max_read_request_count > 1);
    }
    float queries[2 * ROUNDTRIP_DIMENSION];
    fill_query(queries, 0.0f);
    fill_query(queries + ROUNDTRIP_DIMENSION, 20.0f);
    int64_t batch_ids[2] = {0};
    float batch_distances[2] = {0};
    struct PaimonVindexSearchParams batch_params = {
        .top_k = 1,
        .search_width = expected_index_type == PAIMON_VINDEX_INDEX_TYPE_DISKANN
                            ? PAIMON_VINDEX_SEARCH_WIDTH_DISKANN_L_SEARCH
                            : PAIMON_VINDEX_SEARCH_WIDTH_IVF_NPROBE,
        .width = expected_index_type == PAIMON_VINDEX_INDEX_TYPE_DISKANN ? 100 : 4};
    if (paimon_vindex_reader_search_batch(
            reader, queries, 2, batch_params, batch_ids, batch_distances, 2) != 0) {
        fail_ffi("reader search batch failed");
    }
    assert_id_in_cluster(batch_ids[0], 0);
    assert_id_in_cluster(batch_ids[1], 1);
    struct PaimonVindexSearchParamsV2 batch_params_v2 = {
        .top_k = 1,
        .search_width = batch_params.search_width,
        .width = batch_params.width,
        .ivfpq_batch_table_reuse = PAIMON_VINDEX_IVFPQ_BATCH_TABLE_REUSE_OFF,
        .ivfpq_batch_table_reuse_max_bytes = 1};
    if (paimon_vindex_reader_search_batch_v2(
            reader, queries, 2, batch_params_v2, batch_ids, batch_distances, 2) != 0) {
        fail_ffi("reader search batch v2 failed");
    }
    assert_id_in_cluster(batch_ids[0], 0);
    assert_id_in_cluster(batch_ids[1], 1);
    struct PaimonVindexSearchParamsEx batch_params_ex =
        paimon_vindex_search_params_ex_default();
    batch_params_ex.top_k = 1;
    batch_params_ex.search_width = batch_params.search_width;
    batch_params_ex.width = batch_params.width;
    batch_params_ex.ivfpq_batch_table_reuse =
        PAIMON_VINDEX_IVFPQ_BATCH_TABLE_REUSE_OFF;
    batch_params_ex.ivfpq_batch_table_reuse_max_bytes = 1;
    if (paimon_vindex_reader_search_batch_ex(
            reader, queries, 2, &batch_params_ex, batch_ids, batch_distances, 2) != 0) {
        fail_ffi("reader search batch ex failed");
    }
    assert_id_in_cluster(batch_ids[0], 0);
    assert_id_in_cluster(batch_ids[1], 1);
    struct SearchParamsExPrefix batch_params_prefix = {
        .struct_size =
            offsetof(struct SearchParamsExPrefix, width) +
            sizeof(batch_params_prefix.width),
        .top_k = 1,
        .search_width = batch_params.search_width,
        .width = batch_params.width};
    if (paimon_vindex_reader_search_batch_ex(
            reader,
            queries,
            2,
            (const struct PaimonVindexSearchParamsEx *)&batch_params_prefix,
            batch_ids,
            batch_distances,
            2) != 0) {
        fail_ffi("reader search batch ex prefix failed");
    }
    assert_id_in_cluster(batch_ids[0], 0);
    assert_id_in_cluster(batch_ids[1], 1);
    int range_supported = -1;
    ASSERT_TRUE(paimon_vindex_reader_supports_range_search(reader, &range_supported) == 0);
    ASSERT_TRUE(range_supported == (expected_index_type != PAIMON_VINDEX_INDEX_TYPE_DISKANN));
    if (!range_supported) {
        PaimonVindexRangeSearchResult *unsupported = (PaimonVindexRangeSearchResult *)(uintptr_t)1;
        PaimonVindexRangeSearchParams range_params = range_all_params(PAIMON_VINDEX_METRIC_L2);
        ASSERT_TRUE(paimon_vindex_reader_range_search(
            reader, query, ROUNDTRIP_DIMENSION, range_params, &unsupported) != 0);
        ASSERT_TRUE(unsupported == NULL);
    }
    paimon_vindex_reader_free(reader);
    free(buf.data);
    free(data);
    free(ids);
    printf("PASS %s\n", name);
}

static PaimonVindexWriterHandle *new_trained_flat_writer(void) {
    const char *keys[] = {"index.type", "dimension", "nlist", "metric"};
    const char *values[] = {"ivf_flat", "1", "1", "l2"};
    PaimonVindexTrainerHandle *trainer = paimon_vindex_trainer_open(keys, values, 4);
    if (trainer == NULL) {
        fail_ffi("trainer open failed");
    }

    const float data[] = {0.0f, 1.0f};
    const int64_t ids[] = {1, 2};
    if (paimon_vindex_trainer_add_training_vectors(trainer, data, 2) != 0) {
        fail_ffi("trainer add training vectors failed");
    }
    PaimonVindexTrainingHandle *training = paimon_vindex_trainer_finish(trainer);
    if (training == NULL) {
        fail_ffi("trainer finish failed");
    }
    paimon_vindex_trainer_free(trainer);

    PaimonVindexWriterHandle *writer = paimon_vindex_writer_open(training);
    if (writer == NULL) {
        fail_ffi("writer open failed");
    }
    paimon_vindex_training_free(training);
    if (paimon_vindex_writer_add_vectors(writer, ids, data, 2) != 0) {
        fail_ffi("writer add failed");
    }
    return writer;
}

static void test_output_write_callback_error_propagates(void) {
    PaimonVindexWriterHandle *writer = new_trained_flat_writer();
    struct PaimonVindexOutputFile output = {
        .ctx = NULL,
        .write_fn = failing_write,
        .flush_fn = mem_flush,
        .get_pos_fn = NULL,
    };

    ASSERT_TRUE(paimon_vindex_writer_write_index(writer, output) != 0);
    assert_last_error_contains("write callback failed");
    paimon_vindex_writer_free(writer);
    printf("PASS output_write_callback_error_propagates\n");
}

static void test_output_flush_callback_error_propagates(void) {
    PaimonVindexWriterHandle *writer = new_trained_flat_writer();
    struct MemBuffer buf = {0};
    struct PaimonVindexOutputFile output = {
        .ctx = &buf,
        .write_fn = mem_write,
        .flush_fn = failing_flush,
        .get_pos_fn = mem_pos,
    };

    ASSERT_TRUE(paimon_vindex_writer_write_index(writer, output) != 0);
    assert_last_error_contains("flush callback failed");
    paimon_vindex_writer_free(writer);
    free(buf.data);
    printf("PASS output_flush_callback_error_propagates\n");
}

static void test_input_read_ranges_callback_error_propagates(void) {
    struct PaimonVindexInputFile input = {
        .ctx = NULL,
        .read_ranges_fn = failing_read_ranges,
    };

    PaimonVindexReaderHandle *reader = paimon_vindex_reader_open(input);
    ASSERT_TRUE(reader == NULL);
    assert_last_error_contains("read_ranges callback failed");
    printf("PASS input_read_ranges_callback_error_propagates\n");
}

static void test_supported_index_roundtrips(void) {
    const char *flat_keys[] = {"index.type", "dimension", "nlist", "metric"};
    const char *flat_values[] = {"ivf_flat", "8", "4", "l2"};
    run_roundtrip(
        "ivf_flat_roundtrip",
        flat_keys,
        flat_values,
        4,
        PAIMON_VINDEX_INDEX_TYPE_IVF_FLAT,
        0,
        0);

    const char *pq_keys[] = {"index.type", "dimension", "nlist", "metric"};
    const char *pq_values[] = {"ivf_pq", "8", "4", "l2"};
    run_roundtrip(
        "ivf_pq_roundtrip",
        pq_keys,
        pq_values,
        4,
        PAIMON_VINDEX_INDEX_TYPE_IVF_PQ,
        2,
        8);

    const char *rq_keys[] = {"index.type", "dimension", "nlist", "rq.bits", "metric"};
    const char *rq_values[] = {"ivf_rq", "8", "4", "5", "l2"};
    run_roundtrip(
        "ivf_rq_roundtrip",
        rq_keys,
        rq_values,
        5,
        PAIMON_VINDEX_INDEX_TYPE_IVF_RQ,
        0,
        0);

    const char *sq_keys[] = {"index.type", "dimension", "nlist", "metric"};
    const char *sq_values[] = {"ivf_sq", "8", "4", "l2"};
    run_roundtrip(
        "ivf_sq_roundtrip",
        sq_keys,
        sq_values,
        4,
        PAIMON_VINDEX_INDEX_TYPE_IVF_SQ,
        0,
        8);

    const char *diskann_keys[] = {
        "index.type",
        "dimension",
        "metric",
        "pq.m",
        "pq.bits",
        "diskann.max-degree",
        "diskann.build-search-list-size"};
    const char *diskann_values[] = {"diskann", "8", "l2", "4", "4", "8", "16"};
    run_roundtrip(
        "diskann_roundtrip",
        diskann_keys,
        diskann_values,
        7,
        PAIMON_VINDEX_INDEX_TYPE_DISKANN,
        4,
        4);
}

static void test_extensible_search_params_defaults(void) {
    PaimonVindexSearchParamsEx params = paimon_vindex_search_params_ex_default();

    ASSERT_TRUE(
        params.struct_size == PAIMON_VINDEX_SEARCH_PARAMS_EX_V1_SIZE);
    ASSERT_TRUE(params.search_width == PAIMON_VINDEX_SEARCH_WIDTH_AUTO);
    ASSERT_TRUE(
        params.ivfpq_batch_table_reuse ==
        PAIMON_VINDEX_IVFPQ_BATCH_TABLE_REUSE_AUTO);
    ASSERT_TRUE(
        params.ivfpq_batch_table_reuse_max_bytes ==
        PAIMON_VINDEX_DEFAULT_IVFPQ_BATCH_TABLE_REUSE_MAX_BYTES);
    printf("PASS extensible_search_params_defaults\n");
}

static PaimonVindexRangeSearchResultView range_view(PaimonVindexRangeSearchResult *result) {
    PaimonVindexRangeSearchResultView view = {0};
    if (paimon_vindex_range_search_result_view(result, &view) != 0) fail_ffi("range view");
    return view;
}

static void consume_range_fixture(const struct RangeFixture *fixture) {
    struct MemBuffer buffer = {0};
    buffer.data = fixture->index_data;
    buffer.len = fixture->index_len;
    PaimonVindexInputFile input = {.ctx = &buffer, .read_ranges_fn = mem_read_ranges};
    PaimonVindexReaderHandle *reader = paimon_vindex_reader_open(input);
    ASSERT_TRUE(reader != NULL);
    int supported = 0;
    ASSERT_TRUE(paimon_vindex_reader_supports_range_search(reader, &supported) == 0 && supported == 1);
    PaimonVindexRangeSearchResult *result = NULL;
    size_t query_len = fixture->dimension * fixture->query_count;
    int status;
    if (fixture->query_count == 1) {
        status = fixture->filter_len == 0
            ? paimon_vindex_reader_range_search(reader, fixture->queries, query_len, fixture->params, &result)
            : paimon_vindex_reader_range_search_with_roaring_filter(
                  reader, fixture->queries, query_len, fixture->params,
                  fixture->filter, fixture->filter_len, &result);
    } else {
        status = fixture->filter_len == 0
            ? paimon_vindex_reader_range_search_batch(
                  reader, fixture->queries, query_len, fixture->query_count, fixture->params, &result)
            : paimon_vindex_reader_range_search_batch_with_roaring_filter(
                  reader, fixture->queries, query_len, fixture->query_count, fixture->params,
                  fixture->filter, fixture->filter_len, &result);
    }
    if (status != 0) fail_ffi("range oracle");
    paimon_vindex_reader_free(reader);
    PaimonVindexRangeSearchResultView view = range_view(result);
    range_fixture_assert(fixture, &view);
    paimon_vindex_range_search_result_destroy(result);
}

static void test_range_endpoint_raw_results(void) {
    const char *metrics[] = {"l2", "inner_product"};
    const uint32_t metric_codes[] = {PAIMON_VINDEX_METRIC_L2, PAIMON_VINDEX_METRIC_INNER_PRODUCT};
    const float coordinates[][3] = {{3.0f, 4.0f, 5.0f}, {3.0f, 2.5f, 2.0f}};
    const float expected_raw_distances[][2] = {{9.0f, 16.0f}, {-6.0f, -5.0f}};
    const int64_t labels[] = {INT64_C(1) << 40, (INT64_C(1) << 40) + 1, (INT64_C(1) << 40) + 2};
    for (size_t metric_index = 0; metric_index < 2; ++metric_index) {
        float data[3 * RANGE_DIMENSION] = {0};
        float query[RANGE_DIMENSION] = {0};
        for (size_t row = 0; row < 3; ++row) {
            data[row * RANGE_DIMENSION] = coordinates[metric_index][row];
        }
        if (metric_codes[metric_index] == PAIMON_VINDEX_METRIC_INNER_PRODUCT) query[0] = 2.0f;
        const char *keys[] = {"index.type", "dimension", "nlist", "metric"};
        const char *values[] = {"ivf_flat", "8", "1", metrics[metric_index]};
        PaimonVindexTrainerHandle *trainer = paimon_vindex_trainer_open(keys, values, 4);
        ASSERT_TRUE(trainer != NULL);
        ASSERT_TRUE(paimon_vindex_trainer_add_training_vectors(trainer, data, 3) == 0);
        PaimonVindexTrainingHandle *training = paimon_vindex_trainer_finish(trainer);
        ASSERT_TRUE(training != NULL);
        paimon_vindex_trainer_free(trainer);
        PaimonVindexWriterHandle *writer = paimon_vindex_writer_open(training);
        ASSERT_TRUE(writer != NULL);
        paimon_vindex_training_free(training);
        ASSERT_TRUE(paimon_vindex_writer_add_vectors(writer, labels, data, 3) == 0);
        struct MemBuffer buffer = {0};
        PaimonVindexOutputFile output = {
            .ctx = &buffer, .write_fn = mem_write, .flush_fn = mem_flush, .get_pos_fn = mem_pos};
        ASSERT_TRUE(paimon_vindex_writer_write_index(writer, output) == 0);
        paimon_vindex_writer_free(writer);
        PaimonVindexInputFile input = {.ctx = &buffer, .read_ranges_fn = mem_read_ranges};
        PaimonVindexReaderHandle *reader = paimon_vindex_reader_open(input);
        ASSERT_TRUE(reader != NULL);
        PaimonVindexRangeSearchParams params = range_all_params(metric_codes[metric_index]);
        params.nprobe = 1;
        PaimonVindexDistanceEndpoint radius = {4.0, PAIMON_VINDEX_CUT_LE};
        PaimonVindexDistanceEndpoint similarity = {5.0, PAIMON_VINDEX_CUT_GE};
        ASSERT_TRUE(paimon_vindex_distance_band_from_endpoints(
            metric_codes[metric_index], metric_index == 0 ? NULL : &similarity,
            metric_index == 0 ? &radius : NULL, &params.band) == 0);
        PaimonVindexRangeSearchResult *result = NULL;
        ASSERT_TRUE(paimon_vindex_reader_range_search(reader, query, RANGE_DIMENSION, params, &result) == 0);
        paimon_vindex_reader_free(reader);
        free(buffer.data);
        PaimonVindexRangeSearchResultView view = range_view(result);
        range_assert_shape(&view);
        ASSERT_TRUE(view.query_count == 1 && view.hit_count == 2);
        for (size_t expected = 0; expected < 2; ++expected) {
            size_t matches = 0;
            for (size_t hit = 0; hit < view.hit_count; ++hit) {
                if (view.labels[hit] == labels[expected]) {
                    ASSERT_TRUE(range_float_bits(view.raw_distances[hit]) ==
                                range_float_bits(expected_raw_distances[metric_index][expected]));
                    ++matches;
                }
            }
            ASSERT_TRUE(matches == 1);
        }
        paimon_vindex_range_search_result_destroy(result);
        printf("PASS range_endpoint_raw_results %s\n", metrics[metric_index]);
    }
}

static void test_range_endpoints(void) {
    const uint32_t metrics[] = {
        PAIMON_VINDEX_METRIC_L2, PAIMON_VINDEX_METRIC_COSINE, PAIMON_VINDEX_METRIC_INNER_PRODUCT};
    const float candidates[] = {-1.0f, -0.5f, -0.0f, 0.0f, 0.25f, 0.5f, 1.0f, 4.0f};
    for (size_t metric_index = 0; metric_index < 3; ++metric_index) {
        uint32_t metric = metrics[metric_index];
        PaimonVindexRawDistanceBand band;
        ASSERT_TRUE(paimon_vindex_distance_band_from_endpoints(metric, NULL, NULL, &band) == 0);
        ASSERT_TRUE(band.metric == metric);
        ASSERT_TRUE(band.raw_lower_kind == PAIMON_VINDEX_BOUND_UNBOUNDED);
        ASSERT_TRUE(band.raw_upper_kind == PAIMON_VINDEX_BOUND_UNBOUNDED);
        for (uint32_t lower_op = PAIMON_VINDEX_CUT_GE; lower_op <= PAIMON_VINDEX_CUT_GT; ++lower_op) {
            for (uint32_t upper_op = PAIMON_VINDEX_CUT_LE; upper_op <= PAIMON_VINDEX_CUT_LT; ++upper_op) {
                PaimonVindexDistanceEndpoint lower = {0.5, lower_op};
                PaimonVindexDistanceEndpoint upper = {1.0, upper_op};
                ASSERT_TRUE(paimon_vindex_distance_band_from_endpoints(metric, &lower, &upper, &band) == 0);
                for (size_t candidate = 0; candidate < sizeof(candidates) / sizeof(candidates[0]); ++candidate) {
                    float raw_distance = candidates[candidate];
                    if (metric == PAIMON_VINDEX_METRIC_L2 && raw_distance < 0) continue;
                    double value = metric == PAIMON_VINDEX_METRIC_L2 ? (double)sqrtf(raw_distance)
                        : metric == PAIMON_VINDEX_METRIC_INNER_PRODUCT ? -(double)raw_distance : (double)raw_distance;
                    int expected = (lower_op == PAIMON_VINDEX_CUT_GE ? value >= 0.5 : value > 0.5) &&
                        (upper_op == PAIMON_VINDEX_CUT_LE ? value <= 1.0 : value < 1.0);
                    ASSERT_TRUE(expected ==
                        ((band.raw_lower_kind == PAIMON_VINDEX_BOUND_UNBOUNDED || raw_distance >= band.raw_lower) &&
                         (band.raw_upper_kind == PAIMON_VINDEX_BOUND_UNBOUNDED || raw_distance < band.raw_upper)));
                }
            }
        }
        PaimonVindexDistanceEndpoint precise = {nextafter(1.0, 2.0), PAIMON_VINDEX_CUT_GE};
        ASSERT_TRUE(paimon_vindex_distance_band_from_endpoints(metric, &precise, NULL, &band) == 0);
        float raw_boundary = metric == PAIMON_VINDEX_METRIC_INNER_PRODUCT ? -1.0f : 1.0f;
        ASSERT_TRUE(!((band.raw_lower_kind == PAIMON_VINDEX_BOUND_UNBOUNDED || raw_boundary >= band.raw_lower) &&
                      (band.raw_upper_kind == PAIMON_VINDEX_BOUND_UNBOUNDED || raw_boundary < band.raw_upper)));
    }
    PaimonVindexRawDistanceBand band;
    PaimonVindexDistanceEndpoint endpoint = {1.0, PAIMON_VINDEX_CUT_LT};
    ASSERT_TRUE(paimon_vindex_distance_band_from_endpoints(PAIMON_VINDEX_METRIC_L2, &endpoint, NULL, &band) != 0);
    endpoint.op = PAIMON_VINDEX_CUT_GE;
    ASSERT_TRUE(paimon_vindex_distance_band_from_endpoints(PAIMON_VINDEX_METRIC_L2, NULL, &endpoint, &band) != 0);
    endpoint.value = NAN;
    ASSERT_TRUE(paimon_vindex_distance_band_from_endpoints(PAIMON_VINDEX_METRIC_L2, &endpoint, NULL, &band) != 0);
    endpoint.value = INFINITY;
    ASSERT_TRUE(paimon_vindex_distance_band_from_endpoints(PAIMON_VINDEX_METRIC_L2, &endpoint, NULL, &band) != 0);
    endpoint.value = 1;
    endpoint.op = UINT32_MAX;
    ASSERT_TRUE(paimon_vindex_distance_band_from_endpoints(PAIMON_VINDEX_METRIC_L2, &endpoint, NULL, &band) != 0);
    ASSERT_TRUE(paimon_vindex_distance_band_from_endpoints(UINT32_MAX, NULL, NULL, &band) != 0);
    ASSERT_TRUE(paimon_vindex_distance_band_from_endpoints(PAIMON_VINDEX_METRIC_L2, NULL, NULL, NULL) != 0);
    paimon_vindex_range_search_result_destroy(NULL);
    printf("PASS range_endpoints\n");
}

#define ASSERT_RANGE_ERROR(expression) do { \
    result = (PaimonVindexRangeSearchResult *)(uintptr_t)1; \
    ASSERT_TRUE((expression) != 0); \
    ASSERT_TRUE(result == NULL); \
    ASSERT_TRUE(paimon_vindex_last_error() != NULL); \
} while (0)

static void test_range_errors(PaimonVindexReaderHandle *reader, const float *query,
                              PaimonVindexRangeSearchParams params) {
    PaimonVindexRangeSearchResult *result;
    ASSERT_RANGE_ERROR(paimon_vindex_reader_range_search(NULL, query, RANGE_DIMENSION, params, &result));
    ASSERT_RANGE_ERROR(paimon_vindex_reader_range_search(reader, NULL, RANGE_DIMENSION, params, &result));
    ASSERT_RANGE_ERROR(paimon_vindex_reader_range_search(reader, query, RANGE_DIMENSION - 1, params, &result));
    ASSERT_RANGE_ERROR(paimon_vindex_reader_range_search(reader, NULL, 0, params, &result));
    ASSERT_RANGE_ERROR(paimon_vindex_reader_range_search_batch(reader, NULL, 0, 0, params, &result));
    ASSERT_RANGE_ERROR(paimon_vindex_reader_range_search_batch(reader, query, RANGE_DIMENSION, 2, params, &result));
    ASSERT_RANGE_ERROR(paimon_vindex_reader_range_search_batch(
        reader, NULL, 0, SIZE_MAX / RANGE_DIMENSION + 1, params, &result));
    ASSERT_RANGE_ERROR(paimon_vindex_reader_range_search_with_roaring_filter(
        reader, query, RANGE_DIMENSION, params, NULL, 1, &result));
    ASSERT_RANGE_ERROR(paimon_vindex_reader_range_search_with_roaring_filter(
        reader, query, RANGE_DIMENSION, params, NULL, 0, &result));
    ASSERT_RANGE_ERROR(paimon_vindex_reader_range_search_batch_with_roaring_filter(
        reader, query, RANGE_DIMENSION, 1, params, range_filter, 1, &result));
    ASSERT_TRUE(paimon_vindex_reader_range_search(reader, query, RANGE_DIMENSION, params, NULL) != 0);
    int supported;
    ASSERT_TRUE(paimon_vindex_reader_supports_range_search(NULL, &supported) != 0);
    ASSERT_TRUE(paimon_vindex_reader_supports_range_search(reader, NULL) != 0);
    PaimonVindexRangeSearchResultView view;
    ASSERT_TRUE(paimon_vindex_range_search_result_view(NULL, &view) != 0);
    PaimonVindexRangeSearchParams invalid = params;
    invalid.nprobe = 0;
    ASSERT_RANGE_ERROR(paimon_vindex_reader_range_search(reader, query, RANGE_DIMENSION, invalid, &result));
    invalid = params;
    invalid.band.metric = (params.band.metric + 1) % 3;
    ASSERT_RANGE_ERROR(paimon_vindex_reader_range_search(reader, query, RANGE_DIMENSION, invalid, &result));
    invalid.band.metric = UINT32_MAX;
    ASSERT_RANGE_ERROR(paimon_vindex_reader_range_search(reader, query, RANGE_DIMENSION, invalid, &result));
    invalid = params;
    invalid.band.raw_lower_kind = UINT32_MAX;
    ASSERT_RANGE_ERROR(paimon_vindex_reader_range_search(reader, query, RANGE_DIMENSION, invalid, &result));
    invalid = params;
    invalid.band.raw_upper_kind = PAIMON_VINDEX_BOUND_FINITE;
    invalid.band.raw_upper = NAN;
    ASSERT_RANGE_ERROR(paimon_vindex_reader_range_search(reader, query, RANGE_DIMENSION, invalid, &result));
    invalid.band.raw_upper = INFINITY;
    ASSERT_RANGE_ERROR(paimon_vindex_reader_range_search(reader, query, RANGE_DIMENSION, invalid, &result));
    invalid.band.raw_upper = 0;
    invalid.band.raw_lower_kind = PAIMON_VINDEX_BOUND_FINITE;
    invalid.band.raw_lower = 1;
    ASSERT_RANGE_ERROR(paimon_vindex_reader_range_search(reader, query, RANGE_DIMENSION, invalid, &result));
    invalid.band.raw_lower = 0;
    invalid.nprobe = 0;
    ASSERT_RANGE_ERROR(paimon_vindex_reader_range_search(reader, query, RANGE_DIMENSION, invalid, &result));
    float bad_query[RANGE_DIMENSION];
    memcpy(bad_query, query, sizeof(bad_query));
    bad_query[0] = NAN;
    ASSERT_RANGE_ERROR(paimon_vindex_reader_range_search(reader, bad_query, RANGE_DIMENSION, params, &result));
}

static void test_range_matrix(void) {
    const char *index_types[] = {"ivf_flat", "ivf_sq", "ivf_pq", "ivf_rq"};
    const char *metrics[] = {"l2", "inner_product", "cosine"};
    const uint32_t metric_codes[] = {
        PAIMON_VINDEX_METRIC_L2, PAIMON_VINDEX_METRIC_INNER_PRODUCT, PAIMON_VINDEX_METRIC_COSINE};
    float data[RANGE_VECTOR_COUNT * RANGE_DIMENSION];
    int64_t labels[RANGE_VECTOR_COUNT];
    float queries[RANGE_QUERY_COUNT * RANGE_DIMENSION];
    range_fill_data(data, labels, queries);
    for (size_t index_type = 0; index_type < 4; ++index_type) {
        for (size_t metric = 0; metric < 3; ++metric) {
            const char *keys[] = {"index.type", "dimension", "nlist", "metric"};
            const char *values[] = {index_types[index_type], "8", "4", metrics[metric]};
            PaimonVindexTrainerHandle *trainer = paimon_vindex_trainer_open(keys, values, 4);
            if (trainer == NULL) fail_ffi("range trainer open");
            ASSERT_TRUE(paimon_vindex_trainer_add_training_vectors(trainer, data, RANGE_VECTOR_COUNT) == 0);
            PaimonVindexTrainingHandle *training = paimon_vindex_trainer_finish(trainer);
            ASSERT_TRUE(training != NULL);
            paimon_vindex_trainer_free(trainer);
            PaimonVindexWriterHandle *writer = paimon_vindex_writer_open(training);
            ASSERT_TRUE(writer != NULL);
            paimon_vindex_training_free(training);
            ASSERT_TRUE(paimon_vindex_writer_add_vectors(writer, labels, data, RANGE_VECTOR_COUNT) == 0);
            struct MemBuffer buffer = {0};
            PaimonVindexOutputFile output = {
                .ctx = &buffer, .write_fn = mem_write, .flush_fn = mem_flush, .get_pos_fn = mem_pos};
            ASSERT_TRUE(paimon_vindex_writer_write_index(writer, output) == 0);
            paimon_vindex_writer_free(writer);
            PaimonVindexInputFile input = {.ctx = &buffer, .read_ranges_fn = mem_read_ranges};
            PaimonVindexReaderHandle *reader = paimon_vindex_reader_open(input);
            ASSERT_TRUE(reader != NULL);
            int supported = 0;
            ASSERT_TRUE(paimon_vindex_reader_supports_range_search(reader, &supported) == 0 && supported == 1);
            PaimonVindexRangeSearchParams params = range_all_params(metric_codes[metric]);
            PaimonVindexRangeSearchResult *batch = NULL;
            ASSERT_TRUE(paimon_vindex_reader_range_search_batch(
                reader, queries, RANGE_QUERY_COUNT * RANGE_DIMENSION, RANGE_QUERY_COUNT, params, &batch) == 0);
            PaimonVindexRangeSearchResultView batch_view = range_view(batch);
            range_assert_shape(&batch_view);
            ASSERT_TRUE(batch_view.query_count == RANGE_QUERY_COUNT);
            ASSERT_TRUE(batch_view.hit_count == RANGE_VECTOR_COUNT * RANGE_QUERY_COUNT);
            ASSERT_TRUE(batch_view.list_reads > 0);
            for (size_t query = 0; query < RANGE_QUERY_COUNT; ++query) {
                ASSERT_TRUE(batch_view.stats[query].lists_probed == RANGE_NLIST);
                ASSERT_TRUE(batch_view.stats[query].rows_scanned == RANGE_VECTOR_COUNT);
                ASSERT_TRUE(batch_view.stats[query].early_abandoned == 0);
            }
            PaimonVindexRangeSearchResult *single = NULL;
            ASSERT_TRUE(paimon_vindex_reader_range_search(reader, queries, RANGE_DIMENSION, params, &single) == 0);
            PaimonVindexRangeSearchResultView single_view = range_view(single);
            ASSERT_TRUE(single_view.hit_count == RANGE_VECTOR_COUNT);
            PaimonVindexRangeSearchResult *filtered = NULL;
            ASSERT_TRUE(paimon_vindex_reader_range_search_batch_with_roaring_filter(
                reader, queries, RANGE_QUERY_COUNT * RANGE_DIMENSION, RANGE_QUERY_COUNT, params,
                range_filter, sizeof(range_filter), &filtered) == 0);
            PaimonVindexRangeSearchResultView filtered_view = range_view(filtered);
            range_assert_shape(&filtered_view);
            ASSERT_TRUE(filtered_view.hit_count == RANGE_QUERY_COUNT * 2);
            for (size_t hit = 0; hit < filtered_view.hit_count; ++hit) {
                ASSERT_TRUE(filtered_view.labels[hit] == labels[1] || filtered_view.labels[hit] == labels[3]);
            }
            paimon_vindex_range_search_result_destroy(filtered);
            ASSERT_TRUE(paimon_vindex_reader_range_search_with_roaring_filter(
                reader, queries, RANGE_DIMENSION, params, range_filter, sizeof(range_filter), &filtered) == 0);
            filtered_view = range_view(filtered);
            ASSERT_TRUE(filtered_view.hit_count == 2);
            paimon_vindex_range_search_result_destroy(filtered);
            ASSERT_TRUE(paimon_vindex_reader_range_search_with_roaring_filter(
                reader, queries, RANGE_DIMENSION, params, range_empty_filter, sizeof(range_empty_filter), &filtered) == 0);
            filtered_view = range_view(filtered);
            range_assert_shape(&filtered_view);
            ASSERT_TRUE(filtered_view.hit_count == 0 && filtered_view.lims[1] == 0);
            paimon_vindex_range_search_result_destroy(filtered);
            PaimonVindexRangeSearchParams bounded = params;
            float minimum = single_view.raw_distances[0];
            float maximum = minimum;
            for (size_t hit = 1; hit < single_view.hit_count; ++hit) {
                minimum = fminf(minimum, single_view.raw_distances[hit]);
                maximum = fmaxf(maximum, single_view.raw_distances[hit]);
            }
            bounded.band.raw_lower_kind = PAIMON_VINDEX_BOUND_FINITE;
            bounded.band.raw_lower = metric_codes[metric] == PAIMON_VINDEX_METRIC_L2
                ? fmaxf(0, minimum) : minimum;
            bounded.band.raw_upper_kind = PAIMON_VINDEX_BOUND_FINITE;
            bounded.band.raw_upper = minimum + (maximum - minimum) / 2;
            PaimonVindexRangeSearchResult *subset = NULL;
            ASSERT_TRUE(paimon_vindex_reader_range_search_batch(
                reader, queries, RANGE_QUERY_COUNT * RANGE_DIMENSION, RANGE_QUERY_COUNT, bounded, &subset) == 0);
            PaimonVindexRangeSearchResultView subset_view = range_view(subset);
            range_assert_shape(&subset_view);
            ASSERT_TRUE(subset_view.hit_count > 0 && subset_view.hit_count < batch_view.hit_count);
            for (size_t query = 0; query < RANGE_QUERY_COUNT; ++query) {
                size_t expected = 0;
                for (size_t hit = batch_view.lims[query]; hit < batch_view.lims[query + 1]; ++hit) {
                    if (batch_view.raw_distances[hit] >= bounded.band.raw_lower &&
                        batch_view.raw_distances[hit] < bounded.band.raw_upper) {
                        ++expected;
                        int found = 0;
                        for (size_t candidate = subset_view.lims[query]; candidate < subset_view.lims[query + 1]; ++candidate) {
                            if (subset_view.labels[candidate] == batch_view.labels[hit]) found = 1;
                        }
                        ASSERT_TRUE(found);
                    }
                }
                ASSERT_TRUE(subset_view.lims[query + 1] - subset_view.lims[query] == expected);
            }
            paimon_vindex_range_search_result_destroy(subset);
            bounded.band.raw_lower = bounded.band.raw_upper;
            ASSERT_TRUE(paimon_vindex_reader_range_search_batch(
                reader, queries, RANGE_QUERY_COUNT * RANGE_DIMENSION, RANGE_QUERY_COUNT, bounded, &subset) == 0);
            subset_view = range_view(subset);
            range_assert_shape(&subset_view);
            ASSERT_TRUE(subset_view.hit_count == 0 && subset_view.list_reads == 0);
            paimon_vindex_range_search_result_destroy(subset);
            test_range_errors(reader, queries, params);
            ASSERT_TRUE(paimon_vindex_range_search_result_view(single, NULL) != 0);
            paimon_vindex_reader_free(reader);
            free(buffer.data);
            PaimonVindexRangeSearchResultView retained_view = range_view(single);
            ASSERT_TRUE(retained_view.labels == single_view.labels);
            ASSERT_TRUE(retained_view.raw_distances == single_view.raw_distances);
            range_assert_shape(&retained_view);
            range_assert_shape(&batch_view);
            paimon_vindex_range_search_result_destroy(single);
            paimon_vindex_range_search_result_destroy(batch);
            printf("PASS range_matrix %s %s\n", index_types[index_type], metrics[metric]);
        }
    }
}

int main(void) {
    test_extensible_search_params_defaults();
    test_supported_index_roundtrips();
    test_output_write_callback_error_propagates();
    test_output_flush_callback_error_propagates();
    test_input_read_ranges_callback_error_propagates();
    test_range_endpoints();
    test_range_endpoint_raw_results();
    test_range_matrix();
    range_fixture_run_all(consume_range_fixture);
    return 0;
}
