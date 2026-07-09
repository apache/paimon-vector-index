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

struct MemBuffer {
    uint8_t *data;
    size_t len;
    size_t cap;
    size_t pos;
};

enum {
    ROUNDTRIP_DIMENSION = 8,
    ROUNDTRIP_NLIST = 4,
    ROUNDTRIP_PER_LIST = 128,
    ROUNDTRIP_VECTOR_COUNT = ROUNDTRIP_NLIST * ROUNDTRIP_PER_LIST,
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

static int mem_read_at(void *ctx, uint64_t offset, uint8_t *dst, uintptr_t len) {
    struct MemBuffer *buf = (struct MemBuffer *)ctx;
    if (offset > SIZE_MAX || len > SIZE_MAX) {
        return -1;
    }
    size_t off = (size_t)offset;
    size_t n = (size_t)len;
    if (off > buf->len || n > buf->len - off) {
        return -1;
    }
    memcpy(dst, buf->data + off, n);
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

static int failing_read_at(void *ctx, uint64_t offset, uint8_t *dst, uintptr_t len) {
    (void)ctx;
    (void)offset;
    (void)dst;
    (void)len;
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
        uintptr_t expected_hnsw_m) {
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
        .read_at_fn = mem_read_at,
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
    ASSERT_EQ_I64(metadata.nlist, 4);
    ASSERT_EQ_I64(metadata.total_vectors, ROUNDTRIP_VECTOR_COUNT);
    ASSERT_EQ_I64(metadata.pq_m, expected_pq_m);
    ASSERT_EQ_I64(metadata.hnsw_m, expected_hnsw_m);

    if (paimon_vindex_reader_optimize_for_search(reader) != 0) {
        fail_ffi("reader optimize_for_search failed");
    }

    float query[ROUNDTRIP_DIMENSION];
    fill_query(query, 0.0f);
    int64_t result_ids[2] = {0};
    float result_distances[2] = {0};
    struct PaimonVindexSearchParams search_params = {2, 4, 16, 0};
    if (paimon_vindex_reader_search(
            reader, query, search_params, result_ids, result_distances, 2) != 0) {
        fail_ffi("reader search failed");
    }
    assert_id_in_cluster(result_ids[0], 0);
    ASSERT_TRUE(isfinite(result_distances[0]));
    if (expected_index_type == PAIMON_VINDEX_INDEX_TYPE_IVF_RQ) {
        search_params.query_bits = 4;
        if (paimon_vindex_reader_search(
                reader, query, search_params, result_ids, result_distances, 2) != 0) {
            fail_ffi("reader search with query bits failed");
        }
        assert_id_in_cluster(result_ids[0], 0);
        ASSERT_TRUE(isfinite(result_distances[0]));
    }

    float queries[2 * ROUNDTRIP_DIMENSION];
    fill_query(queries, 0.0f);
    fill_query(queries + ROUNDTRIP_DIMENSION, 20.0f);
    int64_t batch_ids[2] = {0};
    float batch_distances[2] = {0};
    struct PaimonVindexSearchParams batch_params = {1, 4, 16, 0};
    if (paimon_vindex_reader_search_batch(
            reader, queries, 2, batch_params, batch_ids, batch_distances, 2) != 0) {
        fail_ffi("reader search batch failed");
    }
    assert_id_in_cluster(batch_ids[0], 0);
    assert_id_in_cluster(batch_ids[1], 1);
    if (expected_index_type == PAIMON_VINDEX_INDEX_TYPE_IVF_RQ) {
        batch_params.query_bits = 8;
        if (paimon_vindex_reader_search_batch(
                reader, queries, 2, batch_params, batch_ids, batch_distances, 2) != 0) {
            fail_ffi("reader search batch with query bits failed");
        }
        assert_id_in_cluster(batch_ids[0], 0);
        assert_id_in_cluster(batch_ids[1], 1);
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

static void test_input_read_callback_error_propagates(void) {
    struct PaimonVindexInputFile input = {
        .ctx = NULL,
        .read_at_fn = failing_read_at,
    };

    PaimonVindexReaderHandle *reader = paimon_vindex_reader_open(input);
    ASSERT_TRUE(reader == NULL);
    assert_last_error_contains("read_at callback failed");
    printf("PASS input_read_callback_error_propagates\n");
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

    const char *pq_keys[] = {"index.type", "dimension", "nlist", "metric", "pq.m"};
    const char *pq_values[] = {"ivf_pq", "8", "4", "l2", "4"};
    run_roundtrip(
        "ivf_pq_roundtrip",
        pq_keys,
        pq_values,
        5,
        PAIMON_VINDEX_INDEX_TYPE_IVF_PQ,
        4,
        0);

    const char *rq_keys[] = {"index.type", "dimension", "nlist", "metric"};
    const char *rq_values[] = {"ivf_rq", "8", "4", "l2"};
    run_roundtrip(
        "ivf_rq_roundtrip",
        rq_keys,
        rq_values,
        4,
        PAIMON_VINDEX_INDEX_TYPE_IVF_RQ,
        0,
        0);

    const char *hnsw_flat_keys[] = {"index.type", "dimension", "nlist", "metric", "hnsw.m"};
    const char *hnsw_flat_values[] = {"ivf_hnsw_flat", "8", "4", "l2", "4"};
    run_roundtrip(
        "ivf_hnsw_flat_roundtrip",
        hnsw_flat_keys,
        hnsw_flat_values,
        5,
        PAIMON_VINDEX_INDEX_TYPE_IVF_HNSW_FLAT,
        0,
        4);

    const char *hnsw_sq_keys[] = {"index.type", "dimension", "nlist", "metric", "hnsw.m"};
    const char *hnsw_sq_values[] = {"ivf_hnsw_sq", "8", "4", "l2", "4"};
    run_roundtrip(
        "ivf_hnsw_sq_roundtrip",
        hnsw_sq_keys,
        hnsw_sq_values,
        5,
        PAIMON_VINDEX_INDEX_TYPE_IVF_HNSW_SQ,
        0,
        4);
}

int main(void) {
    test_supported_index_roundtrips();
    test_output_write_callback_error_propagates();
    test_output_flush_callback_error_propagates();
    test_input_read_callback_error_propagates();
    return 0;
}
