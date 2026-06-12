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

static void fail_ffi(const char *message) {
    const char *err = paimon_vindex_last_error();
    fprintf(stderr, "%s: %s\n", message, err == NULL ? "(no error)" : err);
    abort();
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

static void test_basic_roundtrip(void) {
    const char *keys[] = {"index.type", "dimension", "nlist", "metric"};
    const char *values[] = {"ivf_flat", "2", "2", "l2"};
    PaimonVindexWriterHandle *writer = paimon_vindex_writer_open(keys, values, 4);
    if (writer == NULL) {
        fail_ffi("writer open failed");
    }

    uintptr_t dimension = 0;
    if (paimon_vindex_writer_dimension(writer, &dimension) != 0) {
        fail_ffi("writer dimension failed");
    }
    ASSERT_EQ_I64(dimension, 2);

    const float data[] = {
        0.0f, 0.0f,
        1.0f, 0.0f,
        10.0f, 10.0f,
        11.0f, 10.0f,
    };
    const int64_t ids[] = {100, 101, 200, 201};
    if (paimon_vindex_writer_train(writer, data, 4) != 0) {
        fail_ffi("writer train failed");
    }
    if (paimon_vindex_writer_add_vectors(writer, ids, data, 4) != 0) {
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
    ASSERT_EQ_I64(metadata.index_type, PAIMON_VINDEX_INDEX_TYPE_IVF_FLAT);
    ASSERT_EQ_I64(metadata.metric, PAIMON_VINDEX_METRIC_L2);
    ASSERT_EQ_I64(metadata.dimension, 2);
    ASSERT_EQ_I64(metadata.nlist, 2);
    ASSERT_EQ_I64(metadata.total_vectors, 4);

    if (paimon_vindex_reader_optimize_for_search(reader) != 0) {
        fail_ffi("reader optimize_for_search failed");
    }

    const float query[] = {0.0f, 0.0f};
    int64_t result_ids[2] = {0};
    float result_distances[2] = {0};
    if (paimon_vindex_reader_search(
            reader, query, 2, 2, 0, result_ids, result_distances, 2) != 0) {
        fail_ffi("reader search failed");
    }
    ASSERT_EQ_I64(result_ids[0], 100);
    ASSERT_TRUE(isfinite(result_distances[0]));

    const float queries[] = {0.0f, 0.0f, 10.0f, 10.0f};
    int64_t batch_ids[2] = {0};
    float batch_distances[2] = {0};
    if (paimon_vindex_reader_search_batch(
            reader, queries, 2, 1, 2, 0, batch_ids, batch_distances, 2) != 0) {
        fail_ffi("reader search batch failed");
    }
    ASSERT_EQ_I64(batch_ids[0], 100);
    ASSERT_EQ_I64(batch_ids[1], 200);

    paimon_vindex_reader_free(reader);
    free(buf.data);
    printf("PASS test_basic_roundtrip\n");
}

int main(void) {
    test_basic_roundtrip();
    return 0;
}
