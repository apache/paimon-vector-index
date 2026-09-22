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

#ifndef PAIMON_VINDEX_RANGE_TEST_SUPPORT_H
#define PAIMON_VINDEX_RANGE_TEST_SUPPORT_H

#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    RANGE_DIMENSION = 8,
    RANGE_NLIST = 4,
    RANGE_VECTOR_COUNT = 512,
    RANGE_QUERY_COUNT = 3
};

static const uint8_t range_filter[] = {
    1, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 0, 0,
    58, 48, 0, 0, 1, 0, 0, 0,
    0, 0, 1, 0, 16, 0, 0, 0,
    1, 0, 3, 0
};
static const uint8_t range_empty_filter[8] = {0};

static uint32_t range_float_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float range_float_from_bits(uint32_t bits) {
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static void range_fill_data(float *data, int64_t *labels, float *queries) {
    for (size_t row = 0; row < RANGE_VECTOR_COUNT; ++row) {
        labels[row] = (INT64_C(1) << 40) + (int64_t)row;
        for (size_t dimension = 0; dimension < RANGE_DIMENSION; ++dimension) {
            data[row * RANGE_DIMENSION + dimension] =
                (float)((int)((row * 17 + dimension * 13 + row * dimension) % 101) - 50) / 16.0f;
        }
    }
    for (size_t query = 0; query < RANGE_QUERY_COUNT; ++query) {
        memcpy(queries + query * RANGE_DIMENSION,
               data + query * 71 * RANGE_DIMENSION,
               RANGE_DIMENSION * sizeof(float));
    }
}

static PaimonVindexRangeSearchParams range_all_params(uint32_t metric) {
    PaimonVindexRangeSearchParams params = {{0, 0, 0.0f, 0, 0.0f}, 0};
    params.band.metric = metric;
    params.band.raw_lower_kind = PAIMON_VINDEX_BOUND_UNBOUNDED;
    params.band.raw_upper_kind = PAIMON_VINDEX_BOUND_UNBOUNDED;
    params.nprobe = RANGE_NLIST;
    return params;
}

static void range_assert_shape(const PaimonVindexRangeSearchResultView *view) {
    ASSERT_TRUE(view->lims != NULL);
    ASSERT_TRUE(view->lims[0] == 0);
    ASSERT_TRUE(view->lims[view->query_count] == view->hit_count);
    ASSERT_TRUE(view->hit_count == 0 || (view->labels != NULL && view->raw_distances != NULL));
    ASSERT_TRUE(view->query_count == 0 || view->stats != NULL);
    for (size_t query = 0; query < view->query_count; ++query) {
        ASSERT_TRUE(view->lims[query] <= view->lims[query + 1]);
        ASSERT_TRUE(view->stats[query].rows_committed == view->lims[query + 1] - view->lims[query]);
        ASSERT_TRUE(view->stats[query].rows_scanned >= view->stats[query].rows_committed);
        ASSERT_TRUE(view->stats[query].early_abandoned <= view->stats[query].rows_scanned);
        ASSERT_TRUE(view->stats[query].lists_probed <= RANGE_NLIST);
    }
    for (size_t hit = 0; hit < view->hit_count; ++hit) {
        ASSERT_TRUE(view->labels[hit] >= (INT64_C(1) << 40));
        ASSERT_TRUE(isfinite(view->raw_distances[hit]));
    }
}

static void range_assert_stats_equal(
        const PaimonVindexRangeSearchStats *actual,
        const PaimonVindexRangeSearchStats *expected) {
    ASSERT_TRUE(actual->lists_probed == expected->lists_probed);
    ASSERT_TRUE(actual->rows_scanned == expected->rows_scanned);
    ASSERT_TRUE(actual->rows_committed == expected->rows_committed);
    ASSERT_TRUE(actual->early_abandoned == expected->early_abandoned);
}

struct RangeFixture {
    size_t dimension;
    PaimonVindexRangeSearchParams params;
    size_t query_count;
    size_t filter_len;
    size_t hit_count;
    size_t list_reads;
    float *queries;
    uint8_t *filter;
    size_t *lims;
    int64_t *labels;
    uint32_t *raw_distance_bits;
    PaimonVindexRangeSearchStats *stats;
    uint8_t *index_data;
    size_t index_len;
};

static void *range_fixture_allocate(size_t count, size_t element_size) {
    ASSERT_TRUE(element_size != 0 && count <= SIZE_MAX / element_size);
    void *allocation = calloc(count == 0 ? 1 : count, element_size);
    ASSERT_TRUE(allocation != NULL);
    return allocation;
}

static FILE *range_fixture_open(const char *directory, const char *name, const char *suffix) {
    char filename[4096];
    int written = snprintf(filename, sizeof(filename), "%s/%s%s", directory, name, suffix);
    ASSERT_TRUE(written >= 0 && (size_t)written < sizeof(filename));
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Cannot read range fixture %s\n", filename);
        abort();
    }
    return file;
}

static void range_fixture_load(
        const char *directory, const char *name, const char *index_name,
        struct RangeFixture *fixture) {
    memset(fixture, 0, sizeof(*fixture));
    FILE *file = range_fixture_open(directory, name, ".expected");
    uint32_t raw_lower_bits;
    uint32_t raw_upper_bits;
    ASSERT_TRUE(fscanf(file, "%zu %" SCNu32 " %zu %zu %" SCNu32 " %" SCNu32
                      " %" SCNu32 " %" SCNu32 " %zu %zu %zu",
                      &fixture->dimension, &fixture->params.band.metric,
                      &fixture->query_count, &fixture->params.nprobe,
                      &fixture->params.band.raw_lower_kind, &raw_lower_bits,
                      &fixture->params.band.raw_upper_kind, &raw_upper_bits,
                      &fixture->filter_len, &fixture->hit_count, &fixture->list_reads) == 11);
    fixture->params.band.raw_lower = range_float_from_bits(raw_lower_bits);
    fixture->params.band.raw_upper = range_float_from_bits(raw_upper_bits);
    ASSERT_TRUE(fixture->dimension != 0);
    ASSERT_TRUE(fixture->query_count < SIZE_MAX);
    ASSERT_TRUE(fixture->query_count <= SIZE_MAX / fixture->dimension);
    size_t query_len = fixture->dimension * fixture->query_count;
    fixture->queries = (float *)range_fixture_allocate(query_len, sizeof(float));
    fixture->filter = (uint8_t *)range_fixture_allocate(fixture->filter_len, sizeof(uint8_t));
    fixture->lims = (size_t *)range_fixture_allocate(fixture->query_count + 1, sizeof(size_t));
    fixture->labels = (int64_t *)range_fixture_allocate(fixture->hit_count, sizeof(int64_t));
    fixture->raw_distance_bits = (uint32_t *)range_fixture_allocate(fixture->hit_count, sizeof(uint32_t));
    fixture->stats = (PaimonVindexRangeSearchStats *)range_fixture_allocate(
        fixture->query_count, sizeof(PaimonVindexRangeSearchStats));
    for (size_t element = 0; element < query_len; ++element) {
        uint32_t bits;
        ASSERT_TRUE(fscanf(file, "%" SCNu32, &bits) == 1);
        fixture->queries[element] = range_float_from_bits(bits);
    }
    for (size_t element = 0; element < fixture->filter_len; ++element) {
        unsigned int byte;
        ASSERT_TRUE(fscanf(file, "%u", &byte) == 1 && byte <= UINT8_MAX);
        fixture->filter[element] = (uint8_t)byte;
    }
    for (size_t element = 0; element <= fixture->query_count; ++element) {
        ASSERT_TRUE(fscanf(file, "%zu", &fixture->lims[element]) == 1);
    }
    for (size_t element = 0; element < fixture->hit_count; ++element) {
        ASSERT_TRUE(fscanf(file, "%" SCNd64, &fixture->labels[element]) == 1);
    }
    for (size_t element = 0; element < fixture->hit_count; ++element) {
        ASSERT_TRUE(fscanf(file, "%" SCNu32, &fixture->raw_distance_bits[element]) == 1);
    }
    for (size_t query = 0; query < fixture->query_count; ++query) {
        PaimonVindexRangeSearchStats *stats = &fixture->stats[query];
        ASSERT_TRUE(fscanf(file, "%zu %zu %zu %zu", &stats->lists_probed,
                          &stats->rows_scanned, &stats->rows_committed,
                          &stats->early_abandoned) == 4);
    }
    char trailing;
    ASSERT_TRUE(fscanf(file, " %c", &trailing) == EOF);
    ASSERT_TRUE(fclose(file) == 0);
    file = range_fixture_open(directory, index_name, "");
    ASSERT_TRUE(fseek(file, 0, SEEK_END) == 0);
    long index_len = ftell(file);
    ASSERT_TRUE(index_len > 0);
    fixture->index_len = (size_t)index_len;
    fixture->index_data = (uint8_t *)range_fixture_allocate(fixture->index_len, 1);
    ASSERT_TRUE(fseek(file, 0, SEEK_SET) == 0);
    ASSERT_TRUE(fread(fixture->index_data, 1, fixture->index_len, file) == fixture->index_len);
    ASSERT_TRUE(fclose(file) == 0);
}

struct RangeFixtureHit {
    int64_t label;
    uint32_t raw_distance_bits;
};

static int range_fixture_compare_hits(const void *left, const void *right) {
    const struct RangeFixtureHit *left_hit = (const struct RangeFixtureHit *)left;
    const struct RangeFixtureHit *right_hit = (const struct RangeFixtureHit *)right;
    if (left_hit->label != right_hit->label) return left_hit->label < right_hit->label ? -1 : 1;
    if (left_hit->raw_distance_bits != right_hit->raw_distance_bits) {
        return left_hit->raw_distance_bits < right_hit->raw_distance_bits ? -1 : 1;
    }
    return 0;
}

static void range_fixture_assert(
        const struct RangeFixture *fixture, const PaimonVindexRangeSearchResultView *view) {
    ASSERT_TRUE(view->query_count == fixture->query_count);
    ASSERT_TRUE(view->hit_count == fixture->hit_count);
    ASSERT_TRUE(view->list_reads == fixture->list_reads);
    for (size_t query = 0; query <= fixture->query_count; ++query) {
        ASSERT_TRUE(view->lims[query] == fixture->lims[query]);
    }
    struct RangeFixtureHit *actual = (struct RangeFixtureHit *)range_fixture_allocate(
        fixture->hit_count, sizeof(struct RangeFixtureHit));
    struct RangeFixtureHit *expected = (struct RangeFixtureHit *)range_fixture_allocate(
        fixture->hit_count, sizeof(struct RangeFixtureHit));
    for (size_t hit = 0; hit < fixture->hit_count; ++hit) {
        actual[hit].label = view->labels[hit];
        actual[hit].raw_distance_bits = range_float_bits(view->raw_distances[hit]);
        expected[hit].label = fixture->labels[hit];
        expected[hit].raw_distance_bits = fixture->raw_distance_bits[hit];
    }
    for (size_t query = 0; query < fixture->query_count; ++query) {
        size_t begin = fixture->lims[query];
        size_t count = fixture->lims[query + 1] - begin;
        qsort(actual + begin, count, sizeof(struct RangeFixtureHit), range_fixture_compare_hits);
        qsort(expected + begin, count, sizeof(struct RangeFixtureHit), range_fixture_compare_hits);
        for (size_t hit = begin; hit < begin + count; ++hit) {
            ASSERT_TRUE(range_fixture_compare_hits(&actual[hit], &expected[hit]) == 0);
        }
        range_assert_stats_equal(&view->stats[query], &fixture->stats[query]);
    }
    free(actual);
    free(expected);
}

static void range_fixture_run_all(void (*consume)(const struct RangeFixture *)) {
    const char *directory = getenv("PVI_RANGE_FIXTURES");
    if (directory == NULL) return;
    FILE *manifest = range_fixture_open(directory, "manifest.txt", "");
    char name[256];
    char index_name[256];
    size_t case_count = 0;
    int fields;
    while ((fields = fscanf(manifest, "%255s %255s", name, index_name)) == 2) {
        struct RangeFixture fixture;
        printf("ORACLE %s\n", name);
        range_fixture_load(directory, name, index_name, &fixture);
        consume(&fixture);
        free(fixture.queries);
        free(fixture.filter);
        free(fixture.lims);
        free(fixture.labels);
        free(fixture.raw_distance_bits);
        free(fixture.stats);
        free(fixture.index_data);
        ++case_count;
    }
    ASSERT_TRUE(fields == EOF && case_count != 0);
    ASSERT_TRUE(fclose(manifest) == 0);
    printf("PASS range_core_oracle (%zu cases)\n", case_count);
}

#endif
