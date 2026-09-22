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

#pragma once

extern "C" {
#include "paimon_vindex.h"
}

#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace paimon::vindex {

class Error : public std::runtime_error {
public:
    explicit Error(const std::string& msg) : std::runtime_error(msg) {}
};

inline void check(int result) {
    if (result != 0) {
        const char* err = paimon_vindex_last_error();
        throw Error(err ? err : "unknown vector index error");
    }
}

struct OutputFile {
    std::function<int(const uint8_t*, size_t)> write_fn;
    std::function<int()> flush_fn;
    std::function<int64_t()> get_pos_fn;
};

using ReadRequest = PaimonVindexReadRequest;

struct InputFile {
    // DiskANN batch search may invoke this callback from multiple worker threads.
    std::function<int(ReadRequest* requests, size_t request_count)> read_ranges_fn;
    // Optional positional-read capabilities. Zero leaves the policy unspecified.
    uint64_t estimated_random_read_latency_nanos = 0;
    size_t preferred_window_bytes = 0;
    size_t max_ranges_per_read = 0;
};

namespace detail {

class NativeCallbackScope {
public:
    explicit NativeCallbackScope(const void* context) noexcept
        : context_(context), previous_(current()) {
        current() = this;
    }

    NativeCallbackScope(const NativeCallbackScope&) = delete;
    NativeCallbackScope& operator=(const NativeCallbackScope&) = delete;

    ~NativeCallbackScope() {
        current() = previous_;
    }

    static bool active_for(const void* context) noexcept {
        if (context == nullptr) return false;
        for (auto* scope = current(); scope != nullptr; scope = scope->previous_) {
            if (scope->context_ == context) return true;
        }
        return false;
    }

private:
    static NativeCallbackScope*& current() noexcept {
        static thread_local NativeCallbackScope* scope = nullptr;
        return scope;
    }

    const void* context_;
    NativeCallbackScope* previous_;
};

class NativeHandleMutex {
public:
    void lock() {
        const auto current = std::this_thread::get_id();
        {
            std::lock_guard<std::mutex> state_lock(state_mutex_);
            if (owner_ == current ||
                NativeCallbackScope::active_for(callback_context_)) {
                throw Error("reentrant native-handle operation is not allowed");
            }
        }
        operation_mutex_.lock();
        std::lock_guard<std::mutex> state_lock(state_mutex_);
        owner_ = current;
    }

    bool try_lock() {
        const auto current = std::this_thread::get_id();
        {
            std::lock_guard<std::mutex> state_lock(state_mutex_);
            if (owner_ == current ||
                NativeCallbackScope::active_for(callback_context_)) {
                throw Error("reentrant native-handle operation is not allowed");
            }
        }
        if (!operation_mutex_.try_lock()) return false;
        std::lock_guard<std::mutex> state_lock(state_mutex_);
        owner_ = current;
        return true;
    }

    void unlock() noexcept {
        {
            std::lock_guard<std::mutex> state_lock(state_mutex_);
            owner_ = std::thread::id();
        }
        operation_mutex_.unlock();
    }

    void set_callback_context(const void* context) {
        std::lock_guard<std::mutex> state_lock(state_mutex_);
        callback_context_ = context;
    }

private:
    std::mutex operation_mutex_;
    std::mutex state_mutex_;
    std::thread::id owner_;
    const void* callback_context_ = nullptr;
};

inline int stream_write(void* ctx, const uint8_t* data, size_t len) noexcept {
    try {
        auto* cbs = static_cast<OutputFile*>(ctx);
        return cbs->write_fn(data, len);
    } catch (...) {
        return -1;
    }
}

inline int stream_flush(void* ctx) noexcept {
    try {
        auto* cbs = static_cast<OutputFile*>(ctx);
        if (!cbs->flush_fn) return 0;
        return cbs->flush_fn();
    } catch (...) {
        return -1;
    }
}

inline int64_t stream_get_pos(void* ctx) noexcept {
    try {
        auto* cbs = static_cast<OutputFile*>(ctx);
        if (!cbs->get_pos_fn) return -1;
        return cbs->get_pos_fn();
    } catch (...) {
        return -1;
    }
}

inline int input_read_ranges(
        void* ctx,
        PaimonVindexReadRequest* raw_requests,
        size_t request_count) noexcept {
    NativeCallbackScope callback_scope(ctx);
    try {
        auto* cbs = static_cast<InputFile*>(ctx);
        return cbs->read_ranges_fn(raw_requests, request_count);
    } catch (...) {
        return -1;
    }
}

} // namespace detail

struct Metadata {
    uint32_t index_type = 0;
    size_t dimension = 0;
    size_t nlist = 0;
    uint32_t metric = 0;
    int64_t total_vectors = 0;
    size_t pq_m = 0;
    size_t pq_bits = 0;
    size_t rq_bits = 0;
    size_t diskann_max_degree = 0;
    size_t diskann_build_search_list_size = 0;
    float diskann_alpha = 0.0f;
};

struct ReadPlan {
    uint64_t random_read_latency_nanos = 0;
    size_t window_bytes = 0;
    size_t max_ranges_per_read = 0;
    size_t graph_beam_width = 0;
    size_t filtered_graph_beam_width = 0;
    size_t adjacency_preload_bytes = 0;
    size_t adjacency_cache_bytes = 0;
    size_t raw_vector_cache_bytes = 0;
    size_t memory_budget_bytes = 0;
};

struct SearchResult {
    std::vector<int64_t> ids;
    std::vector<float> distances;
};

using DistanceEndpoint = PaimonVindexDistanceEndpoint;
using RangeSearchStats = PaimonVindexRangeSearchStats;

class DistanceBand {
public:
    static DistanceBand from_endpoints(
            uint32_t metric,
            std::optional<DistanceEndpoint> lower = std::nullopt,
            std::optional<DistanceEndpoint> upper = std::nullopt) {
        PaimonVindexRawDistanceBand raw{};
        check(paimon_vindex_distance_band_from_endpoints(
            metric, lower ? &*lower : nullptr, upper ? &*upper : nullptr, &raw));
        return DistanceBand(raw);
    }

    static DistanceBand from_raw(
            uint32_t metric, uint32_t raw_lower_kind, float raw_lower,
            uint32_t raw_upper_kind, float raw_upper) {
        return DistanceBand({metric, raw_lower_kind, raw_lower, raw_upper_kind, raw_upper});
    }

    uint32_t metric() const { return raw_.metric; }
    uint32_t raw_lower_kind() const { return raw_.raw_lower_kind; }
    float raw_lower() const { return raw_.raw_lower; }
    uint32_t raw_upper_kind() const { return raw_.raw_upper_kind; }
    float raw_upper() const { return raw_.raw_upper; }

    PaimonVindexRawDistanceBand to_ffi() const { return raw_; }

private:
    explicit DistanceBand(PaimonVindexRawDistanceBand raw) : raw_(raw) {}

    PaimonVindexRawDistanceBand raw_;
};

struct RangeSearchParams {
    DistanceBand band = DistanceBand::from_raw(
        PAIMON_VINDEX_METRIC_L2, PAIMON_VINDEX_BOUND_UNBOUNDED, 0.0f,
        PAIMON_VINDEX_BOUND_UNBOUNDED, 0.0f);
    size_t nprobe = 1;

    PaimonVindexRangeSearchParams to_ffi() const {
        return {band.to_ffi(), nprobe};
    }
};

struct RangeSearchResult {
    size_t query_count = 0;
    std::vector<size_t> lims;
    std::vector<int64_t> labels;
    std::vector<float> raw_distances;
    std::vector<RangeSearchStats> stats;
    size_t list_reads = 0;
};

namespace detail {

inline RangeSearchResult copy_range_result(PaimonVindexRangeSearchResult* raw, int status) {
    std::unique_ptr<PaimonVindexRangeSearchResult,
                    decltype(&paimon_vindex_range_search_result_destroy)> guard(
        raw, &paimon_vindex_range_search_result_destroy);
    check(status);
    PaimonVindexRangeSearchResultView view{};
    check(paimon_vindex_range_search_result_view(guard.get(), &view));
    if (view.query_count == std::numeric_limits<size_t>::max() || !view.lims ||
        (view.hit_count != 0 && (!view.labels || !view.raw_distances)) ||
        (view.query_count != 0 && !view.stats)) {
        throw Error("invalid native range result view");
    }
    RangeSearchResult result;
    result.query_count = view.query_count;
    result.lims.assign(view.lims, view.lims + view.query_count + 1);
    if (view.hit_count != 0) {
        result.labels.assign(view.labels, view.labels + view.hit_count);
        result.raw_distances.assign(view.raw_distances, view.raw_distances + view.hit_count);
    }
    if (view.query_count != 0) {
        result.stats.assign(view.stats, view.stats + view.query_count);
    }
    result.list_reads = view.list_reads;
    return result;
}

}

struct SearchParams {
    size_t top_k = 0;
    uint32_t search_width = PAIMON_VINDEX_SEARCH_WIDTH_AUTO;
    size_t width = 0;
    size_t max_initial_filter_expansion_factor = 0;
    uint32_t ivfpq_batch_table_reuse = PAIMON_VINDEX_IVFPQ_BATCH_TABLE_REUSE_AUTO;
    size_t ivfpq_batch_table_reuse_max_bytes =
        PAIMON_VINDEX_DEFAULT_IVFPQ_BATCH_TABLE_REUSE_MAX_BYTES;

    SearchParams(size_t top_k, size_t nprobe)
        : top_k(top_k),
          search_width(PAIMON_VINDEX_SEARCH_WIDTH_IVF_NPROBE),
          width(nprobe) {}

    static SearchParams automatic(size_t top_k) {
        SearchParams params;
        params.top_k = top_k;
        return params;
    }

    static SearchParams diskann(size_t top_k, size_t l_search) {
        SearchParams params;
        params.top_k = top_k;
        params.search_width = PAIMON_VINDEX_SEARCH_WIDTH_DISKANN_L_SEARCH;
        params.width = l_search;
        return params;
    }

    PaimonVindexSearchParams to_ffi() const {
        PaimonVindexSearchParams params;
        params.top_k = top_k;
        params.search_width = search_width;
        params.width = width;
        return params;
    }

    PaimonVindexSearchParamsV2 to_ffi_v2() const {
        PaimonVindexSearchParamsV2 params;
        params.top_k = top_k;
        params.search_width = search_width;
        params.width = width;
        params.ivfpq_batch_table_reuse = ivfpq_batch_table_reuse;
        params.ivfpq_batch_table_reuse_max_bytes = ivfpq_batch_table_reuse_max_bytes;
        return params;
    }

    PaimonVindexSearchParamsEx to_ffi_ex() const {
        auto params = paimon_vindex_search_params_ex_default();
        params.top_k = top_k;
        params.search_width = search_width;
        params.width = width;
        params.max_initial_filter_expansion_factor =
            max_initial_filter_expansion_factor;
        params.ivfpq_batch_table_reuse = ivfpq_batch_table_reuse;
        params.ivfpq_batch_table_reuse_max_bytes = ivfpq_batch_table_reuse_max_bytes;
        return params;
    }

private:
    SearchParams() = default;
};

class Training {
public:
    explicit Training(PaimonVindexTrainingHandle* handle = nullptr) : handle_(handle) {}

    Training(const Training&) = delete;
    Training& operator=(const Training&) = delete;

    Training(Training&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    Training& operator=(Training&& other) noexcept {
        if (this != &other) {
            if (handle_) paimon_vindex_training_free(handle_);
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    ~Training() {
        if (handle_) paimon_vindex_training_free(handle_);
    }

private:
    friend class Writer;

    PaimonVindexTrainingHandle* handle_ = nullptr;
};

class Trainer {
public:
    Trainer(const char* const* keys, const char* const* values, size_t num_options) {
        handle_ = paimon_vindex_trainer_open(keys, values, num_options);
        if (!handle_) throw Error("failed to open vector index trainer");
    }

    explicit Trainer(const std::vector<std::pair<std::string, std::string>>& options) {
        option_keys_.reserve(options.size());
        option_values_.reserve(options.size());
        key_ptrs_.reserve(options.size());
        value_ptrs_.reserve(options.size());
        for (const auto& option : options) {
            option_keys_.push_back(option.first);
            option_values_.push_back(option.second);
        }
        for (size_t i = 0; i < options.size(); i++) {
            key_ptrs_.push_back(option_keys_[i].c_str());
            value_ptrs_.push_back(option_values_[i].c_str());
        }
        handle_ = paimon_vindex_trainer_open(key_ptrs_.data(), value_ptrs_.data(), options.size());
        if (!handle_) throw Error("failed to open vector index trainer");
    }

    Trainer(const Trainer&) = delete;
    Trainer& operator=(const Trainer&) = delete;

    Trainer(Trainer&& other) noexcept
        : handle_(other.handle_),
          option_keys_(std::move(other.option_keys_)),
          option_values_(std::move(other.option_values_)),
          key_ptrs_(std::move(other.key_ptrs_)),
          value_ptrs_(std::move(other.value_ptrs_)) {
        other.handle_ = nullptr;
    }

    Trainer& operator=(Trainer&& other) noexcept {
        if (this != &other) {
            if (handle_) paimon_vindex_trainer_free(handle_);
            handle_ = other.handle_;
            option_keys_ = std::move(other.option_keys_);
            option_values_ = std::move(other.option_values_);
            key_ptrs_ = std::move(other.key_ptrs_);
            value_ptrs_ = std::move(other.value_ptrs_);
            other.handle_ = nullptr;
        }
        return *this;
    }

    ~Trainer() {
        if (handle_) paimon_vindex_trainer_free(handle_);
    }

    size_t dimension() const {
        size_t out = 0;
        check(paimon_vindex_trainer_dimension(handle_, &out));
        return out;
    }

    Trainer& add_training_vectors(const float* data, size_t vector_count) {
        check(paimon_vindex_trainer_add_training_vectors(handle_, data, vector_count));
        return *this;
    }

    // C ABI note: finish consumes the trainer state but leaves the trainer handle owned by caller.
    // This RAII wrapper frees the trainer handle after a successful finish.
    Training finish_training() {
        PaimonVindexTrainingHandle* training = paimon_vindex_trainer_finish(handle_);
        if (!training) {
            const char* err = paimon_vindex_last_error();
            throw Error(err ? err : "failed to finish vector index training");
        }
        paimon_vindex_trainer_free(handle_);
        handle_ = nullptr;
        return Training(training);
    }

    static Training train(
            const std::vector<std::pair<std::string, std::string>>& options,
            const float* data,
            size_t vector_count) {
        Trainer trainer(options);
        trainer.add_training_vectors(data, vector_count);
        return trainer.finish_training();
    }

private:
    PaimonVindexTrainerHandle* handle_ = nullptr;
    std::vector<std::string> option_keys_;
    std::vector<std::string> option_values_;
    std::vector<const char*> key_ptrs_;
    std::vector<const char*> value_ptrs_;
};

class Writer {
public:
    explicit Writer(Training&& training) {
        if (!training.handle_) throw Error("training has already been consumed");
        PaimonVindexTrainingHandle* training_handle = training.handle_;
        training.handle_ = nullptr;
        // C ABI note: writer_open consumes the training state but leaves the handle owned by caller.
        // This RAII wrapper frees the consumed training handle after opening the writer.
        handle_ = paimon_vindex_writer_open(training_handle);
        paimon_vindex_training_free(training_handle);
        if (!handle_) {
            const char* err = paimon_vindex_last_error();
            throw Error(err ? err : "failed to open vector index writer");
        }
    }

    Writer(const Writer&) = delete;
    Writer& operator=(const Writer&) = delete;

    Writer(Writer&& other) noexcept
        : handle_(other.handle_),
          output_(std::move(other.output_)) {
        other.handle_ = nullptr;
    }

    Writer& operator=(Writer&& other) noexcept {
        if (this != &other) {
            if (handle_) paimon_vindex_writer_free(handle_);
            handle_ = other.handle_;
            output_ = std::move(other.output_);
            other.handle_ = nullptr;
        }
        return *this;
    }

    ~Writer() {
        if (handle_) paimon_vindex_writer_free(handle_);
    }

    size_t dimension() const {
        size_t out = 0;
        check(paimon_vindex_writer_dimension(handle_, &out));
        return out;
    }

    void add_vectors(const int64_t* ids, const float* data, size_t vector_count) {
        check(paimon_vindex_writer_add_vectors(handle_, ids, data, vector_count));
    }

    void write_index(OutputFile output) {
        output_ = std::make_shared<OutputFile>(std::move(output));
        PaimonVindexOutputFile raw;
        raw.ctx = output_.get();
        raw.write_fn = detail::stream_write;
        raw.flush_fn = detail::stream_flush;
        raw.get_pos_fn = detail::stream_get_pos;
        check(paimon_vindex_writer_write_index(handle_, raw));
    }

private:
    PaimonVindexWriterHandle* handle_ = nullptr;
    std::shared_ptr<OutputFile> output_;
};

class Reader {
public:
    explicit Reader(InputFile input)
        : Reader(
              std::move(input),
              static_cast<size_t>(4ULL * 1024 * 1024 * 1024)) {}

    Reader(InputFile input, size_t memory_budget_bytes)
        : input_(std::make_shared<InputFile>(std::move(input))) {
        native_handle_mutex_.set_callback_context(input_.get());
        PaimonVindexInputFile raw;
        raw.ctx = input_.get();
        raw.read_ranges_fn = detail::input_read_ranges;
        raw.estimated_random_read_latency_nanos =
                input_->estimated_random_read_latency_nanos;
        raw.preferred_window_bytes = input_->preferred_window_bytes;
        raw.max_ranges_per_read = input_->max_ranges_per_read;
        PaimonVindexReaderOptions options;
        options.memory_budget_bytes = memory_budget_bytes;
        handle_ = paimon_vindex_reader_open_with_options(raw, options);
        if (!handle_) throw Error("failed to open vector index reader");
    }

    Reader(const Reader&) = delete;
    Reader& operator=(const Reader&) = delete;

    Reader(Reader&& other) {
        std::lock_guard<detail::NativeHandleMutex> lock(other.native_handle_mutex_);
        handle_ = other.handle_;
        input_ = std::move(other.input_);
        native_handle_mutex_.set_callback_context(input_.get());
        other.native_handle_mutex_.set_callback_context(nullptr);
        other.handle_ = nullptr;
    }

    Reader& operator=(Reader&& other) {
        if (this != &other) {
            std::scoped_lock lock(native_handle_mutex_, other.native_handle_mutex_);
            if (handle_) paimon_vindex_reader_free(handle_);
            handle_ = other.handle_;
            input_ = std::move(other.input_);
            native_handle_mutex_.set_callback_context(input_.get());
            other.native_handle_mutex_.set_callback_context(nullptr);
            other.handle_ = nullptr;
        }
        return *this;
    }

    ~Reader() {
        std::lock_guard<detail::NativeHandleMutex> lock(native_handle_mutex_);
        if (handle_) paimon_vindex_reader_free(handle_);
    }

    Metadata metadata() const {
        std::lock_guard<detail::NativeHandleMutex> lock(native_handle_mutex_);
        PaimonVindexMetadata raw;
        check(paimon_vindex_reader_metadata(require_open(), &raw));
        Metadata result;
        result.index_type = raw.index_type;
        result.dimension = raw.dimension;
        result.nlist = raw.nlist;
        result.metric = raw.metric;
        result.total_vectors = raw.total_vectors;
        result.pq_m = raw.pq_m;
        result.pq_bits = raw.pq_bits;
        result.rq_bits = raw.rq_bits;
        result.diskann_max_degree = raw.diskann_max_degree;
        result.diskann_build_search_list_size = raw.diskann_build_search_list_size;
        result.diskann_alpha = raw.diskann_alpha;
        return result;
    }

    void optimize_for_search() {
        std::lock_guard<detail::NativeHandleMutex> lock(native_handle_mutex_);
        check(paimon_vindex_reader_optimize_for_search(require_open()));
    }

    void warmup_queries(
            const float* queries, size_t query_count, size_t l_search = 0) {
        std::lock_guard<detail::NativeHandleMutex> lock(native_handle_mutex_);
        check(paimon_vindex_reader_warmup_queries(
            require_open(), queries, query_count, l_search));
    }

    size_t calibrate_search_width(
            const float* queries, size_t query_count, size_t top_k = 10) {
        std::lock_guard<detail::NativeHandleMutex> lock(native_handle_mutex_);
        size_t l_search = 0;
        check(paimon_vindex_reader_calibrate_search_width(
            require_open(), queries, query_count, top_k, &l_search));
        return l_search;
    }

    ReadPlan read_plan() const {
        std::lock_guard<detail::NativeHandleMutex> lock(native_handle_mutex_);
        PaimonVindexReadPlan raw;
        check(paimon_vindex_reader_read_plan(require_open(), &raw));
        ReadPlan result;
        result.random_read_latency_nanos = raw.random_read_latency_nanos;
        result.window_bytes = raw.window_bytes;
        result.max_ranges_per_read = raw.max_ranges_per_read;
        result.graph_beam_width = raw.graph_beam_width;
        result.filtered_graph_beam_width = raw.filtered_graph_beam_width;
        result.adjacency_preload_bytes = raw.adjacency_preload_bytes;
        result.adjacency_cache_bytes = raw.adjacency_cache_bytes;
        result.raw_vector_cache_bytes = raw.raw_vector_cache_bytes;
        result.memory_budget_bytes = raw.memory_budget_bytes;
        return result;
    }

    bool supports_range_search() const {
        std::lock_guard<detail::NativeHandleMutex> lock(native_handle_mutex_);
        int supported = 0;
        check(paimon_vindex_reader_supports_range_search(require_open(), &supported));
        return supported != 0;
    }

    RangeSearchResult range_search(
            const float* query, size_t query_len, RangeSearchParams params) {
        std::lock_guard<detail::NativeHandleMutex> lock(native_handle_mutex_);
        PaimonVindexRangeSearchResult* raw = nullptr;
        int status = paimon_vindex_reader_range_search(
            require_open(), query, query_len, params.to_ffi(), &raw);
        return detail::copy_range_result(raw, status);
    }

    RangeSearchResult range_search(const std::vector<float>& query, RangeSearchParams params) {
        return range_search(query.data(), query.size(), params);
    }

    RangeSearchResult range_search_with_roaring_filter(
            const float* query, size_t query_len, RangeSearchParams params,
            const uint8_t* filter, size_t filter_len) {
        std::lock_guard<detail::NativeHandleMutex> lock(native_handle_mutex_);
        PaimonVindexRangeSearchResult* raw = nullptr;
        int status = paimon_vindex_reader_range_search_with_roaring_filter(
            require_open(), query, query_len, params.to_ffi(), filter, filter_len, &raw);
        return detail::copy_range_result(raw, status);
    }

    RangeSearchResult range_search_with_roaring_filter(
            const std::vector<float>& query, RangeSearchParams params,
            const uint8_t* filter, size_t filter_len) {
        return range_search_with_roaring_filter(query.data(), query.size(), params, filter, filter_len);
    }

    RangeSearchResult range_search_batch(
            const float* queries, size_t queries_len, size_t query_count, RangeSearchParams params) {
        std::lock_guard<detail::NativeHandleMutex> lock(native_handle_mutex_);
        PaimonVindexRangeSearchResult* raw = nullptr;
        int status = paimon_vindex_reader_range_search_batch(
            require_open(), queries, queries_len, query_count, params.to_ffi(), &raw);
        return detail::copy_range_result(raw, status);
    }

    RangeSearchResult range_search_batch(
            const std::vector<float>& queries, size_t query_count, RangeSearchParams params) {
        return range_search_batch(queries.data(), queries.size(), query_count, params);
    }

    RangeSearchResult range_search_batch_with_roaring_filter(
            const float* queries, size_t queries_len, size_t query_count, RangeSearchParams params,
            const uint8_t* filter, size_t filter_len) {
        std::lock_guard<detail::NativeHandleMutex> lock(native_handle_mutex_);
        PaimonVindexRangeSearchResult* raw = nullptr;
        int status = paimon_vindex_reader_range_search_batch_with_roaring_filter(
            require_open(), queries, queries_len, query_count, params.to_ffi(), filter, filter_len, &raw);
        return detail::copy_range_result(raw, status);
    }

    RangeSearchResult range_search_batch_with_roaring_filter(
            const std::vector<float>& queries, size_t query_count, RangeSearchParams params,
            const uint8_t* filter, size_t filter_len) {
        return range_search_batch_with_roaring_filter(
            queries.data(), queries.size(), query_count, params, filter, filter_len);
    }

    SearchResult search(const float* query, SearchParams params) {
        std::lock_guard<detail::NativeHandleMutex> lock(native_handle_mutex_);
        SearchResult result;
        result.ids.resize(params.top_k);
        result.distances.resize(params.top_k);
        auto raw_params = params.to_ffi_ex();
        check(paimon_vindex_reader_search_ex(
            require_open(),
            query,
            &raw_params,
            result.ids.data(),
            result.distances.data(),
            params.top_k));
        return result;
    }

    SearchResult search_with_roaring_filter(
        const float* query,
        SearchParams params,
        const uint8_t* filter,
            size_t filter_len) {
        std::lock_guard<detail::NativeHandleMutex> lock(native_handle_mutex_);
        SearchResult result;
        result.ids.resize(params.top_k);
        result.distances.resize(params.top_k);
        auto raw_params = params.to_ffi_ex();
        check(paimon_vindex_reader_search_with_roaring_filter_ex(
            require_open(),
            query,
            &raw_params,
            filter,
            filter_len,
            result.ids.data(),
            result.distances.data(),
            params.top_k));
        return result;
    }

    SearchResult search_batch(
        const float* queries,
            size_t query_count,
            SearchParams params) {
        std::lock_guard<detail::NativeHandleMutex> lock(native_handle_mutex_);
        const size_t result_len = query_count * params.top_k;
        SearchResult result;
        result.ids.resize(result_len);
        result.distances.resize(result_len);
        auto raw_params = params.to_ffi_ex();
        check(paimon_vindex_reader_search_batch_ex(
            require_open(),
            queries,
            query_count,
            &raw_params,
            result.ids.data(),
            result.distances.data(),
            result_len));
        return result;
    }

    SearchResult search_batch_with_roaring_filter(
        const float* queries,
        size_t query_count,
        SearchParams params,
            const uint8_t* filter,
            size_t filter_len) {
        std::lock_guard<detail::NativeHandleMutex> lock(native_handle_mutex_);
        const size_t result_len = query_count * params.top_k;
        SearchResult result;
        result.ids.resize(result_len);
        result.distances.resize(result_len);
        auto raw_params = params.to_ffi_ex();
        check(paimon_vindex_reader_search_batch_with_roaring_filter_ex(
            require_open(),
            queries,
            query_count,
            &raw_params,
            filter,
            filter_len,
            result.ids.data(),
            result.distances.data(),
            result_len));
        return result;
    }

private:
    PaimonVindexReaderHandle* require_open() const {
        if (!handle_) throw Error("vector index reader is closed");
        return handle_;
    }

    mutable detail::NativeHandleMutex native_handle_mutex_;
    PaimonVindexReaderHandle* handle_ = nullptr;
    std::shared_ptr<InputFile> input_;
};

} // namespace paimon::vindex
