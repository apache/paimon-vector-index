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
#include <memory>
#include <stdexcept>
#include <string>
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

struct InputFile {
    std::function<int(uint64_t offset, uint8_t* buf, size_t len)> read_at_fn;
};

namespace detail {

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

inline int input_read_at(void* ctx, uint64_t offset, uint8_t* buf, size_t len) noexcept {
    try {
        auto* cbs = static_cast<InputFile*>(ctx);
        return cbs->read_at_fn(offset, buf, len);
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
    size_t hnsw_m = 0;
    size_t hnsw_ef_construction = 0;
    size_t hnsw_max_level = 0;
};

struct SearchResult {
    std::vector<int64_t> ids;
    std::vector<float> distances;
};

struct SearchParams {
    size_t top_k = 0;
    size_t nprobe = 0;
    size_t ef_search = 0;
    size_t query_bits = 0;

    SearchParams(size_t top_k, size_t nprobe, size_t ef_search = 0, size_t query_bits = 0)
        : top_k(top_k), nprobe(nprobe), ef_search(ef_search), query_bits(query_bits) {}

    PaimonVindexSearchParams to_ffi() const {
        PaimonVindexSearchParams params;
        params.top_k = top_k;
        params.nprobe = nprobe;
        params.ef_search = ef_search;
        params.query_bits = query_bits;
        return params;
    }
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
    explicit Reader(InputFile input) : input_(std::make_shared<InputFile>(std::move(input))) {
        PaimonVindexInputFile raw;
        raw.ctx = input_.get();
        raw.read_at_fn = detail::input_read_at;
        handle_ = paimon_vindex_reader_open(raw);
        if (!handle_) throw Error("failed to open vector index reader");
    }

    Reader(const Reader&) = delete;
    Reader& operator=(const Reader&) = delete;

    Reader(Reader&& other) noexcept
        : handle_(other.handle_), input_(std::move(other.input_)) {
        other.handle_ = nullptr;
    }

    Reader& operator=(Reader&& other) noexcept {
        if (this != &other) {
            if (handle_) paimon_vindex_reader_free(handle_);
            handle_ = other.handle_;
            input_ = std::move(other.input_);
            other.handle_ = nullptr;
        }
        return *this;
    }

    ~Reader() {
        if (handle_) paimon_vindex_reader_free(handle_);
    }

    Metadata metadata() const {
        PaimonVindexMetadata raw;
        check(paimon_vindex_reader_metadata(handle_, &raw));
        Metadata result;
        result.index_type = raw.index_type;
        result.dimension = raw.dimension;
        result.nlist = raw.nlist;
        result.metric = raw.metric;
        result.total_vectors = raw.total_vectors;
        result.pq_m = raw.pq_m;
        result.hnsw_m = raw.hnsw_m;
        result.hnsw_ef_construction = raw.hnsw_ef_construction;
        result.hnsw_max_level = raw.hnsw_max_level;
        return result;
    }

    void optimize_for_search() {
        check(paimon_vindex_reader_optimize_for_search(handle_));
    }

    SearchResult search(const float* query, SearchParams params) {
        SearchResult result;
        result.ids.resize(params.top_k);
        result.distances.resize(params.top_k);
        check(paimon_vindex_reader_search(
            handle_,
            query,
            params.to_ffi(),
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
        SearchResult result;
        result.ids.resize(params.top_k);
        result.distances.resize(params.top_k);
        check(paimon_vindex_reader_search_with_roaring_filter(
            handle_,
            query,
            params.to_ffi(),
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
        const size_t result_len = query_count * params.top_k;
        SearchResult result;
        result.ids.resize(result_len);
        result.distances.resize(result_len);
        check(paimon_vindex_reader_search_batch(
            handle_,
            queries,
            query_count,
            params.to_ffi(),
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
        const size_t result_len = query_count * params.top_k;
        SearchResult result;
        result.ids.resize(result_len);
        result.distances.resize(result_len);
        check(paimon_vindex_reader_search_batch_with_roaring_filter(
            handle_,
            queries,
            query_count,
            params.to_ffi(),
            filter,
            filter_len,
            result.ids.data(),
            result.distances.data(),
            result_len));
        return result;
    }

private:
    PaimonVindexReaderHandle* handle_ = nullptr;
    std::shared_ptr<InputFile> input_;
};

} // namespace paimon::vindex
