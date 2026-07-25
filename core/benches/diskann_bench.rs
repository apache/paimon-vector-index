// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use paimon_vindex_core::diskann::{
    DiskAnnBuildDistance, DiskAnnBuildParams, DiskAnnBuildStats, DiskAnnIndex,
    DiskAnnRawVectorEncoding, DiskAnnStorageLayout,
};
use paimon_vindex_core::diskann_io::{
    write_diskann_index_with_stats, DiskAnnHeader, DISKANN_HEADER_SIZE, DISKANN_PAGE_SIZE,
};
use paimon_vindex_core::distance::{fvec_l2sqr, MetricType};
use paimon_vindex_core::index::{
    infer_pq_m, StorageProfile, VectorIndexReader, VectorIndexReaderOptions, VectorSearchParams,
    DEFAULT_PQ_CODE_RATIO,
};
use paimon_vindex_core::io::{PosWriter, ReadRequest, SeekRead};
use rayon::prelude::*;
use roaring::RoaringTreemap;
use std::env;
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, Read};
use std::mem::MaybeUninit;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = Dataset::from_env_or_smoke()?;
    let ids = (0..dataset.base.len() / dataset.dimension)
        .map(|id| id as i64)
        .collect::<Vec<_>>();
    let pq_bits = env::var("DISKANN_BENCH_PQ_BITS")
        .unwrap_or_else(|_| "8".to_string())
        .parse::<usize>()?;
    if !matches!(pq_bits, 4 | 8) {
        return Err("DISKANN_BENCH_PQ_BITS must be 4 or 8".into());
    }
    let pq_m = match env::var("DISKANN_BENCH_PQ_M") {
        Ok(value) => value.parse::<usize>()?,
        Err(env::VarError::NotPresent) => {
            let code_ratio = env::var("DISKANN_BENCH_PQ_CODE_RATIO")
                .unwrap_or_else(|_| DEFAULT_PQ_CODE_RATIO.to_string())
                .parse::<f64>()?;
            infer_pq_m(dataset.dimension, pq_bits, code_ratio)?
        }
        Err(error) => return Err(Box::new(error)),
    };
    if pq_m == 0 || pq_m > dataset.dimension {
        return Err("DISKANN_BENCH_PQ_M must be positive and not exceed the dimension".into());
    }
    let storage_layout = match env::var("DISKANN_BENCH_STORAGE_LAYOUT")
        .unwrap_or_else(|_| "compact".to_string())
        .as_str()
    {
        "compact" => DiskAnnStorageLayout::Compact,
        "interleaved" => DiskAnnStorageLayout::Interleaved,
        _ => return Err("DISKANN_BENCH_STORAGE_LAYOUT must be compact or interleaved".into()),
    };
    let raw_vector_encoding = match env::var("DISKANN_BENCH_RAW_VECTOR_ENCODING")
        .unwrap_or_else(|_| "f32".to_string())
        .as_str()
    {
        "f32" => DiskAnnRawVectorEncoding::F32,
        "f16" => DiskAnnRawVectorEncoding::F16,
        _ => return Err("DISKANN_BENCH_RAW_VECTOR_ENCODING must be f32 or f16".into()),
    };
    let build_distance = match env::var("DISKANN_BENCH_BUILD_DISTANCE")
        .unwrap_or_else(|_| "product_quantized".to_string())
        .as_str()
    {
        "full_precision" => DiskAnnBuildDistance::FullPrecision,
        "product_quantized" => DiskAnnBuildDistance::ProductQuantized,
        _ => {
            return Err(
                "DISKANN_BENCH_BUILD_DISTANCE must be full_precision or product_quantized".into(),
            )
        }
    };
    let build_params = DiskAnnBuildParams {
        storage_layout,
        raw_vector_encoding,
        build_distance,
        ..DiskAnnBuildParams::default()
    };
    let build_start = Instant::now();
    let mut index = DiskAnnIndex::with_pq_bits(
        dataset.dimension,
        MetricType::L2,
        pq_m,
        pq_bits,
        build_params,
    );
    let pq_training_started = Instant::now();
    index.train(&dataset.base, ids.len());
    let pq_training_time = pq_training_started.elapsed();
    index.add(&dataset.base, &ids);
    let (index_file, mut output) = TemporaryIndexFile::create()?;
    let build_stats = write_diskann_index_with_stats(&index, &mut PosWriter::new(&mut output))?;
    output.sync_all()?;
    let file_bytes = output.metadata()?.len();
    drop(output);
    let (adjacency_section_bytes, adjacency_pages) =
        read_adjacency_layout_metrics(index_file.path())?;
    let build_time = build_start.elapsed();
    let peak_rss_bytes = peak_resident_set_bytes()?;
    let concurrency = env::var("DISKANN_BENCH_CONCURRENCY")
        .unwrap_or_else(|_| "1".to_string())
        .parse::<usize>()?;
    if concurrency == 0 {
        return Err("DISKANN_BENCH_CONCURRENCY must be greater than zero".into());
    }
    let reader_memory_budget_bytes = env::var("DISKANN_BENCH_READER_MEMORY_BUDGET_BYTES")
        .unwrap_or_else(|_| (4usize * 1024 * 1024 * 1024).to_string())
        .parse::<usize>()?;
    let warmup_query_count = env::var("DISKANN_BENCH_WARMUP_QUERIES")
        .unwrap_or_else(|_| "8".to_string())
        .parse::<usize>()?
        .min(dataset.queries.len() / dataset.dimension);

    println!(
        "storage_profile,storage_layout,raw_vector_encoding,build_distance,l_search,n,nq,d,pq_bits,k,concurrency,reader_memory_budget_bytes,warmup_query_count,warmup_ms,warmup_pread_rounds,warmup_pread_ranges,warmup_pread_bytes,recall_at_1,recall_at_10,build_ms,graph_shards,pq_train_ms,pq_encode_ms,vamana_init_ms,vamana_pass_one_ms,vamana_pass_two_ms,connectivity_repair_ms,locality_remap_ms,resident_serialize_ms,adjacency_serialize_ms,vector_serialize_ms,peak_rss_bytes,first_query_us,p50_query_us,p95_query_us,p99_query_us,warm_query_us,qps,pread_rounds,pread_ranges,max_ranges_per_round,max_in_flight_rounds,max_in_flight_ranges,pread_bytes,pread_wait_ms,adjacency_cache_hits,adjacency_cache_misses,adjacency_cache_waits,adjacency_cache_evictions,adjacency_cache_lock_acquisitions,adjacency_cache_lock_wait_ns,query_adjacency_cache_peak_bytes,query_adjacency_cache_evictions,rerank_candidate_references,rerank_unique_windows,raw_vector_cache_hits,raw_vector_cache_misses,raw_vector_cache_evictions,parallel_session_queries,simulated_rtt_ms,adjacency_section_bytes,adjacency_pages,file_bytes"
    );
    for storage_profile in [
        StorageProfile::Memory,
        StorageProfile::LocalStorage,
        StorageProfile::RemoteStorage,
        StorageProfile::ObjectStore,
    ] {
        for l_search in [50, 100, 200] {
            run_profile(
                &dataset,
                ProfileRun {
                    index_path: index_file.path(),
                    file_bytes,
                    adjacency_section_bytes,
                    adjacency_pages,
                    pq_bits,
                    storage_layout,
                    raw_vector_encoding,
                    build_distance,
                    storage_profile,
                    l_search,
                    concurrency,
                    reader_memory_budget_bytes,
                    warmup_query_count,
                    build_time,
                    pq_training_time,
                    build_stats,
                    peak_rss_bytes,
                },
            )?;
        }
    }
    if env::var_os("DISKANN_BENCH_FILTERED_MATRIX").is_some() {
        run_filtered_matrix(&dataset, index_file.path(), reader_memory_budget_bytes)?;
    }
    Ok(())
}

fn read_adjacency_layout_metrics(path: &Path) -> io::Result<(u64, u64)> {
    let mut header_bytes = [0u8; DISKANN_HEADER_SIZE];
    File::open(path)?.read_exact(&mut header_bytes)?;
    let header = DiskAnnHeader::decode(&header_bytes)?;
    let adjacency_section_bytes = header.sections.adjacency.length;
    Ok((
        adjacency_section_bytes,
        adjacency_section_bytes / u64::from(DISKANN_PAGE_SIZE),
    ))
}

fn run_filtered_matrix(
    dataset: &Dataset,
    index_path: &Path,
    reader_memory_budget_bytes: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let stats = Arc::new(Mutex::new(ReadStats::default()));
    let source = InstrumentedStore {
        inner: BenchmarkFileSource::open(index_path)?,
        stats: Arc::clone(&stats),
        round_trip_latency: Duration::ZERO,
    };
    let mut adaptive = VectorIndexReader::open_with_options(
        source,
        VectorIndexReaderOptions::new(StorageProfile::LocalStorage, reader_memory_budget_bytes),
    )?;
    adaptive.optimize_for_search()?;
    let mut exhaustive = VectorIndexReader::open_with_options(
        BenchmarkFileSource::open(index_path)?,
        VectorIndexReaderOptions::new(StorageProfile::ObjectStore, reader_memory_budget_bytes),
    )?;
    exhaustive.optimize_for_search()?;
    let vector_count = dataset.base.len() / dataset.dimension;
    let query_count = dataset.queries.len() / dataset.dimension;
    let params = VectorSearchParams::with_l_search(10, 200);
    println!(
        "filtered_distribution,selectivity_percent,matching_nodes,actual_strategy,graph_fallbacks,recall_at_10_vs_exhaustive,elapsed_ms,pread_rounds,pread_ranges,max_ranges_per_round,pread_bytes,pq_distance_evaluations,pq_code_loads,query_chunks,rerank_chunks,rerank_candidate_references,rerank_unique_windows,adjacency_cache_hits,adjacency_cache_misses,adjacency_cache_waits,adjacency_cache_evictions,raw_vector_cache_hits,raw_vector_cache_misses,raw_vector_cache_evictions,parallel_exact_rerank_chunks,parallel_exact_rerank_references"
    );
    for distribution in ["random", "clustered"] {
        let mut ordered_nodes = (0..vector_count).collect::<Vec<_>>();
        if distribution == "random" {
            ordered_nodes.sort_unstable_by_key(|node| benchmark_mix(*node as u64));
        }
        for selectivity_basis_points in [1usize, 10, 100, 1000, 5000, 10_000] {
            let matching_count = vector_count
                .saturating_mul(selectivity_basis_points)
                .div_ceil(10_000)
                .max(1);
            let mut filter = RoaringTreemap::new();
            filter.extend(
                ordered_nodes[..matching_count]
                    .iter()
                    .map(|node| *node as u64),
            );
            let mut filter_bytes = Vec::new();
            filter.serialize_into(&mut filter_bytes)?;
            *stats.lock().unwrap() = ReadStats::default();
            let started = Instant::now();
            let (adaptive_ids, _) = adaptive.search_batch_with_roaring_filter(
                &dataset.queries,
                query_count,
                params,
                &filter_bytes,
            )?;
            let elapsed = started.elapsed();
            let (exhaustive_ids, _) = exhaustive.search_batch_with_roaring_filter(
                &dataset.queries,
                query_count,
                params,
                &filter_bytes,
            )?;
            let recall = recall_against_exhaustive(&adaptive_ids, &exhaustive_ids, params.top_k);
            let search_stats = adaptive
                .diskann_search_stats()
                .expect("DiskANN benchmark reader diagnostics");
            let strategy = if search_stats.filtered_graph_queries == 0 {
                "scan"
            } else if search_stats.filtered_graph_fallbacks == 0 {
                "graph"
            } else {
                "graph_fallback"
            };
            let snapshot = *stats.lock().unwrap();
            println!(
                "{distribution},{selectivity:.2},{matching_count},{strategy},{fallbacks},{recall:.4},{elapsed_ms},{rounds},{ranges},{max_ranges},{bytes},{pq_evaluations},{pq_loads},{query_chunks},{rerank_chunks},{candidate_references},{unique_windows},{adjacency_hits},{adjacency_misses},{adjacency_waits},{adjacency_evictions},{cache_hits},{cache_misses},{cache_evictions},{parallel_rerank_chunks},{parallel_rerank_references}",
                selectivity = selectivity_basis_points as f64 / 100.0,
                fallbacks = search_stats.filtered_graph_fallbacks,
                elapsed_ms = elapsed.as_millis(),
                rounds = snapshot.rounds,
                ranges = snapshot.ranges,
                max_ranges = snapshot.max_ranges_per_round,
                bytes = snapshot.bytes,
                pq_evaluations = search_stats.pq_distance_evaluations,
                pq_loads = search_stats.pq_code_loads,
                query_chunks = search_stats.query_chunks,
                rerank_chunks = search_stats.rerank_chunks,
                candidate_references = search_stats.rerank_candidate_references,
                unique_windows = search_stats.rerank_unique_windows,
                adjacency_hits = search_stats.adjacency_cache_hits,
                adjacency_misses = search_stats.adjacency_cache_misses,
                adjacency_waits = search_stats.adjacency_cache_waits,
                adjacency_evictions = search_stats.adjacency_cache_evictions,
                cache_hits = search_stats.raw_vector_cache_hits,
                cache_misses = search_stats.raw_vector_cache_misses,
                cache_evictions = search_stats.raw_vector_cache_evictions,
                parallel_rerank_chunks = search_stats.parallel_exact_rerank_chunks,
                parallel_rerank_references = search_stats.parallel_exact_rerank_references,
            );
            if search_stats.filtered_graph_queries != 0
                && env::var_os("DISKANN_BENCH_ACCEPTANCE").is_some()
                && recall + 0.01 + f64::EPSILON < 1.0
            {
                return Err(format!(
                    "filtered {distribution} Recall@10 {:.4} is more than 0.01 below exhaustive at {:.2}% selectivity",
                    recall,
                    selectivity_basis_points as f64 / 100.0
                )
                .into());
            }
        }
    }
    Ok(())
}

fn benchmark_mix(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn recall_against_exhaustive(actual: &[i64], expected: &[i64], top_k: usize) -> f64 {
    let mut hits = 0usize;
    let mut total = 0usize;
    for (actual, expected) in actual.chunks_exact(top_k).zip(expected.chunks_exact(top_k)) {
        let expected = expected
            .iter()
            .copied()
            .filter(|row_id| *row_id >= 0)
            .collect::<Vec<_>>();
        total += expected.len();
        hits += actual
            .iter()
            .filter(|row_id| **row_id >= 0 && expected.contains(row_id))
            .count();
    }
    if total == 0 {
        1.0
    } else {
        hits as f64 / total as f64
    }
}

struct ProfileRun<'a> {
    index_path: &'a Path,
    file_bytes: u64,
    adjacency_section_bytes: u64,
    adjacency_pages: u64,
    pq_bits: usize,
    storage_layout: DiskAnnStorageLayout,
    raw_vector_encoding: DiskAnnRawVectorEncoding,
    build_distance: DiskAnnBuildDistance,
    storage_profile: StorageProfile,
    l_search: usize,
    concurrency: usize,
    reader_memory_budget_bytes: usize,
    warmup_query_count: usize,
    build_time: Duration,
    pq_training_time: Duration,
    build_stats: DiskAnnBuildStats,
    peak_rss_bytes: u64,
}

fn run_profile(dataset: &Dataset, run: ProfileRun<'_>) -> Result<(), Box<dyn std::error::Error>> {
    let ProfileRun {
        index_path,
        file_bytes,
        adjacency_section_bytes,
        adjacency_pages,
        pq_bits,
        storage_layout,
        raw_vector_encoding,
        build_distance,
        storage_profile,
        l_search,
        concurrency: requested_concurrency,
        reader_memory_budget_bytes,
        warmup_query_count,
        build_time,
        pq_training_time,
        build_stats,
        peak_rss_bytes,
    } = run;
    let stats = Arc::new(Mutex::new(ReadStats::default()));
    let round_trip_latency = Duration::from_millis(
        env::var("DISKANN_BENCH_SIMULATED_RTT_MS")
            .unwrap_or_else(|_| "0".to_string())
            .parse::<u64>()?,
    );
    let query_count = dataset.queries.len() / dataset.dimension;
    let concurrency = requested_concurrency
        .min(query_count.max(1))
        .min(rayon::current_num_threads());
    let source = InstrumentedStore {
        inner: BenchmarkFileSource::open(index_path)?,
        stats: Arc::clone(&stats),
        round_trip_latency,
    };
    let mut reader = VectorIndexReader::open_with_options(
        source,
        VectorIndexReaderOptions::new(storage_profile, reader_memory_budget_bytes),
    )?;
    *stats.lock().unwrap() = ReadStats::default();
    let warmup_started = Instant::now();
    reader.warmup_queries(
        &dataset.queries[..warmup_query_count * dataset.dimension],
        warmup_query_count,
        l_search,
    )?;
    let warmup_time = warmup_started.elapsed();
    let warmup_snapshot = *stats.lock().unwrap();
    *stats.lock().unwrap() = ReadStats::default();

    let params = VectorSearchParams::with_l_search(10, l_search);
    let started = Instant::now();
    let mut outcomes = Vec::with_capacity(query_count);
    let first_query_started = Instant::now();
    let (first_ids, _) = reader.search(&dataset.queries[..dataset.dimension], params)?;
    let first_search_stats = reader
        .diskann_search_stats()
        .expect("DiskANN benchmark reader diagnostics");
    let mut adjacency_cache_hits = first_search_stats.adjacency_cache_hits;
    let mut adjacency_cache_misses = first_search_stats.adjacency_cache_misses;
    let mut adjacency_cache_waits = first_search_stats.adjacency_cache_waits;
    let mut adjacency_cache_evictions = first_search_stats.adjacency_cache_evictions;
    let mut adjacency_cache_lock_acquisitions =
        first_search_stats.adjacency_cache_lock_acquisitions;
    let mut adjacency_cache_lock_wait_nanos = first_search_stats.adjacency_cache_lock_wait_nanos;
    let mut query_adjacency_cache_peak_bytes = first_search_stats.query_adjacency_cache_peak_bytes;
    let mut query_adjacency_cache_evictions = first_search_stats.query_adjacency_cache_evictions;
    let mut rerank_candidate_references = first_search_stats.rerank_candidate_references;
    let mut rerank_unique_windows = first_search_stats.rerank_unique_windows;
    let mut raw_vector_cache_hits = first_search_stats.raw_vector_cache_hits;
    let mut raw_vector_cache_misses = first_search_stats.raw_vector_cache_misses;
    let mut raw_vector_cache_evictions = first_search_stats.raw_vector_cache_evictions;
    let mut parallel_session_queries = first_search_stats.parallel_session_queries;
    outcomes.push(QueryOutcome {
        query_index: 0,
        latency: first_query_started.elapsed(),
        result_ids: first_ids,
    });
    if concurrency == 1 {
        for query_index in 1..query_count {
            let query = &dataset.queries
                [query_index * dataset.dimension..(query_index + 1) * dataset.dimension];
            let query_started = Instant::now();
            let (result_ids, _) = reader.search(query, params)?;
            let search_stats = reader
                .diskann_search_stats()
                .expect("DiskANN benchmark reader diagnostics");
            adjacency_cache_hits =
                adjacency_cache_hits.saturating_add(search_stats.adjacency_cache_hits);
            adjacency_cache_misses =
                adjacency_cache_misses.saturating_add(search_stats.adjacency_cache_misses);
            adjacency_cache_waits =
                adjacency_cache_waits.saturating_add(search_stats.adjacency_cache_waits);
            adjacency_cache_evictions =
                adjacency_cache_evictions.saturating_add(search_stats.adjacency_cache_evictions);
            adjacency_cache_lock_acquisitions = adjacency_cache_lock_acquisitions
                .saturating_add(search_stats.adjacency_cache_lock_acquisitions);
            adjacency_cache_lock_wait_nanos = adjacency_cache_lock_wait_nanos
                .saturating_add(search_stats.adjacency_cache_lock_wait_nanos);
            query_adjacency_cache_peak_bytes =
                query_adjacency_cache_peak_bytes.max(search_stats.query_adjacency_cache_peak_bytes);
            query_adjacency_cache_evictions = query_adjacency_cache_evictions
                .saturating_add(search_stats.query_adjacency_cache_evictions);
            rerank_candidate_references = rerank_candidate_references
                .saturating_add(search_stats.rerank_candidate_references);
            rerank_unique_windows =
                rerank_unique_windows.saturating_add(search_stats.rerank_unique_windows);
            raw_vector_cache_hits =
                raw_vector_cache_hits.saturating_add(search_stats.raw_vector_cache_hits);
            raw_vector_cache_misses =
                raw_vector_cache_misses.saturating_add(search_stats.raw_vector_cache_misses);
            raw_vector_cache_evictions =
                raw_vector_cache_evictions.saturating_add(search_stats.raw_vector_cache_evictions);
            parallel_session_queries =
                parallel_session_queries.saturating_add(search_stats.parallel_session_queries);
            outcomes.push(QueryOutcome {
                query_index,
                latency: query_started.elapsed(),
                result_ids,
            });
        }
    } else {
        for batch_start in (1..query_count).step_by(concurrency) {
            let batch_end = (batch_start + concurrency).min(query_count);
            let batch_count = batch_end - batch_start;
            let batch_queries =
                &dataset.queries[batch_start * dataset.dimension..batch_end * dataset.dimension];
            let batch_started = Instant::now();
            let (batch_ids, _) = reader.search_batch(batch_queries, batch_count, params)?;
            let search_stats = reader
                .diskann_search_stats()
                .expect("DiskANN benchmark reader diagnostics");
            adjacency_cache_hits =
                adjacency_cache_hits.saturating_add(search_stats.adjacency_cache_hits);
            adjacency_cache_misses =
                adjacency_cache_misses.saturating_add(search_stats.adjacency_cache_misses);
            adjacency_cache_waits =
                adjacency_cache_waits.saturating_add(search_stats.adjacency_cache_waits);
            adjacency_cache_evictions =
                adjacency_cache_evictions.saturating_add(search_stats.adjacency_cache_evictions);
            adjacency_cache_lock_acquisitions = adjacency_cache_lock_acquisitions
                .saturating_add(search_stats.adjacency_cache_lock_acquisitions);
            adjacency_cache_lock_wait_nanos = adjacency_cache_lock_wait_nanos
                .saturating_add(search_stats.adjacency_cache_lock_wait_nanos);
            query_adjacency_cache_peak_bytes =
                query_adjacency_cache_peak_bytes.max(search_stats.query_adjacency_cache_peak_bytes);
            query_adjacency_cache_evictions = query_adjacency_cache_evictions
                .saturating_add(search_stats.query_adjacency_cache_evictions);
            rerank_candidate_references = rerank_candidate_references
                .saturating_add(search_stats.rerank_candidate_references);
            rerank_unique_windows =
                rerank_unique_windows.saturating_add(search_stats.rerank_unique_windows);
            raw_vector_cache_hits =
                raw_vector_cache_hits.saturating_add(search_stats.raw_vector_cache_hits);
            raw_vector_cache_misses =
                raw_vector_cache_misses.saturating_add(search_stats.raw_vector_cache_misses);
            raw_vector_cache_evictions =
                raw_vector_cache_evictions.saturating_add(search_stats.raw_vector_cache_evictions);
            parallel_session_queries =
                parallel_session_queries.saturating_add(search_stats.parallel_session_queries);
            let batch_latency = batch_started.elapsed();
            for batch_index in 0..batch_count {
                outcomes.push(QueryOutcome {
                    query_index: batch_start + batch_index,
                    latency: batch_latency,
                    result_ids: batch_ids
                        [batch_index * params.top_k..(batch_index + 1) * params.top_k]
                        .to_vec(),
                });
            }
        }
    }
    let elapsed = started.elapsed();
    outcomes.sort_unstable_by_key(|outcome| outcome.query_index);
    let first_query = outcomes
        .first()
        .map(|outcome| outcome.latency)
        .unwrap_or_default();
    let mut recall_1_hits = 0usize;
    let mut recall_10_hits = 0usize;
    let mut recall_10_total = 0usize;
    let mut query_latencies = Vec::with_capacity(query_count);
    for outcome in &outcomes {
        query_latencies.push(outcome.latency);
        let truth = &dataset.ground_truth[outcome.query_index];
        recall_1_hits +=
            usize::from(outcome.result_ids.first().copied() == truth.first().map(|id| *id as i64));
        let truth_at_10 = truth.iter().take(10).copied().collect::<Vec<_>>();
        recall_10_total += truth_at_10.len();
        recall_10_hits += outcome
            .result_ids
            .iter()
            .filter(|&&id| id >= 0 && truth_at_10.contains(&(id as u32)))
            .count();
    }
    let warm_query = if query_count > 1 {
        query_latencies.iter().skip(1).copied().sum::<Duration>() / (query_count - 1) as u32
    } else {
        first_query
    };
    let snapshot = *stats.lock().unwrap();
    let recall_at_1 = recall_1_hits as f64 / query_count as f64;
    let recall_at_10 = if recall_10_total == 0 {
        0.0
    } else {
        recall_10_hits as f64 / recall_10_total as f64
    };
    let profile_name = match storage_profile {
        StorageProfile::Auto => "auto",
        StorageProfile::Memory => "memory",
        StorageProfile::LocalStorage => "local_storage",
        StorageProfile::RemoteStorage => "remote_storage",
        StorageProfile::ObjectStore => "object_store",
    };
    let p50_query = percentile(&query_latencies, 50);
    let p95_query = percentile(&query_latencies, 95);
    let p99_query = percentile(&query_latencies, 99);
    let observed_peak_rss_bytes = peak_resident_set_bytes()?.max(peak_rss_bytes);
    println!(
        "{profile},{storage_layout},{raw_vector_encoding},{build_distance},{l_search},{n},{nq},{d},{pq_bits},10,{concurrency},{reader_memory_budget_bytes},{warmup_query_count},{warmup_ms},{warmup_rounds},{warmup_ranges},{warmup_bytes},{recall_1:.4},{recall_10:.4},{build_ms},{graph_shards},{pq_train_ms},{pq_encode_ms},{vamana_init_ms},{vamana_pass_one_ms},{vamana_pass_two_ms},{connectivity_repair_ms},{locality_remap_ms},{resident_serialize_ms},{adjacency_serialize_ms},{vector_serialize_ms},{peak_rss},{first_us},{p50_us},{p95_us},{p99_us},{warm_us},{qps:.2},{rounds},{ranges},{round_qd},{in_flight_rounds},{in_flight_ranges},{read_bytes},{wait_ms:.3},{adjacency_hits},{adjacency_misses},{adjacency_waits},{adjacency_evictions},{adjacency_lock_acquisitions},{adjacency_lock_wait_nanos},{query_adjacency_peak_bytes},{query_adjacency_evictions},{rerank_candidate_references},{rerank_unique_windows},{cache_hits},{cache_misses},{cache_evictions},{parallel_session_queries},{rtt_ms},{adjacency_section_bytes},{adjacency_pages},{file_bytes}",
        profile = profile_name,
        storage_layout = match storage_layout {
            DiskAnnStorageLayout::Compact => "compact",
            DiskAnnStorageLayout::Interleaved => "interleaved",
        },
        raw_vector_encoding = match raw_vector_encoding {
            DiskAnnRawVectorEncoding::F32 => "f32",
            DiskAnnRawVectorEncoding::F16 => "f16",
        },
        build_distance = match build_distance {
            DiskAnnBuildDistance::FullPrecision => "full_precision",
            DiskAnnBuildDistance::ProductQuantized => "product_quantized",
        },
        l_search = l_search,
        n = dataset.base.len() / dataset.dimension,
        nq = query_count,
        d = dataset.dimension,
        warmup_query_count = warmup_query_count,
        warmup_ms = warmup_time.as_millis(),
        warmup_rounds = warmup_snapshot.rounds,
        warmup_ranges = warmup_snapshot.ranges,
        warmup_bytes = warmup_snapshot.bytes,
        recall_1 = recall_at_1,
        recall_10 = recall_at_10,
        build_ms = build_time.as_millis(),
        graph_shards = build_stats.graph_shards,
        pq_train_ms = pq_training_time.as_millis(),
        pq_encode_ms = build_stats.pq_encoding.as_millis(),
        vamana_init_ms = build_stats.vamana_initialization.as_millis(),
        vamana_pass_one_ms = build_stats.vamana_pass_one.as_millis(),
        vamana_pass_two_ms = build_stats.vamana_pass_two.as_millis(),
        connectivity_repair_ms = build_stats.connectivity_repair.as_millis(),
        locality_remap_ms = build_stats.locality_remap.as_millis(),
        resident_serialize_ms = build_stats.resident_serialization.as_millis(),
        adjacency_serialize_ms = build_stats.adjacency_serialization.as_millis(),
        vector_serialize_ms = build_stats.vector_serialization.as_millis(),
        peak_rss = observed_peak_rss_bytes,
        first_us = first_query.as_micros(),
        p50_us = p50_query.as_micros(),
        p95_us = p95_query.as_micros(),
        p99_us = p99_query.as_micros(),
        warm_us = warm_query.as_micros(),
        qps = query_count as f64 / elapsed.as_secs_f64(),
        rounds = snapshot.rounds,
        ranges = snapshot.ranges,
        round_qd = snapshot.max_ranges_per_round,
        in_flight_rounds = snapshot.max_in_flight_rounds,
        in_flight_ranges = snapshot.max_in_flight_ranges,
        read_bytes = snapshot.bytes,
        wait_ms = snapshot.wait.as_secs_f64() * 1000.0,
        adjacency_hits = adjacency_cache_hits,
        adjacency_misses = adjacency_cache_misses,
        adjacency_waits = adjacency_cache_waits,
        adjacency_evictions = adjacency_cache_evictions,
        adjacency_lock_acquisitions = adjacency_cache_lock_acquisitions,
        adjacency_lock_wait_nanos = adjacency_cache_lock_wait_nanos,
        query_adjacency_peak_bytes = query_adjacency_cache_peak_bytes,
        query_adjacency_evictions = query_adjacency_cache_evictions,
        rerank_candidate_references = rerank_candidate_references,
        rerank_unique_windows = rerank_unique_windows,
        cache_hits = raw_vector_cache_hits,
        cache_misses = raw_vector_cache_misses,
        cache_evictions = raw_vector_cache_evictions,
        rtt_ms = round_trip_latency.as_millis(),
    );

    if env::var_os("DISKANN_BENCH_ACCEPTANCE").is_some() {
        if recall_at_10 < 0.90 {
            return Err(format!("Recall@10 {:.4} is below 0.90", recall_at_10).into());
        }
        if build_time > Duration::from_secs(2 * 60 * 60) {
            return Err("DiskANN build exceeded two hours".into());
        }
        if observed_peak_rss_bytes > 4 * 1024 * 1024 * 1024 {
            return Err(
                format!("DiskANN peak RSS {} exceeds 4 GiB", observed_peak_rss_bytes).into(),
            );
        }
        if matches!(
            storage_profile,
            StorageProfile::RemoteStorage | StorageProfile::ObjectStore
        ) && snapshot.rounds > query_count.saturating_mul(8)
        {
            return Err(format!(
                "coalesced-range reads {} exceed seven graph plus one rerank round per query",
                snapshot.rounds
            )
            .into());
        }
    }
    Ok(())
}

struct QueryOutcome {
    query_index: usize,
    latency: Duration,
    result_ids: Vec<i64>,
}

/// Benchmark-only adapter for exercising local positional I/O.
///
/// Production integrations provide their own `SeekRead` implementation so the
/// core API does not expose or assume `std::fs::File`.
struct BenchmarkFileSource {
    file: File,
}

impl BenchmarkFileSource {
    fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        Ok(Self {
            file: File::open(path)?,
        })
    }
}

impl SeekRead for BenchmarkFileSource {
    fn pread(&mut self, ranges: &mut [ReadRequest<'_>]) -> io::Result<()> {
        #[cfg(any(unix, windows))]
        {
            ranges
                .par_iter_mut()
                .try_for_each(|range| benchmark_read_exact_at(&self.file, range.buf, range.pos))
        }

        #[cfg(not(any(unix, windows)))]
        {
            let old_pos = std::io::Seek::stream_position(&mut self.file)?;
            for range in ranges {
                std::io::Seek::seek(&mut self.file, std::io::SeekFrom::Start(range.pos))?;
                std::io::Read::read_exact(&mut self.file, range.buf)?;
            }
            std::io::Seek::seek(&mut self.file, std::io::SeekFrom::Start(old_pos))?;
            Ok(())
        }
    }

    fn try_clone_reader(&self) -> io::Result<Option<Self>> {
        Ok(Some(Self {
            file: self.file.try_clone()?,
        }))
    }
}

#[cfg(unix)]
fn benchmark_read_exact_at(file: &File, buf: &mut [u8], pos: u64) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(buf, pos)
}

#[cfg(windows)]
fn benchmark_read_exact_at(file: &File, mut buf: &mut [u8], mut pos: u64) -> io::Result<()> {
    use std::os::windows::fs::FileExt;

    while !buf.is_empty() {
        let read = file.seek_read(buf, pos)?;
        if read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "failed to fill whole buffer",
            ));
        }
        pos = pos
            .checked_add(read as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "read offset overflow"))?;
        buf = &mut buf[read..];
    }
    Ok(())
}

#[derive(Clone, Copy, Default)]
struct ReadStats {
    rounds: usize,
    ranges: usize,
    bytes: usize,
    wait: Duration,
    max_ranges_per_round: usize,
    in_flight_rounds: usize,
    in_flight_ranges: usize,
    max_in_flight_rounds: usize,
    max_in_flight_ranges: usize,
}

struct InstrumentedStore {
    inner: BenchmarkFileSource,
    stats: Arc<Mutex<ReadStats>>,
    round_trip_latency: Duration,
}

impl SeekRead for InstrumentedStore {
    fn pread(&mut self, ranges: &mut [ReadRequest<'_>]) -> io::Result<()> {
        let range_count = ranges.len();
        let byte_count = ranges.iter().map(|range| range.buf.len()).sum::<usize>();
        {
            let mut stats = self.stats.lock().unwrap();
            stats.in_flight_rounds += 1;
            stats.in_flight_ranges += range_count;
            stats.max_in_flight_rounds = stats.max_in_flight_rounds.max(stats.in_flight_rounds);
            stats.max_in_flight_ranges = stats.max_in_flight_ranges.max(stats.in_flight_ranges);
        }
        let started = Instant::now();
        if !self.round_trip_latency.is_zero() {
            thread::sleep(self.round_trip_latency);
        }
        let result = self.inner.pread(ranges);
        let mut stats = self.stats.lock().unwrap();
        stats.in_flight_rounds -= 1;
        stats.in_flight_ranges -= range_count;
        stats.rounds += 1;
        stats.ranges += range_count;
        stats.max_ranges_per_round = stats.max_ranges_per_round.max(range_count);
        stats.bytes += byte_count;
        stats.wait += started.elapsed();
        result
    }

    fn try_clone_reader(&self) -> io::Result<Option<Self>> {
        let Some(inner) = self.inner.try_clone_reader()? else {
            return Ok(None);
        };
        Ok(Some(Self {
            inner,
            stats: Arc::clone(&self.stats),
            round_trip_latency: self.round_trip_latency,
        }))
    }
}

struct TemporaryIndexFile {
    path: PathBuf,
}

impl TemporaryIndexFile {
    fn create() -> io::Result<(Self, File)> {
        let directory = env::var_os("DISKANN_BENCH_INDEX_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(env::temp_dir);
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(io::Error::other)?
            .as_nanos();
        for attempt in 0..100 {
            let path = directory.join(format!(
                "paimon-vindex-diskann-{}-{}-{}.dann",
                std::process::id(),
                timestamp,
                attempt
            ));
            match OpenOptions::new()
                .create_new(true)
                .read(true)
                .write(true)
                .open(&path)
            {
                Ok(file) => return Ok((Self { path }, file)),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        }
        Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "failed to allocate a unique DiskANN benchmark index file",
        ))
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TemporaryIndexFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn percentile(samples: &[Duration], percentile: usize) -> Duration {
    if samples.is_empty() {
        return Duration::ZERO;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let index = (sorted.len() - 1) * percentile.min(100) / 100;
    sorted[index]
}

#[cfg(unix)]
fn peak_resident_set_bytes() -> io::Result<u64> {
    let mut usage = MaybeUninit::<libc::rusage>::zeroed();
    let status = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if status != 0 {
        return Err(io::Error::last_os_error());
    }
    let max_rss = unsafe { usage.assume_init() }.ru_maxrss as u64;
    #[cfg(target_os = "macos")]
    return Ok(max_rss);
    #[cfg(not(target_os = "macos"))]
    return Ok(max_rss.saturating_mul(1024));
}

#[cfg(not(unix))]
fn peak_resident_set_bytes() -> io::Result<u64> {
    Ok(0)
}

struct Dataset {
    dimension: usize,
    base: Vec<f32>,
    queries: Vec<f32>,
    ground_truth: Vec<Vec<u32>>,
}

impl Dataset {
    fn from_env_or_smoke() -> Result<Self, Box<dyn std::error::Error>> {
        match (
            env::var_os("DISKANN_BASE_FVECS"),
            env::var_os("DISKANN_QUERY_FVECS"),
            env::var_os("DISKANN_GROUND_TRUTH_IVECS"),
        ) {
            (Some(base), Some(queries), Some(truth)) => {
                let (dimension, base) = read_fvecs(Path::new(&base))?;
                let (query_dimension, queries) = read_fvecs(Path::new(&queries))?;
                if query_dimension != dimension {
                    return Err("base/query fvec dimensions differ".into());
                }
                let ground_truth = read_ivecs(Path::new(&truth))?;
                if ground_truth.len() != queries.len() / dimension {
                    return Err("query and ground-truth counts differ".into());
                }
                Ok(Self {
                    dimension,
                    base,
                    queries,
                    ground_truth,
                })
            }
            (None, None, None) => Self::smoke(),
            _ => Err("set all three DISKANN_* dataset paths or none of them".into()),
        }
    }

    fn smoke() -> Result<Self, Box<dyn std::error::Error>> {
        let n = env::var("DISKANN_BENCH_SMOKE_N")
            .unwrap_or_else(|_| "512".to_string())
            .parse::<usize>()?;
        let nq = env::var("DISKANN_BENCH_SMOKE_NQ")
            .unwrap_or_else(|_| "16".to_string())
            .parse::<usize>()?;
        let dimension = env::var("DISKANN_BENCH_SMOKE_DIMENSION")
            .unwrap_or_else(|_| "32".to_string())
            .parse::<usize>()?;
        if n == 0 || nq == 0 || dimension == 0 {
            return Err("DiskANN smoke n, nq, and dimension must be positive".into());
        }
        let mut state = 42u64;
        let mut base = vec![0.0f32; n * dimension];
        for value in &mut base {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *value = (state >> 32) as f32 / u32::MAX as f32;
        }
        let mut queries = Vec::with_capacity(nq * dimension);
        let mut ground_truth = Vec::with_capacity(nq);
        for query in 0..nq {
            let source = query * 31 % n;
            let query_vector = &base[source * dimension..(source + 1) * dimension];
            queries.extend_from_slice(query_vector);
            let mut exact = base
                .chunks_exact(dimension)
                .enumerate()
                .map(|(node, vector)| (fvec_l2sqr(query_vector, vector), node as u32))
                .collect::<Vec<_>>();
            exact.select_nth_unstable_by(10.min(n).saturating_sub(1), |left, right| {
                left.0
                    .total_cmp(&right.0)
                    .then_with(|| left.1.cmp(&right.1))
            });
            exact.truncate(10.min(n));
            exact.sort_unstable_by(|left, right| {
                left.0
                    .total_cmp(&right.0)
                    .then_with(|| left.1.cmp(&right.1))
            });
            ground_truth.push(exact.into_iter().map(|(_, node)| node).collect());
        }
        Ok(Self {
            dimension,
            base,
            queries,
            ground_truth,
        })
    }
}

fn read_fvecs(path: &Path) -> io::Result<(usize, Vec<f32>)> {
    let rows = read_i32_records(path)?;
    let dimension = rows.first().map(Vec::len).unwrap_or(0);
    if dimension == 0 || rows.iter().any(|row| row.len() != dimension) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid fvecs shape",
        ));
    }
    Ok((
        dimension,
        rows.into_iter()
            .flatten()
            .map(|bits| f32::from_bits(bits as u32))
            .collect(),
    ))
}

fn read_ivecs(path: &Path) -> io::Result<Vec<Vec<u32>>> {
    Ok(read_i32_records(path)?
        .into_iter()
        .map(|row| row.into_iter().map(|value| value as u32).collect())
        .collect())
}

fn read_i32_records(path: &Path) -> io::Result<Vec<Vec<i32>>> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut rows = Vec::new();
    loop {
        let mut length = [0u8; 4];
        match reader.read_exact(&mut length) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(error) => return Err(error),
        }
        let length = i32::from_le_bytes(length);
        if length <= 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid record length",
            ));
        }
        let mut row = Vec::with_capacity(length as usize);
        for _ in 0..length {
            let mut value = [0u8; 4];
            reader.read_exact(&mut value)?;
            row.push(i32::from_le_bytes(value));
        }
        rows.push(row);
    }
    Ok(rows)
}
