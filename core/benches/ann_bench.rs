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

#[path = "support/ann_bench_support.rs"]
mod ann_bench_support;

use ann_bench_support::{
    add_fixed_round_latency, default_training_vector_count, inspect_i32_records,
    inspect_public_dataset, parse_storage_case_names, resolve_shape_value, should_isolate_indexes,
};
use paimon_vindex_core::diskann::{
    DiskAnnBuildDistance, DiskAnnBuildParams, DiskAnnRawVectorEncoding,
};
use paimon_vindex_core::distance::{fvec_l2sqr, MetricType};
use paimon_vindex_core::index::{
    infer_pq_m, VectorIndexConfig, VectorIndexReader, VectorIndexReaderOptions, VectorIndexTrainer,
    VectorIndexWriter, VectorSearchParams, DEFAULT_PQ_CODE_RATIO,
};
use paimon_vindex_core::io::{PosWriter, ReadRequest, SeekRead, SeekReadCapabilities};
use paimon_vindex_core::rq::{is_supported_rq_bits, DEFAULT_RQ_BITS};
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::env;
use std::fs::{self, File};
use std::io::{self, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

const ISOLATED_CHILD_ENV: &str = "ANN_BENCH_ISOLATED_CHILD";
const SUPPRESS_CSV_HEADER_ENV: &str = "ANN_BENCH_SUPPRESS_CSV_HEADER";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if run_isolated_public_indexes()? {
        return Ok(());
    }
    run_benchmark()
}

fn run_isolated_public_indexes() -> Result<bool, Box<dyn std::error::Error>> {
    let dataset_paths = DatasetPaths::from_env()?;
    let indexes = selected_indexes()?;
    let is_child = env::var_os(ISOLATED_CHILD_ENV).is_some();
    let reuses_index = env::var_os("ANN_REUSE_INDEX_PATH").is_some();
    if !should_isolate_indexes(
        dataset_paths.is_some(),
        indexes.len(),
        is_child,
        reuses_index,
    ) {
        return Ok(false);
    }

    println!("{}", CsvRow::header());
    io::stdout().flush()?;
    let executable = env::current_exe()?;
    for index in indexes {
        eprintln!("running public benchmark in isolated process: {index}");
        let status = Command::new(&executable)
            .env("ANN_INDEXES", &index)
            .env(ISOLATED_CHILD_ENV, "1")
            .env(SUPPRESS_CSV_HEADER_ENV, "1")
            .status()?;
        if !status.success() {
            return Err(
                format!("isolated ANN benchmark process for {index} exited with {status}").into(),
            );
        }
    }
    Ok(true)
}

fn run_benchmark() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::from_env()?;
    eprintln!(
        "workload: n={} train_n={} d={} pq_m={} raw_bytes={} ({:.3} GiB), indexes={}, storage_cases={}",
        config.n,
        config.train_n,
        config.d,
        config.pq_m,
        config.raw_dataset_bytes()?,
        config.raw_dataset_bytes()? as f64 / GIB as f64,
        config.indexes.join(","),
        config
            .storage_cases
            .iter()
            .map(|storage| storage.name)
            .collect::<Vec<_>>()
            .join(",")
    );
    let dataset_started = Instant::now();
    let dataset = Dataset::load(&config)?;
    eprintln!(
        "loaded dataset={} ({:.3} GiB) in {} ms",
        config.dataset_name,
        config.raw_dataset_bytes()? as f64 / GIB as f64,
        dataset_started.elapsed().as_millis()
    );
    let ground_truth = if let Some(ground_truth) = dataset.ground_truth.clone() {
        eprintln!(
            "loaded published ground truth: {} queries, top-{}",
            ground_truth.len(),
            ground_truth.first().map_or(0, Vec::len)
        );
        ground_truth
    } else {
        let truth_started = Instant::now();
        let ground_truth = exact_ground_truth(&dataset, config.k);
        eprintln!(
            "exact ground truth: {} queries in {} ms",
            config.nq,
            truth_started.elapsed().as_millis()
        );
        ground_truth
    };

    if env::var_os(SUPPRESS_CSV_HEADER_ENV).is_none() {
        println!("{}", CsvRow::header());
    }
    if let Some(path) = &config.reuse_index_path {
        let mut specs = index_specs(&config);
        if specs.len() != 1 {
            return Err("ANN_REUSE_INDEX_PATH requires exactly one ANN_INDEXES value".into());
        }
        let spec = specs.pop().expect("validated one ANN index specification");
        let built = BuiltIndex {
            name: spec.name,
            path: path.clone(),
            searches: spec.searches,
            build_time_ms: 0,
            train_time_ms: 0,
            add_time_ms: 0,
            write_time_ms: 0,
            file_bytes: fs::metadata(path)?.len(),
            peak_rss_bytes: peak_resident_set_bytes()?,
        };
        run_built_index(&config, &dataset, &ground_truth, &built)?;
        return Ok(());
    }

    let ids = (0..config.n as i64).collect::<Vec<_>>();
    let workspace = prepare_workspace(&config.output_dir)?;
    for spec in index_specs(&config) {
        let built = build_index(&config, &dataset, &ids, &workspace, spec)?;
        eprintln!(
            "built {} in {} ms: train={} ms, add={} ms, write={} ms ({} bytes, peak RSS {} bytes)",
            built.name,
            built.build_time_ms,
            built.train_time_ms,
            built.add_time_ms,
            built.write_time_ms,
            built.file_bytes,
            built.peak_rss_bytes
        );
        run_built_index(&config, &dataset, &ground_truth, &built)?;
        if !config.keep_indexes {
            fs::remove_file(&built.path)?;
        }
    }
    Ok(())
}

fn run_built_index(
    config: &Config,
    dataset: &Dataset,
    ground_truth: &[Vec<i64>],
    built: &BuiltIndex,
) -> Result<(), Box<dyn std::error::Error>> {
    for search in &built.searches {
        for storage in &config.storage_cases {
            run_query_case(config, dataset, ground_truth, built, *search, *storage)?;
        }
    }
    Ok(())
}

const GIB: u64 = 1024 * 1024 * 1024;
const ALL_INDEX_NAMES: [&str; 5] = ["IVF_FLAT", "IVF_SQ", "IVF_PQ", "IVF_RQ", "DISKANN"];

#[derive(Clone)]
struct Config {
    dataset_name: String,
    dataset_paths: Option<DatasetPaths>,
    n: usize,
    train_n: usize,
    nq: usize,
    d: usize,
    k: usize,
    nlist: usize,
    nprobe: usize,
    pq_m: usize,
    diskann_l_searches: Vec<usize>,
    diskann_memory_budget_bytes: usize,
    diskann_build_distance: DiskAnnBuildDistance,
    diskann_raw_vector_encoding: DiskAnnRawVectorEncoding,
    rq_bits: usize,
    clusters: usize,
    noise_dimensions: usize,
    seed: u64,
    indexes: Vec<String>,
    storage_cases: Vec<StorageCase>,
    keep_indexes: bool,
    reuse_index_path: Option<PathBuf>,
    output_dir: PathBuf,
}

impl Config {
    fn from_env() -> Result<Self, Box<dyn std::error::Error>> {
        let dataset_paths = DatasetPaths::from_env()?;
        let public_shape = dataset_paths
            .as_ref()
            .map(|paths| inspect_public_dataset(&paths.base, &paths.queries, &paths.ground_truth))
            .transpose()?;
        let dataset_name = env::var("ANN_DATASET_NAME").unwrap_or_else(|_| {
            if dataset_paths.is_some() {
                "public-fvecs".to_string()
            } else {
                "clustered-synthetic".to_string()
            }
        });
        if dataset_name.contains(',') || dataset_name.contains('\n') {
            return Err("ANN_DATASET_NAME must not contain commas or newlines".into());
        }
        let d = resolve_shape_value(
            "ANN_D",
            read_optional_env("ANN_D")?,
            public_shape.map(|shape| shape.dimension),
            64,
        )?;
        let n = match env::var("ANN_DATA_GIB") {
            Ok(value) => {
                if public_shape.is_some() {
                    return Err(
                        "ANN_DATA_GIB cannot be combined with public fvecs dataset paths".into(),
                    );
                }
                let gib = value.parse::<f64>()?;
                if !gib.is_finite() || gib <= 0.0 {
                    return Err("ANN_DATA_GIB must be finite and greater than zero".into());
                }
                let requested_bytes = gib * GIB as f64;
                if requested_bytes > usize::MAX as f64 {
                    return Err("ANN_DATA_GIB exceeds addressable memory".into());
                }
                let vector_bytes = d
                    .checked_mul(size_of::<f32>())
                    .ok_or("ANN_D byte size overflows usize")?;
                (requested_bytes as usize) / vector_bytes
            }
            Err(env::VarError::NotPresent) => resolve_shape_value(
                "ANN_N",
                read_optional_env("ANN_N")?,
                public_shape.map(|shape| shape.vector_count),
                20_000,
            )?,
            Err(error) => return Err(Box::new(error)),
        };
        let nq = resolve_shape_value(
            "ANN_NQ",
            read_optional_env("ANN_NQ")?,
            public_shape.map(|shape| shape.query_count),
            64,
        )?;
        let k = read_env("ANN_K", 10)?;
        let nlist = read_env("ANN_NLIST", 64)?;
        let nprobe = read_env("ANN_NPROBE", 8)?;
        let train_n = match read_optional_env("ANN_TRAIN_N")? {
            Some(train_n) => train_n,
            None => default_training_vector_count(n, nlist)?,
        };
        if let Some(shape) = public_shape {
            if shape.ground_truth_width < k {
                return Err(format!(
                    "public ground-truth width {} is smaller than ANN_K={k}",
                    shape.ground_truth_width
                )
                .into());
            }
        }
        let pq_m = match env::var("ANN_PQ_M") {
            Ok(value) => value.parse()?,
            Err(env::VarError::NotPresent) => {
                let code_ratio = read_env("ANN_PQ_CODE_RATIO", DEFAULT_PQ_CODE_RATIO)?;
                infer_pq_m(d, 8, code_ratio)?
            }
            Err(error) => return Err(Box::new(error)),
        };
        let diskann_l_search = read_env("ANN_DISKANN_L_SEARCH", 100)?;
        let diskann_l_searches =
            read_usize_list_env("ANN_DISKANN_L_SEARCHES", &[diskann_l_search])?;
        let diskann_memory_budget_bytes = read_env(
            "ANN_DISKANN_MEMORY_BUDGET_BYTES",
            DiskAnnBuildParams::default().memory_budget_bytes,
        )?;
        let diskann_build_distance = match env::var("ANN_DISKANN_BUILD_DISTANCE")
            .unwrap_or_else(|_| "product_quantized".to_string())
            .to_ascii_lowercase()
            .as_str()
        {
            "full_precision" | "full-precision" => DiskAnnBuildDistance::FullPrecision,
            "product_quantized" | "product-quantized" | "pq" => {
                DiskAnnBuildDistance::ProductQuantized
            }
            value => {
                return Err(format!(
                    "ANN_DISKANN_BUILD_DISTANCE must be full_precision or product_quantized, got {value}"
                )
                .into())
            }
        };
        let diskann_raw_vector_encoding = match env::var("ANN_DISKANN_RAW_VECTOR_ENCODING")
            .unwrap_or_else(|_| {
                match DiskAnnBuildParams::default().raw_vector_encoding {
                    DiskAnnRawVectorEncoding::F32 => "f32",
                    DiskAnnRawVectorEncoding::F16 => "f16",
                }
                .to_string()
            })
            .to_ascii_lowercase()
            .as_str()
        {
            "f32" => DiskAnnRawVectorEncoding::F32,
            "f16" => DiskAnnRawVectorEncoding::F16,
            value => {
                return Err(format!(
                    "ANN_DISKANN_RAW_VECTOR_ENCODING must be f32 or f16, got {value}"
                )
                .into())
            }
        };
        let rq_bits = read_env("ANN_RQ_BITS", DEFAULT_RQ_BITS)?;
        let clusters = read_env("ANN_CLUSTERS", 32)?;
        let noise_dimensions = read_env("ANN_NOISE_DIMENSIONS", d)?;
        let seed = read_env("ANN_SEED", 42)?;
        let indexes = selected_indexes()?;
        let storage_cases = selected_storage_cases()?;
        let keep_indexes = read_bool_env("ANN_KEEP_INDEXES", false)?;
        let reuse_index_path = env::var("ANN_REUSE_INDEX_PATH").ok().map(PathBuf::from);
        let output_dir = env::var("ANN_OUTPUT_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| env::temp_dir().join("paimon-ann-bench"));

        if n == 0
            || train_n == 0
            || nq == 0
            || d == 0
            || k == 0
            || nlist == 0
            || nprobe == 0
            || clusters == 0
            || noise_dimensions == 0
        {
            return Err(
                "ANN_N/ANN_DATA_GIB, ANN_TRAIN_N, ANN_NQ, ANN_D, ANN_K, ANN_NLIST, ANN_NPROBE, ANN_CLUSTERS, and ANN_NOISE_DIMENSIONS must be > 0".into(),
            );
        }
        if noise_dimensions > d {
            return Err(format!(
                "ANN_NOISE_DIMENSIONS ({noise_dimensions}) must be <= ANN_D ({d})"
            )
            .into());
        }
        if train_n > n {
            return Err(format!("ANN_TRAIN_N ({train_n}) must be <= vector count ({n})").into());
        }
        if nlist > n {
            return Err(format!("ANN_NLIST ({nlist}) must be <= ANN_N ({n})").into());
        }
        if nprobe > nlist {
            return Err(format!("ANN_NPROBE ({nprobe}) must be <= ANN_NLIST ({nlist})").into());
        }
        let needs_pq = indexes
            .iter()
            .any(|name| matches!(name.as_str(), "IVF_PQ" | "DISKANN"));
        if needs_pq && pq_m == 0 {
            return Err("ANN_PQ_M must be > 0 for IVF_PQ or DISKANN".into());
        }
        if needs_pq && !d.is_multiple_of(pq_m) {
            return Err(format!("ANN_D ({d}) must be divisible by ANN_PQ_M ({pq_m})").into());
        }
        if !is_supported_rq_bits(rq_bits) {
            return Err(format!("ANN_RQ_BITS ({rq_bits}) must be in 1..=8").into());
        }

        Ok(Self {
            dataset_name,
            dataset_paths,
            n,
            train_n,
            nq,
            d,
            k,
            nlist,
            nprobe,
            pq_m,
            diskann_l_searches,
            diskann_memory_budget_bytes,
            diskann_build_distance,
            diskann_raw_vector_encoding,
            rq_bits,
            clusters,
            noise_dimensions,
            seed,
            indexes,
            storage_cases,
            keep_indexes,
            reuse_index_path,
            output_dir,
        })
    }

    fn raw_dataset_bytes(&self) -> Result<usize, Box<dyn std::error::Error>> {
        self.n
            .checked_mul(self.d)
            .and_then(|elements| elements.checked_mul(size_of::<f32>()))
            .ok_or_else(|| "ANN raw dataset byte size overflows usize".into())
    }
}

struct Dataset {
    data: Vec<f32>,
    queries: Vec<f32>,
    dimension: usize,
    ground_truth: Option<Vec<Vec<i64>>>,
}

impl Dataset {
    fn load(config: &Config) -> io::Result<Self> {
        if let Some(paths) = &config.dataset_paths {
            return Self::fvecs(config, paths);
        }
        Self::clustered(config)
    }

    fn fvecs(config: &Config, paths: &DatasetPaths) -> io::Result<Self> {
        let (dimension, data) = read_fvecs(&paths.base)?;
        let (query_dimension, queries) = read_fvecs(&paths.queries)?;
        if dimension != config.d || query_dimension != config.d {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "ANN_D={} does not match base/query fvec dimensions {dimension}/{query_dimension}",
                    config.d
                ),
            ));
        }
        let n = data.len() / dimension;
        let nq = queries.len() / dimension;
        if n != config.n || nq != config.nq {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "ANN_N/ANN_NQ={}/{} do not match base/query fvec counts {n}/{nq}",
                    config.n, config.nq
                ),
            ));
        }
        let ground_truth = read_ivecs(&paths.ground_truth)?;
        if ground_truth.len() != config.nq {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "ANN_NQ={} does not match ground-truth count {}",
                    config.nq,
                    ground_truth.len()
                ),
            ));
        }
        let truth_width = ground_truth.first().map_or(0, Vec::len);
        if truth_width < config.k || ground_truth.iter().any(|row| row.len() != truth_width) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "ground truth must have a fixed width of at least ANN_K={}",
                    config.k
                ),
            ));
        }
        if let Some(invalid) = ground_truth
            .iter()
            .flatten()
            .find(|row_id| **row_id < 0 || **row_id as usize >= config.n)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("ground-truth row ID {invalid} is outside [0, {})", config.n),
            ));
        }
        Ok(Self {
            data,
            queries,
            dimension,
            ground_truth: Some(ground_truth),
        })
    }

    fn clustered(config: &Config) -> io::Result<Self> {
        let mut rng = Lcg::new(config.seed);
        let center_elements = config
            .clusters
            .checked_mul(config.d)
            .ok_or_else(|| io::Error::other("ANN center shape overflows usize"))?;
        let mut centers = try_zeroed_f32(center_elements, "cluster centers")?;
        for value in &mut centers {
            *value = rng.next_signed_f32() * 15.0;
        }

        let data_elements = config
            .n
            .checked_mul(config.d)
            .ok_or_else(|| io::Error::other("ANN dataset shape overflows usize"))?;
        let mut data = try_zeroed_f32(data_elements, "dataset")?;
        for row in 0..config.n {
            let cluster = row % config.clusters;
            for component in 0..config.d {
                let noise = if is_noise_dimension(component, config.d, config.noise_dimensions) {
                    rng.next_signed_f32()
                } else {
                    0.0
                };
                data[row * config.d + component] = centers[cluster * config.d + component] + noise;
            }
        }

        let query_elements = config
            .nq
            .checked_mul(config.d)
            .ok_or_else(|| io::Error::other("ANN query shape overflows usize"))?;
        let mut queries = try_zeroed_f32(query_elements, "queries")?;
        for query in 0..config.nq {
            let cluster = (query * 17 + 3) % config.clusters;
            for component in 0..config.d {
                let noise = if is_noise_dimension(component, config.d, config.noise_dimensions) {
                    rng.next_signed_f32()
                } else {
                    0.0
                };
                queries[query * config.d + component] =
                    centers[cluster * config.d + component] + noise;
            }
        }
        Ok(Self {
            data,
            queries,
            dimension: config.d,
            ground_truth: None,
        })
    }
}

#[derive(Clone)]
struct DatasetPaths {
    base: PathBuf,
    queries: PathBuf,
    ground_truth: PathBuf,
}

impl DatasetPaths {
    fn from_env() -> Result<Option<Self>, Box<dyn std::error::Error>> {
        match (
            env::var_os("ANN_BASE_FVECS"),
            env::var_os("ANN_QUERY_FVECS"),
            env::var_os("ANN_GROUND_TRUTH_IVECS"),
        ) {
            (Some(base), Some(queries), Some(ground_truth)) => Ok(Some(Self {
                base: PathBuf::from(base),
                queries: PathBuf::from(queries),
                ground_truth: PathBuf::from(ground_truth),
            })),
            (None, None, None) => Ok(None),
            _ => Err(
                "set ANN_BASE_FVECS, ANN_QUERY_FVECS, and ANN_GROUND_TRUTH_IVECS together".into(),
            ),
        }
    }
}

fn is_noise_dimension(component: usize, dimension: usize, noise_dimensions: usize) -> bool {
    if noise_dimensions == dimension {
        return true;
    }
    let component = component as u128;
    let noise_dimensions = noise_dimensions as u128;
    let dimension = dimension as u128;
    (component + 1) * noise_dimensions / dimension != component * noise_dimensions / dimension
}

fn try_zeroed_f32(elements: usize, name: &str) -> io::Result<Vec<f32>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|error| io::Error::other(format!("failed to allocate ANN {name}: {error}")))?;
    values.resize(elements, 0.0);
    Ok(values)
}

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_signed_f32(&mut self) -> f32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        (self.state >> 32) as f32 / u32::MAX as f32 * 2.0 - 1.0
    }
}

struct IndexSpec {
    name: &'static str,
    config: VectorIndexConfig,
    searches: Vec<VectorSearchParams>,
}

fn index_specs(config: &Config) -> Vec<IndexSpec> {
    let ivf_search = VectorSearchParams::new(config.k, config.nprobe);
    vec![
        IndexSpec {
            name: "IVF_FLAT",
            config: VectorIndexConfig::IvfFlat {
                dimension: config.d,
                nlist: config.nlist,
                metric: MetricType::L2,
                use_approximate_coarse_assignment: true,
            },
            searches: vec![ivf_search],
        },
        IndexSpec {
            name: "IVF_SQ",
            config: VectorIndexConfig::IvfSq {
                dimension: config.d,
                nlist: config.nlist,
                metric: MetricType::L2,
                use_approximate_coarse_assignment: true,
            },
            searches: vec![ivf_search],
        },
        IndexSpec {
            name: "IVF_PQ",
            config: VectorIndexConfig::IvfPq {
                dimension: config.d,
                nlist: config.nlist,
                m: config.pq_m,
                metric: MetricType::L2,
                use_opq: false,
                use_approximate_coarse_assignment: true,
            },
            searches: vec![ivf_search],
        },
        IndexSpec {
            name: "IVF_RQ",
            config: VectorIndexConfig::IvfRq {
                dimension: config.d,
                nlist: config.nlist,
                bits: config.rq_bits,
                metric: MetricType::L2,
                use_approximate_coarse_assignment: true,
            },
            searches: vec![ivf_search],
        },
        IndexSpec {
            name: "DISKANN",
            config: VectorIndexConfig::DiskAnn {
                dimension: config.d,
                metric: MetricType::L2,
                pq_m: config.pq_m,
                pq_bits: 8,
                build: DiskAnnBuildParams {
                    memory_budget_bytes: config.diskann_memory_budget_bytes,
                    build_distance: config.diskann_build_distance,
                    raw_vector_encoding: config.diskann_raw_vector_encoding,
                    ..DiskAnnBuildParams::default()
                },
            },
            searches: config
                .diskann_l_searches
                .iter()
                .map(|l_search| VectorSearchParams::with_l_search(config.k, *l_search))
                .collect(),
        },
    ]
    .into_iter()
    .filter(|spec| config.indexes.iter().any(|name| name == spec.name))
    .collect()
}

struct BuiltIndex {
    name: &'static str,
    path: PathBuf,
    searches: Vec<VectorSearchParams>,
    build_time_ms: u128,
    train_time_ms: u128,
    add_time_ms: u128,
    write_time_ms: u128,
    file_bytes: u64,
    peak_rss_bytes: u64,
}

fn build_index(
    config: &Config,
    dataset: &Dataset,
    ids: &[i64],
    workspace: &Path,
    spec: IndexSpec,
) -> io::Result<BuiltIndex> {
    let path = workspace.join(format!("{}.index", spec.name.to_ascii_lowercase()));
    let started = Instant::now();
    let training_elements = config
        .train_n
        .checked_mul(config.d)
        .ok_or_else(|| io::Error::other("ANN training shape overflows usize"))?;
    eprintln!(
        "training {} with {} sampled vectors",
        spec.name, config.train_n
    );
    let train_started = Instant::now();
    let training = VectorIndexTrainer::train(
        spec.config,
        &dataset.data[..training_elements],
        config.train_n,
    )?;
    let train_time_ms = train_started.elapsed().as_millis();
    let mut writer = VectorIndexWriter::new(training);
    eprintln!("adding {} vectors to {}", config.n, spec.name);
    let add_started = Instant::now();
    writer.add_vectors(ids, &dataset.data, config.n)?;
    let add_time_ms = add_started.elapsed().as_millis();
    let mut file = File::create(&path)?;
    eprintln!("building and serializing {}", spec.name);
    let write_started = Instant::now();
    writer.write(&mut PosWriter::new(&mut file))?;
    file.sync_all()?;
    let write_time_ms = write_started.elapsed().as_millis();
    let build_time_ms = started.elapsed().as_millis();
    let file_bytes = file.metadata()?.len();
    let peak_rss_bytes = peak_resident_set_bytes()?;
    Ok(BuiltIndex {
        name: spec.name,
        path,
        searches: spec.searches,
        build_time_ms,
        train_time_ms,
        add_time_ms,
        write_time_ms,
        file_bytes,
        peak_rss_bytes,
    })
}

#[derive(Clone, Copy)]
struct StorageCase {
    name: &'static str,
    latency: Duration,
    virtualize_sequential_latency: bool,
}

impl StorageCase {
    fn local_ssd() -> Self {
        Self {
            name: "local_ssd_warm_cache",
            latency: Duration::ZERO,
            virtualize_sequential_latency: false,
        }
    }

    fn remote_cache_2ms() -> Self {
        Self {
            name: "remote_cache_2ms",
            latency: Duration::from_millis(2),
            virtualize_sequential_latency: false,
        }
    }

    fn object_store_20ms() -> Self {
        Self {
            name: "object_store_20ms",
            latency: Duration::from_millis(20),
            virtualize_sequential_latency: true,
        }
    }

    fn from_name(name: &str) -> Self {
        match name {
            "local_ssd_warm_cache" => Self::local_ssd(),
            "remote_cache_2ms" => Self::remote_cache_2ms(),
            "object_store_20ms" => Self::object_store_20ms(),
            _ => unreachable!("storage case names are validated before conversion"),
        }
    }
}

fn run_query_case(
    config: &Config,
    dataset: &Dataset,
    ground_truth: &[Vec<i64>],
    index: &BuiltIndex,
    search: VectorSearchParams,
    storage: StorageCase,
) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!(
        "querying {} with storage={} l_search={}",
        index.name,
        storage.name,
        search.configured_diskann_l_search().unwrap_or(0)
    );
    let io_pool = if index.name == "DISKANN" {
        Some(Arc::new(
            ThreadPoolBuilder::new()
                .num_threads(rayon::current_num_threads())
                .thread_name(|index| format!("ann-bench-io-{index}"))
                .build()?,
        ))
    } else {
        None
    };
    let stats = Arc::new(Mutex::new(IoStats::default()));
    let source = InstrumentedFile {
        file: Arc::new(File::open(&index.path)?),
        io_pool: io_pool.clone(),
        stats: Arc::clone(&stats),
        latency: storage.latency,
        latency_enabled: Arc::new(AtomicBool::new(true)),
    };
    let latency_enabled = Arc::clone(&source.latency_enabled);
    if storage.virtualize_sequential_latency {
        latency_enabled.store(false, Ordering::Relaxed);
    }
    let optimize_started = Instant::now();
    let mut reader = VectorIndexReader::open_with_options(
        source,
        VectorIndexReaderOptions::new(4 * 1024 * 1024 * 1024),
    )?;
    reader.optimize_for_search()?;
    let optimize_elapsed = optimize_started.elapsed();
    let optimize_stats = snapshot_stats(&stats);
    let optimize_ms = if storage.virtualize_sequential_latency {
        add_fixed_round_latency(optimize_elapsed, optimize_stats.rounds, storage.latency)
            .as_millis()
    } else {
        optimize_elapsed.as_millis()
    };

    let first_rounds_before = snapshot_stats(&stats).rounds;
    let first_started = Instant::now();
    reader.search(&dataset.queries[..config.d], search)?;
    let first_elapsed = first_started.elapsed();
    let first_query_us = if storage.virtualize_sequential_latency {
        add_fixed_round_latency(
            first_elapsed,
            snapshot_stats(&stats)
                .rounds
                .saturating_sub(first_rounds_before),
            storage.latency,
        )
        .as_micros()
    } else {
        first_elapsed.as_micros()
    };

    eprintln!(
        "running sequential queries for {} storage={} l_search={}",
        index.name,
        storage.name,
        search.configured_diskann_l_search().unwrap_or(0)
    );
    reset_stats(&stats);
    let mut latencies = Vec::with_capacity(config.nq);
    let mut sequential_rq_stats = paimon_vindex_core::index::IVFRQSearchStats::default();
    let mut virtual_sequential_elapsed = Duration::ZERO;
    let sequential_started = Instant::now();
    for (query_index, query) in dataset.queries.chunks_exact(config.d).enumerate() {
        let rounds_before = snapshot_stats(&stats).rounds;
        let started = Instant::now();
        reader.search(query, search).map_err(|error| {
            io::Error::new(
                error.kind(),
                format!("sequential query {query_index} failed: {error}"),
            )
        })?;
        if let Some(stats) = reader.ivfrq_search_stats() {
            sequential_rq_stats.merge(stats);
        }
        let elapsed = started.elapsed();
        let elapsed = if storage.virtualize_sequential_latency {
            add_fixed_round_latency(
                elapsed,
                snapshot_stats(&stats).rounds.saturating_sub(rounds_before),
                storage.latency,
            )
        } else {
            elapsed
        };
        virtual_sequential_elapsed = virtual_sequential_elapsed.saturating_add(elapsed);
        latencies.push(elapsed);
    }
    let sequential_elapsed = if storage.virtualize_sequential_latency {
        virtual_sequential_elapsed
    } else {
        sequential_started.elapsed()
    };
    let sequential_stats = snapshot_stats(&stats);

    eprintln!(
        "running batch queries for {} storage={} l_search={}",
        index.name,
        storage.name,
        search.configured_diskann_l_search().unwrap_or(0)
    );
    let batch_stats = Arc::new(Mutex::new(IoStats::default()));
    let batch_source = InstrumentedFile {
        file: Arc::new(File::open(&index.path)?),
        io_pool,
        stats: Arc::clone(&batch_stats),
        latency: storage.latency,
        latency_enabled: Arc::new(AtomicBool::new(!storage.virtualize_sequential_latency)),
    };
    let batch_latency_enabled = Arc::clone(&batch_source.latency_enabled);
    let mut batch_reader = VectorIndexReader::open_with_options(
        batch_source,
        VectorIndexReaderOptions::new(4 * 1024 * 1024 * 1024),
    )?;
    batch_reader.optimize_for_search()?;
    batch_latency_enabled.store(true, Ordering::Relaxed);
    reset_stats(&batch_stats);
    let batch_started = Instant::now();
    let (result_ids, _) = batch_reader.search_batch(&dataset.queries, config.nq, search)?;
    let batch_elapsed = batch_started.elapsed();
    let rq_stats = batch_reader.ivfrq_search_stats().unwrap_or_default();
    let batch_stats = snapshot_stats(&batch_stats);
    let rq_refine_ratio = if rq_stats.eligible_vectors == 0 {
        0.0
    } else {
        rq_stats.refined_vectors as f64 / rq_stats.eligible_vectors as f64
    };
    if index.name == "IVF_RQ" {
        eprintln!(
            "IVF-RQ sequential scan stats: scanned={} refined={} ({:.2}%) final={} seeded_lists={} parallel_list_tasks={}",
            sequential_rq_stats.scanned_vectors,
            sequential_rq_stats.refined_vectors,
            if sequential_rq_stats.eligible_vectors == 0 {
                0.0
            } else {
                sequential_rq_stats.refined_vectors as f64
                    / sequential_rq_stats.eligible_vectors as f64
                    * 100.0
            },
            sequential_rq_stats.final_distance_evaluations,
            sequential_rq_stats.seeded_lists,
            sequential_rq_stats.parallel_list_tasks,
        );
        eprintln!(
            "IVF-RQ batch scan stats: scanned={} eligible={} refined={} ({:.2}%) final={} refined_coarse_lookups={} extra_plane_lookups={} fastscan_blocks={} scalar_blocks={} seeded_lists={} parallel_list_tasks={}",
            rq_stats.scanned_vectors,
            rq_stats.eligible_vectors,
            rq_stats.refined_vectors,
            rq_refine_ratio * 100.0,
            rq_stats.final_distance_evaluations,
            rq_stats.refined_coarse_byte_lookups,
            rq_stats.extra_plane_byte_lookups,
            rq_stats.fastscan_blocks,
            rq_stats.scalar_blocks,
            rq_stats.seeded_lists,
            rq_stats.parallel_list_tasks,
        );
    }

    let recall_at_10 = recall_at_k(&result_ids, ground_truth, config.k);
    let p50 = percentile(&latencies, 50);
    let p95 = percentile(&latencies, 95);
    let build_distance = match config.diskann_build_distance {
        DiskAnnBuildDistance::FullPrecision => "full_precision",
        DiskAnnBuildDistance::ProductQuantized => "product_quantized",
    };
    let raw_vector_encoding = match (index.name, config.diskann_raw_vector_encoding) {
        ("DISKANN", DiskAnnRawVectorEncoding::F32) => "f32",
        ("DISKANN", DiskAnnRawVectorEncoding::F16) => "f16",
        _ => "none",
    };
    println!(
        "{dataset},{index},{storage},{n},{train_n},{raw_dataset_bytes},{nq},{d},{k},{nlist},{nprobe},{pq_m},{rq_bits},{build_distance},{raw_vector_encoding},{l_search},{build_ms},{train_ms},{add_ms},{write_ms},{peak_rss_bytes},{optimize_ms},{optimize_rounds},{optimize_ranges},{optimize_bytes},{file_bytes},{recall:.4},{first_us},{p50_us},{p95_us},{sequential_qps:.2},{seq_rounds},{seq_ranges},{seq_bytes},{batch_ms},{batch_qps:.2},{batch_rounds},{batch_ranges},{batch_bytes},{rq_seq_scanned},{rq_seq_refined},{rq_seq_final},{rq_seq_seeded_lists},{rq_seq_parallel_list_tasks},{rq_scanned},{rq_eligible},{rq_refined},{rq_refine_ratio:.6},{rq_final},{rq_refined_coarse_lookups},{rq_extra_plane_lookups},{rq_fastscan_blocks},{rq_scalar_blocks},{rq_seeded_lists},{rq_parallel_list_tasks}",
        dataset = config.dataset_name,
        index = index.name,
        storage = storage.name,
        n = config.n,
        train_n = config.train_n,
        raw_dataset_bytes = config.raw_dataset_bytes()?,
        nq = config.nq,
        d = config.d,
        k = config.k,
        nlist = if index.name == "DISKANN" {
            1
        } else {
            config.nlist
        },
        nprobe = search.configured_ivf_nprobe().unwrap_or(0),
        pq_m = config.pq_m,
        rq_bits = if index.name == "IVF_RQ" { config.rq_bits } else { 0 },
        build_distance = build_distance,
        raw_vector_encoding = raw_vector_encoding,
        l_search = search.configured_diskann_l_search().unwrap_or(0),
        build_ms = index.build_time_ms,
        train_ms = index.train_time_ms,
        add_ms = index.add_time_ms,
        write_ms = index.write_time_ms,
        peak_rss_bytes = index.peak_rss_bytes,
        optimize_rounds = optimize_stats.rounds,
        optimize_ranges = optimize_stats.ranges,
        optimize_bytes = optimize_stats.bytes,
        file_bytes = index.file_bytes,
        recall = recall_at_10,
        first_us = first_query_us,
        p50_us = p50.as_micros(),
        p95_us = p95.as_micros(),
        sequential_qps = config.nq as f64 / sequential_elapsed.as_secs_f64(),
        seq_rounds = sequential_stats.rounds,
        seq_ranges = sequential_stats.ranges,
        seq_bytes = sequential_stats.bytes,
        batch_ms = batch_elapsed.as_millis(),
        batch_qps = config.nq as f64 / batch_elapsed.as_secs_f64(),
        batch_rounds = batch_stats.rounds,
        batch_ranges = batch_stats.ranges,
        batch_bytes = batch_stats.bytes,
        rq_seq_scanned = sequential_rq_stats.scanned_vectors,
        rq_seq_refined = sequential_rq_stats.refined_vectors,
        rq_seq_final = sequential_rq_stats.final_distance_evaluations,
        rq_seq_seeded_lists = sequential_rq_stats.seeded_lists,
        rq_seq_parallel_list_tasks = sequential_rq_stats.parallel_list_tasks,
        rq_scanned = rq_stats.scanned_vectors,
        rq_eligible = rq_stats.eligible_vectors,
        rq_refined = rq_stats.refined_vectors,
        rq_refine_ratio = rq_refine_ratio,
        rq_final = rq_stats.final_distance_evaluations,
        rq_refined_coarse_lookups = rq_stats.refined_coarse_byte_lookups,
        rq_extra_plane_lookups = rq_stats.extra_plane_byte_lookups,
        rq_fastscan_blocks = rq_stats.fastscan_blocks,
        rq_scalar_blocks = rq_stats.scalar_blocks,
        rq_seeded_lists = rq_stats.seeded_lists,
        rq_parallel_list_tasks = rq_stats.parallel_list_tasks,
    );
    Ok(())
}

struct CsvRow;

impl CsvRow {
    fn header() -> &'static str {
        "dataset,index,storage,n,train_n,raw_dataset_bytes,nq,d,k,nlist,nprobe,pq_m,rq_bits,diskann_build_distance,diskann_raw_vector_encoding,l_search,build_ms,train_ms,add_ms,write_ms,peak_rss_bytes,optimize_ms,optimize_pread_rounds,optimize_pread_ranges,optimize_pread_bytes,file_bytes,recall_at_10,first_query_us,p50_query_us,p95_query_us,sequential_qps,sequential_pread_rounds,sequential_pread_ranges,sequential_pread_bytes,batch_ms,batch_qps,batch_pread_rounds,batch_pread_ranges,batch_pread_bytes,rq_sequential_scanned_vectors,rq_sequential_refined_vectors,rq_sequential_final_distance_evaluations,rq_sequential_seeded_lists,rq_sequential_parallel_list_tasks,rq_scanned_vectors,rq_eligible_vectors,rq_refined_vectors,rq_refine_ratio,rq_final_distance_evaluations,rq_refined_coarse_byte_lookups,rq_extra_plane_byte_lookups,rq_fastscan_blocks,rq_scalar_blocks,rq_seeded_lists,rq_parallel_list_tasks"
    }
}

fn exact_ground_truth(dataset: &Dataset, k: usize) -> Vec<Vec<i64>> {
    dataset
        .queries
        .par_chunks_exact(dataset.dimension)
        .map(|query| {
            let mut distances = dataset
                .data
                .chunks_exact(dataset.dimension)
                .enumerate()
                .map(|(row, vector)| (fvec_l2sqr(query, vector), row as i64))
                .collect::<Vec<_>>();
            let retained = k.min(distances.len());
            if retained != 0 {
                distances.select_nth_unstable_by(retained - 1, |left, right| {
                    left.0
                        .total_cmp(&right.0)
                        .then_with(|| left.1.cmp(&right.1))
                });
                distances.truncate(retained);
                distances.sort_unstable_by(|left, right| {
                    left.0
                        .total_cmp(&right.0)
                        .then_with(|| left.1.cmp(&right.1))
                });
            }
            distances.into_iter().map(|(_, row)| row).collect()
        })
        .collect()
}

fn recall_at_k(actual: &[i64], expected: &[Vec<i64>], k: usize) -> f64 {
    let mut hits = 0usize;
    let mut total = 0usize;
    for (actual, expected) in actual.chunks_exact(k).zip(expected) {
        let expected = &expected[..expected.len().min(k)];
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

fn percentile(samples: &[Duration], percentile: usize) -> Duration {
    if samples.is_empty() {
        return Duration::ZERO;
    }
    let mut samples = samples.to_vec();
    samples.sort_unstable();
    samples[(samples.len() - 1) * percentile.min(100) / 100]
}

#[derive(Debug, Clone, Copy, Default)]
struct IoStats {
    rounds: usize,
    ranges: usize,
    bytes: usize,
}

fn reset_stats(stats: &Mutex<IoStats>) {
    *stats.lock().unwrap() = IoStats::default();
}

fn snapshot_stats(stats: &Mutex<IoStats>) -> IoStats {
    *stats.lock().unwrap()
}

#[derive(Clone)]
struct InstrumentedFile {
    file: Arc<File>,
    io_pool: Option<Arc<ThreadPool>>,
    stats: Arc<Mutex<IoStats>>,
    latency: Duration,
    latency_enabled: Arc<AtomicBool>,
}

impl SeekRead for InstrumentedFile {
    fn pread(&mut self, ranges: &mut [ReadRequest<'_>]) -> io::Result<()> {
        let range_count = ranges.len();
        let byte_count = ranges.iter().map(|range| range.buf.len()).sum::<usize>();
        if self.latency_enabled.load(Ordering::Relaxed) && !self.latency.is_zero() {
            thread::sleep(self.latency);
        }
        let mut read_ranges = || {
            ranges
                .par_iter_mut()
                .try_for_each(|range| read_exact_at(&self.file, range.buf, range.pos))
        };
        if let Some(io_pool) = &self.io_pool {
            io_pool.install(read_ranges)?;
        } else {
            read_ranges()?;
        }
        let mut stats = self.stats.lock().unwrap();
        stats.rounds += 1;
        stats.ranges += range_count;
        stats.bytes += byte_count;
        Ok(())
    }

    fn try_clone_reader(&self) -> io::Result<Option<Self>> {
        Ok(Some(self.clone()))
    }

    fn read_capabilities(&self) -> SeekReadCapabilities {
        SeekReadCapabilities {
            estimated_random_read_latency_nanos: u64::try_from(self.latency.as_nanos())
                .unwrap_or(u64::MAX),
            ..SeekReadCapabilities::default()
        }
    }
}

#[cfg(unix)]
fn read_exact_at(file: &File, mut buf: &mut [u8], mut pos: u64) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    while !buf.is_empty() {
        let read = file.read_at(buf, pos)?;
        if read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "benchmark positional read reached EOF",
            ));
        }
        pos = pos
            .checked_add(read as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "read offset overflow"))?;
        buf = &mut buf[read..];
    }
    Ok(())
}

#[cfg(windows)]
fn read_exact_at(file: &File, mut buf: &mut [u8], mut pos: u64) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        let read = file.seek_read(buf, pos)?;
        if read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "benchmark positional read reached EOF",
            ));
        }
        pos = pos
            .checked_add(read as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "read offset overflow"))?;
        buf = &mut buf[read..];
    }
    Ok(())
}

fn read_fvecs(path: &Path) -> io::Result<(usize, Vec<f32>)> {
    let (rows, dimension) = inspect_i32_records(path)?;
    let elements = rows
        .checked_mul(dimension)
        .ok_or_else(|| io::Error::other("fvec shape overflows usize"))?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|error| io::Error::other(format!("failed to allocate fvec data: {error}")))?;
    let mut reader = BufReader::with_capacity(8 * 1024 * 1024, File::open(path)?);
    let mut header = [0u8; 4];
    let mut row = vec![0u8; dimension * size_of::<f32>()];
    for row_index in 0..rows {
        reader.read_exact(&mut header)?;
        let actual_dimension = i32::from_le_bytes(header);
        if actual_dimension != dimension as i32 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "fvec row {row_index} has dimension {actual_dimension}, expected {dimension}"
                ),
            ));
        }
        reader.read_exact(&mut row)?;
        values.extend(
            row.chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte chunk"))),
        );
    }
    Ok((dimension, values))
}

fn read_ivecs(path: &Path) -> io::Result<Vec<Vec<i64>>> {
    let (rows, width) = inspect_i32_records(path)?;
    let mut result = Vec::with_capacity(rows);
    let mut reader = BufReader::with_capacity(1024 * 1024, File::open(path)?);
    let mut header = [0u8; 4];
    let mut row = vec![0u8; width * size_of::<i32>()];
    for row_index in 0..rows {
        reader.read_exact(&mut header)?;
        let actual_width = i32::from_le_bytes(header);
        if actual_width != width as i32 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("ivec row {row_index} has width {actual_width}, expected {width}"),
            ));
        }
        reader.read_exact(&mut row)?;
        result.push(
            row.chunks_exact(4)
                .map(|bytes| i32::from_le_bytes(bytes.try_into().expect("four-byte chunk")) as i64)
                .collect(),
        );
    }
    Ok(result)
}

fn read_env<T>(name: &str, default: T) -> Result<T, Box<dyn std::error::Error>>
where
    T: std::str::FromStr,
    T::Err: std::error::Error + 'static,
{
    match env::var(name) {
        Ok(value) => Ok(value.parse()?),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(Box::new(error)),
    }
}

fn read_optional_env<T>(name: &str) -> Result<Option<T>, Box<dyn std::error::Error>>
where
    T: std::str::FromStr,
    T::Err: std::error::Error + 'static,
{
    match env::var(name) {
        Ok(value) => Ok(Some(value.parse()?)),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(Box::new(error)),
    }
}

fn read_bool_env(name: &str, default: bool) -> Result<bool, Box<dyn std::error::Error>> {
    match env::var(name) {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" => Ok(true),
            "0" | "false" | "no" => Ok(false),
            _ => Err(format!("{name} must be one of 1, 0, true, false, yes, or no").into()),
        },
        Err(env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(Box::new(error)),
    }
}

fn read_usize_list_env(
    name: &str,
    default: &[usize],
) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let value = match env::var(name) {
        Ok(value) => value,
        Err(env::VarError::NotPresent) => return Ok(default.to_vec()),
        Err(error) => return Err(Box::new(error)),
    };
    let mut values = Vec::new();
    for item in value.split(',') {
        let parsed = item.trim().parse::<usize>()?;
        if parsed == 0 {
            return Err(format!("{name} values must be > 0").into());
        }
        if !values.contains(&parsed) {
            values.push(parsed);
        }
    }
    if values.is_empty() {
        return Err(format!("{name} must contain at least one value").into());
    }
    Ok(values)
}

fn selected_indexes() -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let value = env::var("ANN_INDEXES").unwrap_or_else(|_| "all".to_string());
    if value.trim().eq_ignore_ascii_case("all") {
        return Ok(ALL_INDEX_NAMES
            .iter()
            .map(|name| (*name).to_string())
            .collect());
    }
    let mut selected = Vec::new();
    for value in value.split(',') {
        let normalized = value.trim().to_ascii_uppercase().replace('-', "_");
        if !ALL_INDEX_NAMES.contains(&normalized.as_str()) {
            return Err(format!(
                "unknown ANN_INDEXES value '{value}'; expected all or a comma-separated subset of {}",
                ALL_INDEX_NAMES.join(",")
            )
            .into());
        }
        if !selected.contains(&normalized) {
            selected.push(normalized);
        }
    }
    if selected.is_empty() {
        return Err("ANN_INDEXES must select at least one index".into());
    }
    Ok(selected)
}

fn selected_storage_cases() -> Result<Vec<StorageCase>, Box<dyn std::error::Error>> {
    let value = env::var("ANN_STORAGE_CASES").ok();
    Ok(parse_storage_case_names(value.as_deref())?
        .into_iter()
        .map(StorageCase::from_name)
        .collect())
}

#[cfg(unix)]
fn peak_resident_set_bytes() -> io::Result<u64> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    let result = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if result != 0 {
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

fn prepare_workspace(output_dir: &Path) -> io::Result<PathBuf> {
    let workspace = output_dir.join(format!("{}", std::process::id()));
    if workspace.exists() {
        fs::remove_dir_all(&workspace)?;
    }
    fs::create_dir_all(&workspace)?;
    Ok(workspace)
}
