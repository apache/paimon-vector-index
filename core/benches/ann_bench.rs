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

use paimon_vindex_core::distance::MetricType;
use paimon_vindex_core::hnsw::HnswBuildParams;
use paimon_vindex_core::io::{write_index, IVFPQIndexReader, PosWriter};
use paimon_vindex_core::ivfhnswflat::IVFHNSWFlatIndex;
use paimon_vindex_core::ivfhnswflat_io::{
    search_batch_ivfhnswflat_reader, write_ivfhnswflat_index, IVFHNSWFlatIndexReader,
};
use paimon_vindex_core::ivfhnswsq::IVFHNSWSQIndex;
use paimon_vindex_core::ivfhnswsq_io::{
    search_batch_ivfhnswsq_reader, write_ivfhnswsq_index, IVFHNSWSQIndexReader,
};
use paimon_vindex_core::ivfpq::{search_batch_reader, IVFPQIndex};
use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = Config::from_env()?;
    let dataset = Dataset::clustered(cfg.n, cfg.nq, cfg.d, cfg.clusters, cfg.seed);
    let ids: Vec<i64> = (0..cfg.n as i64).collect();
    let workspace = prepare_workspace(&cfg.output_dir)?;

    println!("{}", CsvRow::header());
    run_paimon_ivfpq(&cfg, &dataset, &ids, &workspace)?;
    run_paimon_ivfhnswflat(&cfg, &dataset, &ids, &workspace)?;
    run_paimon_ivfhnswsq(&cfg, &dataset, &ids, &workspace)?;

    Ok(())
}

#[derive(Clone)]
struct Config {
    n: usize,
    nq: usize,
    d: usize,
    k: usize,
    nlist: usize,
    nprobe: usize,
    pq_m: usize,
    hnsw_m: usize,
    hnsw_ef_construction: usize,
    hnsw_ef_search: usize,
    clusters: usize,
    seed: u64,
    output_dir: PathBuf,
}

impl Config {
    fn from_env() -> Result<Self, Box<dyn std::error::Error>> {
        let n = read_env("ANN_N", 10_000)?;
        let nq = read_env("ANN_NQ", 100)?;
        let d = read_env("ANN_D", 128)?;
        let k = read_env("ANN_K", 10)?;
        let nlist = read_env("ANN_NLIST", 64)?;
        let nprobe = read_env("ANN_NPROBE", 8)?;
        let pq_m = read_env("ANN_PQ_M", 16)?;
        let hnsw_m = read_env("ANN_HNSW_M", 20)?;
        let hnsw_ef_construction = read_env("ANN_HNSW_EF_CONSTRUCTION", 150)?;
        let hnsw_ef_search = read_env("ANN_HNSW_EF_SEARCH", 80)?;
        let clusters = read_env("ANN_CLUSTERS", 32)?;
        let seed = read_env("ANN_SEED", 42)?;
        let output_dir = env::var("ANN_OUTPUT_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| env::temp_dir().join("paimon-ann-bench"));

        if n == 0 || nq == 0 || d == 0 || k == 0 || nlist == 0 || nprobe == 0 {
            return Err(
                "ANN_N, ANN_NQ, ANN_D, ANN_K, ANN_NLIST, and ANN_NPROBE must be > 0".into(),
            );
        }
        if nlist > n {
            return Err(format!("ANN_NLIST ({}) must be <= ANN_N ({})", nlist, n).into());
        }
        if nprobe > nlist {
            return Err(format!("ANN_NPROBE ({}) must be <= ANN_NLIST ({})", nprobe, nlist).into());
        }
        if d % pq_m != 0 {
            return Err(format!("ANN_D ({}) must be divisible by ANN_PQ_M ({})", d, pq_m).into());
        }

        Ok(Self {
            n,
            nq,
            d,
            k,
            nlist,
            nprobe,
            pq_m,
            hnsw_m,
            hnsw_ef_construction,
            hnsw_ef_search,
            clusters,
            seed,
            output_dir,
        })
    }

    fn hnsw_params(&self) -> HnswBuildParams {
        HnswBuildParams {
            m: self.hnsw_m,
            ef_construction: self.hnsw_ef_construction,
            ..HnswBuildParams::default()
        }
        .sanitized()
    }
}

struct Dataset {
    data: Vec<f32>,
    queries: Vec<f32>,
}

impl Dataset {
    fn clustered(n: usize, nq: usize, d: usize, clusters: usize, seed: u64) -> Self {
        let mut rng = Lcg::new(seed);
        let mut centers = vec![0.0f32; clusters * d];
        for value in &mut centers {
            *value = rng.next_f32() * 30.0;
        }

        let mut data = vec![0.0f32; n * d];
        for i in 0..n {
            let cluster = i % clusters;
            for j in 0..d {
                data[i * d + j] = centers[cluster * d + j] + rng.next_f32();
            }
        }

        let mut queries = vec![0.0f32; nq * d];
        for qi in 0..nq {
            let source = (qi * 9973) % n;
            queries[qi * d..(qi + 1) * d].copy_from_slice(&data[source * d..(source + 1) * d]);
        }

        Self { data, queries }
    }
}

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    }
}

struct CsvRow {
    engine: &'static str,
    index: &'static str,
    n: usize,
    nq: usize,
    d: usize,
    k: usize,
    nlist: usize,
    nprobe: usize,
    ef_search: Option<usize>,
    build_ms: u128,
    read_ms: u128,
    first_query_ms: u128,
    search_ms: u128,
    qps: f64,
    disk_bytes: u64,
    disk_scope: &'static str,
    note: &'static str,
}

impl CsvRow {
    fn header() -> &'static str {
        "engine,index,n,nq,d,k,nlist,nprobe,ef_search,build_ms,read_ms,first_query_ms,search_ms,qps,disk_bytes,disk_scope,note"
    }
}

impl std::fmt::Display for CsvRow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{:.2},{},{},{}",
            self.engine,
            self.index,
            self.n,
            self.nq,
            self.d,
            self.k,
            self.nlist,
            self.nprobe,
            self.ef_search
                .map(|value| value.to_string())
                .unwrap_or_default(),
            self.build_ms,
            self.read_ms,
            self.first_query_ms,
            self.search_ms,
            self.qps,
            self.disk_bytes,
            self.disk_scope,
            self.note,
        )
    }
}

fn run_paimon_ivfpq(
    cfg: &Config,
    dataset: &Dataset,
    ids: &[i64],
    workspace: &Path,
) -> io::Result<()> {
    let path = workspace.join("paimon_ivfpq.index");
    let start = Instant::now();
    let mut index = IVFPQIndex::new(cfg.d, cfg.nlist, cfg.pq_m, MetricType::L2, false);
    index.train(&dataset.data, cfg.n);
    index.add(&dataset.data, ids, cfg.n);
    index.build_search_structures();
    index.build_precomputed_table();
    write_to_file(&path, |writer| write_index(&index, writer))?;
    let build = start.elapsed();
    drop(index);

    let start = Instant::now();
    let file = fs::File::open(&path)?;
    let mut reader = IVFPQIndexReader::open(file)?;
    reader.ensure_loaded()?;
    let read = start.elapsed();

    let first_query =
        time_first_query(|| reader.search(&dataset.queries[..cfg.d], cfg.k, cfg.nprobe))?;
    let search = time_search(|| {
        search_batch_reader(&mut reader, &dataset.queries, cfg.nq, cfg.k, cfg.nprobe).map(|_| ())
    })?;

    print_row(
        cfg,
        "paimon",
        "IVF_PQ",
        None,
        build,
        read,
        first_query,
        search,
        path.metadata()?.len(),
        "index_bytes",
        "",
    );
    Ok(())
}

fn run_paimon_ivfhnswflat(
    cfg: &Config,
    dataset: &Dataset,
    ids: &[i64],
    workspace: &Path,
) -> io::Result<()> {
    let path = workspace.join("paimon_ivfhnswflat.index");
    let start = Instant::now();
    let mut index = IVFHNSWFlatIndex::new(cfg.d, cfg.nlist, MetricType::L2, cfg.hnsw_params());
    index.train(&dataset.data, cfg.n);
    index.add(&dataset.data, ids, cfg.n);
    index.build_graphs()?;
    write_to_file(&path, |writer| write_ivfhnswflat_index(&index, writer))?;
    let build = start.elapsed();
    drop(index);

    let start = Instant::now();
    let file = fs::File::open(&path)?;
    let mut reader = IVFHNSWFlatIndexReader::open(file)?;
    reader.ensure_loaded()?;
    let read = start.elapsed();

    let first_query = time_first_query(|| {
        reader.search(
            &dataset.queries[..cfg.d],
            cfg.k,
            cfg.nprobe,
            cfg.hnsw_ef_search,
        )
    })?;
    let search = time_search(|| {
        search_batch_ivfhnswflat_reader(
            &mut reader,
            &dataset.queries,
            cfg.nq,
            cfg.k,
            cfg.nprobe,
            cfg.hnsw_ef_search,
        )
        .map(|_| ())
    })?;

    print_row(
        cfg,
        "paimon",
        "IVF_HNSW_FLAT",
        Some(cfg.hnsw_ef_search),
        build,
        read,
        first_query,
        search,
        path.metadata()?.len(),
        "index_bytes",
        "",
    );
    Ok(())
}

fn run_paimon_ivfhnswsq(
    cfg: &Config,
    dataset: &Dataset,
    ids: &[i64],
    workspace: &Path,
) -> io::Result<()> {
    let path = workspace.join("paimon_ivfhnswsq.index");
    let start = Instant::now();
    let mut index = IVFHNSWSQIndex::new(cfg.d, cfg.nlist, MetricType::L2, cfg.hnsw_params());
    index.train(&dataset.data, cfg.n);
    index.add(&dataset.data, ids, cfg.n);
    index.build_graphs()?;
    write_to_file(&path, |writer| write_ivfhnswsq_index(&index, writer))?;
    let build = start.elapsed();
    drop(index);

    let start = Instant::now();
    let file = fs::File::open(&path)?;
    let mut reader = IVFHNSWSQIndexReader::open(file)?;
    reader.ensure_loaded()?;
    let read = start.elapsed();

    let first_query = time_first_query(|| {
        reader.search(
            &dataset.queries[..cfg.d],
            cfg.k,
            cfg.nprobe,
            cfg.hnsw_ef_search,
        )
    })?;
    let search = time_search(|| {
        search_batch_ivfhnswsq_reader(
            &mut reader,
            &dataset.queries,
            cfg.nq,
            cfg.k,
            cfg.nprobe,
            cfg.hnsw_ef_search,
        )
        .map(|_| ())
    })?;

    print_row(
        cfg,
        "paimon",
        "IVF_HNSW_SQ",
        Some(cfg.hnsw_ef_search),
        build,
        read,
        first_query,
        search,
        path.metadata()?.len(),
        "index_bytes",
        "",
    );
    Ok(())
}

fn write_to_file(
    path: &Path,
    write: impl FnOnce(&mut PosWriter<&mut fs::File>) -> io::Result<()>,
) -> io::Result<()> {
    let mut file = fs::File::create(path)?;
    let mut writer = PosWriter::new(&mut file);
    write(&mut writer)
}

fn time_first_query<T>(query: impl FnOnce() -> io::Result<T>) -> io::Result<Duration> {
    let start = Instant::now();
    query()?;
    Ok(start.elapsed())
}

fn time_search(search: impl FnOnce() -> io::Result<()>) -> io::Result<Duration> {
    let start = Instant::now();
    search()?;
    Ok(start.elapsed())
}

#[allow(clippy::too_many_arguments)]
fn print_row(
    cfg: &Config,
    engine: &'static str,
    index: &'static str,
    ef_search: Option<usize>,
    build: Duration,
    read: Duration,
    first_query: Duration,
    search: Duration,
    disk_bytes: u64,
    disk_scope: &'static str,
    note: &'static str,
) {
    println!(
        "{}",
        CsvRow {
            engine,
            index,
            n: cfg.n,
            nq: cfg.nq,
            d: cfg.d,
            k: cfg.k,
            nlist: cfg.nlist,
            nprobe: cfg.nprobe,
            ef_search,
            build_ms: build.as_millis(),
            read_ms: read.as_millis(),
            first_query_ms: first_query.as_millis(),
            search_ms: search.as_millis(),
            qps: cfg.nq as f64 / search.as_secs_f64(),
            disk_bytes,
            disk_scope,
            note,
        }
    );
}

fn read_env<T>(name: &str, default: T) -> Result<T, Box<dyn std::error::Error>>
where
    T: std::str::FromStr,
    T::Err: std::error::Error + 'static,
{
    match env::var(name) {
        Ok(value) => Ok(value.parse()?),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(err) => Err(Box::new(err)),
    }
}

fn prepare_workspace(output_dir: &Path) -> io::Result<PathBuf> {
    let workspace = output_dir.join(format!("{}", std::process::id()));
    if workspace.exists() {
        fs::remove_dir_all(&workspace)?;
    }
    fs::create_dir_all(&workspace)?;
    Ok(workspace)
}
