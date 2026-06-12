use paimon_vindex_core::distance::MetricType;
use paimon_vindex_core::hnsw::HnswBuildParams;
use paimon_vindex_core::index::{
    VectorIndexConfig, VectorIndexReader, VectorIndexWriter, VectorSearchParams,
};
use paimon_vindex_core::io::PosWriter;
use roaring::RoaringTreemap;
use std::env;
use std::io::{self, Cursor};
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = Config::from_env()?;
    cfg.validate()?;

    let (data, queries) = generate_clustered_data(cfg.n, cfg.nq, cfg.d, cfg.clusters, cfg.seed);
    let ids: Vec<i64> = (0..cfg.n as i64).collect();

    let start = Instant::now();
    let mut writer = VectorIndexWriter::new(VectorIndexConfig::IvfHnswSq {
        dimension: cfg.d,
        nlist: cfg.nlist,
        metric: MetricType::L2,
        hnsw: cfg.hnsw_params(),
    })?;
    writer.train(&data, cfg.n)?;
    writer.add_vectors(&ids, &data, cfg.n)?;
    let mut index_bytes = Vec::new();
    writer.write(&mut PosWriter::new(&mut index_bytes))?;
    let build = start.elapsed();

    println!(
        "n={},nq={},d={},k={},nlist={},nprobe={},ef_search={},index_bytes={},build_ms={}",
        cfg.n,
        cfg.nq,
        cfg.d,
        cfg.k,
        cfg.nlist,
        cfg.nprobe,
        cfg.ef_search,
        index_bytes.len(),
        build.as_millis()
    );
    println!("filter_stride,allowed,optimized,warmup_ms,search_ms,us_per_query,qps");

    for stride in &cfg.filter_strides {
        let filter_bytes = filter_bytes(cfg.n, *stride)?;
        let allowed = cfg.n.div_ceil(*stride);
        let baseline = run_case(&cfg, &index_bytes, &queries, &filter_bytes, false)?;
        let optimized = run_case(&cfg, &index_bytes, &queries, &filter_bytes, true)?;
        assert_same_results(*stride, &baseline.result, &optimized.result);
        print_row(
            *stride,
            allowed,
            false,
            baseline.warmup,
            baseline.search,
            cfg.nq,
        );
        print_row(
            *stride,
            allowed,
            true,
            optimized.warmup,
            optimized.search,
            cfg.nq,
        );
    }

    Ok(())
}

struct Config {
    n: usize,
    nq: usize,
    d: usize,
    k: usize,
    nlist: usize,
    nprobe: usize,
    ef_search: usize,
    hnsw_m: usize,
    hnsw_ef_construction: usize,
    hnsw_max_level: usize,
    clusters: usize,
    seed: u64,
    filter_strides: Vec<usize>,
}

impl Config {
    fn from_env() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            n: read_env("FILTER_BENCH_N", 50_000)?,
            nq: read_env("FILTER_BENCH_NQ", 500)?,
            d: read_env("FILTER_BENCH_D", 128)?,
            k: read_env("FILTER_BENCH_K", 10)?,
            nlist: read_env("FILTER_BENCH_NLIST", 64)?,
            nprobe: read_env("FILTER_BENCH_NPROBE", 32)?,
            ef_search: read_env("FILTER_BENCH_EF_SEARCH", 80)?,
            hnsw_m: read_env("FILTER_BENCH_HNSW_M", 20)?,
            hnsw_ef_construction: read_env("FILTER_BENCH_HNSW_EF_CONSTRUCTION", 150)?,
            hnsw_max_level: read_env("FILTER_BENCH_HNSW_MAX_LEVEL", 7)?,
            clusters: read_env("FILTER_BENCH_CLUSTERS", 32)?,
            seed: read_env("FILTER_BENCH_SEED", 42)?,
            filter_strides: read_strides("FILTER_BENCH_FILTER_STRIDES", &[1, 4, 16, 64])?,
        })
    }

    fn validate(&self) -> Result<(), Box<dyn std::error::Error>> {
        if self.n == 0 || self.nq == 0 || self.d == 0 || self.k == 0 {
            return Err("FILTER_BENCH_N, NQ, D, and K must be greater than 0".into());
        }
        if self.nlist == 0 || self.nprobe == 0 || self.nprobe > self.nlist || self.nlist > self.n {
            return Err("FILTER_BENCH_NLIST/NPROBE must satisfy 0 < nprobe <= nlist <= n".into());
        }
        if self.clusters == 0 {
            return Err("FILTER_BENCH_CLUSTERS must be greater than 0".into());
        }
        if self.filter_strides.is_empty() || self.filter_strides.contains(&0) {
            return Err("FILTER_BENCH_FILTER_STRIDES must contain positive integers".into());
        }
        Ok(())
    }

    fn hnsw_params(&self) -> HnswBuildParams {
        HnswBuildParams {
            m: self.hnsw_m,
            ef_construction: self.hnsw_ef_construction,
            max_level: self.hnsw_max_level,
        }
        .sanitized()
    }
}

struct CaseResult {
    warmup: Duration,
    search: Duration,
    result: (Vec<i64>, Vec<f32>),
}

fn run_case(
    cfg: &Config,
    index_bytes: &[u8],
    queries: &[f32],
    filter_bytes: &[u8],
    optimized: bool,
) -> io::Result<CaseResult> {
    let mut reader = VectorIndexReader::open(Cursor::new(index_bytes.to_vec()))?;
    let warmup_start = Instant::now();
    if optimized {
        reader.optimize_for_search()?;
    }
    let warmup = warmup_start.elapsed();

    let params = VectorSearchParams::with_ef_search(cfg.k, cfg.nprobe, cfg.ef_search);
    let _ = reader.search_batch_with_roaring_filter(queries, cfg.nq, params, filter_bytes)?;

    let start = Instant::now();
    let result = reader.search_batch_with_roaring_filter(queries, cfg.nq, params, filter_bytes)?;
    let search = start.elapsed();
    Ok(CaseResult {
        warmup,
        search,
        result,
    })
}

fn filter_bytes(n: usize, stride: usize) -> io::Result<Vec<u8>> {
    let mut filter = RoaringTreemap::new();
    for id in (0..n as u64).step_by(stride) {
        filter.insert(id);
    }
    let mut bytes = Vec::new();
    filter.serialize_into(&mut bytes)?;
    Ok(bytes)
}

fn assert_same_results(
    stride: usize,
    expected: &(Vec<i64>, Vec<f32>),
    actual: &(Vec<i64>, Vec<f32>),
) {
    assert_eq!(
        actual.0, expected.0,
        "ids should match for stride {}",
        stride
    );
    assert_eq!(
        actual.1.len(),
        expected.1.len(),
        "distance count should match for stride {}",
        stride
    );
    for (actual, expected) in actual.1.iter().zip(expected.1.iter()) {
        assert!(
            (actual - expected).abs() < 1e-4,
            "distance {} should match {} for stride {}",
            actual,
            expected,
            stride
        );
    }
}

fn print_row(
    filter_stride: usize,
    allowed: usize,
    optimized: bool,
    warmup: Duration,
    search: Duration,
    nq: usize,
) {
    println!(
        "{},{},{},{},{},{:.2},{:.2}",
        filter_stride,
        allowed,
        optimized,
        warmup.as_millis(),
        search.as_millis(),
        search.as_secs_f64() * 1_000_000.0 / nq as f64,
        nq as f64 / search.as_secs_f64(),
    );
}

fn generate_clustered_data(
    n: usize,
    nq: usize,
    d: usize,
    clusters: usize,
    seed: u64,
) -> (Vec<f32>, Vec<f32>) {
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
    (data, queries)
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

fn read_strides(name: &str, default: &[usize]) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    match env::var(name) {
        Ok(value) => value
            .split(',')
            .map(|part| {
                part.trim()
                    .parse::<usize>()
                    .map_err(|e| -> Box<dyn std::error::Error> { Box::new(e) })
            })
            .collect(),
        Err(env::VarError::NotPresent) => Ok(default.to_vec()),
        Err(err) => Err(Box::new(err)),
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
