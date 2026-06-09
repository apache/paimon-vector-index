use paimon_vindex_core::distance::{fvec_distance, MetricType};
use paimon_vindex_core::hnsw::HnswBuildParams;
use paimon_vindex_core::io::{write_index, PosWriter};
use paimon_vindex_core::ivfflat::IVFFlatIndex;
use paimon_vindex_core::ivfflat_io::write_ivfflat_index;
use paimon_vindex_core::ivfhnswflat::IVFHNSWFlatIndex;
use paimon_vindex_core::ivfhnswflat_io::write_ivfhnswflat_index;
use paimon_vindex_core::ivfhnswsq::IVFHNSWSQIndex;
use paimon_vindex_core::ivfhnswsq_io::write_ivfhnswsq_index;
use paimon_vindex_core::ivfpq::IVFPQIndex;
use std::collections::HashSet;
use std::time::Instant;

fn main() {
    run_scenario(Scenario {
        name: "small-lists",
        d: 64,
        n: 20_000,
        nq: 50,
        k: 10,
        nlist: 64,
        pq_m: 8,
        nprobes: &[1, 4, 8, 16, 32, 64],
        hnsw_build_ef: 80,
        hnsw_search_efs: &[80],
    });

    println!();

    run_scenario(Scenario {
        name: "large-lists",
        d: 64,
        n: 50_000,
        nq: 50,
        k: 10,
        nlist: 8,
        pq_m: 8,
        nprobes: &[1, 2, 4, 8],
        hnsw_build_ef: 200,
        hnsw_search_efs: &[80, 160, 320],
    });
}

struct Scenario<'a> {
    name: &'a str,
    d: usize,
    n: usize,
    nq: usize,
    k: usize,
    nlist: usize,
    pq_m: usize,
    nprobes: &'a [usize],
    hnsw_build_ef: usize,
    hnsw_search_efs: &'a [usize],
}

fn run_scenario(s: Scenario<'_>) {
    println!("=== IVF Recall Attribution Benchmark ===");
    println!(
        "scenario: {}, n={}, nq={}, d={}, nlist={}, avg_list={}, k={}, metric=L2",
        s.name,
        s.n,
        s.nq,
        s.d,
        s.nlist,
        s.n / s.nlist,
        s.k
    );

    let data = generate_clustered_data(s.n, s.d, 32, 42);
    let ids: Vec<i64> = (0..s.n as i64).collect();
    let queries = &data[..s.nq * s.d];

    let start = Instant::now();
    let ground_truth = brute_force_ground_truth(&data, queries, s.n, s.nq, s.d, s.k);
    println!("ground truth: {:.2}s", start.elapsed().as_secs_f64());

    let start = Instant::now();
    let mut ivfpq = IVFPQIndex::new(s.d, s.nlist, s.pq_m, MetricType::L2, false);
    ivfpq.train(&data, s.n);
    ivfpq.add(&data, &ids, s.n);
    ivfpq.build_precomputed_table();
    println!("build IVF-PQ: {:.2}s", start.elapsed().as_secs_f64());

    let start = Instant::now();
    let mut ivfflat = IVFFlatIndex::new(s.d, s.nlist, MetricType::L2);
    ivfflat.train(&data, s.n);
    ivfflat.add(&data, &ids, s.n);
    println!("build IVF-FLAT: {:.2}s", start.elapsed().as_secs_f64());

    let start = Instant::now();
    let mut ivfhnswflat = IVFHNSWFlatIndex::new(
        s.d,
        s.nlist,
        MetricType::L2,
        HnswBuildParams {
            m: 16,
            ef_construction: s.hnsw_build_ef,
            max_level: 7,
        },
    );
    ivfhnswflat.train(&data, s.n);
    ivfhnswflat.add(&data, &ids, s.n);
    ivfhnswflat.build_graphs().unwrap();
    println!("build IVF-HNSW-FLAT: {:.2}s", start.elapsed().as_secs_f64());

    let start = Instant::now();
    let mut ivfhnswsq = IVFHNSWSQIndex::new(
        s.d,
        s.nlist,
        MetricType::L2,
        HnswBuildParams {
            m: 16,
            ef_construction: s.hnsw_build_ef,
            max_level: 7,
        },
    );
    ivfhnswsq.train(&data, s.n);
    ivfhnswsq.add(&data, &ids, s.n);
    ivfhnswsq.build_graphs().unwrap();
    println!("build IVF-HNSW-SQ: {:.2}s", start.elapsed().as_secs_f64());
    print_sizes(&ivfpq, &ivfflat, &ivfhnswflat, &ivfhnswsq);

    println!();
    println!(
        "index      nprobe  ef      recall@{}  query_ms  us/query",
        s.k
    );
    println!("---------  ------  ------  ---------  --------  --------");

    for &nprobe in s.nprobes {
        let mut distances = vec![0.0f32; s.nq * s.k];
        let mut labels = vec![0i64; s.nq * s.k];
        let start = Instant::now();
        ivfpq.search(queries, s.nq, s.k, nprobe, &mut distances, &mut labels);
        let elapsed = start.elapsed();
        print_row(
            "IVF-PQ",
            nprobe,
            None,
            recall_at_k(&labels, &ground_truth, s.nq, s.k),
            elapsed,
            s.nq,
        );

        let mut distances = vec![0.0f32; s.nq * s.k];
        let mut labels = vec![0i64; s.nq * s.k];
        let start = Instant::now();
        ivfflat.search(queries, s.nq, s.k, nprobe, &mut distances, &mut labels);
        let elapsed = start.elapsed();
        print_row(
            "IVF-FLAT",
            nprobe,
            None,
            recall_at_k(&labels, &ground_truth, s.nq, s.k),
            elapsed,
            s.nq,
        );

        for &ef_search in s.hnsw_search_efs {
            let mut distances = vec![0.0f32; s.nq * s.k];
            let mut labels = vec![0i64; s.nq * s.k];
            let start = Instant::now();
            ivfhnswflat.search(
                queries,
                s.nq,
                s.k,
                nprobe,
                ef_search,
                &mut distances,
                &mut labels,
            );
            let elapsed = start.elapsed();
            print_row(
                "IVF-HNSW",
                nprobe,
                Some(ef_search),
                recall_at_k(&labels, &ground_truth, s.nq, s.k),
                elapsed,
                s.nq,
            );

            let mut distances = vec![0.0f32; s.nq * s.k];
            let mut labels = vec![0i64; s.nq * s.k];
            let start = Instant::now();
            ivfhnswsq.search(
                queries,
                s.nq,
                s.k,
                nprobe,
                ef_search,
                &mut distances,
                &mut labels,
            );
            let elapsed = start.elapsed();
            print_row(
                "IVF-HSQ",
                nprobe,
                Some(ef_search),
                recall_at_k(&labels, &ground_truth, s.nq, s.k),
                elapsed,
                s.nq,
            );
        }
    }
}

fn print_sizes(
    ivfpq: &IVFPQIndex,
    ivfflat: &IVFFlatIndex,
    ivfhnswflat: &IVFHNSWFlatIndex,
    ivfhnswsq: &IVFHNSWSQIndex,
) {
    let mut pq = Vec::new();
    write_index(ivfpq, &mut PosWriter::new(&mut pq)).unwrap();
    let mut flat = Vec::new();
    write_ivfflat_index(ivfflat, &mut PosWriter::new(&mut flat)).unwrap();
    let mut hnswflat = Vec::new();
    write_ivfhnswflat_index(ivfhnswflat, &mut PosWriter::new(&mut hnswflat)).unwrap();
    let mut hnswsq = Vec::new();
    write_ivfhnswsq_index(ivfhnswsq, &mut PosWriter::new(&mut hnswsq)).unwrap();

    println!(
        "serialized sizes: IVF-PQ={:.2} MiB, IVF-FLAT={:.2} MiB, IVF-HNSW-FLAT={:.2} MiB, IVF-HNSW-SQ={:.2} MiB",
        bytes_to_mib(pq.len()),
        bytes_to_mib(flat.len()),
        bytes_to_mib(hnswflat.len()),
        bytes_to_mib(hnswsq.len())
    );
}

fn bytes_to_mib(bytes: usize) -> f64 {
    bytes as f64 / 1024.0 / 1024.0
}

fn print_row(
    index: &str,
    nprobe: usize,
    ef: Option<usize>,
    recall: f64,
    elapsed: std::time::Duration,
    nq: usize,
) {
    let ms = elapsed.as_secs_f64() * 1000.0;
    let ef = ef.map(|v| v.to_string()).unwrap_or_else(|| "-".to_string());
    println!(
        "{:<9}  {:>6}  {:>6}  {:>8.2}%  {:>8.2}  {:>8.1}",
        index,
        nprobe,
        ef,
        recall * 100.0,
        ms,
        ms * 1000.0 / nq as f64
    );
}

fn recall_at_k(labels: &[i64], ground_truth: &[Vec<i64>], nq: usize, k: usize) -> f64 {
    let mut hits = 0usize;
    for qi in 0..nq {
        let gt: HashSet<i64> = ground_truth[qi].iter().copied().collect();
        hits += labels[qi * k..(qi + 1) * k]
            .iter()
            .filter(|id| gt.contains(id))
            .count();
    }
    hits as f64 / (nq * k) as f64
}

fn brute_force_ground_truth(
    data: &[f32],
    queries: &[f32],
    n: usize,
    nq: usize,
    d: usize,
    k: usize,
) -> Vec<Vec<i64>> {
    (0..nq)
        .map(|qi| {
            let query = &queries[qi * d..(qi + 1) * d];
            let mut distances: Vec<(f32, i64)> = (0..n)
                .map(|i| {
                    let vector = &data[i * d..(i + 1) * d];
                    (fvec_distance(query, vector, MetricType::L2), i as i64)
                })
                .collect();
            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            distances[..k].iter().map(|&(_, id)| id).collect()
        })
        .collect()
}

fn generate_clustered_data(n: usize, d: usize, num_clusters: usize, seed: u64) -> Vec<f32> {
    let mut rng_state = seed;
    let mut next = || {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng_state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    };

    let mut centers = vec![0.0f32; num_clusters * d];
    for value in &mut centers {
        *value = next() * 30.0;
    }

    let mut data = vec![0.0f32; n * d];
    for i in 0..n {
        let cluster = i % num_clusters;
        for j in 0..d {
            data[i * d + j] = centers[cluster * d + j] + next();
        }
    }
    data
}
