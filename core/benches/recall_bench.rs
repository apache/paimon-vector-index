use paimon_vindex_core::distance::{fvec_distance, MetricType};
use paimon_vindex_core::ivfflat::IVFFlatIndex;
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

    println!();
    println!("index      nprobe  recall@{}  query_ms  us/query", s.k);
    println!("---------  ------  ---------  --------  --------");

    for &nprobe in s.nprobes {
        let mut distances = vec![0.0f32; s.nq * s.k];
        let mut labels = vec![0i64; s.nq * s.k];
        let start = Instant::now();
        ivfpq.search(queries, s.nq, s.k, nprobe, &mut distances, &mut labels);
        let elapsed = start.elapsed();
        print_row(
            "IVF-PQ",
            nprobe,
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
            recall_at_k(&labels, &ground_truth, s.nq, s.k),
            elapsed,
            s.nq,
        );
    }
}

fn print_row(index: &str, nprobe: usize, recall: f64, elapsed: std::time::Duration, nq: usize) {
    let ms = elapsed.as_secs_f64() * 1000.0;
    println!(
        "{:<9}  {:>6}  {:>8.2}%  {:>8.2}  {:>8.1}",
        index,
        nprobe,
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
