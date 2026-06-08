use paimon_vindex_core::distance::MetricType;
use paimon_vindex_core::fastscan::{fastscan_4bit, pack_codes_block_layout};
use paimon_vindex_core::ivfpq::IVFPQIndex;
use std::collections::HashSet;
use std::time::Instant;

fn main() {
    println!("=== Paimon IVF-PQ Full Benchmark ===\n");

    let d = 128;
    let nlist = 256;
    let n = 100_000;
    let nprobe = 8;
    let k = 10;
    let nq = 100;

    println!(
        "Dataset: {}K vectors, d={}, nlist={}, nprobe={}, k={}",
        n / 1000,
        d,
        nlist,
        nprobe,
        k
    );
    println!();

    // Generate data
    let mut rng_state: u64 = 42;
    let mut next = || -> f32 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng_state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    };
    let num_clusters = 16;
    let mut centers = vec![0.0f32; num_clusters * d];
    for v in centers.iter_mut() {
        *v = next() * 50.0;
    }
    let mut data = vec![0.0f32; n * d];
    for i in 0..n {
        let c = i % num_clusters;
        for j in 0..d {
            data[i * d + j] = centers[c * d + j] + next();
        }
    }
    let ids: Vec<i64> = (0..n as i64).collect();
    let queries = &data[..nq * d];

    // === 8-bit M=16 ===
    let m8 = 16;
    print!("Building 8-bit (M={})...", m8);
    let start = Instant::now();
    let mut idx8 = IVFPQIndex::new(d, nlist, m8, MetricType::L2, false);
    idx8.train(&data, n);
    idx8.add(&data, &ids, n);
    idx8.build_search_structures();
    idx8.build_precomputed_table();
    let build8 = start.elapsed();
    println!(" {:.2}s", build8.as_secs_f64());

    // === 4-bit M=32 (same storage) ===
    let m4 = 32;
    print!("Building 4-bit (M={})...", m4);
    let start = Instant::now();
    let mut idx4 = IVFPQIndex::with_nbits(d, nlist, m4, 4, MetricType::L2, false);
    idx4.train(&data, n);
    idx4.add(&data, &ids, n);
    idx4.build_search_structures();
    idx4.build_precomputed_table();
    let build4 = start.elapsed();
    println!(" {:.2}s", build4.as_secs_f64());

    // === Query: 8-bit ===
    let mut d8 = vec![0.0f32; nq * k];
    let mut l8 = vec![0i64; nq * k];
    let start = Instant::now();
    for _ in 0..5 {
        idx8.search(queries, nq, k, nprobe, &mut d8, &mut l8);
    }
    let q8 = start.elapsed().as_secs_f64() / 5.0;

    // === Query: 4-bit (standard scan) ===
    let mut d4 = vec![0.0f32; nq * k];
    let mut l4 = vec![0i64; nq * k];
    let start = Instant::now();
    for _ in 0..5 {
        idx4.search(queries, nq, k, nprobe, &mut d4, &mut l4);
    }
    let q4 = start.elapsed().as_secs_f64() / 5.0;

    // === Query: 4-bit FastScan (block layout) ===
    // Simulate: pack one list's codes and scan with fastscan
    let biggest_list = idx4
        .codes
        .iter()
        .enumerate()
        .max_by_key(|(_, c)| c.len())
        .map(|(i, _)| i)
        .unwrap();
    let list_n = idx4.ids[biggest_list].len();
    let cs4 = idx4.pq.code_size();
    let packed = pack_codes_block_layout(&idx4.codes[biggest_list], list_n, cs4);

    // Build distance table for benchmark
    let mut sim_table = vec![0.0f32; m4 * 16];
    let query0 = &data[0..d];
    // compute residual
    let centroid = &idx4.quantizer_centroids[biggest_list * d..(biggest_list + 1) * d];
    let residual: Vec<f32> = (0..d).map(|j| query0[j] - centroid[j]).collect();
    idx4.pq
        .compute_distance_table(&residual, MetricType::L2, &mut sim_table);

    let mut fs_dists = vec![0.0f32; list_n];
    let start = Instant::now();
    for _ in 0..100 {
        fastscan_4bit(&sim_table, &packed, list_n, m4, &mut fs_dists);
    }
    let fs_us = start.elapsed().as_micros() as f64 / 100.0;

    // === Recall ===
    let nq_r = nq.min(20);
    let mut recall_4 = 0usize;
    let mut recall_8 = 0usize;
    for qi in 0..nq_r {
        let query = &data[qi * d..(qi + 1) * d];
        let mut bf: Vec<(f32, i64)> = (0..n)
            .map(|i| {
                let mut dist = 0.0f32;
                for j in 0..d {
                    let diff = query[j] - data[i * d + j];
                    dist += diff * diff;
                }
                (dist, i as i64)
            })
            .collect();
        bf.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let gt: HashSet<i64> = bf[..k].iter().map(|&(_, id)| id).collect();
        let base = qi * k;
        recall_4 += l4[base..base + k]
            .iter()
            .filter(|id| gt.contains(id))
            .count();
        recall_8 += l8[base..base + k]
            .iter()
            .filter(|id| gt.contains(id))
            .count();
    }

    let codes_4: usize = idx4.codes.iter().map(|c| c.len()).sum();
    let codes_8: usize = idx8.codes.iter().map(|c| c.len()).sum();

    // === Print results ===
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║              Paimon IVF-PQ Performance Summary                   ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!(
        "║                    8-bit (M={})       4-bit (M={})              ║",
        m8, m4
    );
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!(
        "║ Storage/vec:       {} bytes           {} bytes                  ║",
        codes_8 / n,
        codes_4 / n
    );
    println!(
        "║ Build time:        {:.2}s              {:.2}s                   ║",
        build8.as_secs_f64(),
        build4.as_secs_f64()
    );
    println!(
        "║ Query (nq={}):    {:.1}ms ({:.0}μs/q)    {:.1}ms ({:.0}μs/q)   ║",
        nq,
        q8 * 1000.0,
        q8 * 1e6 / nq as f64,
        q4 * 1000.0,
        q4 * 1e6 / nq as f64
    );
    println!(
        "║ Recall@{}:         {:.1}%              {:.1}%                   ║",
        k,
        recall_8 as f64 / (nq_r * k) as f64 * 100.0,
        recall_4 as f64 / (nq_r * k) as f64 * 100.0
    );
    println!(
        "║ FastScan (1 list): -                  {:.0}μs ({} vecs)         ║",
        fs_us, list_n
    );
    println!("╚══════════════════════════════════════════════════════════════════╝");
}
