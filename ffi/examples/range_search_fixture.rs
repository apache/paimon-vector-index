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
use paimon_vindex_core::index::{
    VectorIndexConfig, VectorIndexReader, VectorIndexTrainer, VectorIndexWriter,
};
use paimon_vindex_core::io::PosWriter;
use paimon_vindex_core::range::{Bound, DistanceBand, VectorRangeSearchParams};
use roaring::RoaringTreemap;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, Cursor, Write};
use std::path::Path;

const DIMENSION: usize = 16;
const ROWS: usize = 256;
const NPROBE: usize = 3;

fn main() -> io::Result<()> {
    let output = std::env::args_os()
        .nth(1)
        .ok_or_else(|| io::Error::other("usage: range_search_fixture OUTPUT_DIR"))?;
    let output = Path::new(&output);
    fs::create_dir_all(output)?;
    let mut manifest = File::create(output.join("manifest.txt"))?;
    let vectors = (0..ROWS)
        .flat_map(|row| {
            (0..DIMENSION)
                .map(move |dimension| ((row * 13 + dimension * 7) % 67) as f32 / 31.0 - 1.0)
        })
        .collect::<Vec<_>>();
    let ids = (0..ROWS)
        .map(|row| (1_i64 << 40) + row as i64)
        .collect::<Vec<_>>();
    let queries = [0, 17, 129]
        .into_iter()
        .flat_map(|row| {
            vectors[row * DIMENSION..(row + 1) * DIMENSION]
                .iter()
                .copied()
        })
        .collect::<Vec<_>>();
    let allow_list = ids
        .iter()
        .enumerate()
        .filter(|(row, _)| row % 2 == 1)
        .map(|(_, &label)| label as u64)
        .collect::<RoaringTreemap>();
    let mut filter = Vec::new();
    allow_list.serialize_into(&mut filter)?;
    let mut empty_filter = Vec::new();
    RoaringTreemap::new().serialize_into(&mut empty_filter)?;
    let mut case_count = 0;

    for family in ["ivf_flat", "ivf_sq", "ivf_pq", "ivf_rq"] {
        for (metric_name, metric, metric_code, lower, upper) in [
            ("l2", MetricType::L2, 0, 0.15, 1.75),
            ("cosine", MetricType::Cosine, 2, -0.15, 0.45),
            ("inner_product", MetricType::InnerProduct, 1, -4.0, 0.25),
        ] {
            let options = HashMap::from([
                ("index.type".to_string(), family.to_string()),
                ("dimension".to_string(), DIMENSION.to_string()),
                ("metric".to_string(), metric_name.to_string()),
                ("nlist".to_string(), NPROBE.to_string()),
            ]);
            let config = VectorIndexConfig::from_options(&options)?;
            let training = VectorIndexTrainer::train(config, &vectors, ROWS)?;
            let mut writer = VectorIndexWriter::new(training);
            writer.add_vectors(&ids, &vectors, ROWS)?;
            let mut bytes = Vec::new();
            writer.write(&mut PosWriter::new(&mut bytes))?;
            let index_name = format!("{family}-{metric_name}.index");
            fs::write(output.join(&index_name), &bytes)?;

            for (band_name, low, high) in [
                ("bounded", Bound::Finite(lower), Bound::Finite(upper)),
                ("unbounded", Bound::Unbounded, Bound::Unbounded),
                ("empty", Bound::Finite(0.0), Bound::Finite(0.0)),
            ] {
                for query_count in [1, 3] {
                    for (filter_name, filter_bytes) in
                        [("all", None), ("filtered", Some(filter.as_slice()))]
                    {
                        let name = format!(
                            "{family}-{metric_name}-{band_name}-{query_count}-{filter_name}"
                        );
                        write_case(
                            output,
                            &name,
                            &bytes,
                            metric,
                            metric_code,
                            low,
                            high,
                            &queries[..query_count * DIMENSION],
                            query_count,
                            filter_bytes,
                        )?;
                        writeln!(manifest, "{name} {index_name}")?;
                        case_count += 1;
                    }
                }
            }
            for query_count in [1, 3] {
                let name = format!("{family}-{metric_name}-empty-filter-{query_count}");
                write_case(
                    output,
                    &name,
                    &bytes,
                    metric,
                    metric_code,
                    Bound::Unbounded,
                    Bound::Unbounded,
                    &queries[..query_count * DIMENSION],
                    query_count,
                    Some(empty_filter.as_slice()),
                )?;
                writeln!(manifest, "{name} {index_name}")?;
                case_count += 1;
            }
        }
    }
    println!(
        "Generated {case_count} core range-search oracle cases in {}",
        output.display()
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_case(
    output: &Path,
    name: &str,
    index: &[u8],
    metric: MetricType,
    metric_code: u32,
    lower: Bound,
    upper: Bound,
    queries: &[f32],
    query_count: usize,
    filter: Option<&[u8]>,
) -> io::Result<()> {
    let mut reader = VectorIndexReader::open(Cursor::new(index.to_vec()))?;
    assert!(reader.supports_range_search());
    let params =
        VectorRangeSearchParams::new(DistanceBand::from_raw(lower, upper, metric)?, NPROBE);
    let result = match (query_count == 1, filter) {
        (true, None) => reader.range_search(queries, params),
        (true, Some(filter)) => reader.range_search_with_roaring_filter(queries, params, filter),
        (false, None) => reader.range_search_batch(queries, query_count, params),
        (false, Some(filter)) => {
            reader.range_search_batch_with_roaring_filter(queries, query_count, params, filter)
        }
    }?;
    let encode = |bound| match bound {
        Bound::Unbounded => (0, 0),
        Bound::Finite(value) => (1, value.to_bits()),
    };
    let (lower_kind, lower_bits) = encode(lower);
    let (upper_kind, upper_bits) = encode(upper);
    let filter = filter.unwrap_or(&[]);
    let mut file = File::create(output.join(format!("{name}.expected")))?;
    writeln!(file, "{DIMENSION} {metric_code} {query_count} {NPROBE} {lower_kind} {lower_bits} {upper_kind} {upper_bits} {} {} {}", filter.len(), result.labels().len(), result.call_stats().list_reads())?;
    for query in queries {
        writeln!(file, "{}", query.to_bits())?;
    }
    for byte in filter {
        writeln!(file, "{byte}")?;
    }
    for offset in result.lims() {
        writeln!(file, "{offset}")?;
    }
    for label in result.labels() {
        writeln!(file, "{label}")?;
    }
    for distance in result.raw_distances() {
        writeln!(file, "{}", distance.to_bits())?;
    }
    for query in 0..query_count {
        let stats = result.query(query).stats;
        writeln!(
            file,
            "{} {} {} {}",
            stats.lists_probed(),
            stats.rows_scanned(),
            stats.rows_committed(),
            stats.early_abandoned()
        )?;
    }
    Ok(())
}
