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

use paimon_vindex_core::index::{
    IndexType, VectorIndexReader, VectorIndexTrainer, VectorIndexTraining, VectorIndexWriter,
    VectorSearchParams,
};
use paimon_vindex_core::io::PosWriter;
use std::collections::HashMap;
use std::fs;
use std::io::{self, Cursor, Write};
use std::path::Path;
use std::time::Instant;

// Quality-only calibration of a shared model; final timings must use the Spark benchmark.
// Arguments: data-dir, query-IDs text file, NEW output-dir, key=value native options.
// model-file=PATH reuses a native model or Paimon routing envelope; probes=2,4 limits the sweep.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args: Vec<_> = std::env::args().skip(1).collect();
    // Cargo appends this flag even for a harness-free benchmark.
    if args.last().is_some_and(|arg| arg == "--bench") {
        args.pop();
    }
    if args.len() < 3 {
        return Err("expected data-dir query-ids output-dir [key=value ...]".into());
    }
    let data_dir = Path::new(&args[0]);
    let ids: Vec<usize> = fs::read_to_string(&args[1])?
        .split_whitespace()
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    if ids.is_empty() {
        return Err("query IDs must not be empty".into());
    }
    let out = Path::new(&args[2]);
    let (d, data) = read_records(&data_dir.join("base.fvecs"))?;
    let (qd, queries) = read_records(&data_dir.join("query.fvecs"))?;
    let (width, truth) = read_records(&data_dir.join("groundtruth.ivecs"))?;
    if d != qd
        || width < 10
        || ids
            .iter()
            .any(|&i| i >= queries.len() / d || i >= truth.len() / width)
    {
        return Err("invalid query/ground truth shape or query IDs".into());
    }
    let n = data.len() / d;
    let data: Vec<f32> = data.into_iter().map(f32::from_bits).collect();
    let queries: Vec<f32> = queries.into_iter().map(f32::from_bits).collect();
    let mut options: HashMap<String, String> = [
        ("index.type", "ivf_pq"),
        ("dimension", "128"),
        ("nlist", "64"),
        ("pq.m", "32"),
        ("metric", "l2"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect();
    options.insert("dimension".into(), d.to_string());
    for arg in &args[3..] {
        let (k, v) = arg.split_once('=').ok_or("option must be key=value")?;
        options.insert(k.into(), v.into());
    }
    let model_file = options.remove("model-file");
    let requested_probes = options.remove("probes");
    let started = Instant::now();
    let training = if let Some(path) = model_file {
        let bytes = fs::read(path)?;
        if bytes.starts_with(b"PIVFTM01") {
            VectorIndexTraining::deserialize(&bytes)?
        } else {
            let header_bytes = bytes.get(..4).ok_or("truncated model envelope")?;
            let header = u32::from_be_bytes(header_bytes.try_into()?) as usize;
            let start = header
                .checked_add(4)
                .ok_or("model envelope length overflow")?;
            let payload = bytes
                .get(start..)
                .ok_or("truncated model envelope metadata")?;
            VectorIndexTraining::deserialize(payload)?
        }
    } else {
        VectorIndexTrainer::centroid_from_options(&options)?
            .add_training_vectors(&data, n)?
            .finish()?
    };
    if training.index_type() != IndexType::IvfPq || training.dimension() != d {
        return Err("expected an IVF-PQ model matching the dataset dimension".into());
    }
    let train_seconds = started.elapsed().as_secs_f64();
    fs::create_dir(out)?;
    fs::write(out.join("model.native"), training.serialize()?)?;
    let nlist = training.nlist()?;
    let mut writer = VectorIndexWriter::new(training);
    let started = Instant::now();
    for start in (0..n).step_by(32768) {
        let end = (start + 32768).min(n);
        let row_ids: Vec<_> = (start..end).map(|i| i as i64).collect();
        writer.add_vectors(&row_ids, &data[start * d..end * d], end - start)?;
    }
    let mut index = Vec::new();
    writer.write(&mut PosWriter::new(&mut index))?;
    fs::write(out.join("index.bin"), &index)?;
    let add_seconds = started.elapsed().as_secs_f64();
    drop(writer);
    let index_bytes = index.len();
    let mut reader = VectorIndexReader::open(Cursor::new(index))?;
    reader.optimize_for_search()?;
    let mut csv = fs::File::create(out.join("queries.csv"))?;
    writeln!(csv, "nprobe,query_id,hits_at_10")?;
    let explicit_probes = requested_probes.is_some();
    let mut probes: Vec<usize> = if let Some(value) = requested_probes {
        value.split(',').map(str::parse).collect::<Result<_, _>>()?
    } else {
        vec![1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48, nlist]
    };
    if probes
        .iter()
        .any(|&p| p == 0 || (explicit_probes && p > nlist))
    {
        return Err("explicit nprobe must be in [1, model nlist]".into());
    }
    probes.retain(|p| *p <= nlist);
    probes.sort_unstable();
    probes.dedup();
    for probe in probes {
        let mut hits = 0;
        for &id in &ids {
            let (result, _) = reader.search(
                &queries[id * d..(id + 1) * d],
                VectorSearchParams::new(10, probe),
            )?;
            let count = result
                .iter()
                .filter(|&&i| i >= 0 && truth[id * width..id * width + 10].contains(&(i as u32)))
                .count();
            hits += count;
            writeln!(csv, "{probe},{id},{count}")?;
        }
        csv.flush()?;
        println!(
            "nprobe={probe}, recall={:.6}",
            hits as f64 / (10 * ids.len()) as f64
        );
    }
    let metadata = reader.metadata();
    fs::write(
        out.join("build.txt"),
        format!("options={options:?}\nmetadata={metadata:?}\ntrain_seconds={train_seconds}\nadd_seconds={add_seconds}\nindex_bytes={index_bytes}\n"),
    )?;
    println!("train_seconds={train_seconds:.3}, add_seconds={add_seconds:.3}, bytes={index_bytes}");
    Ok(())
}

fn read_records(path: &Path) -> io::Result<(usize, Vec<u32>)> {
    let bytes = fs::read(path)?;
    if bytes.len() < 4 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "empty vecs"));
    }
    let d = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
    let row_bytes = d
        .checked_add(1)
        .and_then(|fields| fields.checked_mul(4))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "dimension overflow"))?;
    if d == 0 || !bytes.len().is_multiple_of(row_bytes) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid vecs shape",
        ));
    }
    let mut values = Vec::with_capacity(bytes.len() / row_bytes * d);
    for row in bytes.chunks_exact(row_bytes) {
        if u32::from_le_bytes(row[..4].try_into().unwrap()) as usize != d {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "dimension mismatch",
            ));
        }
        values.extend(
            row[4..]
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap())),
        );
    }
    Ok((d, values))
}
