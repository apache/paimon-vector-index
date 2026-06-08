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

//! Disk-based shuffler for large-scale IVF-PQ index building.
//! Inspired by Lance's shuffler: write vectors sequentially with partition IDs,
//! then read back grouped by partition for PQ encoding.

#[cfg(test)]
use std::cell::Cell;
use std::collections::{HashMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

type PartitionData = (Vec<Vec<i64>>, Vec<Vec<f32>>);

/// Record format: [partition_id: u32][row_id: i64][vector: f32 * dim]
const RECORD_OVERHEAD: usize = 4 + 8; // partition_id + row_id
const MAX_OPEN_PARTITION_WRITERS: usize = 64;

/// Disk-based shuffler that accumulates vectors with partition assignments,
/// then reads them back grouped by partition.
pub struct DiskShuffler {
    path: PathBuf,
    partition_paths: Vec<PathBuf>,
    partition_writers: HashMap<usize, BufWriter<File>>,
    writer_lru: VecDeque<usize>,
    partition_opened: Vec<bool>,
    dim: usize,
    record_size: usize,
    count: usize,
    partition_counts: Vec<usize>,
    partition_offsets: Vec<u64>,
    #[cfg(test)]
    bytes_read: Cell<u64>,
    finalized: bool,
}

impl DiskShuffler {
    /// Create a new shuffler with a temp file.
    pub fn new(dim: usize, nlist: usize) -> io::Result<Self> {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("ivfpq-shuffle-{}-{}.bin", std::process::id(), id));
        let mut partition_paths = Vec::with_capacity(nlist);
        for partition_id in 0..nlist {
            partition_paths.push(std::env::temp_dir().join(format!(
                "ivfpq-shuffle-{}-{}-part-{}.bin",
                std::process::id(),
                id,
                partition_id
            )));
        }

        Ok(DiskShuffler {
            path,
            partition_paths,
            partition_writers: HashMap::new(),
            writer_lru: VecDeque::new(),
            partition_opened: vec![false; nlist],
            dim,
            record_size: RECORD_OVERHEAD + dim * 4,
            count: 0,
            partition_counts: vec![0; nlist],
            partition_offsets: vec![0; nlist],
            #[cfg(test)]
            bytes_read: Cell::new(0),
            finalized: false,
        })
    }

    /// Write a vector with its partition assignment and row ID.
    pub fn write_vector(
        &mut self,
        partition_id: u32,
        row_id: i64,
        vector: &[f32],
    ) -> io::Result<()> {
        if vector.len() != self.dim {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "vector length {} does not match expected dim {}",
                    vector.len(),
                    self.dim
                ),
            ));
        }
        if partition_id as usize >= self.partition_counts.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "partition_id {} out of range (nlist={})",
                    partition_id,
                    self.partition_counts.len()
                ),
            ));
        }
        if self.finalized {
            return Err(io::Error::other("shuffler has already been finalized"));
        }
        let partition_idx = partition_id as usize;
        let writer = self.partition_writer(partition_idx)?;
        writer.write_all(&partition_id.to_le_bytes())?;
        writer.write_all(&row_id.to_le_bytes())?;
        for &v in vector {
            writer.write_all(&v.to_le_bytes())?;
        }
        self.partition_counts[partition_idx] += 1;
        self.count += 1;
        Ok(())
    }

    /// Finalize writing and return partition counts.
    pub fn finish_write(&mut self) -> io::Result<()> {
        if self.finalized {
            return Ok(());
        }

        for writer in self.partition_writers.values_mut() {
            writer.flush()?;
        }
        self.partition_writers.clear();
        self.writer_lru.clear();

        let mut out = BufWriter::with_capacity(8 * 1024 * 1024, File::create(&self.path)?);
        let mut offset = 0u64;
        for partition_id in 0..self.partition_counts.len() {
            self.partition_offsets[partition_id] = offset;
            if self.partition_counts[partition_id] == 0 {
                continue;
            }
            let mut input = BufReader::with_capacity(
                8 * 1024 * 1024,
                File::open(&self.partition_paths[partition_id])?,
            );
            let copied = io::copy(&mut input, &mut out)?;
            offset += copied;
        }
        out.flush()?;
        self.finalized = true;
        Ok(())
    }

    /// Read all vectors for a specific partition.
    /// Returns (row_ids, vectors) where vectors is flat [count * dim].
    pub fn read_partition(&self, partition_id: u32) -> io::Result<(Vec<i64>, Vec<f32>)> {
        self.validate_finalized()?;
        let partition_id = self.validate_partition_id(partition_id)?;
        let count = self.partition_counts[partition_id];
        if count == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        let mut ids = Vec::with_capacity(count);
        let mut vectors = Vec::with_capacity(count * self.dim);

        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(self.partition_offsets[partition_id]))?;
        let mut reader = BufReader::with_capacity(8 * 1024 * 1024, file);
        let mut record_buf = vec![0u8; self.record_size];

        for _ in 0..count {
            reader.read_exact(&mut record_buf)?;
            self.add_bytes_read(self.record_size as u64);
            let row_id = decode_row_id(&record_buf);
            ids.push(row_id);
            decode_vector(&record_buf, self.dim, &mut vectors);
        }

        Ok((ids, vectors))
    }

    /// Read all partitions at once (for moderate datasets that fit in memory after PQ encoding).
    /// Returns (ids_per_list, vectors_per_list).
    pub fn read_all_partitions(&self) -> io::Result<PartitionData> {
        self.validate_finalized()?;
        let nlist = self.partition_counts.len();
        let mut all_ids: Vec<Vec<i64>> = vec![Vec::new(); nlist];
        let mut all_vectors: Vec<Vec<f32>> = vec![Vec::new(); nlist];

        // Pre-allocate
        for p in 0..nlist {
            all_ids[p].reserve(self.partition_counts[p]);
            all_vectors[p].reserve(self.partition_counts[p] * self.dim);
        }

        let file = File::open(&self.path)?;
        let mut reader = BufReader::with_capacity(8 * 1024 * 1024, file);
        let mut record_buf = vec![0u8; self.record_size];

        for _ in 0..self.count {
            reader.read_exact(&mut record_buf)?;
            self.add_bytes_read(self.record_size as u64);
            let pid =
                u32::from_le_bytes([record_buf[0], record_buf[1], record_buf[2], record_buf[3]])
                    as usize;
            let row_id = decode_row_id(&record_buf);
            all_ids[pid].push(row_id);
            decode_vector(&record_buf, self.dim, &mut all_vectors[pid]);
        }

        Ok((all_ids, all_vectors))
    }

    pub fn total_count(&self) -> usize {
        self.count
    }

    pub fn partition_counts(&self) -> &[usize] {
        &self.partition_counts
    }

    #[cfg(test)]
    fn bytes_read(&self) -> u64 {
        self.bytes_read.get()
    }

    fn validate_partition_id(&self, partition_id: u32) -> io::Result<usize> {
        let partition_id = partition_id as usize;
        if partition_id >= self.partition_counts.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "partition_id {} out of range (nlist={})",
                    partition_id,
                    self.partition_counts.len()
                ),
            ));
        }
        Ok(partition_id)
    }

    fn validate_finalized(&self) -> io::Result<()> {
        if !self.finalized {
            return Err(io::Error::other(
                "finish_write must be called before reading",
            ));
        }
        Ok(())
    }

    #[cfg(test)]
    fn add_bytes_read(&self, bytes: u64) {
        self.bytes_read.set(self.bytes_read.get() + bytes);
    }

    #[cfg(not(test))]
    fn add_bytes_read(&self, _bytes: u64) {}

    fn partition_writer(&mut self, partition_id: usize) -> io::Result<&mut BufWriter<File>> {
        if !self.partition_writers.contains_key(&partition_id) {
            if self.partition_writers.len() == MAX_OPEN_PARTITION_WRITERS {
                if let Some(evicted_partition_id) = self.writer_lru.pop_front() {
                    if let Some(mut writer) = self.partition_writers.remove(&evicted_partition_id) {
                        writer.flush()?;
                    }
                }
            }

            let file = if self.partition_opened[partition_id] {
                OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&self.partition_paths[partition_id])?
            } else {
                self.partition_opened[partition_id] = true;
                OpenOptions::new()
                    .create(true)
                    .write(true)
                    .truncate(true)
                    .open(&self.partition_paths[partition_id])?
            };
            self.partition_writers.insert(
                partition_id,
                BufWriter::with_capacity(8 * 1024 * 1024, file),
            );
        }

        if let Some(pos) = self
            .writer_lru
            .iter()
            .position(|&cached_partition_id| cached_partition_id == partition_id)
        {
            self.writer_lru.remove(pos);
        }
        self.writer_lru.push_back(partition_id);

        self.partition_writers
            .get_mut(&partition_id)
            .ok_or_else(|| io::Error::other("failed to open partition writer"))
    }
}

impl Drop for DiskShuffler {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
        for path in &self.partition_paths {
            let _ = std::fs::remove_file(path);
        }
    }
}

fn decode_row_id(record_buf: &[u8]) -> i64 {
    i64::from_le_bytes([
        record_buf[4],
        record_buf[5],
        record_buf[6],
        record_buf[7],
        record_buf[8],
        record_buf[9],
        record_buf[10],
        record_buf[11],
    ])
}

fn decode_vector(record_buf: &[u8], dim: usize, out: &mut Vec<f32>) {
    for i in 0..dim {
        let off = RECORD_OVERHEAD + i * 4;
        let v = f32::from_le_bytes([
            record_buf[off],
            record_buf[off + 1],
            record_buf[off + 2],
            record_buf[off + 3],
        ]);
        out.push(v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_vector_validates_dim() {
        let mut shuffler = DiskShuffler::new(4, 2).unwrap();
        let err = shuffler.write_vector(0, 1, &[1.0, 2.0]).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn test_write_vector_validates_partition_id() {
        let mut shuffler = DiskShuffler::new(4, 2).unwrap();
        let err = shuffler
            .write_vector(5, 1, &[1.0, 2.0, 3.0, 4.0])
            .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn test_new_does_not_open_one_file_per_partition() {
        let shuffler = DiskShuffler::new(4, 1024).unwrap();

        assert!(shuffler.partition_writers.is_empty());
        assert_eq!(shuffler.partition_counts().len(), 1024);
    }

    #[test]
    fn test_write_vector_limits_open_partition_writers() {
        let nlist = MAX_OPEN_PARTITION_WRITERS + 8;
        let mut shuffler = DiskShuffler::new(4, nlist).unwrap();

        for partition_id in 0..nlist {
            shuffler
                .write_vector(
                    partition_id as u32,
                    partition_id as i64,
                    &[1.0, 2.0, 3.0, 4.0],
                )
                .unwrap();
        }

        assert_eq!(shuffler.partition_writers.len(), MAX_OPEN_PARTITION_WRITERS);
    }

    #[test]
    fn test_read_partition_requires_finish_write() {
        let mut shuffler = DiskShuffler::new(4, 2).unwrap();
        shuffler.write_vector(0, 1, &[1.0, 2.0, 3.0, 4.0]).unwrap();

        let err = shuffler.read_partition(0).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::Other);
    }

    #[test]
    fn test_read_partition_validates_partition_id() {
        let mut shuffler = DiskShuffler::new(4, 2).unwrap();
        shuffler.finish_write().unwrap();

        let err = shuffler.read_partition(5).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn test_shuffler_roundtrip() {
        let dim = 4;
        let nlist = 3;
        let mut shuffler = DiskShuffler::new(dim, nlist).unwrap();

        // Write vectors to different partitions
        shuffler
            .write_vector(0, 100, &[1.0, 2.0, 3.0, 4.0])
            .unwrap();
        shuffler
            .write_vector(1, 200, &[5.0, 6.0, 7.0, 8.0])
            .unwrap();
        shuffler
            .write_vector(0, 300, &[9.0, 10.0, 11.0, 12.0])
            .unwrap();
        shuffler
            .write_vector(2, 400, &[13.0, 14.0, 15.0, 16.0])
            .unwrap();
        shuffler.finish_write().unwrap();

        assert_eq!(shuffler.partition_counts(), &[2, 1, 1]);

        // Read partition 0
        let (ids, vecs) = shuffler.read_partition(0).unwrap();
        assert_eq!(ids, vec![100, 300]);
        assert_eq!(vecs.len(), 2 * dim);
        assert_eq!(&vecs[0..4], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&vecs[4..8], &[9.0, 10.0, 11.0, 12.0]);

        // Read all
        let (all_ids, all_vecs) = shuffler.read_all_partitions().unwrap();
        assert_eq!(all_ids[0], vec![100, 300]);
        assert_eq!(all_ids[1], vec![200]);
        assert_eq!(all_ids[2], vec![400]);
        assert_eq!(&all_vecs[1][..], &[5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_read_partition_uses_partition_index() {
        let dim = 2;
        let nlist = 3;
        let mut shuffler = DiskShuffler::new(dim, nlist).unwrap();

        for i in 0..10 {
            shuffler
                .write_vector((i % nlist) as u32, i as i64, &[i as f32, i as f32 + 0.5])
                .unwrap();
        }
        shuffler.finish_write().unwrap();

        let partition_id = 1;
        let partition_count = shuffler.partition_counts()[partition_id] as u64;
        let expected_bytes = partition_count * shuffler.record_size as u64;

        let before = shuffler.bytes_read();
        let (ids, vectors) = shuffler.read_partition(partition_id as u32).unwrap();
        let bytes_read = shuffler.bytes_read() - before;

        assert_eq!(ids, vec![1, 4, 7]);
        assert_eq!(vectors.len(), ids.len() * dim);
        assert_eq!(
            bytes_read, expected_bytes,
            "read_partition should read only the selected partition's records"
        );
    }
}
