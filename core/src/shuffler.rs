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

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

type PartitionData = (Vec<Vec<i64>>, Vec<Vec<f32>>);

/// Record format: [partition_id: u32][row_id: i64][vector: f32 * dim]
const RECORD_OVERHEAD: usize = 4 + 8; // partition_id + row_id

/// Disk-based shuffler that accumulates vectors with partition assignments,
/// then reads them back grouped by partition.
pub struct DiskShuffler {
    path: PathBuf,
    writer: Option<BufWriter<File>>,
    dim: usize,
    record_size: usize,
    count: usize,
    partition_counts: Vec<usize>,
}

impl DiskShuffler {
    /// Create a new shuffler with a temp file.
    pub fn new(dim: usize, nlist: usize) -> io::Result<Self> {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("ivfpq-shuffle-{}-{}.bin", std::process::id(), id));
        let file = File::create(&path)?;
        let writer = BufWriter::with_capacity(8 * 1024 * 1024, file);

        Ok(DiskShuffler {
            path,
            writer: Some(writer),
            dim,
            record_size: RECORD_OVERHEAD + dim * 4,
            count: 0,
            partition_counts: vec![0; nlist],
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
        let writer = self.writer.as_mut().unwrap();
        writer.write_all(&partition_id.to_le_bytes())?;
        writer.write_all(&row_id.to_le_bytes())?;
        for &v in vector {
            writer.write_all(&v.to_le_bytes())?;
        }
        self.partition_counts[partition_id as usize] += 1;
        self.count += 1;
        Ok(())
    }

    /// Finalize writing and return partition counts.
    pub fn finish_write(&mut self) -> io::Result<()> {
        if let Some(w) = self.writer.take() {
            drop(w); // flush and close
        }
        Ok(())
    }

    /// Read all vectors for a specific partition.
    /// Returns (row_ids, vectors) where vectors is flat [count * dim].
    pub fn read_partition(&self, partition_id: u32) -> io::Result<(Vec<i64>, Vec<f32>)> {
        let count = self.partition_counts[partition_id as usize];
        if count == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        let mut ids = Vec::with_capacity(count);
        let mut vectors = Vec::with_capacity(count * self.dim);

        let file = File::open(&self.path)?;
        let mut reader = BufReader::with_capacity(8 * 1024 * 1024, file);
        let mut record_buf = vec![0u8; self.record_size];

        for _ in 0..self.count {
            reader.read_exact(&mut record_buf)?;
            let pid =
                u32::from_le_bytes([record_buf[0], record_buf[1], record_buf[2], record_buf[3]]);
            if pid == partition_id {
                let row_id = i64::from_le_bytes([
                    record_buf[4],
                    record_buf[5],
                    record_buf[6],
                    record_buf[7],
                    record_buf[8],
                    record_buf[9],
                    record_buf[10],
                    record_buf[11],
                ]);
                ids.push(row_id);
                for i in 0..self.dim {
                    let off = RECORD_OVERHEAD + i * 4;
                    let v = f32::from_le_bytes([
                        record_buf[off],
                        record_buf[off + 1],
                        record_buf[off + 2],
                        record_buf[off + 3],
                    ]);
                    vectors.push(v);
                }
            }
        }

        Ok((ids, vectors))
    }

    /// Read all partitions at once (for moderate datasets that fit in memory after PQ encoding).
    /// Returns (ids_per_list, vectors_per_list).
    pub fn read_all_partitions(&self) -> io::Result<PartitionData> {
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
            let pid =
                u32::from_le_bytes([record_buf[0], record_buf[1], record_buf[2], record_buf[3]])
                    as usize;
            let row_id = i64::from_le_bytes([
                record_buf[4],
                record_buf[5],
                record_buf[6],
                record_buf[7],
                record_buf[8],
                record_buf[9],
                record_buf[10],
                record_buf[11],
            ]);
            all_ids[pid].push(row_id);
            for i in 0..self.dim {
                let off = RECORD_OVERHEAD + i * 4;
                let v = f32::from_le_bytes([
                    record_buf[off],
                    record_buf[off + 1],
                    record_buf[off + 2],
                    record_buf[off + 3],
                ]);
                all_vectors[pid].push(v);
            }
        }

        Ok((all_ids, all_vectors))
    }

    pub fn total_count(&self) -> usize {
        self.count
    }

    pub fn partition_counts(&self) -> &[usize] {
        &self.partition_counts
    }
}

impl Drop for DiskShuffler {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
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
}
