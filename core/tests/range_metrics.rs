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

use paimon_vindex_core::distance::{
    fvec_distance, fvec_inner_product, fvec_l2sqr, fvec_norm_l2sqr, fvec_normalize, MetricType,
};
use paimon_vindex_core::index::{IndexType, VectorIndexReader, VectorSearchParams};
use paimon_vindex_core::io::{write_index, IVFPQIndexReader, PosWriter};
use paimon_vindex_core::ivfflat::IVFFlatIndex;
use paimon_vindex_core::ivfflat_io::{write_ivfflat_index, IVFFlatIndexReader};
use paimon_vindex_core::ivfpq::IVFPQIndex;
use paimon_vindex_core::ivfrq::IVFRQIndex;
use paimon_vindex_core::ivfrq_io::{write_ivfrq_index, IVFRQIndexReader};
use paimon_vindex_core::ivfsq::IVFSQIndex;
use paimon_vindex_core::ivfsq_io::{write_ivfsq_index, IVFSQIndexReader};
use paimon_vindex_core::range::{
    Bound, CutOperator, DistanceBand, DistanceEndpoint, RangeSearchResult, VectorRangeSearchParams,
};
use paimon_vindex_core::rq::RQRotation;
use paimon_vindex_core::sq::ScalarQuantizer;
use roaring::RoaringTreemap;
use std::collections::BTreeSet;
use std::io::{self, Cursor};

const DIMENSION: usize = 16;
const LISTS: usize = 3;
const ROWS: usize = 37;
const METRICS: [MetricType; 3] = [MetricType::L2, MetricType::Cosine, MetricType::InnerProduct];

#[derive(Clone, Copy, Debug)]
enum Family {
    Flat,
    Sq,
    Pq(usize, bool, bool),
    Rq(usize),
}

const FAMILIES: [Family; 13] = [
    Family::Flat,
    Family::Sq,
    Family::Pq(4, false, false),
    Family::Pq(4, false, true),
    Family::Pq(4, true, false),
    Family::Pq(4, true, true),
    Family::Pq(8, false, false),
    Family::Pq(8, false, true),
    Family::Pq(8, true, false),
    Family::Pq(8, true, true),
    Family::Rq(1),
    Family::Rq(4),
    Family::Rq(8),
];

enum Source {
    Flat(IVFFlatIndex),
    Sq(IVFSQIndex),
    Pq(IVFPQIndex),
    Rq(IVFRQIndex),
}

fn normalized(mut vector: Vec<f32>, metric: MetricType) -> Vec<f32> {
    if metric == MetricType::Cosine {
        fvec_normalize(&mut vector);
    }
    vector
}

fn fixture(family: Family, metric: MetricType) -> Source {
    let centroids = (0..LISTS)
        .flat_map(|list| {
            normalized(
                (0..DIMENSION)
                    .map(|dimension| if dimension == list { 4.0 } else { 0.0 })
                    .collect(),
                metric,
            )
        })
        .collect::<Vec<_>>();
    let vectors = (0..LISTS)
        .flat_map(|list| {
            (0..ROWS).flat_map(move |row| {
                normalized(
                    (0..DIMENSION)
                        .map(|dimension| {
                            if dimension == list {
                                4.0
                            } else {
                                ((row * 13 + dimension * 7) % 31) as f32 / 31.0 - 0.5
                            }
                        })
                        .collect(),
                    metric,
                )
            })
        })
        .collect::<Vec<_>>();
    let ids = (0..LISTS * ROWS)
        .map(|row| {
            if row % 5 == 0 {
                -1000 - row as i64
            } else {
                1000 + row as i64
            }
        })
        .collect::<Vec<_>>();
    match family {
        Family::Flat => {
            let mut index = IVFFlatIndex::new(DIMENSION, LISTS, metric);
            index.set_quantizer_centroids(centroids);
            index.add(&vectors, &ids, ids.len());
            Source::Flat(index)
        }
        Family::Sq => {
            let mut index = IVFSQIndex::new(DIMENSION, LISTS, metric);
            index.set_quantizer_centroids(centroids);
            index.sq = ScalarQuantizer::with_bounds(DIMENSION, -0.6, 0.6);
            index.list_sqs = (0..LISTS).map(|_| index.sq.clone()).collect();
            index.add(&vectors, &ids, ids.len());
            Source::Sq(index)
        }
        Family::Pq(bits, opq, residual) => {
            let mut index = IVFPQIndex::with_nbits(DIMENSION, LISTS, 4, bits, metric, opq);
            index.by_residual = residual;
            index.set_quantizer_centroids(centroids);
            let codebook = (0..4 * index.pq.ksub() * 4)
                .map(|position| ((position * 17 + position / 7) % 101) as f32 / 100.0 - 0.5)
                .collect();
            index.pq.set_centroids(codebook);
            if let Some(rotation) = &mut index.opq {
                rotation.rotation = vec![0.0; DIMENSION * DIMENSION];
                for dimension in 0..DIMENSION {
                    rotation.rotation[dimension * DIMENSION + (dimension + 4) % DIMENSION] =
                        if dimension % 2 == 0 { 1.0 } else { -1.0 };
                }
                rotation.is_trained = true;
            }
            for list in 0..LISTS {
                index.ids[list] = ids[list * ROWS..(list + 1) * ROWS].to_vec();
                index.codes[list] = (0..ROWS * index.pq.code_size())
                    .map(|position| (position * 31 + list * 19) as u8)
                    .collect();
            }
            Source::Pq(index)
        }
        Family::Rq(bits) => {
            let mut index = IVFRQIndex::with_bits(DIMENSION, LISTS, bits, metric);
            index.set_quantizer_centroids(centroids);
            index.add(&vectors, &ids, ids.len());
            Source::Rq(index)
        }
    }
}

impl Source {
    fn bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        let mut writer = PosWriter::new(&mut bytes);
        match self {
            Self::Flat(index) => write_ivfflat_index(index, &mut writer),
            Self::Sq(index) => write_ivfsq_index(index, &mut writer),
            Self::Pq(index) => write_index(index, &mut writer),
            Self::Rq(index) => write_ivfrq_index(index, &mut writer),
        }
        .unwrap();
        bytes
    }

    fn centroids(&self) -> &[f32] {
        match self {
            Self::Flat(index) => index.quantizer_centroids(),
            Self::Sq(index) => index.quantizer_centroids(),
            Self::Pq(index) => index.quantizer_centroids(),
            Self::Rq(index) => index.quantizer_centroids(),
        }
    }

    fn query(&self, query: &[f32], metric: MetricType) -> Vec<f32> {
        let mut query = normalized(query.to_vec(), metric);
        if let Self::Pq(index) = self {
            if let Some(opq) = &index.opq {
                let mut rotated = vec![0.0; DIMENSION];
                opq.apply(&query, &mut rotated);
                query = rotated;
            }
        }
        query
    }

    fn lists(&self, query: &[f32], nprobe: usize) -> Vec<usize> {
        let mut lists = (0..LISTS).collect::<Vec<_>>();
        lists.sort_by(|&left, &right| {
            fvec_l2sqr(
                query,
                &self.centroids()[left * DIMENSION..(left + 1) * DIMENSION],
            )
            .total_cmp(&fvec_l2sqr(
                query,
                &self.centroids()[right * DIMENSION..(right + 1) * DIMENSION],
            ))
            .then(left.cmp(&right))
        });
        lists.truncate(nprobe.min(LISTS));
        lists
    }

    fn oracle(&self, raw_query: &[f32], metric: MetricType, nprobe: usize) -> Vec<(i64, f32)> {
        let query = self.query(raw_query, metric);
        let mut rows = Vec::new();
        for list in self.lists(&query, nprobe) {
            let centroid = &self.centroids()[list * DIMENSION..(list + 1) * DIMENSION];
            match self {
                Self::Flat(index) => {
                    for (&id, vector) in index.ids[list]
                        .iter()
                        .zip(index.vectors[list].chunks_exact(DIMENSION))
                    {
                        rows.push((id, fvec_distance(&query, vector, metric)));
                    }
                }
                Self::Sq(index) => {
                    let sq = &index.list_sqs[list];
                    for (&id, code) in index.ids[list]
                        .iter()
                        .zip(index.codes[list].chunks_exact(DIMENSION))
                    {
                        let mut dot = 0.0;
                        let mut norm: f32 = 0.0;
                        let mut squared = 0.0;
                        for dimension in 0..DIMENSION {
                            let decoded = (centroid[dimension] + sq.mins[dimension])
                                + code[dimension] as f32
                                    * ((sq.maxs[dimension] - sq.mins[dimension]) * (1.0 / 255.0));
                            dot += query[dimension] * decoded;
                            norm += decoded * decoded;
                            squared += (query[dimension] - decoded).powi(2);
                        }
                        let denominator = fvec_norm_l2sqr(&query).sqrt() * norm.sqrt();
                        rows.push((
                            id,
                            match metric {
                                MetricType::L2 => squared,
                                MetricType::InnerProduct => -dot,
                                MetricType::Cosine => {
                                    if denominator > 0.0 {
                                        1.0 - dot / denominator
                                    } else {
                                        1.0
                                    }
                                }
                            },
                        ));
                    }
                }
                Self::Pq(index) => {
                    for (&id, code) in index.ids[list]
                        .iter()
                        .zip(index.codes[list].chunks_exact(index.pq.code_size()))
                    {
                        let mut decoded = vec![0.0; DIMENSION];
                        index.pq.decode(code, &mut decoded);
                        let mut distance = 0.0;
                        for sub in 0..index.pq.m() {
                            let range = index.pq.chunk_range(sub);
                            let sub_query = query[range.clone()]
                                .iter()
                                .zip(&centroid[range.clone()])
                                .map(|(&value, &center)| {
                                    if index.by_residual && metric != MetricType::InnerProduct {
                                        value - center
                                    } else {
                                        value
                                    }
                                })
                                .collect::<Vec<_>>();
                            let mut sub_distance = if metric == MetricType::InnerProduct {
                                -fvec_inner_product(&sub_query, &decoded[range])
                            } else {
                                fvec_l2sqr(&sub_query, &decoded[range])
                            };
                            if sub == 0 && index.by_residual && metric == MetricType::InnerProduct {
                                sub_distance -= fvec_inner_product(&query, centroid);
                            }
                            distance += sub_distance;
                        }
                        if metric == MetricType::Cosine {
                            distance *= 0.5;
                        }
                        rows.push((id, distance));
                    }
                }
                Self::Rq(index) => {
                    let rotation =
                        RQRotation::new(DIMENSION, index.rotation_seed, index.rotation_rounds);
                    let mut rotated = vec![0.0; index.padded_d];
                    rotation.rotate(&query, &mut rotated, &mut vec![0.0; index.padded_d]);
                    let sum = rotated.iter().sum::<f32>();
                    let coarse = fvec_l2sqr(&query, centroid);
                    let add = match metric {
                        MetricType::L2 => coarse,
                        MetricType::Cosine => 0.5 * coarse,
                        MetricType::InnerProduct => {
                            -0.5 * (fvec_norm_l2sqr(&query) + fvec_norm_l2sqr(centroid) - coarse)
                        }
                    };
                    for (position, &id) in index.ids[list].iter().enumerate() {
                        let code = &index.codes[list]
                            [position * index.code_size()..(position + 1) * index.code_size()];
                        let subset = |plane: usize, byte: usize| {
                            (0..8)
                                .rev()
                                .filter(|bit| {
                                    code[plane * index.plane_size() + byte] & (1 << bit) != 0
                                })
                                .fold(0.0, |total, bit| total + rotated[byte * 8 + bit])
                        };
                        let mut unsigned = (0..index.plane_size())
                            .map(|byte| subset(0, byte))
                            .sum::<f32>()
                            * (1usize << (index.bits - 1)) as f32;
                        for plane in 1..index.bits {
                            for byte in 0..index.plane_size() {
                                unsigned += (1usize << (index.bits - 1 - plane)) as f32
                                    * subset(plane, byte);
                            }
                        }
                        let factors = index.factors[list][position].full;
                        let distance = factors.f_add
                            + add
                            + factors.f_rescale
                                * (unsigned - ((1usize << index.bits) - 1) as f32 * 0.5 * sum);
                        rows.push((id, distance));
                    }
                }
            }
        }
        rows
    }
}

fn direct(
    bytes: &[u8],
    family: Family,
    queries: &[f32],
    batch: bool,
    params: VectorRangeSearchParams,
    filter: Option<&[u8]>,
) -> io::Result<RangeSearchResult> {
    macro_rules! run {
        ($reader:ty) => {{
            let mut reader = <$reader>::open(Cursor::new(bytes.to_vec()))?;
            match (batch, filter) {
                (false, None) => reader.range_search(queries, params),
                (false, Some(filter)) => {
                    reader.range_search_with_roaring_filter(queries, params, filter)
                }
                (true, None) => {
                    reader.range_search_batch(queries, queries.len() / DIMENSION, params)
                }
                (true, Some(filter)) => reader.range_search_batch_with_roaring_filter(
                    queries,
                    queries.len() / DIMENSION,
                    params,
                    filter,
                ),
            }
        }};
    }
    match family {
        Family::Flat => run!(IVFFlatIndexReader<Cursor<Vec<u8>>>),
        Family::Sq => run!(IVFSQIndexReader<Cursor<Vec<u8>>>),
        Family::Pq(..) => run!(IVFPQIndexReader<Cursor<Vec<u8>>>),
        Family::Rq(..) => run!(IVFRQIndexReader<Cursor<Vec<u8>>>),
    }
}

fn pairs(rows: impl IntoIterator<Item = (i64, f32)>) -> Vec<(i64, u32)> {
    let mut rows = rows
        .into_iter()
        .map(|(id, distance)| (id, distance.to_bits()))
        .collect::<Vec<_>>();
    rows.sort_unstable();
    rows
}

fn result_pairs(result: &RangeSearchResult, query: usize) -> Vec<(i64, u32)> {
    pairs(
        result
            .query(query)
            .labels
            .iter()
            .copied()
            .zip(result.query(query).distances.iter().copied()),
    )
}

fn band(metric: MetricType) -> DistanceBand {
    DistanceBand::new(Bound::Unbounded, Bound::Unbounded, metric).unwrap()
}

fn filter_bytes(ids: impl IntoIterator<Item = i64>) -> Vec<u8> {
    let filter = ids
        .into_iter()
        .map(|id| id as u64)
        .collect::<RoaringTreemap>();
    let mut bytes = Vec::new();
    filter.serialize_into(&mut bytes).unwrap();
    bytes
}

#[test]
fn all_families_match_independent_metric_oracles_and_four_entry_points() {
    for metric in METRICS {
        for family in FAMILIES {
            if metric == MetricType::L2 && matches!(family, Family::Sq) {
                continue;
            }
            let source = fixture(family, metric);
            let bytes = source.bytes();
            let queries = (0..3 * DIMENSION)
                .map(|position| ((position * 13) % 37) as f32 / 17.0 - 0.9)
                .collect::<Vec<_>>();
            for nprobe in [1, LISTS] {
                let oracles = queries
                    .chunks_exact(DIMENSION)
                    .map(|query| source.oracle(query, metric, nprobe))
                    .collect::<Vec<_>>();
                let mut distances = oracles
                    .iter()
                    .flatten()
                    .map(|row| row.1)
                    .collect::<Vec<_>>();
                distances.sort_by(f32::total_cmp);
                let lower = distances[distances.len() / 4];
                let upper = distances[distances.len() * 3 / 4];
                let cuts =
                    DistanceBand::new(Bound::Finite(lower), Bound::Finite(upper), metric).unwrap();
                for band in [band(metric), cuts] {
                    let params = VectorRangeSearchParams::new(band, nprobe);
                    let allowed = oracles
                        .iter()
                        .flatten()
                        .filter(|row| row.0 >= 0 && row.0 % 3 != 0)
                        .map(|row| row.0)
                        .collect::<BTreeSet<_>>();
                    let filter = filter_bytes(allowed.iter().copied());
                    for selected in [None, Some(filter.as_slice())] {
                        let mut reader =
                            VectorIndexReader::open(Cursor::new(bytes.clone())).unwrap();
                        assert!(reader.supports_range_search());
                        let batch = if let Some(filter) = selected {
                            reader
                                .range_search_batch_with_roaring_filter(&queries, 3, params, filter)
                        } else {
                            reader.range_search_batch(&queries, 3, params)
                        }
                        .unwrap();
                        let typed =
                            direct(&bytes, family, &queries, true, params, selected).unwrap();
                        let unique = queries
                            .chunks_exact(DIMENSION)
                            .flat_map(|query| source.lists(&source.query(query, metric), nprobe))
                            .collect::<BTreeSet<_>>();
                        assert_eq!(batch.call_stats().list_reads(), unique.len());
                        for (query_index, query) in queries.chunks_exact(DIMENSION).enumerate() {
                            let expected = pairs(oracles[query_index].iter().copied().filter(
                                |(id, distance)| {
                                    band.admit(*distance)
                                        && (selected.is_none() || allowed.contains(id))
                                },
                            ));
                            assert_eq!(
                                result_pairs(&batch, query_index),
                                expected,
                                "{family:?} {metric:?} query={query_index}"
                            );
                            assert_eq!(result_pairs(&typed, query_index), expected);
                            let single = if let Some(filter) = selected {
                                reader.range_search_with_roaring_filter(query, params, filter)
                            } else {
                                reader.range_search(query, params)
                            }
                            .unwrap();
                            assert_eq!(result_pairs(&single, 0), expected);
                            assert_eq!(
                                result_pairs(
                                    &direct(&bytes, family, query, false, params, selected)
                                        .unwrap(),
                                    0
                                ),
                                expected
                            );
                            let stats = batch.query(query_index).stats;
                            assert_eq!(stats.rows_committed(), expected.len());
                            assert_eq!(stats.lists_probed(), nprobe);
                            assert_eq!(
                                stats.rows_scanned(),
                                oracles[query_index]
                                    .iter()
                                    .filter(|row| selected.is_none() || allowed.contains(&row.0))
                                    .count()
                            );
                            if metric != MetricType::L2
                                || matches!(family, Family::Rq(_) | Family::Pq(..))
                            {
                                assert_eq!(stats.early_abandoned(), 0);
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn endpoints_match_public_predicates_at_ulps_zeros_and_extremes() {
    let mut values = vec![
        -f32::MAX,
        -f32::MIN_POSITIVE,
        -f32::from_bits(1),
        -0.0,
        0.0,
        f32::from_bits(1),
        f32::MIN_POSITIVE,
        f32::MAX,
    ];
    for center in [-2.0f32, -0.5, 0.5, 1.0, 2.0] {
        values.extend([
            f32::from_bits(center.to_bits() - 1),
            center,
            f32::from_bits(center.to_bits() + 1),
        ]);
    }
    for metric in METRICS {
        for &stored in &values {
            if metric == MetricType::L2 && stored < 0.0 {
                continue;
            }
            let public = match metric {
                MetricType::L2 => f64::from(stored.sqrt()),
                MetricType::Cosine => f64::from(stored),
                MetricType::InnerProduct => -f64::from(stored),
            };
            assert_eq!(metric.public_distance(stored), public);
            let epsilon = public.abs().max(f64::MIN_POSITIVE) * f64::EPSILON;
            for value in [public - epsilon, public, public + epsilon] {
                for op in [
                    CutOperator::Ge,
                    CutOperator::Gt,
                    CutOperator::Le,
                    CutOperator::Lt,
                ] {
                    let endpoint = Some(DistanceEndpoint { value, op });
                    let (lower, upper) = if matches!(op, CutOperator::Ge | CutOperator::Gt) {
                        (endpoint, None)
                    } else {
                        (None, endpoint)
                    };
                    let derived = DistanceBand::from_endpoints(lower, upper, metric);
                    let band = match derived {
                        Ok(band) => band,
                        Err(error) => {
                            assert_eq!(metric, MetricType::L2);
                            assert_eq!(error.kind(), io::ErrorKind::Unsupported);
                            continue;
                        }
                    };
                    for &candidate in &values {
                        if metric == MetricType::L2 && candidate < 0.0 {
                            continue;
                        }
                        let displayed = metric.public_distance(candidate);
                        let expected = match op {
                            CutOperator::Ge => displayed >= value,
                            CutOperator::Gt => displayed > value,
                            CutOperator::Le => displayed <= value,
                            CutOperator::Lt => displayed < value,
                        };
                        assert_eq!(
                            band.admit(candidate),
                            expected,
                            "{metric:?} {op:?} {value} candidate={candidate}"
                        );
                    }
                }
            }
        }
    }
    for metric in [MetricType::Cosine, MetricType::InnerProduct] {
        for &stored in &values {
            let value = metric.public_distance(stored);
            let singleton = DistanceBand::from_endpoints(
                Some(DistanceEndpoint {
                    value,
                    op: CutOperator::Ge,
                }),
                Some(DistanceEndpoint {
                    value,
                    op: CutOperator::Le,
                }),
                metric,
            )
            .unwrap();
            for &candidate in &values {
                assert_eq!(
                    singleton.admit(candidate),
                    metric.public_distance(candidate) == value,
                    "{metric:?} singleton={value} candidate={candidate}"
                );
            }
        }
        for (value, op) in [(-f64::MAX, CutOperator::Lt), (f64::MAX, CutOperator::Gt)] {
            let endpoint = Some(DistanceEndpoint { value, op });
            let (lower, upper) = if op == CutOperator::Gt {
                (endpoint, None)
            } else {
                (None, endpoint)
            };
            assert!(DistanceBand::from_endpoints(lower, upper, metric)
                .unwrap()
                .is_empty());
        }
        for value in [-f64::MAX, 0.0, f64::MAX] {
            let empty = DistanceBand::from_endpoints(
                Some(DistanceEndpoint {
                    value,
                    op: CutOperator::Gt,
                }),
                Some(DistanceEndpoint {
                    value,
                    op: CutOperator::Lt,
                }),
                metric,
            )
            .unwrap();
            assert!(values.iter().all(|&value| !empty.admit(value)));
        }
    }
}

#[test]
fn capability_validation_empty_bands_and_topk_are_preserved() {
    for metric in METRICS {
        assert!(!IndexType::DiskAnn.supports_range_search(metric));
        for family in FAMILIES {
            let bytes = fixture(family, metric).bytes();
            let query = vec![0.2; DIMENSION];
            let mut reader = VectorIndexReader::open(Cursor::new(bytes.clone())).unwrap();
            let params = VectorRangeSearchParams::new(band(metric), LISTS);
            let topk = reader
                .search(&query, VectorSearchParams::new(7, LISTS))
                .unwrap();
            reader.range_search(&query, params).unwrap();
            assert_eq!(
                reader
                    .search(&query, VectorSearchParams::new(7, LISTS))
                    .unwrap(),
                topk
            );
            let empty = VectorRangeSearchParams::new(
                DistanceBand::new(Bound::Finite(0.5), Bound::Finite(0.5), metric).unwrap(),
                LISTS,
            );
            assert_eq!(
                direct(&bytes, family, &query[..DIMENSION - 1], false, empty, None)
                    .unwrap_err()
                    .kind(),
                io::ErrorKind::InvalidInput
            );
            let wrong_metric = if metric == MetricType::L2 {
                MetricType::Cosine
            } else {
                MetricType::L2
            };
            assert_eq!(
                direct(
                    &bytes,
                    family,
                    &query,
                    false,
                    VectorRangeSearchParams::new(band(wrong_metric), 1),
                    None
                )
                .unwrap_err()
                .kind(),
                io::ErrorKind::InvalidInput
            );
            for batch in [false, true] {
                let result = direct(&bytes, family, &query, batch, empty, None).unwrap();
                assert!(result.labels().is_empty());
                assert_eq!(result.call_stats().list_reads(), 0);
                assert_eq!(result.query(0).stats.lists_probed(), 0);
                for invalid in [
                    vec![f32::NAN; DIMENSION],
                    vec![f32::INFINITY; DIMENSION],
                    vec![f32::NEG_INFINITY; DIMENSION],
                ] {
                    assert_eq!(
                        direct(&bytes, family, &invalid, batch, empty, None)
                            .unwrap_err()
                            .kind(),
                        io::ErrorKind::InvalidInput
                    );
                }
                assert_eq!(
                    direct(&bytes, family, &query, batch, empty, Some(&[255]))
                        .unwrap_err()
                        .kind(),
                    io::ErrorKind::InvalidInput
                );
                assert_eq!(
                    direct(
                        &bytes,
                        family,
                        &query,
                        batch,
                        VectorRangeSearchParams::new(band(metric), 0),
                        None
                    )
                    .unwrap_err()
                    .kind(),
                    io::ErrorKind::InvalidInput
                );
            }
        }
    }
}

#[test]
fn nonfinite_consumed_data_is_rejected_without_poisoning_filtered_rows() {
    for metric in [MetricType::Cosine, MetricType::InnerProduct] {
        for family in FAMILIES {
            for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut source = fixture(family, metric);
                match &mut source {
                    Source::Flat(index) => index.vectors[0][0] = invalid,
                    Source::Sq(index) => {
                        index.list_sqs[0] = ScalarQuantizer::with_bounds(DIMENSION, 1.0e38, 1.0e38);
                    }
                    Source::Pq(index) => {
                        let mut centroids = index.pq.centroids().to_vec();
                        let code = index.codes[0][0] as usize;
                        centroids[code * index.pq.dsub()] = invalid;
                        index.pq.set_centroids(centroids);
                    }
                    Source::Rq(index) => {
                        index.factors[0][0].coarse.f_add = invalid;
                        index.factors[0][0].full.f_add = invalid;
                    }
                }
                let query = vec![4.0; DIMENSION];
                let bytes = source.bytes();
                let params = VectorRangeSearchParams::new(band(metric), LISTS);
                let empty_filter = filter_bytes([]);
                for batch in [false, true] {
                    let queries = if batch {
                        query.repeat(2)
                    } else {
                        query.clone()
                    };
                    assert_eq!(
                        direct(&bytes, family, &queries, batch, params, None)
                            .unwrap_err()
                            .kind(),
                        io::ErrorKind::InvalidData,
                        "{family:?} {metric:?}"
                    );
                    let empty =
                        direct(&bytes, family, &queries, batch, params, Some(&empty_filter))
                            .unwrap();
                    assert!(empty.labels().is_empty());
                    assert_eq!(empty.query(0).stats.rows_scanned(), 0);
                }
                let good = source
                    .oracle(&query, metric, LISTS)
                    .into_iter()
                    .find(|&(id, distance)| id >= 0 && distance.is_finite())
                    .unwrap();
                let only_good = filter_bytes([good.0]);
                let result =
                    direct(&bytes, family, &query, false, params, Some(&only_good)).unwrap();
                assert_eq!(result.labels(), &[good.0]);
            }
        }
    }
}

#[test]
fn cosine_zero_queries_still_validate_consumed_row_norms() {
    let metric = MetricType::Cosine;
    let mut flat = IVFFlatIndex::new(DIMENSION, 1, metric);
    flat.set_quantizer_centroids(vec![0.0; DIMENSION]);
    flat.ids[0] = vec![1, 2];
    flat.vectors[0] = vec![0.0; 2 * DIMENSION];
    flat.vectors[0][DIMENSION..].fill(1.0e20);

    let mut sq = IVFSQIndex::new(DIMENSION, 1, metric);
    sq.set_quantizer_centroids(vec![0.0; DIMENSION]);
    sq.sq = ScalarQuantizer::with_bounds(DIMENSION, 0.0, 1.0e20);
    sq.list_sqs = vec![sq.sq.clone()];
    sq.ids[0] = vec![1, 2];
    sq.codes[0] = vec![0; 2 * DIMENSION];
    sq.codes[0][DIMENSION..].fill(255);

    for (family, source) in [
        (Family::Flat, Source::Flat(flat)),
        (Family::Sq, Source::Sq(sq)),
    ] {
        let bytes = source.bytes();
        let params = VectorRangeSearchParams::new(band(metric), 1);
        let bad = filter_bytes([2]);
        let good = filter_bytes([1]);
        for count in [1, 2] {
            let queries = vec![0.0; count * DIMENSION];
            for filter in [None, Some(bad.as_slice())] {
                assert_eq!(
                    direct(&bytes, family, &queries, count > 1, params, filter)
                        .unwrap_err()
                        .kind(),
                    io::ErrorKind::InvalidData
                );
            }
            let result = direct(&bytes, family, &queries, count > 1, params, Some(&good)).unwrap();
            for query in 0..count {
                assert_eq!(result.query(query).labels, &[1]);
                assert_eq!(result.query(query).distances, &[1.0]);
                assert_eq!(result.query(query).stats.rows_scanned(), 1);
            }
        }
    }
}

#[test]
fn nonfinite_coarse_data_and_finite_query_overflow_fail_loud() {
    for metric in [MetricType::Cosine, MetricType::InnerProduct] {
        for family in FAMILIES {
            for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.0e20] {
                let source = fixture(family, metric);
                let mut reader = VectorIndexReader::open(Cursor::new(source.bytes())).unwrap();
                macro_rules! corrupt {
                    ($reader:expr) => {{
                        $reader.ensure_loaded().unwrap();
                        $reader.quantizer_centroids[DIMENSION * (LISTS - 1)] = invalid;
                    }};
                }
                match &mut reader {
                    VectorIndexReader::IvfFlat(reader) => corrupt!(reader),
                    VectorIndexReader::IvfSq(reader) => corrupt!(reader),
                    VectorIndexReader::IvfPq(reader) => corrupt!(reader),
                    VectorIndexReader::IvfRq(reader) => corrupt!(reader),
                    VectorIndexReader::DiskAnn(_) => unreachable!(),
                }
                let params = VectorRangeSearchParams::new(band(metric), 1);
                assert_eq!(
                    reader
                        .range_search(&[0.0; DIMENSION], params)
                        .unwrap_err()
                        .kind(),
                    io::ErrorKind::InvalidData
                );
            }
            let source = fixture(family, metric);
            assert_eq!(
                direct(
                    &source.bytes(),
                    family,
                    &[f32::MAX; DIMENSION],
                    false,
                    VectorRangeSearchParams::new(band(metric), 1),
                    None
                )
                .unwrap_err()
                .kind(),
                io::ErrorKind::InvalidData
            );
        }
    }
}

#[test]
fn flat_zero_vectors_and_inner_product_extrema_use_public_semantics() {
    for metric in [MetricType::Cosine, MetricType::InnerProduct] {
        let mut index = IVFFlatIndex::new(DIMENSION, 1, metric);
        index.set_quantizer_centroids(vec![0.0; DIMENSION]);
        let values = if metric == MetricType::Cosine {
            vec![0.0, -1.0, 1.0]
        } else {
            vec![-f32::MAX, -1.0, 0.0, 1.0, f32::MAX]
        };
        for (position, &value) in values.iter().enumerate() {
            index.ids[0].push(position as i64);
            let mut vector = vec![0.0; DIMENSION];
            vector[0] = value;
            index.vectors[0].extend(vector);
        }
        let bytes = Source::Flat(index).bytes();
        for first in [0.0, 1.0] {
            let mut query = vec![0.0; DIMENSION];
            query[0] = first;
            let params = VectorRangeSearchParams::new(band(metric), 1);
            let result = direct(&bytes, Family::Flat, &query, false, params, None).unwrap();
            assert_eq!(result.labels().len(), values.len());
            for (&id, &distance) in result.labels().iter().zip(result.distances()) {
                let value = values[id as usize];
                let expected = if metric == MetricType::Cosine {
                    if value == 0.0 || first == 0.0 {
                        1.0
                    } else {
                        1.0 - value
                    }
                } else {
                    -(first * value)
                };
                assert_eq!(distance, expected);
            }
            for public in [-f64::from(f32::MAX), 0.0, 1.0, f64::from(f32::MAX)] {
                let selected = DistanceBand::from_endpoints(
                    Some(DistanceEndpoint {
                        value: public,
                        op: CutOperator::Ge,
                    }),
                    None,
                    metric,
                )
                .unwrap();
                let expected = pairs(
                    result
                        .labels()
                        .iter()
                        .copied()
                        .zip(result.distances().iter().copied())
                        .filter(|&(_, distance)| metric.public_distance(distance) >= public),
                );
                let found = direct(
                    &bytes,
                    Family::Flat,
                    &query,
                    false,
                    VectorRangeSearchParams::new(selected, 1),
                    None,
                )
                .unwrap();
                assert_eq!(result_pairs(&found, 0), expected);
            }
        }
    }
}

#[test]
fn quantized_zero_queries_keep_estimator_membership() {
    for family in FAMILIES {
        if matches!(family, Family::Flat) {
            continue;
        }
        let source = fixture(family, MetricType::Cosine);
        let query = [0.0; DIMENSION];
        let expected = source.oracle(&query, MetricType::Cosine, LISTS);
        let result = direct(
            &source.bytes(),
            family,
            &query,
            false,
            VectorRangeSearchParams::new(band(MetricType::Cosine), LISTS),
            None,
        )
        .unwrap();
        assert_eq!(result_pairs(&result, 0), pairs(expected));
    }
}

#[test]
fn pq_streaming_shares_reads_filters_and_propagates_errors() {
    use paimon_vindex_core::io::{ReadRequest, SeekRead};
    use paimon_vindex_core::ivfpq::RowIdFilter;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    struct LimitedReader {
        data: Cursor<Vec<u8>>,
        calls: Arc<AtomicUsize>,
    }
    impl SeekRead for LimitedReader {
        fn pread(&mut self, requests: &mut [ReadRequest<'_>]) -> io::Result<()> {
            assert!(
                requests
                    .iter()
                    .map(|request| request.buf.len())
                    .sum::<usize>()
                    <= 64 * 1024 * 1024
            );
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.data.pread(requests)
        }
    }
    struct CountingFilter {
        calls: AtomicUsize,
        last: i64,
    }
    impl RowIdFilter for CountingFilter {
        fn contains(&self, id: i64) -> bool {
            self.calls.fetch_add(1, Ordering::Relaxed);
            id == 0 || id == self.last
        }
    }
    for bits in [4, 8] {
        let dimension = 256;
        let code_size = dimension * bits / 8;
        let count = 64 * 1024 * 1024 / code_size + 1;
        let mut index =
            IVFPQIndex::with_nbits(dimension, 1, dimension, bits, MetricType::Cosine, false);
        index.set_quantizer_centroids(vec![0.0; dimension]);
        index
            .pq
            .set_centroids(vec![0.25; dimension * index.pq.ksub()]);
        index.ids[0] = (0..count as i64).collect();
        index.codes[0] = vec![0; count * code_size];
        let bytes = Source::Pq(index).bytes();
        let calls = Arc::new(AtomicUsize::new(0));
        let mut reader = IVFPQIndexReader::open(LimitedReader {
            data: Cursor::new(bytes),
            calls: calls.clone(),
        })
        .unwrap();
        let query = vec![1.0; dimension];
        let filter = CountingFilter {
            calls: AtomicUsize::new(0),
            last: count as i64 - 1,
        };
        let params = VectorRangeSearchParams::new(band(MetricType::Cosine), 1);
        let result = reader
            .range_search_batch_with_filter(&query.repeat(2), 2, params, Some(&filter))
            .unwrap();
        assert_eq!(result.call_stats().list_reads(), 1);
        assert_eq!(filter.calls.load(Ordering::Relaxed), count);
        assert!(calls.load(Ordering::Relaxed) > 3);
        for query_index in 0..2 {
            assert_eq!(result.query(query_index).labels, &[0, count as i64 - 1]);
            assert_eq!(result.query(query_index).stats.rows_scanned(), 2);
            assert_eq!(result.query(query_index).stats.early_abandoned(), 0);
            assert_eq!(result.query(query_index).distances, &[4.5, 4.5]);
        }
        let mut centroids = reader.pq.centroids().to_vec();
        centroids[0] = f32::NAN;
        reader.pq.set_centroids(centroids);
        assert_eq!(
            reader
                .range_search_with_filter(&query, params, Some(&filter))
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidData
        );
    }
}
