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

use super::*;
use paimon_vindex_core::range::{Bound, CutOperator, DistanceBand, DistanceEndpoint};

#[test]
fn raw_names_preserve_c_abi_layout() {
    use std::mem::{align_of, offset_of, size_of};

    assert_eq!(size_of::<PaimonVindexRawDistanceBand>(), 20);
    assert_eq!(align_of::<PaimonVindexRawDistanceBand>(), 4);
    assert_eq!(offset_of!(PaimonVindexRawDistanceBand, metric), 0);
    assert_eq!(offset_of!(PaimonVindexRawDistanceBand, raw_lower_kind), 4);
    assert_eq!(offset_of!(PaimonVindexRawDistanceBand, raw_lower), 8);
    assert_eq!(offset_of!(PaimonVindexRawDistanceBand, raw_upper_kind), 12);
    assert_eq!(offset_of!(PaimonVindexRawDistanceBand, raw_upper), 16);

    let word = size_of::<usize>();
    assert_eq!(size_of::<PaimonVindexRangeSearchResultView>(), 7 * word);
    assert_eq!(
        align_of::<PaimonVindexRangeSearchResultView>(),
        align_of::<usize>()
    );
    assert_eq!(
        offset_of!(PaimonVindexRangeSearchResultView, query_count),
        0
    );
    assert_eq!(
        offset_of!(PaimonVindexRangeSearchResultView, hit_count),
        word
    );
    assert_eq!(
        offset_of!(PaimonVindexRangeSearchResultView, lims),
        2 * word
    );
    assert_eq!(
        offset_of!(PaimonVindexRangeSearchResultView, labels),
        3 * word
    );
    assert_eq!(
        offset_of!(PaimonVindexRangeSearchResultView, raw_distances),
        4 * word
    );
    assert_eq!(
        offset_of!(PaimonVindexRangeSearchResultView, stats),
        5 * word
    );
    assert_eq!(
        offset_of!(PaimonVindexRangeSearchResultView, list_reads),
        6 * word
    );
}

#[test]
fn range_endpoints_match_core_for_every_metric_and_operator() {
    for (metric_code, metric) in [
        (PAIMON_VINDEX_METRIC_L2, MetricType::L2),
        (PAIMON_VINDEX_METRIC_COSINE, MetricType::Cosine),
        (PAIMON_VINDEX_METRIC_INNER_PRODUCT, MetricType::InnerProduct),
    ] {
        for (lower_code, lower_op) in [
            (PAIMON_VINDEX_CUT_GE, CutOperator::Ge),
            (PAIMON_VINDEX_CUT_GT, CutOperator::Gt),
        ] {
            for (upper_code, upper_op) in [
                (PAIMON_VINDEX_CUT_LE, CutOperator::Le),
                (PAIMON_VINDEX_CUT_LT, CutOperator::Lt),
            ] {
                for (lower_value, upper_value) in [
                    (0.1, 1.1),
                    (1.0, 2.0),
                    (1.0_f64.next_down(), 2.0_f64.next_up()),
                    (1.0_f64.next_up(), 2.0_f64.next_down()),
                ] {
                    let lower = PaimonVindexDistanceEndpoint {
                        value: lower_value,
                        op: lower_code,
                    };
                    let upper = PaimonVindexDistanceEndpoint {
                        value: upper_value,
                        op: upper_code,
                    };
                    let expected = DistanceBand::from_endpoints(
                        Some(DistanceEndpoint {
                            value: lower.value,
                            op: lower_op,
                        }),
                        Some(DistanceEndpoint {
                            value: upper.value,
                            op: upper_op,
                        }),
                        metric,
                    )
                    .unwrap();
                    let mut actual = std::mem::MaybeUninit::uninit();
                    assert_eq!(
                        unsafe {
                            paimon_vindex_distance_band_from_endpoints(
                                metric_code,
                                &lower,
                                &upper,
                                actual.as_mut_ptr(),
                            )
                        },
                        0
                    );
                    let actual = unsafe { actual.assume_init() };
                    assert_eq!(actual.metric, metric_code);
                    for (kind, value, expected) in [
                        (
                            actual.raw_lower_kind,
                            actual.raw_lower,
                            expected.raw_lower(),
                        ),
                        (
                            actual.raw_upper_kind,
                            actual.raw_upper,
                            expected.raw_upper(),
                        ),
                    ] {
                        match expected {
                            Bound::Unbounded => assert_eq!(kind, PAIMON_VINDEX_BOUND_UNBOUNDED),
                            Bound::Finite(expected) => {
                                assert_eq!(kind, PAIMON_VINDEX_BOUND_FINITE);
                                assert_eq!(value.to_bits(), expected.to_bits());
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn range_errors_leave_no_owned_result() {
    let params = PaimonVindexRangeSearchParams {
        band: PaimonVindexRawDistanceBand {
            metric: 0,
            raw_lower_kind: 0,
            raw_lower: 0.0,
            raw_upper_kind: 0,
            raw_upper: 0.0,
        },
        nprobe: 1,
    };
    let mut result = ptr::dangling_mut::<PaimonVindexRangeSearchResult>();
    assert_eq!(
        unsafe {
            paimon_vindex_reader_range_search(ptr::null_mut(), ptr::null(), 0, params, &mut result)
        },
        -1
    );
    assert!(result.is_null());
    unsafe { paimon_vindex_range_search_result_destroy(result) };
}

struct RangeInput {
    bytes: Vec<u8>,
    fail_reads: bool,
}

struct RangeReader {
    handle: *mut PaimonVindexReaderHandle,
    input: Box<RangeInput>,
}

impl RangeReader {
    fn new() -> Self {
        let config = VectorIndexConfig::from_options(&HashMap::from([
            ("index.type".to_string(), "ivf_flat".to_string()),
            ("dimension".to_string(), "2".to_string()),
            ("nlist".to_string(), "1".to_string()),
            ("metric".to_string(), "l2".to_string()),
        ]))
        .unwrap();
        let vectors = [1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0];
        let training = VectorIndexTrainer::train(config, &vectors, 4).unwrap();
        let mut writer = VectorIndexWriter::new(training);
        writer
            .add_vectors(&[1, 2, i64::MAX, -1099511627776], &vectors, 4)
            .unwrap();
        let mut bytes = Vec::new();
        writer
            .write(&mut paimon_vindex_core::io::PosWriter::new(&mut bytes))
            .unwrap();
        let mut input = Box::new(RangeInput {
            bytes,
            fail_reads: false,
        });
        let handle = unsafe {
            paimon_vindex_reader_open(PaimonVindexInputFile {
                ctx: (&mut *input as *mut RangeInput).cast(),
                read_ranges_fn: Some(range_read),
                estimated_random_read_latency_nanos: 0,
                preferred_window_bytes: 0,
                max_ranges_per_read: 0,
            })
        };
        assert!(!handle.is_null());
        Self { handle, input }
    }
}

impl Drop for RangeReader {
    fn drop(&mut self) {
        unsafe { paimon_vindex_reader_free(self.handle) };
    }
}

unsafe extern "C" fn range_read(
    ctx: *mut c_void,
    requests: *mut PaimonVindexReadRequest,
    count: usize,
) -> c_int {
    let input = unsafe { &*ctx.cast::<RangeInput>() };
    if input.fail_reads {
        return -1;
    }
    for request in unsafe { slice::from_raw_parts_mut(requests, count) } {
        let Ok(start) = usize::try_from(request.offset) else {
            return -1;
        };
        let Some(end) = start.checked_add(request.len) else {
            return -1;
        };
        let Some(source) = input.bytes.get(start..end) else {
            return -1;
        };
        unsafe { slice::from_raw_parts_mut(request.buf, request.len) }.copy_from_slice(source);
    }
    0
}

fn unbounded_params() -> PaimonVindexRangeSearchParams {
    PaimonVindexRangeSearchParams {
        band: PaimonVindexRawDistanceBand {
            metric: 0,
            raw_lower_kind: 0,
            raw_lower: f32::NAN,
            raw_upper_kind: 0,
            raw_upper: f32::NAN,
        },
        nprobe: 1,
    }
}

#[test]
fn range_result_outlives_reader_and_preserves_signed_labels_and_statistics() {
    let reader = RangeReader::new();
    let mut supported = 0;
    assert_eq!(
        unsafe { paimon_vindex_reader_supports_range_search(reader.handle, &mut supported) },
        0
    );
    assert_eq!(supported, 1);
    let queries = [1.0, 0.0, 0.0, 1.0];
    let mut result = ptr::null_mut();
    assert_eq!(
        unsafe {
            paimon_vindex_reader_range_search_batch(
                reader.handle,
                queries.as_ptr(),
                queries.len(),
                2,
                unbounded_params(),
                &mut result,
            )
        },
        0
    );
    let mut expected_reader =
        VectorIndexReader::open(std::io::Cursor::new(reader.input.bytes.clone())).unwrap();
    let expected = expected_reader
        .range_search_batch(
            &queries,
            2,
            paimon_vindex_core::range::VectorRangeSearchParams::new(
                DistanceBand::from_raw(Bound::Unbounded, Bound::Unbounded, MetricType::L2).unwrap(),
                1,
            ),
        )
        .unwrap();
    drop(reader);
    let mut view = std::mem::MaybeUninit::uninit();
    assert_eq!(
        unsafe { paimon_vindex_range_search_result_view(result, view.as_mut_ptr()) },
        0
    );
    let view = unsafe { view.assume_init() };
    assert_eq!(view.query_count, 2);
    assert_eq!(view.hit_count, 8);
    assert_eq!(
        unsafe { slice::from_raw_parts(view.lims, 3) },
        expected.lims()
    );
    assert_eq!(
        unsafe { slice::from_raw_parts(view.labels, view.hit_count) },
        expected.labels()
    );
    assert_eq!(
        unsafe { slice::from_raw_parts(view.raw_distances, view.hit_count) },
        expected.raw_distances()
    );
    assert_eq!(view.list_reads, expected.call_stats().list_reads());
    for (query, actual) in unsafe { slice::from_raw_parts(view.stats, 2) }
        .iter()
        .enumerate()
    {
        let stats = expected.query(query).stats;
        assert_eq!(actual.lists_probed, stats.lists_probed());
        assert_eq!(actual.rows_scanned, stats.rows_scanned());
        assert_eq!(actual.rows_committed, stats.rows_committed());
        assert_eq!(actual.early_abandoned, stats.early_abandoned());
    }
    unsafe { paimon_vindex_range_search_result_destroy(result) };
}

#[test]
fn range_rejects_invalid_lengths_before_dereferencing_inputs() {
    let reader = RangeReader::new();
    let mut result = ptr::null_mut();
    for (data, len, count) in [
        (ptr::null(), 2, 1),
        (ptr::null(), 0, 0),
        (ptr::dangling(), usize::MAX, usize::MAX),
        (ptr::dangling(), usize::MAX - 1, usize::MAX / 2),
        (ptr::dangling(), 1, 1),
        (ptr::without_provenance(1), 2, 1),
    ] {
        assert_eq!(
            unsafe {
                paimon_vindex_reader_range_search_batch(
                    reader.handle,
                    data,
                    len,
                    count,
                    unbounded_params(),
                    &mut result,
                )
            },
            -1
        );
        assert!(result.is_null());
    }
    let query = [1.0, 0.0];
    assert_eq!(
        unsafe {
            paimon_vindex_reader_range_search_with_roaring_filter(
                reader.handle,
                query.as_ptr(),
                2,
                unbounded_params(),
                ptr::dangling(),
                usize::MAX,
                &mut result,
            )
        },
        -1
    );
    assert!(result.is_null());
    assert_eq!(
        unsafe {
            paimon_vindex_reader_range_search(
                reader.handle,
                query.as_ptr(),
                2,
                unbounded_params(),
                ptr::null_mut(),
            )
        },
        -1
    );
}

#[test]
fn range_validation_and_io_errors_do_not_transfer_results() {
    let mut reader = RangeReader::new();
    let query = [1.0, 0.0];
    let mut result = ptr::null_mut();
    let mut params = unbounded_params();
    params.nprobe = 0;
    assert_eq!(
        unsafe {
            paimon_vindex_reader_range_search(reader.handle, query.as_ptr(), 2, params, &mut result)
        },
        -1
    );
    params = unbounded_params();
    for metric in [1, 2, u32::MAX] {
        params.band.metric = metric;
        assert_eq!(
            unsafe {
                paimon_vindex_reader_range_search(
                    reader.handle,
                    query.as_ptr(),
                    2,
                    params,
                    &mut result,
                )
            },
            -1
        );
    }
    params = unbounded_params();
    for kind in [2, u32::MAX] {
        params.band.raw_lower_kind = kind;
        assert_eq!(
            unsafe {
                paimon_vindex_reader_range_search(
                    reader.handle,
                    query.as_ptr(),
                    2,
                    params,
                    &mut result,
                )
            },
            -1
        );
    }
    params = unbounded_params();
    params.band.raw_lower_kind = PAIMON_VINDEX_BOUND_FINITE;
    for value in [f32::NAN, f32::INFINITY, -1.0] {
        params.band.raw_lower = value;
        assert_eq!(
            unsafe {
                paimon_vindex_reader_range_search(
                    reader.handle,
                    query.as_ptr(),
                    2,
                    params,
                    &mut result,
                )
            },
            -1
        );
    }
    assert_eq!(
        unsafe {
            paimon_vindex_reader_range_search_with_roaring_filter(
                reader.handle,
                query.as_ptr(),
                2,
                unbounded_params(),
                ptr::null(),
                0,
                &mut result,
            )
        },
        -1
    );
    assert_eq!(
        unsafe {
            paimon_vindex_reader_range_search(
                reader.handle,
                [f32::NAN, 0.0].as_ptr(),
                2,
                unbounded_params(),
                &mut result,
            )
        },
        -1
    );
    reader.input.fail_reads = true;
    assert_eq!(
        unsafe {
            paimon_vindex_reader_range_search(
                reader.handle,
                query.as_ptr(),
                2,
                unbounded_params(),
                &mut result,
            )
        },
        -1
    );
    assert!(result.is_null());
    assert!(!paimon_vindex_last_error().is_null());
}

#[test]
fn range_endpoint_errors_leave_output_unchanged() {
    let mut output = unbounded_params().band;
    output.metric = 77;
    for endpoint in [
        PaimonVindexDistanceEndpoint {
            value: f64::NAN,
            op: PAIMON_VINDEX_CUT_GE,
        },
        PaimonVindexDistanceEndpoint {
            value: 1.0,
            op: PAIMON_VINDEX_CUT_LE,
        },
        PaimonVindexDistanceEndpoint { value: 1.0, op: 99 },
        PaimonVindexDistanceEndpoint {
            value: 1.0e30,
            op: PAIMON_VINDEX_CUT_GE,
        },
    ] {
        assert_eq!(
            unsafe {
                paimon_vindex_distance_band_from_endpoints(0, &endpoint, ptr::null(), &mut output)
            },
            -1
        );
        assert_eq!(output.metric, 77);
    }
    assert_eq!(
        unsafe {
            paimon_vindex_distance_band_from_endpoints(77, ptr::null(), ptr::null(), &mut output)
        },
        -1
    );
    assert_eq!(
        unsafe {
            paimon_vindex_distance_band_from_endpoints(0, ptr::null(), ptr::null(), ptr::null_mut())
        },
        -1
    );
    assert_eq!(
        unsafe { paimon_vindex_range_search_result_view(ptr::null(), ptr::null_mut()) },
        -1
    );
}
