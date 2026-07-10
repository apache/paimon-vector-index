# Implementation Plan: IVF-RQ Search Performance

## Tasks

### Task 1: Build the byte LUT incrementally
- **Files**: `core/src/rq.rs`
- **Changes**: Replace the repeated eight-dimension accumulation with a
  subset-pattern recurrence, retaining exact `f32` semantics.
- **Verify**: `cargo test -p paimon-vindex-core rq:: --lib`

### Task 2: Make result and centroid selection proportional to the requested size
- **Files**: `core/src/topk.rs`, `core/src/kmeans.rs`
- **Changes**: Implement an indexed max heap and select the coarse top prefix
  before sorting it.
- **Verify**: `cargo test -p paimon-vindex-core topk:: kmeans:: --lib`

### Task 3: Batch RQ inverted-list reads
- **Files**: `core/src/ivfrq_io.rs`
- **Changes**: Add a memory-bounded multi-list reader and use it in batch
  search; add a counting-reader regression test for one multi-range read.
- **Verify**: `cargo test -p paimon-vindex-core ivfrq_io:: --lib`

### Task 4: Verify behavior and benchmark the hot paths
- **Files**: `core/src/rq.rs`, `core/src/topk.rs`, `core/src/kmeans.rs`,
  `core/src/ivfrq_io.rs`
- **Changes**: Add result-equivalence tests and run the full core test suite.
- **Verify**: `cargo test -p paimon-vindex-core --lib` and
  `cargo bench -p paimon-vindex-core --bench ann_bench -- --nocapture`
