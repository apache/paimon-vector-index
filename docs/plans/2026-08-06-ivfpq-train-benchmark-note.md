# IVF-PQ Training Benchmark Note — 2026-08-06

Machine: Apple M3 Pro (11 cores), macOS 25.4.0, rustc 1.94.0, `--release`.
Workload: `TRAIN_N=244606 TRAIN_D=768 TRAIN_NLIST=1024 TRAIN_PQ_M=96`,
`MetricType::InnerProduct`, `use_opq=false`, synthetic normalized vectors
(`core/examples/ivfpq_train.rs`). All numbers from this machine only; the
production gate (16 CPUs) must be re-measured on the target shard.

`train_total_s` is the authoritative `IVFPQIndex::train` wall time.
`mirror_*` phases are attribution only.

| commit | change | threads | train_total_s | mirror_coarse_s | mirror_pq_s |
|---|---|---|---|---|---|
| 7772a3c | baseline | 1 | 60.57 | 2.87 | 57.79 |
| 7772a3c | baseline | 11 | 13.29 | 2.82 | 10.14 |
| 9ea4278 | parallel assignment (Phase 2) | 1 | 66.65 | 3.09 | 61.77 |
| 9ea4278 | parallel assignment (Phase 2) | 11 | 11.98 | 1.95 | 9.50 |
| faf1573 | batched splits (Phase 3) | 1 | 63.58 | 2.83 | 59.08 |
| faf1573 | batched splits (Phase 3) | 11 | 10.82 | 0.68 | 10.18 |

Baseline peak RSS: 2.29 GiB (1 thread), 2.49 GiB (11 threads).
Checksum unchanged through Phase 2 (bit-identical training); changed at
Phase 3 as expected (split schedule changes), covered by the recall gate.

Findings:

- PQ sub-quantizer training dominates end-to-end train time (~95% at 1
  thread, ~85% at 11 threads). It is already Rayon-parallel across the 96
  sub-quantizers and scales ~5.7x on 11 cores.
- Coarse hierarchical k-means improved 2.82s -> 0.68s (4.2x) from Phases
  2+3, but is only ~21% of the total, capping end-to-end gains: 13.29s ->
  10.82s (1.23x) on this machine.
- The 2x total-train success criterion is NOT reachable through coarse
  k-means alone. The next candidate is the PQ path (per-iteration argmin
  scan and kmeans++ init inside each sub-quantizer); per the plan, a native
  CPU profile should precede any further change.

Recall gate (synthetic regression check, `recall_bench`
`inner-product-hierarchical`, n=100k, d=64, nlist=1024): IVF-FLAT and
IVF-SQ recall are equivalent before/after (within run-to-run noise;
IVF-FLAT\@nprobe=8: 66.2% -> 72.4%). Pre-existing issue (not caused by this
change, identical before/after): IVF-PQ recall with InnerProduct is ~0%
and decreases as nprobe grows, which suggests an ordering-direction defect
in the IP+PQ search path; needs a separate investigation.
