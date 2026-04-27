# profi Benchmark

This document defines the performance baseline for `profi`. The numbers below
are intended for operational decisions: choosing a default mode, setting
expectations for diagnostic mode, and identifying where further profiling work
should focus.

## Summary

- Production default: `--kernel-mode=anonymous`, `--probe-profile=lean`.
- Anonymous mode is the default for continuous production monitoring.
- Full mode provides exact kernel-name visibility and is intended for targeted
  diagnostics on selected pods.
- In-kernel aggregation is the main reason anonymous mode remains practical:
  `AGGREGATED`, `LAUNCH_AGG`, and `NCCL_AGG` keep event volume away from the
  userspace ring-buffer path.
- Benchmark runs should report zero ring-buffer drops and zero cardinality-limit
  hits.
- Per-pod full-mode promotion works and should be used selectively through the
  `profi/mode: full` pod annotation.

## Tested Modes

| Mode | Intended use | Kernel names | Expected overhead profile |
|---|---|---:|---|
| `off` | Minimum-overhead CUDA/NCCL API monitoring | No | Lowest observability and lowest overhead |
| `anonymous` | Production monitoring default | No | Near measurement noise on representative inference workloads |
| `full` | Investigation of specific pods | Yes | Workload-dependent; use for targeted diagnostic windows |

## Release Baseline

| Area | Baseline |
|---|---|
| Runtime library | `libbpf-rs` |
| Build-time skeleton generation | `libbpf-cargo` |
| eBPF source | `src/bpf/profi.bpf.c` |
| Default kernel mode | `anonymous` |
| Default probe profile | `lean` |
| Diagnostic mode | `full`, preferably via per-pod annotation |

## Benchmark Results

### Qwen3.5-35B TP=2 Anonymous Stress

**Workload**

| Field | Value |
|---|---|
| Model | Qwen3.5-35B |
| Tensor parallelism | TP=2 |
| Load | 500 prompts, 512/512 tokens |
| Concurrency | 64 |
| Profiler mode | `anonymous` + `lean` |

**Result**

| Metric | With profiler | Without profiler | Overhead |
|---|---:|---:|---:|
| Request throughput | 10.41 req/s | 10.49 req/s | `-0.8%` |
| Output token throughput | 5328 tok/s | 5369 tok/s | `-0.8%` |
| Total token throughput | 10655 tok/s | 10737 tok/s | `-0.8%` |
| Mean TTFT | 406.8 ms | 383.9 ms | `+6.0%` |
| Median TTFT | 419.1 ms | 393.8 ms | `+6.4%` |
| Mean TPOT | 10.97 ms | 10.92 ms | `+0.5%` |
| Mean ITL | 10.98 ms | 10.91 ms | `+0.6%` |

**Profiler health**

| Metric | Value |
|---|---:|
| `profi_dropped_events_total` | 0 |
| Cardinality limit hits | 0 |
| NCCL hangs | 0 |
| CUDA calls handled | about 456K in 48s |
| NCCL calls handled | about 9.3K in 48s |

Interpretation: anonymous mode has negligible throughput impact and small TTFT
impact on this workload.

### Qwen3-14B TP=2 Full + Lean

**Workload**

| Field | Value |
|---|---|
| Runtime | vLLM |
| Model | Qwen/Qwen3-14B |
| Tensor parallelism | TP=2 |
| Load | 1000 prompts |
| Concurrency | 128 |
| Request rate | `inf` |
| Profiler mode | `full` + `lean` |

**Result**

| Metric | Baseline | Full + lean | Overhead |
|---|---:|---:|---:|
| Mean TTFT | 149.26 ms | 160.70 ms | `+7.7%` |
| Median TTFT | 149.95 ms | 164.51 ms | `+9.7%` |
| P99 TTFT | 332.23 ms | 327.81 ms | `-1.3%` |
| Throughput | 86.17 req/s | 85.05 req/s | `-1.3%` |
| Mean TPOT | 10.09 ms | 10.12 ms | `+0.3%` |
| P99 TPOT | 10.50 ms | 10.47 ms | approximately `0%` |

**Profiler health**

| Metric | Value |
|---|---:|
| `profi_dropped_events_total` | 0 |
| `profi_system_ring_buffer_drops_rate` | 0 |
| `profi_system_launch_agg_drops_total` | 0 |
| `profi_tracked_pids` | 2 |
| `profi_nccl_calls_total` sum | 10,274 |
| `profi_cuda_kernel_launches_total` sum | 543,124 across 573 series |

NCCL byte-volume claims require separate validation of
`profi_nccl_bytes_total` argument decoding.

Interpretation: full mode is suitable for targeted kernel-level diagnostics on
this workload shape.

### DeepSeek-V4 TP=8

**Workload**

| Field | Value |
|---|---|
| Runtime | SGLang |
| Model | DeepSeek-V4-Flash-FP |
| Tensor parallelism | TP=8 |
| Decoding | EAGLE speculative decoding |
| Load shape | Closed-loop, `rate=inf` |
| Profiler mode | `anonymous` + `lean`, `full` + `lean` |

**Result**

| Mode | Result |
|---|---:|
| `anonymous` + `lean` | within measurement noise |
| `full` + `lean` | `-9%` TTFT delta |

Interpretation: anonymous mode is appropriate for continuous production
monitoring on this kernel-launch-dense workload. Full mode remains targeted,
but exact kernel names are practical for bounded diagnostic windows.

## Operational Guidance

- Use `anonymous` mode as the default for production GPU nodes.
- Use `off` only when CUDA/NCCL API metrics are enough and latency sensitivity
  is stricter than kernel-launch observability.
- Use `full` for targeted investigation, preferably on individual pods and for
  bounded time windows.
- Promote individual pods with the annotation:

```yaml
metadata:
  annotations:
    profi/mode: full
```

- Keep `--probe-profile=lean` for production. Use broader probe coverage only
  when debugging a specific behavior.
- Treat ring-buffer and aggregation drop metrics as release gates:
  `profi_dropped_events_total`, `profi_system_ring_buffer_drops_rate`, and
  `profi_system_launch_agg_drops_total` should remain zero in benchmark runs.

## Limitations

- Negative overhead values are within normal benchmark variance and should be
  read as "no measurable regression", not as guaranteed speedup.
- `profi_nccl_bytes_total` needs separate validation before NCCL byte-volume
  metrics are used in performance claims.

## Optimization Backlog

1. Remove remaining per-event allocation/copy from detailed ring-buffer paths.
2. Reduce syscall and allocation pressure in per-CPU map draining.
3. Stream `/proc/*/maps` discovery instead of reading full files into memory.
4. Archive raw benchmark output alongside this summary for release runs.
