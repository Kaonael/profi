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

### Qwen3.5-35B TP=2 Anonymous Stress (SGLang)

**Workload**

| Field | Value |
|---|---|
| Model | Qwen3.6-35B-A3B-FP8 |
| Runtime | SGLang |
| Tensor parallelism | TP=2 |
| Load | 5000 prompts, 512/512 tokens |
| Concurrency | 256 |
| Profiler mode | `anonymous` + `lean` |

**Result**

| Metric | With profiler | Without profiler | Overhead |
|---|---:|---:|---:|
| Request throughput | 14.43 req/s | 15.24 req/s | `-5.3%` |
| Output token throughput | 1849 tok/s | 1953 tok/s | `-5.3%` |
| Total token throughput | 9178 tok/s | 9692 tok/s | `-5.3%` |
| Median TTFT | 1640.8 ms | 1547.4 ms | `+6.0%` |
| Median TPOT | 129.5 ms | 122.8 ms | `+5.4%` |

Interpretation: anonymous mode has a measurable but acceptable (~5-6%) impact on throughput and latency for this high-concurrency SGLang workload.

### Qwen3.5-35B TP=2 Full + Lean (vLLM)

**Workload**

| Field | Value |
|---|---|
| Runtime | vLLM |
| Model | Qwen3.6-35B-A3B-FP8 |
| Tensor parallelism | TP=2 |
| Load | 2000 prompts |
| Concurrency | 256 |
| Request rate | `13` (rate-limited) |
| Profiler mode | `full` + `lean` |

**Result**

| Metric | Baseline | Full + lean | Overhead |
|---|---:|---:|---:|
| Median TTFT | 187.5 ms | 207.8 ms | `+10.8%` |
| Median TPOT | 40.2 ms | 46.7 ms | `+16.1%` |
| Throughput | 12.73 req/s | 12.71 req/s | `-0.1%` |

Interpretation: in vLLM, `full` mode overhead is more visible in per-token latency (TPOT) due to the high frequency of kernel launches being traced, while throughput remains stable under rate-limited load.

### DeepSeek-V4 TP=8 (SGLang)

**Workload**

| Field | Value |
|---|---|
| Runtime | SGLang |
| Model | DeepSeek-V4-Flash |
| Tensor parallelism | TP=8 |
| Load shape | 1000 prompts, Concurrency 256 |
| Profiler mode | `anonymous` + `lean`, `full` + `lean` |

**Result**

| Mode | Throughput Overhead | Median TTFT Delta |
|---|---:|---:|
| `anonymous` + `lean` | `-0.9%` | `+0.4%` |
| `full` + `lean` | `-3.3%` | `+4.8%` |

Interpretation: anonymous mode is highly efficient on large TP=8 configurations, with overhead remaining below 1%. Full mode adds about 3-5% overhead, making it very practical for diagnostic windows.


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
