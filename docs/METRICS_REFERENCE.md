# profi-exporter Metrics Reference

## Overview

profi-exporter collects metrics via eBPF uprobes attached dynamically to `libcudart.so`, `libcuda.so`, and `libnccl.so` inside running containers. Metrics are exported in Prometheus format on port `9401/metrics`.

Metric availability depends on the `--kernel-mode` flag:

| Mode | Description |
|---|---|
| `off` | Runtime API + NCCL only. No Driver API probes. |
| `anonymous` | Runtime API + NCCL + kernel launch counting without name resolution. **Default.** |
| `full` | All probes enabled, including kernel name resolution via `cuModuleGetFunction`. |

---

## HTTP Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/metrics` | GET | Prometheus text format metrics |
| `/health` | GET | Liveness probe — `200 ok` if event loop heartbeat < 30s, else `503` |
| `/ready` | GET | Readiness probe — `200 ready` if libs attached + ring buffer open + K8s ready |
| `/status` | GET | JSON: `attached_libraries`, `tracked_pids`, `events_processed`, `uptime_seconds`, `kernel_mode` |

---

## Export paths

profi always ships two export paths that read the same Prometheus registry atomically — values agree by construction.

| Path | Activation | Transport | Interval |
|---|---|---|---|
| Prometheus `/metrics` | always on | HTTP pull on `:9401` | scrape-driven |
| OpenTelemetry OTLP | auto-enabled when `--otlp-endpoint` (or `OTEL_EXPORTER_OTLP_ENDPOINT`) is set | gRPC (4317) or HTTP/protobuf (4318) push | `--otlp-interval-secs` (default `60`) |

Metric **names** (`profi_*`) stay unchanged on both paths. Label **keys** are remapped to OpenTelemetry semantic conventions on the OTLP path only — the Prometheus tables below use the original label names.

| Prometheus label | OTLP attribute |
|---|---|
| `pod` | `k8s.pod.name` |
| `namespace` | `k8s.namespace.name` |
| `container` | `k8s.container.name` |
| `gpu` | `gpu.id` |
| `gpu_uuid` | `gpu.uuid` |
| `gpu_model` | `gpu.model` |
| `pid` | `process.pid` |
| `comm` | `process.command` |

profi-specific labels (`operation`, `kernel`, `kernel_class`, `phase`, `direction`, `stream`, `error_code`, `clock_type`, `type`, `state`, `error_type`, `location`, `reason`, `library`) are prefixed with `profi.*` on OTLP (`profi.operation`, `profi.kernel`, …).

All self-observability metrics live under the `profi_system_*` prefix (`profi_system_uptime_seconds`, `profi_system_event_loop_*`, `profi_system_http_*`, etc.) and are filtered out of OTLP by a single `starts_with("profi_system_")` check in the OTLP bridge — they stay on `/metrics` for local debugging only. Any future self-obs metric added under that prefix is excluded automatically.

See [root README](../README.md#otlp-export) for CLI flags, env vars, TLS/mTLS, and a collector config example.

---

## CUDA Runtime API

### `profi_cuda_calls_total`

**Type:** counter
**Description:** Total number of CUDA Runtime API calls.

**Purpose:** Primary metric for GPU process activity. Use it to understand which operations dominate the workload (compute vs memory), compare activity across pods and GPU workers, and detect anomalous behavior (sudden spike or drop in call rate).

**Labels:**

| Label | Description | Example |
|---|---|---|
| `operation` | CUDA/NCCL function name | `cudaLaunchKernel`, `cudaMemcpyAsync`, `cudaStreamSync`, `cudaMemsetAsync`, `cudaGraphLaunch`, `cuModuleLoadData`, `ncclAllGather` |
| `pid` | Host-level process PID | `89444` |
| `comm` | Thread name from `/proc/comm` (truncated 16 chars) | `sglang::schedul`, `""` |
| `namespace` | Kubernetes namespace | `default` |
| `pod` | Pod name | `inference-worker-0` |
| `container` | Container name | `caas-workload` |
| `gpu` | GPU index (0-based) | `4`, `7` |
| `gpu_uuid` | GPU UUID | `GPU-c3745ad3-...` |
| `stream` | CUDA stream | `default`, `0x322849c0` |

**Example queries:**
```promql
# Kernel launch throughput per pod
rate(profi_cuda_calls_total{operation="cudaLaunchKernel"}[1m])

# Memory vs compute operation ratio
sum by (pod) (rate(profi_cuda_calls_total{operation=~"cudaMemcpy.*"}[1m]))
/
sum by (pod) (rate(profi_cuda_calls_total{operation="cudaLaunchKernel"}[1m]))
```

---

### `profi_cuda_duration_bucket_total`

**Type:** counter histogram bucket
**Description:** Cumulative latency buckets for CUDA Runtime API calls emitted from aggregate eBPF maps.
**Buckets:** 1us, 5us, 10us, 50us, 100us, 500us, 1ms, 5ms, 10ms, 50ms, 100ms, 500ms, 1s

**Purpose:** Detects slowdowns at the individual CUDA operation level. High `cudaStreamSync` latency indicates a GPU stall. High `cudaMemcpyAsync` latency indicates a memory bottleneck.

**Labels:** `operation`, `namespace`, `pod`, `gpu`

`profi_cuda_duration_sum_seconds_total` and
`profi_cuda_duration_count_total` expose the matching cumulative sum and count.
Build dashboards and alerts from `*_bucket_total`, `*_sum_seconds_total`, and
`*_count_total`; these are the low-overhead aggregate path and are populated by
default. `profi_cuda_duration_seconds` is reserved for detailed ring-buffer
events and is not the default source of truth.

**Example queries:**
```promql
# P99 latency of cudaLaunchKernel
histogram_quantile(0.99,
  rate(profi_cuda_duration_bucket_total{operation="cudaLaunchKernel"}[5m])
)

# Mean stream sync latency. Requires --probe-profile=full because sync probes
# are diagnostic-only in the lean production profile.
rate(profi_cuda_duration_sum_seconds_total{operation="cudaStreamSync"}[1m])
/ rate(profi_cuda_duration_count_total{operation="cudaStreamSync"}[1m])
```

---

### `profi_cuda_memcpy_bytes_total`

**Type:** counter
**Description:** Bytes transferred via cudaMemcpy/cudaMemcpyAsync/cudaMemcpyPeer/cudaMemset/cudaMemsetAsync.

**Purpose:** Reveals data transfer patterns: PCIe usage (h2d/d2h) vs intra-GPU movement (d2d). In healthy LLM inference, d2d dominates. High h2d/d2h indicates PCIe bottleneck.

**Labels:** `direction`, `pid`, `comm`, `namespace`, `pod`, `container`, `gpu`, `gpu_uuid`, `stream`

| `direction` | Description |
|---|---|
| `h2d` | Host -> Device (CPU RAM -> GPU VRAM) |
| `d2h` | Device -> Host (GPU VRAM -> CPU RAM) |
| `d2d` | Device -> Device (intra-GPU) |
| `h2h` | Host -> Host (via CUDA) |
| `p2p` | Peer-to-peer (cudaMemcpyPeer, packed src/dst device) |
| `unknown` | CUDA memcpy kind outside the known enum range |

**Example queries:**
```promql
# PCIe h2d bandwidth (MB/s)
rate(profi_cuda_memcpy_bytes_total{direction="h2d"}[1m]) / 1e6

# d2d traffic share (healthy inference = >95%)
sum by (pod) (rate(profi_cuda_memcpy_bytes_total{direction="d2d"}[1m]))
/ sum by (pod) (rate(profi_cuda_memcpy_bytes_total[1m]))
```

---

### `profi_cuda_malloc_bytes_total`

**Type:** counter
**Description:** Bytes allocated via cudaMalloc/cudaMallocHost/cudaMallocManaged.

**Purpose:** Monitors dynamic GPU memory allocation. In production LLM inference, cudaMalloc is called mostly at startup. Significant growth during steady-state signals unexpected dynamic allocation or a memory leak.

**Labels:** `pid`, `comm`, `namespace`, `pod`, `container`, `gpu`, `gpu_uuid`

**Example queries:**
```promql
# Dynamic GPU memory allocation growth over time
increase(profi_cuda_malloc_bytes_total[10m])
```

---

### `profi_cuda_active_memory_bytes`

**Type:** gauge
**Description:** Net GPU memory allocated (cudaMalloc minus cudaFree) since profiler start. May be negative if allocations preceded profiler attach.

**Purpose:** Real-time view of GPU memory utilization per process. Useful for detecting memory leaks (steadily growing) or fragmentation (high allocated but OOM errors).

**Labels:** `pid`, `comm`, `namespace`, `pod`, `container`, `gpu`, `gpu_uuid`

**Example queries:**
```promql
# GPU memory usage per pod
sum by (pod) (profi_cuda_active_memory_bytes) / 1e9
```

---

### `profi_cuda_errors_total`

**Type:** counter
**Description:** CUDA/NCCL API calls returning non-zero error codes.

**Purpose:** Detects GPU errors at the API level. Any non-zero value indicates a problem (OOM, invalid arguments, driver issues). Critical for SLA monitoring.

**Labels:** `operation`, `pid`, `comm`, `namespace`, `pod`, `container`, `gpu`, `gpu_uuid`, `error_code`

**Example queries:**
```promql
# Alert on any CUDA errors
increase(profi_cuda_errors_total[5m]) > 0
```

---

## CUDA Kernel Tracing

> Available only with `--kernel-mode=anonymous` and `--kernel-mode=full`.

### `profi_cuda_kernel_launches_total`

**Type:** counter
**Description:** Number of CUDA kernel launches, broken down by kernel name (full mode) or aggregated (anonymous mode).

**Purpose:** In `full` mode, reveals which CUDA kernels dominate the workload (attention, matmul, softmax). In `anonymous` mode, provides total launch rate without name resolution overhead.

**Labels:** `kernel`, `pid`, `comm`, `namespace`, `pod`, `container`, `gpu`, `gpu_uuid`, `kernel_class`, `phase`

| Label | Description |
|---|---|
| `kernel` | Kernel function name (normalized) or `anonymous` |
| `kernel_class` | `attention`, `gemm`, `collective`, `activation`, `memory`, `sampling`, `other` |
| `phase` | NVTX inference phase: `prefill`, `decode`, `attention`, `mlp`, `norm`, `other`, `""` |

**Example queries:**
```promql
# Top-10 kernels by launch rate
topk(10, sum by (kernel) (rate(profi_cuda_kernel_launches_total[1m])))

# Prefill vs decode kernel launch ratio
sum(rate(profi_cuda_kernel_launches_total{phase="prefill"}[1m]))
/ sum(rate(profi_cuda_kernel_launches_total{phase="decode"}[1m]))
```

---

### `profi_cuda_kernel_duration_bucket_total`

**Type:** counter histogram bucket
**Description:** Cumulative latency buckets for aggregate CUDA kernel launches by name.
**Buckets:** 1us, 5us, 10us, 50us, 100us, 500us, 1ms, 10ms

> Only available with `--kernel-mode=full`.

**Labels:** `kernel`, `namespace`, `pod`, `gpu`, `kernel_class`, `phase`

`profi_cuda_kernel_duration_sum_seconds_total` and
`profi_cuda_kernel_duration_count_total` expose the matching cumulative sum and
count.

**Example queries:**
```promql
# P99 latency of attention kernels
histogram_quantile(0.99,
  rate(profi_cuda_kernel_duration_bucket_total{kernel_class="attention"}[5m])
)
```

---

## NCCL (Multi-GPU Communication)

### `profi_nccl_calls_total`

**Type:** counter
**Description:** Total number of NCCL collective operation calls.

**Purpose:** Monitors inter-GPU communication for Tensor Parallel and Pipeline Parallel inference.

**Labels:** `operation`, `pid`, `comm`, `namespace`, `pod`, `container`, `gpu`, `gpu_uuid`

| `operation` | Description |
|---|---|
| `ncclAllReduce` | Sum tensors across all GPUs (TP) |
| `ncclAllGather` | Gather distributed tensor (TP) |
| `ncclReduceScatter` | Reduce and scatter across GPUs (TP) |
| `ncclBroadcast` | Broadcast from one GPU |
| `ncclSend` / `ncclRecv` | Point-to-point transfer (PP) |

---

### `profi_nccl_bytes_total`

**Type:** counter
**Description:** Bytes transferred via NCCL collective operations.

**Purpose:** Measures actual inter-GPU bandwidth consumption. Reveals whether NCCL is a bottleneck.

**Labels:** `operation`, `pid`, `comm`, `namespace`, `pod`, `container`, `gpu`, `gpu_uuid`

**Example queries:**
```promql
# Total inter-GPU bandwidth (GB/s)
sum by (pod) (rate(profi_nccl_bytes_total[1m])) / 1e9
```

---

### `profi_nccl_duration_bucket_total`

**Type:** counter histogram bucket
**Description:** Cumulative latency buckets for aggregate NCCL collective operations.
**Buckets:** 10us, 50us, 100us, 500us, 1ms, 5ms, 10ms, 50ms, 100ms, 500ms, 1s

**Labels:** `operation`, `namespace`, `pod`, `gpu`

`profi_nccl_duration_sum_seconds_total` and
`profi_nccl_duration_count_total` expose the matching cumulative sum and count.

**Example queries:**
```promql
# P99 AllGather latency
histogram_quantile(0.99,
  rate(profi_nccl_duration_bucket_total{operation="ncclAllGather"}[5m])
)
```

---

### `profi_nccl_hang_detected_total`

**Type:** counter
**Description:** NCCL collective operations that exceeded `--nccl-hang-timeout`.

**Labels:** `operation`, `pid`

---

### `profi_nccl_stale_entries`

**Type:** gauge
**Description:** Number of in-flight NCCL entries currently exceeding the hang timeout.

---

### `profi_nccl_straggler_ratio`

**Type:** gauge
**Description:** Ratio of this GPU's NCCL latency to the group median. Values above `1.5` indicate a likely straggler.

**Labels:** `pid`, `gpu`, `gpu_uuid`, `operation`

---

## Infrastructure

### `profi_gpu_info`

**Type:** gauge (always = 1)
**Description:** GPU device inventory on the node.

**Labels:** `gpu`, `gpu_uuid`, `gpu_model`

**Example queries:**
```promql
# Total GPU count across all nodes
count(profi_gpu_info)

# Join to add GPU model to other metrics
profi_cuda_calls_total * on(gpu) group_left(gpu_model) profi_gpu_info
```

---

### `profi_gpu_temperature_celsius`

**Type:** gauge
**Description:** GPU die temperature from NVML.

**Labels:** `gpu`, `gpu_uuid`

---

### `profi_gpu_power_watts`

**Type:** gauge
**Description:** GPU power draw from NVML.

**Labels:** `gpu`, `gpu_uuid`

---

### `profi_gpu_clock_mhz`

**Type:** gauge
**Description:** GPU clock speed from NVML.

**Labels:** `gpu`, `gpu_uuid`, `clock_type` (`sm`, `mem`)

---

### `profi_gpu_utilization_ratio`

**Type:** gauge
**Description:** GPU or memory utilization from NVML, normalized to `0..1`.

**Labels:** `gpu`, `gpu_uuid`, `type` (`gpu`, `memory`)

---

### `profi_gpu_memory_bytes`

**Type:** gauge
**Description:** GPU VRAM usage from NVML.

**Labels:** `gpu`, `gpu_uuid`, `state` (`used`, `free`, `total`)

---

### `profi_gpu_ecc_errors_total`

**Type:** counter
**Description:** GPU ECC memory error counters from NVML.

**Labels:** `gpu`, `gpu_uuid`, `error_type`, `location`

---

### `profi_gpu_throttle_active`

**Type:** gauge
**Description:** GPU throttle reasons currently active (`1` = active, `0` = inactive).

**Labels:** `gpu`, `gpu_uuid`, `reason`

---

### `profi_tracked_pids`

**Type:** gauge
**Description:** Number of unique CUDA processes currently observed by the profiler.

---

### `profi_dropped_events_total`

**Type:** counter
**Description:** Events dropped due to eBPF RingBuf overflow.

**Purpose:** Data integrity health metric. Any value above 0 means events are lost and counters become underestimates. Must be zero in steady state.

---

## Self-Observability Metrics

### `profi_system_discovery_scan_duration_seconds`

**Type:** histogram
**Description:** Time spent scanning `/proc` for CUDA libraries.
**Buckets:** 1ms, 5ms, 10ms, 50ms, 100ms, 500ms, 1s, 5s

---

### `profi_system_discovery_attached_libraries`

**Type:** gauge
**Description:** Number of attached library instances by library type.

**Labels:** `library` (`libcudart.so`, `libnccl.so`, `libcuda.so`, `libnvtx3interop.so`)

---

### `profi_system_event_loop_process_duration_seconds`

**Type:** histogram
**Description:** Time spent processing events in one ring buffer batch.
**Buckets:** 1us, 5us, 10us, 50us, 100us, 500us, 1ms

---

### `profi_system_aggregated_map_drain_duration_seconds`

**Type:** histogram
**Description:** Time spent draining the eBPF aggregated PerCpuHashMap.
**Buckets:** 10us, 50us, 100us, 500us, 1ms, 5ms, 10ms

---

### `profi_system_metric_handle_cache_size`

**Type:** gauge
**Description:** Number of entries in the metric handle cache.

---

### `profi_system_prometheus_encode_duration_seconds`

**Type:** histogram
**Description:** Time spent encoding Prometheus metrics text format on each scrape.
**Buckets:** 100us, 500us, 1ms, 5ms, 10ms, 50ms, 100ms

---

### `profi_system_uptime_seconds`

**Type:** gauge
**Description:** Seconds since profi process started.

---

### `profi_system_ring_buffer_drops_rate`

**Type:** gauge
**Description:** Ring buffer event drops per second over the last drain interval. Non-zero triggers adaptive drain frequency increase.

---

### `profi_cardinality_limit_drops_total`

**Type:** counter
**Description:** Events dropped because cardinality limit (`--max-time-series`) was exceeded. Indicates too many unique label combinations.

---

### `profi_system_kernel_name_resolve_failures_total`

**Type:** counter
**Description:** Lazy kernel-name reads from `/proc/<pid>/mem` that failed.

**Labels:** `reason`

---

### `profi_system_launch_agg_drops_total`

**Type:** counter
**Description:** `cuLaunchKernel` events dropped because the `LAUNCH_AGG` eBPF map was full.

---

### `profi_system_http_auth_success_total`

**Type:** counter
**Description:** Successful `/metrics` authentications by method.

**Labels:** `method`

---

### `profi_system_http_auth_failures_total`

**Type:** counter
**Description:** Rejected `/metrics` requests by reason.

**Labels:** `reason`

---

### `profi_system_http_tokenreview_cache_total`

**Type:** counter
**Description:** Kubernetes TokenReview cache outcomes.

**Labels:** `result`

---

### `profi_system_http_tokenreview_latency_seconds`

**Type:** histogram
**Description:** Latency of TokenReview calls to the Kubernetes API.
**Buckets:** 1ms, 5ms, 10ms, 50ms, 100ms, 500ms, 1s, 5s

---

### `profi_system_http_tls_handshakes_total`

**Type:** counter
**Description:** TLS handshakes on the `/metrics` endpoint.

**Labels:** `result`

---

## Label Reference

| Label | Type | Description | Source |
|---|---|---|---|
| `operation` | string | CUDA/NCCL function name | uprobe function name |
| `pid` | int | Host process PID | uprobe context |
| `comm` | string | Thread name (16 chars max) | `/proc/<pid>/comm` |
| `namespace` | string | Kubernetes namespace | K8s pod enrichment |
| `pod` | string | Pod name | K8s pod enrichment |
| `container` | string | Container name | K8s pod enrichment |
| `gpu` | int | GPU index (0-based) | NVIDIA device minor / procfs enrichment |
| `gpu_uuid` | string | GPU UUID | `/proc/driver/nvidia/gpus` |
| `gpu_model` | string | GPU model name | `/proc/driver/nvidia/gpus` |
| `stream` | string | CUDA stream (`default` or hex address) | CUDA context |
| `direction` | string | Memcpy direction: `h2d`, `d2h`, `d2d`, `h2h`, `p2p`, `unknown` | cudaMemcpyKind |
| `kernel` | string | CUDA kernel name (normalized) or `anonymous` | cuModuleGetFunction |
| `kernel_class` | string | `attention`, `gemm`, `collective`, `activation`, `memory`, `sampling`, `other` | regex classification |
| `phase` | string | Inference phase from NVTX: `prefill`, `decode`, `attention`, `mlp`, `norm`, `other`, `""` | NVTX range name classification |
| `error_code` | int | CUDA error code (non-zero) | uprobe return value |
| `library` | string | Library type | discovery scan |
| `clock_type` | string | GPU clock type: `sm`, `mem` | NVML |
| `type` | string | Utilization type: `gpu`, `memory` | NVML |
| `state` | string | Memory state: `used`, `free`, `total` | NVML |
| `error_type` | string | ECC error counter type | NVML |
| `location` | string | ECC error memory location | NVML |
| `reason` | string | Throttle, auth failure, or resolver failure reason | NVML / HTTP auth / kernel resolver |

---

## Metric Availability by --kernel-mode

| Metric | `off` | `anonymous` | `full` |
|---|:---:|:---:|:---:|
| `profi_cuda_calls_total` | yes | yes | yes |
| `profi_cuda_duration_bucket_total` | yes | yes | yes |
| `profi_cuda_memcpy_bytes_total` | yes | yes | yes |
| `profi_cuda_malloc_bytes_total` | yes | yes | yes |
| `profi_cuda_active_memory_bytes` | yes | yes | yes |
| `profi_cuda_errors_total` | yes | yes | yes |
| `profi_nccl_calls_total` | yes | yes | yes |
| `profi_nccl_bytes_total` | yes | yes | yes |
| `profi_nccl_duration_bucket_total` | yes | yes | yes |
| `profi_cuda_kernel_launches_total` | - | yes (anonymous) | yes (named) |
| `profi_cuda_kernel_duration_bucket_total` | - | - | yes |
| `profi_gpu_info` | yes | yes | yes |
| `profi_gpu_temperature_celsius` | yes | yes | yes |
| `profi_gpu_power_watts` | yes | yes | yes |
| `profi_gpu_clock_mhz` | yes | yes | yes |
| `profi_gpu_utilization_ratio` | yes | yes | yes |
| `profi_gpu_memory_bytes` | yes | yes | yes |
| `profi_gpu_ecc_errors_total` | yes | yes | yes |
| `profi_gpu_throttle_active` | yes | yes | yes |
| `profi_tracked_pids` | yes | yes | yes |
| `profi_dropped_events_total` | yes | yes | yes |
| `profi_nccl_hang_detected_total` | yes | yes | yes |
| `profi_nccl_stale_entries` | yes | yes | yes |
| `profi_nccl_straggler_ratio` | yes | yes | yes |
| `profi_system_ring_buffer_drops_rate` | yes | yes | yes |
| `profi_cardinality_limit_drops_total` | yes | yes | yes |
| `profi_system_discovery_scan_duration_seconds` | yes | yes | yes |
| `profi_system_discovery_attached_libraries` | yes | yes | yes |
| `profi_system_event_loop_process_duration_seconds` | yes | yes | yes |
| `profi_system_aggregated_map_drain_duration_seconds` | yes | yes | yes |
| `profi_system_metric_handle_cache_size` | yes | yes | yes |
| `profi_system_prometheus_encode_duration_seconds` | yes | yes | yes |
| `profi_system_uptime_seconds` | yes | yes | yes |
| `profi_system_kernel_name_resolve_failures_total` | - | - | yes |
| `profi_system_launch_agg_drops_total` | - | yes | yes |
| `profi_system_http_auth_success_total` | yes | yes | yes |
| `profi_system_http_auth_failures_total` | yes | yes | yes |
| `profi_system_http_tokenreview_cache_total` | yes | yes | yes |
| `profi_system_http_tokenreview_latency_seconds` | yes | yes | yes |
| `profi_system_http_tls_handshakes_total` | yes | yes | yes |

---

## CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--pid` | `0` (all) | Target PID (0 = all processes) |
| `--cudart` | `/usr/local/cuda/lib64/libcudart.so` | Path to libcudart.so |
| `--listen` | `0.0.0.0:9401` | Prometheus listen address |
| `--report-interval` | `0` (disabled) | Terminal report interval (seconds) |
| `--proc-path` | `/proc` | Proc filesystem mount point |
| `--node-name` | env `NODE_NAME` | K8s node name (enables pod enrichment) |
| `--refresh-interval` | `10` | Library discovery interval (seconds) |
| `--gc-interval` | `60` | Stale PID cleanup interval (seconds) |
| `--kernel-mode` | `anonymous` | `full`, `anonymous`, or `off` |
| `--probe-profile` | `lean` | `lean` production probe set or `full` diagnostic CUDA Runtime probes |
| `--enable-nvtx-tracing` | `false` | Enable NVTX range phase tracking |
| `--max-time-series` | `50000` | Cardinality limit: max cached metric handles |
| `--max-streams-per-pid` | `32` | Streams per PID before collapse to `default` |
| `--max-kernels-per-pid` | `512` | Kernel names per PID before collapse |
| `--entries-size` | `10240` | eBPF ENTRIES map max entries |
| `--aggregated-size` | `2048` | eBPF AGGREGATED map max entries |
| `--launch-agg-size` | `8192` | eBPF LAUNCH_AGG map max entries |
| `--malloc-sizes-size` | `131072` | eBPF MALLOC_SIZES map max entries |
| `--sample-rate` | `1` (off) | Sampling rate N for aggregatable events (1/N) |
| `--detailed-launches` | `false` | Emit per-launch ringbuf events for exact launch histograms |

---

## Cardinality (single pod, TP=2)

| Mode | Total metric lines | Unique series | Primary driver |
|---|---|---|---|
| `off` | ~288 | ~55 | operations x pids x directions |
| `anonymous` | ~288 | ~55 | same (kernel launches aggregated) |
| `full` | ~1838 | ~325 | unique kernel names (~138 series) |

Series grow linearly with the number of pods: approximately **160 series/pod** in `full` mode, **28 series/pod** in `off`/`anonymous` mode.

Cardinality is bounded by `--max-time-series` (default 50,000), `--max-streams-per-pid` (default 32), and `--max-kernels-per-pid` (default 512). Excess series are collapsed; drops are tracked in `profi_cardinality_limit_drops_total`.

---

## Typical Use Cases

### Production monitoring (`--kernel-mode=off` or `anonymous`)

```promql
# GPU compute activity proxy (kernel launch rate)
sum by (pod, gpu) (rate(profi_cuda_calls_total{operation="cudaLaunchKernel"}[1m]))

# Inter-GPU bandwidth (GB/s)
sum by (pod) (rate(profi_nccl_bytes_total[1m])) / 1e9

# Host-to-device PCIe bandwidth (MB/s)
rate(profi_cuda_memcpy_bytes_total{direction="h2d"}[1m]) / 1e6

# Active GPU memory per pod
sum by (pod) (profi_cuda_active_memory_bytes) / 1e9

# Profiler health
profi_dropped_events_total > 0
profi_tracked_pids
profi_system_ring_buffer_drops_rate
```

### Performance debugging (`--kernel-mode=full`)

```promql
# Slowest kernels by mean duration
topk(10,
  rate(profi_cuda_kernel_duration_sum_seconds_total[5m])
  / rate(profi_cuda_kernel_duration_count_total[5m])
)

# NCCL latency spike detection
histogram_quantile(0.99, rate(profi_nccl_duration_bucket_total[1m]))
> 2 * histogram_quantile(0.99, rate(profi_nccl_duration_bucket_total[10m]))

# Prefill vs decode kernel time split
histogram_quantile(0.95,
  sum by (phase, le) (rate(profi_cuda_kernel_duration_bucket_total[5m]))
)
```

### Alerting

```promql
# Data loss — all counters are now underestimates
increase(profi_dropped_events_total[5m]) > 0

# Cardinality explosion
increase(profi_cardinality_limit_drops_total[5m]) > 0

# Profiler lost process tracking on a GPU node
profi_tracked_pids == 0

# Abnormal NCCL latency (potential interconnect degradation)
histogram_quantile(0.99,
  rate(profi_nccl_duration_bucket_total{operation="ncclAllGather"}[2m])
) > 0.1

# CUDA errors detected
increase(profi_cuda_errors_total[5m]) > 0

# NVML reported active GPU throttling
profi_gpu_throttle_active == 1

# NCCL hang detector fired
increase(profi_nccl_hang_detected_total[5m]) > 0
```
