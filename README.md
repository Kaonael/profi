# profi

eBPF-based CUDA/NCCL profiler for LLM inference on Kubernetes. Attaches libbpf-based uprobes dynamically to `libcudart.so`, `libcuda.so`, and `libnccl.so` inside running containers without code changes or restarts, aggregates hot-path events in kernel maps, and exports Prometheus metrics.

## Features

- Zero-instrumentation: attaches to running processes via `/proc` scan
- Per-pod, per-GPU, per-PID metric enrichment via Kubernetes API
- Tracks CUDA Runtime API, Driver API, and NCCL collectives
- Three kernel tracing modes with different overhead/visibility tradeoffs
- Runs as a DaemonSet on GPU nodes, exposes metrics on port `9401`

## Architecture

```
[GPU Container]                         [profi-exporter DaemonSet]
libcudart.so ── uprobe/uprobe_multi ──► eBPF aggregate maps
libcuda.so   ── uprobe/uprobe_multi ──► AGGREGATED / LAUNCH_AGG / NCCL_AGG
libnccl.so   ── uprobe/uprobe_multi ──► ring buffer for detailed/error events
                                            │
                                            ▼
                                   libbpf-rs userspace reader
                                            │
                         ┌──────────────────┴──────────────────┐
                         │                                     │
                  K8s/NVML enrichment                  Prometheus /metrics
                  pod/container/GPU                    OTLP metrics export
```

## Requirements

- Linux kernel ≥ 6.6 (`uprobe_multi` is required for low-overhead NCCL attach)
- `clang` and libbpf headers for building the embedded eBPF skeleton
- `clang-format` and `clang-tidy` for local C/eBPF checks
- `helm` for chart lint/render checks
- `prek` for local Git hooks
- Runtime capabilities: `SYS_ADMIN`, `BPF`, `PERFMON`, `SYS_PTRACE`
- Kubernetes runtime settings: `hostPID: true`
- Host mounts: `/proc`, `/sys/kernel/debug`, `/sys/fs/bpf`
- NVIDIA runtime access for NVML: `runtimeClassName: nvidia`,
  `NVIDIA_VISIBLE_DEVICES=all`, `NVIDIA_DRIVER_CAPABILITIES=compute,utility`

## Quick Start

### Kubernetes (Helm)

```bash
helm install profi deploy/profi/ \
  --namespace monitoring --create-namespace \
  --set image.repository=ghcr.io/kaonael/profi-exporter \
  --set image.tag=v0.0.1
```

See `deploy/profi/values.yaml` for all configurable knobs (kernel mode,
OTLP, mTLS, PrometheusRule alerts, Grafana dashboard).

### Local (for development)

```bash
# Build userspace binary. The BPF C program is compiled by build.rs via
# libbpf-cargo and embedded as a skeleton; clang must be in PATH.
make build

# Run (requires root or CAP_BPF/CAP_PERFMON)
sudo ./target/release/profi \
  --listen 0.0.0.0:9401
```

### Local checks

This repository uses the standard `.pre-commit-config.yaml` format and
[`prek`](https://prek.j178.dev/) as the runner for local Git hooks. Install
`prek` once, then install the repository hooks:

```bash
make prek-install
```

Run the full local check suite before opening a PR or cutting a release:

```bash
make prek-run
```

The hook suite mirrors CI: file hygiene checks, `cargo check`, `cargo fmt`,
`clang-format`, `clang-tidy`, direct `clang -target bpf` compilation,
`cargo clippy`, `cargo test`, `helm lint`, and `helm template`.

### Project Layout

```text
.
├── Cargo.toml
├── build.rs
├── src/
│   ├── main.rs
│   ├── lib.rs
│   ├── bpf.rs
│   └── bpf/
│       ├── profi.bpf.c
│       ├── profi_events.h
│       └── vmlinux.h
├── tests/
├── benches/
└── target/
```

`build.rs` uses `libbpf-cargo` to compile `src/bpf/profi.bpf.c` and generate
the Rust skeleton included from `src/bpf.rs`.

### Docker

```bash
make docker-build
make docker-push
```

Published release images are available from GitHub Container Registry:

```bash
docker pull ghcr.io/kaonael/profi-exporter:v0.0.1
```

The `Publish Image` GitHub Actions workflow pushes images on `main`/`master`,
on `v*` tags, and on manual dispatch. Release tags publish matching image tags,
for example `v0.0.1-alpha` publishes
`ghcr.io/kaonael/profi-exporter:v0.0.1-alpha`.

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--listen <ADDR>` | `0.0.0.0:9401` | Prometheus metrics listen address |
| `--proc-path <PATH>` | `/proc` | Path to procfs (use `/host/proc` in containers) |
| `--kernel-mode <MODE>` | `anonymous` | Kernel tracing mode: `off`, `anonymous`, `full` (see below) |
| `--probe-profile <PROFILE>` | `lean` | CUDA Runtime probe profile: `lean` for production, `full` for diagnostic probes |
| `--cudart <PATH>` | `/usr/local/cuda/lib64/libcudart.so` | Path to libcudart for initial attach. If not found, discovery via `/proc` is used |
| `--pid <PID>` | `0` (all) | Attach to a specific PID only |
| `--node-name <NAME>` | `$NODE_NAME` env | Kubernetes node name for pod enrichment. Set via `fieldRef: spec.nodeName` |
| `--refresh-interval <SECS>` | `10` | Interval for `/proc` library discovery scan and K8s pod list refresh |
| `--gc-interval <SECS>` | `60` | Interval for stale PID cleanup (evicts PIDs that have exited) |
| `--report-interval <SECS>` | `0` (off) | If >0, print periodic throughput report to stdout |
| `--enable-nvtx-tracing` | off | Enable NVTX range tracing. Higher overhead, use for debugging only |
| `--disable-nvml` | off | Disable NVML GPU hardware monitoring |
| `--nvml-interval <SECS>` | `5` | NVML polling interval |
| `--nccl-hang-timeout <SECS>` | `60` | Seconds before an in-flight NCCL collective is considered hung; `0` disables |
| `--max-time-series <N>` | `50000` | Global cap on distinct Prometheus label tuples |
| `--max-streams-per-pid <N>` | `32` | Per-PID stream cardinality cap before collapsing to `stream="default"` |
| `--max-kernels-per-pid <N>` | `512` | Per-PID kernel-name cap before collapsing to `kernel="other"` |
| `--sample-rate <N>` | `1` | Sample 1 in N aggregatable events; `1` disables sampling |
| `--entries-size <N>` | `10240` | `INFLIGHT` eBPF map max entries |
| `--aggregated-size <N>` | `2048` | `AGGREGATED` eBPF map max entries |
| `--launch-agg-size <N>` | `8192` | `LAUNCH_AGG` eBPF map max entries |
| `--malloc-sizes-size <N>` | `131072` | Active allocation tracking map max entries |
| `--detailed-launches` | off | Emit a ringbuf event for every kernel launch instead of relying only on `LAUNCH_AGG` |

### --kernel-mode

Controls Driver API probe attachment and kernel name resolution. Hot-path kernel launches and CUDA/NCCL durations are aggregated in eBPF maps; the ring buffer is reserved for detailed events, errors, and diagnostics.

| Mode | Driver API probes | Kernel name resolution | Performance profile | Recommended for |
|---|---|---|---|---|
| `off` | No | No | Lowest overhead | Latency-sensitive API/NCCL monitoring |
| `anonymous` | Yes | No (count only) | Near measurement noise on representative inference workloads | General production monitoring |
| `full` | Yes | Yes | Workload-dependent; optimized for targeted diagnosis | Kernel-level analysis of selected pods |

See [docs/BENCHMARK.md](docs/BENCHMARK.md) for detailed overhead measurements.

### Environment Variables

| Variable | Equivalent flag | Description |
|---|---|---|
| `NODE_NAME` | `--node-name` | Kubernetes node name, used for K8s pod enrichment |
| `RUST_LOG` | — | Log level: `error`, `warn`, `info`, `debug`, `trace` |

## OTLP export

An OpenTelemetry OTLP push exporter is always compiled in and auto-activates when an endpoint is configured. See [docs/OTEL.md](docs/OTEL.md) for full configuration details, label mapping, and examples.

## Securing `/metrics`

profi serves `/metrics` over plain HTTP by default.
For production, two independent dimensions enable TLS and client
authentication. Both are off by default.

### Flags

| Flag | Env var | Default | Description |
|---|---|---|---|
| `--metrics-tls-mode` | `PROFI_METRICS_TLS_MODE` | `off` | `off` / `server` / `mtls` |
| `--metrics-tls-cert` | `PROFI_METRICS_TLS_CERT` | — | Server cert chain (PEM) |
| `--metrics-tls-key` | `PROFI_METRICS_TLS_KEY` | — | Server private key (PEM) |
| `--metrics-tls-client-ca` | `PROFI_METRICS_TLS_CLIENT_CA` | — | CA for `mtls` mode |
| `--metrics-auth-mode` | `PROFI_METRICS_AUTH_MODE` | `off` | `off` / `bearer` / `mtls-or-bearer` |
| `--metrics-auth-audience` | `PROFI_METRICS_AUTH_AUDIENCE` | — | Required TokenReview audience (optional) |
| `--metrics-auth-cache-ttl` | `PROFI_METRICS_AUTH_CACHE_TTL` | `60` | Cache TTL for successful TokenReviews |
| `--metrics-auth-cache-size` | `PROFI_METRICS_AUTH_CACHE_SIZE` | `1024` | LRU cap |

`/health` and `/ready` always bypass L7 auth (for kubelet probes). When
TLS is enabled the probes use HTTPS.

Bearer tokens are validated via the Kubernetes `TokenReview` API. See
[docs/SECURITY.md](docs/SECURITY.md) for the full matrix and threat model.

### Example: Helm values with cert-manager + mTLS-or-Bearer

```yaml
profi:
  metricsTls:
    mode: mtls
    certManager:
      enabled: true
      issuerName: internal-ca
  metricsAuth:
    mode: mtls-or-bearer
    cacheTtlSeconds: 60
```

## Metrics

See [docs/METRICS_REFERENCE.md](docs/METRICS_REFERENCE.md) for full documentation of all exported metrics, labels, and example PromQL queries.

| Metric | Type | Description |
|---|---|---|
| `profi_cuda_calls_total` | counter | CUDA Runtime API call counts |
| `profi_cuda_duration_bucket_total` | counter histogram bucket | Aggregate CUDA API latency buckets |
| `profi_cuda_duration_sum_seconds_total` | counter | Aggregate CUDA API latency sum |
| `profi_cuda_duration_count_total` | counter | Aggregate CUDA API latency count |
| `profi_cuda_memcpy_bytes_total` | counter | Bytes transferred (h2d/d2h/d2d/h2h) |
| `profi_cuda_malloc_bytes_total` | counter | Bytes allocated via cudaMalloc |
| `profi_cuda_kernel_launches_total` | counter | Kernel launches by name or anonymous |
| `profi_cuda_kernel_duration_bucket_total` | counter histogram bucket | Aggregate per-kernel latency buckets (`full` mode only) |
| `profi_nccl_calls_total` | counter | NCCL collective call counts |
| `profi_nccl_bytes_total` | counter | NCCL inter-GPU bytes transferred |
| `profi_nccl_duration_bucket_total` | counter histogram bucket | Aggregate NCCL latency buckets |
| `profi_gpu_info` | gauge | GPU device inventory |
| `profi_tracked_pids` | gauge | Number of observed CUDA processes |
| `profi_dropped_events_total` | counter | eBPF ring buffer overflow events |

## Build

```bash
# Userspace binary. The BPF C skeleton is generated by build.rs via libbpf-cargo.
make build

# Rust + C formatting and lint checks
make lint

# C/eBPF-only checks
make c-fmt-check
make c-tidy
make bpf-compile

# Optional kernel verifier load check. Requires privileges and a writable BPF FS.
make bpf-verify

# Tests
make test

# Container image
make docker-build IMG=ghcr.io/kaonael/profi-exporter:v0.0.1-alpha
make docker-push  IMG=ghcr.io/kaonael/profi-exporter:v0.0.1-alpha
```

## Performance

Representative libbpf/libbpf-rs benchmark results:

| Workload | Mode | Throughput impact | Latency impact | Notes |
|---|---|---|---|---|
| Qwen3.5-35B TP=2 stress | `anonymous` + `lean` | -0.8% | +6.0% mean TTFT | Stress run |
| Qwen3-14B TP=2 | `full` + `lean` | -1.3% | +7.7% mean TTFT | Full-mode diagnostic run |
| DeepSeek-V4 TP=8 | `anonymous` + `lean` | within noise | +1.4% median TTFT | Kernel-launch-dense workload |
| DeepSeek-V4 TP=8 | `full` + `lean` | no observed regression | -9% TTFT delta | Hot-path validation |

See [docs/BENCHMARK.md](docs/BENCHMARK.md) for methodology and caveats.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
