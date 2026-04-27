# profi Helm Chart

[profi](https://github.com/Kaonael/profi) is an eBPF-based, zero-instrumentation
CUDA/NCCL profiler for LLM inference and training workloads on Kubernetes.

This chart installs profi as a DaemonSet on every node that advertises an
NVIDIA GPU (`nvidia.com/gpu.present=true`), exposes Prometheus metrics on
port `9401`, and optionally ships a ServiceMonitor, PrometheusRule (with
recording + alerting rules), and a Grafana dashboard as a ConfigMap.

## TL;DR

```bash
helm install profi ./deploy/profi \
  --namespace gpu-profi --create-namespace
```

## Prerequisites

| Requirement | Why |
|---|---|
| Kubernetes ≥ 1.24 | Tested on Kubernetes clusters with NVIDIA GPU Operator |
| Linux kernel ≥ 6.6 on nodes | `uprobe_multi` support for low-overhead NCCL attach |
| NVIDIA GPU Operator **or** `nvidia.com/gpu.present=true` label on GPU nodes | Node selector target |
| Prometheus Operator (kube-prometheus-stack, etc.) | Only if `serviceMonitor.enabled=true` or `prometheusRule.enabled=true` |
| Grafana with sidecar dashboard provisioner | Only if `grafanaDashboard.enabled=true` |

Nothing else is required — profi bundles the libbpf-generated eBPF skeleton,
NVML bindings, and all probe definitions into a single static binary inside
the container.

## GPU Operator coexistence

profi and [NVIDIA GPU Operator](https://github.com/NVIDIA/gpu-operator) are
**complementary, not competing**.

| Layer | DCGM-Exporter (GPU Operator) | profi |
|---|---|---|
| Source | NVML + DCGM counters | eBPF uprobes on libcudart / libcuda / libnccl |
| Visibility | **Hardware** (SM util, memory, power, ECC) | **Application** (every CUDA/NCCL API call, kernel launches, memcpy bytes) |
| Overhead | Low (polling-based) | Near measurement noise in `anonymous`; workload-dependent in targeted `full` mode |
| Pod-level enrichment | No — per-GPU | Yes — per pod/namespace/container |
| LLM-specific features | No | NCCL hang detection, straggler detection, NVTX phase tracking |

Both DaemonSets can run side-by-side on the same node. profi selects
`nvidia.com/gpu.present=true` (the label the GPU Operator sets) and tolerates
`nvidia.com/gpu:NoSchedule`, so nothing special is needed.

If you don't run the GPU Operator, either:

1. label your GPU nodes manually (`kubectl label node <n> nvidia.com/gpu.present=true`), or
2. change `nodeSelector` in `values.yaml` to whatever label you use.

## What gets installed

With default values:

- `DaemonSet/profi` — one pod per GPU node
- `Service/profi` — headless (one endpoint per pod) on `:9401`
- `ServiceAccount/profi` + `ClusterRole`/`ClusterRoleBinding` (list/watch pods for k8s label enrichment)
- `ServiceMonitor/profi` — Prometheus Operator scrape target (15s interval)
- `PrometheusRule/profi` — 11 recording rules + 10 alerts (see `deploy/runbooks/`)
- `ConfigMap/profi-dashboard` — Grafana dashboard (picked up by the
  `grafana_dashboard=1` sidecar label)

## Values reference

Short reference below; full inline documentation lives in
[`values.yaml`](values.yaml).

### Profi runtime (maps 1:1 to CLI flags)

| Value | Default | Flag |
|---|---|---|
| `profi.kernelMode` | `anonymous` | `--kernel-mode` |
| `profi.probeProfile` | `lean` | `--probe-profile` |
| `profi.enableNvtxTracing` | `false` | `--enable-nvtx-tracing` |
| `profi.disableNvml` | `false` | `--disable-nvml` |
| `profi.cudartPath` | `/usr/local/cuda/lib64/libcudart.so` | `--cudart` |
| `profi.procPath` | `/host/proc` | `--proc-path` |
| `profi.listen` | `0.0.0.0:9401` | `--listen` |
| `profi.intervals.refreshSeconds` | `10` | `--refresh-interval` |
| `profi.intervals.gcSeconds` | `60` | `--gc-interval` |
| `profi.intervals.nvmlSeconds` | `5` | `--nvml-interval` |
| `profi.reportIntervalSeconds` | `0` | `--report-interval` |
| `profi.cardinality.maxTimeSeries` | `50000` | `--max-time-series` |
| `profi.cardinality.maxStreamsPerPid` | `32` | `--max-streams-per-pid` |
| `profi.cardinality.maxKernelsPerPid` | `512` | `--max-kernels-per-pid` |
| `profi.cardinality.sampleRate` | `1` | `--sample-rate` |
| `profi.cardinality.detailedLaunches` | `false` | `--detailed-launches` |
| `profi.ebpfMaps.entriesSize` | `10240` | `--entries-size` |
| `profi.ebpfMaps.aggregatedSize` | `2048` | `--aggregated-size` |
| `profi.ebpfMaps.launchAggSize` | `8192` | `--launch-agg-size` |
| `profi.ebpfMaps.mallocSizesSize` | `131072` | `--malloc-sizes-size` |
| `profi.nccl.hangTimeoutSeconds` | `60` | `--nccl-hang-timeout` |
| `profi.logLevel` | `info` | `RUST_LOG` env |
| `profi.otlp.endpoint` | `""` | `--otlp-endpoint` (empty = disabled) |
| `profi.otlp.protocol` | `grpc` | `--otlp-protocol` (`grpc` or `http/protobuf`) |
| `profi.otlp.intervalSeconds` | `60` | `--otlp-interval-secs` |
| `profi.otlp.timeoutSeconds` | `10` | `--otlp-timeout-secs` |
| `profi.otlp.serviceName` | `profi` | `--otlp-service-name` |
| `profi.otlp.caCert` / `clientCert` / `clientKey` | `""` | TLS / mTLS certs (paths inside the pod) |
| `profi.otlp.insecure` | `false` | `--otlp-insecure` (localhost collectors only) |
| `profi.otlp.resourceAttrs` | `""` | `--otlp-resource-attrs` (extra `k=v,k=v` on the Resource) |
| `profi.extraArgs` | `[]` | escape hatch for new flags |

### Image / scheduling / RBAC

| Value | Default | Notes |
|---|---|---|
| `image.repository` | `ghcr.io/kaonael/profi-exporter` | override for internal registries |
| `image.tag` | `""` | falls back to `Chart.AppVersion` |
| `resources.requests` | `50m / 64Mi` | |
| `resources.limits` | `500m / 256Mi` | raise mem on busy nodes (>200 pods) |
| `hostPID` | `true` | **required** for /proc scan |
| `nodeSelector` | `{nvidia.com/gpu.present: "true"}` | GPU Operator label |
| `tolerations` | `nvidia.com/gpu:NoSchedule` | |
| `nvidia.enabled` | `true` | inject NVIDIA runtime env without reserving GPUs |
| `nvidia.runtimeClassName` | `nvidia` | required for NVML driver libraries on GPU Operator clusters |
| `nvidia.visibleDevices` | `all` | exposes GPU devices to NVML |
| `nvidia.driverCapabilities` | `compute,utility` | `utility` provides `libnvidia-ml.so.1` |
| `priorityClassName` | `system-node-critical` | |

### Monitoring

| Value | Default | Notes |
|---|---|---|
| `serviceMonitor.enabled` | `true` | set `false` if no Prometheus Operator |
| `serviceMonitor.interval` | `15s` | |
| `prometheusRule.enabled` | `true` | |
| `prometheusRule.recording.enabled` | `true` | 11 recording rules, 30s evaluation |
| `prometheusRule.alerts.*.enabled` | `true` | each alert individually toggleable |
| `prometheusRule.runbookBaseUrl` | `https://github.com/Kaonael/profi/blob/main/deploy/runbooks` | set to your internal URL |
| `grafanaDashboard.enabled` | `true` | ConfigMap with `grafana_dashboard=1` label |
| `grafanaDashboard.folderName` | `GPU / profi` | |

## OTLP export

profi always ships with the OpenTelemetry OTLP push exporter compiled in.
It stays dormant until `profi.otlp.endpoint` is set; Prometheus `/metrics`
is unaffected either way (both paths read the same registry atomically, so
values match by construction).

### Send to a collector alongside Prometheus

```yaml
# values.yaml
profi:
  otlp:
    endpoint: "http://otel-collector.observability.svc:4317"
    protocol: grpc
    intervalSeconds: 30
    serviceName: profi
    resourceAttrs: "deployment.environment=prod,team=ml-infra"

extraEnv:
  # Bearer token etc. — keep out of values.yaml; use a Secret.
  - name: OTEL_EXPORTER_OTLP_HEADERS
    valueFrom:
      secretKeyRef:
        name: otlp-auth
        key: headers
```

### mTLS with a corporate collector

```yaml
profi:
  otlp:
    endpoint: "https://otel-gateway.corp:4317"
    caCert: /etc/profi/tls/ca.crt
    clientCert: /etc/profi/tls/client.crt
    clientKey: /etc/profi/tls/client.key

extraVolumes:
  - name: otlp-tls
    secret:
      secretName: profi-otlp-tls
extraVolumeMounts:
  - name: otlp-tls
    mountPath: /etc/profi/tls
    readOnly: true
```

### Cardinality in managed backends

profi emits up to `profi.cardinality.maxTimeSeries` (default 50k) series.
Vendors like Datadog/New Relic bill on series count — if cost is a concern,
drop high-cardinality labels (`pid`, `comm`, `stream`) in the collector
before export. Example collector snippet:

```yaml
processors:
  attributes/drop-high-cardinality:
    actions:
      - key: process.pid
        action: delete
      - key: process.command
        action: delete
      - key: profi.cuda.stream
        action: delete

service:
  pipelines:
    metrics:
      processors: [attributes/drop-high-cardinality, batch]
```

Label name reference (Prometheus → OTel semconv) is in the root
[README.md](../../README.md#otlp-export).

## Kernel mode trade-offs

```
       performance profile       kernel labels        use when
─────────────────────────────────────────────────────────────────────────
off      lowest overhead          none                 API/NCCL fleet overview
anon     near measurement noise   kernel_class, phase  DEFAULT — production inference
full     workload-dependent       kernel name (exact)  targeted kernel debugging
```

Benchmark data and TTFT measurements live in
[`../../docs/BENCHMARK.md`](../../docs/BENCHMARK.md).

## Alerts

All 10 alerts include a `runbook_url` annotation pointing to a specific
playbook in [`deploy/runbooks/`](../runbooks/):

| Alert | Severity | Runbook |
|---|---|---|
| `ProfiNCCLHangDetected` | critical | [nccl-hang.md](../runbooks/nccl-hang.md) |
| `ProfiCUDAErrors` | critical | [event-loss.md](../runbooks/event-loss.md) |
| `ProfiExporterDown` | critical | [event-loss.md](../runbooks/event-loss.md) |
| `ProfiNCCLStraggler` | warning | [nccl-straggler.md](../runbooks/nccl-straggler.md) |
| `ProfiGPUThrottling` | warning | [nccl-straggler.md](../runbooks/nccl-straggler.md) |
| `ProfiDroppedEvents` | warning | [event-loss.md](../runbooks/event-loss.md) |
| `ProfiRingBufferDropRate` | warning | [event-loss.md](../runbooks/event-loss.md) |
| `ProfiCardinalityDrops` | warning | [cardinality-explosion.md](../runbooks/cardinality-explosion.md) |
| `ProfiNoTrackedPids` | warning | [event-loss.md](../runbooks/event-loss.md) |
| `ProfiEventLoopSlow` | info | [event-loss.md](../runbooks/event-loss.md) |

Disable any alert by setting `prometheusRule.alerts.<name>.enabled=false`.

## Troubleshooting

- **Pods stuck in `CrashLoopBackOff` with `failed to load eBPF program`**:
  kernel is unsupported (< 6.6) or `BPF`/`PERFMON` capability is not allowed by
  your Pod Security Standards / SELinux.
- **Pods running but no metrics**: check `profi_tracked_pids` — if zero, the
  node has no CUDA workloads (expected), or `hostPID`/`CAP_SYS_PTRACE` is
  missing.
- **Cardinality explosion**: see [cardinality-explosion.md](../runbooks/cardinality-explosion.md).
- **Uninstall**: `helm uninstall profi -n gpu-profi`. Maps and uprobes are
  cleaned up automatically when the pod exits.

## Upgrade

```bash
helm upgrade profi ./deploy/profi -n gpu-profi -f my-values.yaml
```

- `DaemonSet` uses `RollingUpdate` by default.
- No CRDs are installed by this chart (it relies on
  `monitoring.coreos.com/v1` CRDs from Prometheus Operator); upgrades are
  a standard rolling replace.
- Document breaking changes in release notes for the chart version being
  installed.
