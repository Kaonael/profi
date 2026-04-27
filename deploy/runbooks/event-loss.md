# Runbook: profi Event Loss & Exporter Health

Covers the following alerts, which all share remediation mechanics:

- `ProfiDroppedEvents` — ring buffer overflow (eBPF → userspace)
- `ProfiRingBufferDropRate` — sustained drop rate > threshold
- `ProfiNoTrackedPids` — discovery sees no CUDA processes
- `ProfiExporterDown` — Prometheus cannot scrape the pod
- `ProfiEventLoopSlow` — profi's own processing loop is slow
- `ProfiCUDAErrors` — CUDA API returning non-zero error codes

## Symptom

One of the alerts above fired. The common thread: either **profi is losing
signal** (first five) or **applications are erroring** (last one).

## Impact

- **Undercounted metrics**: rates, histograms, and memory counters are all
  lower than reality. Dashboards look fine but are lying.
- **Missed critical alerts**: hang detection, straggler detection, cardinality
  limits all depend on profi seeing every event.
- **`ProfiCUDAErrors`** is different — it means the *application* is broken
  (OOM, illegal memory access, driver reset); profi itself is healthy.

## Diagnosis by alert

### `ProfiDroppedEvents` / `ProfiRingBufferDropRate`

The eBPF RingBuf (per-CPU) is too small for the event rate.

```promql
rate(profi_dropped_events_total[1m])
profi_system_ring_buffer_drops_rate
```

Check the event rate we're trying to sustain:

```promql
sum by (pod) (rate(profi_cuda_calls_total[1m]))
```

If a single pod pushes millions of `cudaLaunchKernel` calls/sec, the default
`--entries-size=10240` will overflow.

### `ProfiNoTrackedPids`

```promql
profi_tracked_pids
```

If this is zero while GPU workloads exist on the node, **discovery is broken**.
Usual causes:

- `hostPID: true` missing from the DaemonSet (pod can't see host PIDs)
- `/host/proc` volume not mounted or wrong path
- `CAP_SYS_PTRACE` missing from securityContext (can't read other pids' maps)
- The node genuinely has no CUDA workloads (expected — not a bug)

Verify:

```bash
POD=$(kubectl -n <ns> get pod -l app.kubernetes.io/name=profi -o name --field-selector spec.nodeName=<node> | head -1)
kubectl -n <ns> exec "$POD" -- ls /host/proc | head
kubectl -n <ns> port-forward "$POD" 9401:9401
curl -s http://127.0.0.1:9401/status
```

The `/status` endpoint returns JSON with `attached_libraries`, `tracked_pids`,
`events_processed`, `uptime_seconds`, and `kernel_mode`.

### `ProfiExporterDown`

Prometheus `up == 0` for the profi job. Either:

1. The pod itself is `CrashLoopBackOff` / `ImagePullBackOff`. `kubectl describe
   pod <pod>` to investigate.
2. The `/metrics` endpoint is returning 500. Usually an eBPF verifier failure
   or NVML init failure on an unusual host. Logs:
   ```bash
   kubectl -n <ns> logs <pod> --tail=100
   ```
3. The pod is fine but Prometheus can't reach it — Service / ServiceMonitor /
   NetworkPolicy misconfiguration.

### `ProfiEventLoopSlow`

```promql
histogram_quantile(0.99, rate(profi_system_event_loop_process_duration_seconds_bucket[5m]))
```

profi's own processing loop p99 > threshold (default 1ms). If the loop is
slow, it's *also* the reason events eventually drop. Co-fires with
`ProfiDroppedEvents`. Causes:

- Very high event rate from a noisy pod
- `--kernel-mode=full` on a workload that issues many unique kernel names
  (symbol resolution is expensive)
- CPU-starved pod (requests too low, host under pressure)

### `ProfiCUDAErrors`

This is an **application** signal, not a profi bug. Inspect labels:

- `error_code=2` → `cudaErrorMemoryAllocation` — workload OOM'd; app will likely
  crash or fail requests. Check `profi_cuda_active_memory_bytes` and
  `profi_gpu_memory_bytes{state="used"}`.
- `error_code=700` → `cudaErrorIllegalAddress` — kernel bug or memory corruption;
  the CUDA context is dead, the process will not recover without restart.
- `error_code=719` → `cudaErrorLaunchFailure` — kernel crashed at runtime; the
  context is dead.
- Others: see [CUDA Error Codes](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html).

## Mitigation

### Raise map sizes

Start here for `ProfiDroppedEvents` / `ProfiRingBufferDropRate`:

```yaml
# values.yaml
profi:
  ebpfMaps:
    entriesSize: 32768        # 4x default
    aggregatedSize: 8192      # 4x default
    launchAggSize: 32768      # raise if profi_system_launch_agg_drops_total increases
    mallocSizesSize: 262144   # 2x default (raise for PyTorch workloads)
```

`helm upgrade` — each pod will restart with larger maps. Kernel memory cost is
small (a few MB per map per CPU).

### Drop to a lighter kernel mode

If map sizes don't help, reduce the data volume:

```yaml
profi:
  kernelMode: anonymous   # from 'full'  (removes per-kernel-name labels)
# or
  kernelMode: off         # from 'anonymous'  (disables kernel tracing entirely)
```

Per `docs/BENCHMARK.md`, `full` → `anonymous` removes per-kernel-name labels
and is the first mitigation for kernel-name cardinality; `anonymous` → `off`
disables kernel tracing entirely.

### Enable sampling (last resort)

```yaml
profi:
  cardinality:
    sampleRate: 2   # keep 1-in-2 events
```

Scales drops down linearly but also scales *measurements* down — rates must
be multiplied by `sampleRate` at query time. Prefer larger maps over sampling.

### Fix `NoTrackedPids`

Verify the DaemonSet has `hostPID: true`, `/proc` mounted at `profi.procPath`
(default `/host/proc`), and `CAP_SYS_PTRACE`. The Helm chart sets all three
by default — this alert usually means a custom values override disabled one.

### Fix `ExporterDown`

- `CrashLoopBackOff` on startup: `kubectl logs --previous` — typically
  `failed to load eBPF program` on an unsupported kernel (<5.8) or missing
  `BPF` capability.
- `CrashLoopBackOff` after running: OOMKilled. Raise `resources.limits.memory`
  (default 256Mi is conservative — 512Mi is safe for busy nodes).

### Fix `CUDAErrors` (application-side)

- OOM: reduce batch size, shard model, or pick a bigger GPU.
- Illegal address / launch failure: escalate to application team; profi only
  observed the symptom.

## References

- eBPF map sizing: `src/main.rs` (flags `--entries-size`, `--aggregated-size`, `--launch-agg-size`, `--malloc-sizes-size`)
- Overhead reference: `docs/BENCHMARK.md`
- CUDA error codes: [cudaError enum](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html)
- Related alerts: `ProfiCardinalityDrops` (same root cause, different surface) → see `cardinality-explosion.md`
