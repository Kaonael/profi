# Runbook: NCCL Hang Detected (`ProfiNCCLHangDetected`)

## Symptom

Alert `ProfiNCCLHangDetected` has fired. profi observed an in-flight NCCL
collective that exceeded `--nccl-hang-timeout` (default 60s). Alert labels:

- `pid` — the process that issued the stuck collective
- `operation` — e.g. `ncclAllReduce`, `ncclAllGather`, `ncclBroadcast`, `ncclSend`, `ncclRecv`
- `pod`, `namespace`, `instance` — where it's running

## Impact

**Distributed training/inference is stuck.** An NCCL collective cannot complete
until every participating rank joins. One hung rank blocks the whole group,
wasting GPU-hours proportional to the world size. For inference serving, this
manifests as requests piling up with no output tokens.

## Diagnosis

### 1. Identify the stuck collective

```promql
profi_nccl_hang_detected_total
```

Look at the labels to find `pid`, `operation`, and (via the `pod` label on
nearby `profi_nccl_calls_total`) the pod.

```promql
profi_nccl_stale_entries > 0
```

counts currently-stale in-flight entries; confirms the hang is ongoing, not
cleared.

### 2. Cross-check with straggler detection

```promql
topk(5, profi_nccl_straggler_ratio)
```

A straggler that went from >1.5x to "entry stuck forever" is the usual
progression: one slow rank eventually becomes a hard hang.

### 3. Inspect the pod

```bash
kubectl -n <namespace> exec <pod> -- nvidia-smi
kubectl -n <namespace> logs <pod> --tail=200 | grep -Ei 'nccl|cuda|hang|timeout'
# If NCCL_DEBUG is not already enabled, you'll see little — enable on next restart
```

Check for:

- NCCL warnings about connection timeouts / rank not responding
- A GPU that's in `P0` state at 100% utilization but with 0 MiB/s memory
  bandwidth (stuck kernel)
- Another GPU on a peer pod missing from the same collective

### 4. Check the peers

Distributed workloads typically run as `StatefulSet` (`rank-0`, `rank-1`, ...).
List all peers:

```bash
kubectl -n <namespace> get pods -l <your-selector> -o wide
```

A hang usually shows **one** pod as the culprit (missing heartbeat, unhealthy,
OOMKilled) while the others are waiting on it.

### 5. Driver / hardware status

```bash
kubectl -n <namespace> exec <pod> -- nvidia-smi -q -d ECC,TEMPERATURE,POWER
```

Look for `Pending Page Retirement`, XIDs in `dmesg` on the host, or thermal
throttling. Cross-reference:

```promql
profi_gpu_ecc_errors_total
profi_gpu_throttle_active == 1
```

## Mitigation

### Immediate (restore service)

1. **Bounce the inference pod** — `kubectl -n <ns> delete pod <pod>`. In a
   multi-replica deployment, traffic fails over while the replacement starts.
2. If the same pod is repeatedly getting stuck, cordon the node and drain:
   ```bash
   kubectl cordon <node>
   kubectl drain <node> --ignore-daemonsets --delete-emptydir-data
   ```

### Root-cause mitigation (pick based on diagnosis)

- **Symptom of a slow peer**: raise NCCL timeout on the application
  (`NCCL_ASYNC_ERROR_HANDLING=1`, `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`), so
  the app surfaces the hang as an exception instead of blocking forever.
- **NVLink/NIC fabric flap**: check the host `dmesg` for `NVLink` errors; if
  persistent, schedule the node out of service.
- **Driver mismatch between pods**: verify `nvidia-smi --version` is identical
  across all ranks; a mixed driver version matrix causes hangs.
- **Container resource starvation** (CPU throttling causing the CPU-side NCCL
  proxy threads to stall): check `container_cpu_cfs_throttled_seconds_total`;
  raise the pod's CPU requests.

### Escalation

- **Hung GPU visible in `nvidia-smi` but unkillable** (driver state stuck):
  node reboot is the only fix. Open infra ticket.
- **Repeats on the same GPU across pod restarts**: the GPU is suspect.
  Cordon + RMA.
- **Repeats across multiple nodes in the same time window**: suspect a fabric
  issue (NVLink switch, InfiniBand fabric). Page network on-call.

## References

- profi metric sources: `src/main.rs`, `src/metrics.rs`
- `--nccl-hang-timeout` configuration: `values.yaml` → `profi.nccl.hangTimeoutSeconds`
- NCCL env for app-side timeout: [NCCL_ASYNC_ERROR_HANDLING](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- Related alerts: `ProfiNCCLStraggler` (precursor), `ProfiCUDAErrors` (sometimes co-fires)
