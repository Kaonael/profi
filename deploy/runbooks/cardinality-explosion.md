# Runbook: Cardinality Explosion (`ProfiCardinalityDrops`)

## Symptom

Alert `ProfiCardinalityDrops` has fired. profi hit the `--max-time-series`
cap (default 50000) and is *silently dropping new label combinations*. The
counter:

```promql
increase(profi_cardinality_limit_drops_total[5m]) > 0
```

is incrementing.

## Impact

- Some workloads are no longer being observed at all. Every new distinct
  label tuple beyond the cap is dropped on the floor.
- Prometheus itself is likely bloated — large per-series memory, slow queries,
  scrape time stretching.
- Alerts based on `profi_*` metrics may under-fire for the unobserved
  workloads.

## Diagnosis

### 1. Which pod is noisy?

```promql
topk(10,
  count by (namespace, pod) ({__name__=~"profi_.*", __name__!="profi_gpu_info"})
)
```

This tells you which pods contribute the most distinct time series. One pod
with 10k+ series almost always explains it.

### 2. Which label is exploding?

Usually it's one of: `stream`, `kernel`, or `pid`. Count distinct values of
each on the noisy pod:

```promql
count(count by (stream) (profi_cuda_calls_total{pod="<noisy-pod>"}))
count(count by (kernel) (profi_cuda_kernel_launches_total{pod="<noisy-pod>"}))
count(count by (pid)    (profi_cuda_calls_total{pod="<noisy-pod>"}))
```

Interpretation:

- **`stream` > 32**: profi should already have collapsed these to `"default"`
  (via `--max-streams-per-pid`). If you see >32 values, the limit was raised.
- **`kernel` > 512**: same — `--max-kernels-per-pid` collapses the overflow
  to `"other"`. Workload is using a lot of unique kernel names (usually a
  torch compile / JIT pattern).
- **`pid` unbounded**: a pod that spawns many short-lived processes
  (training job with lots of worker restarts, Ray actors, etc.).

### 3. Workload classification

- **torch.compile / triton JIT**: generates unique kernel names per compile.
- **Custom CUDA kernels with templated names**: every template instantiation
  is a distinct kernel.
- **Fork-heavy pipelines**: high pid turnover.
- **Multi-tenant inference**: many small pods, each with reasonable
  cardinality, but in aggregate you overflow the node-level cap.

## Mitigation

### Quick: lower per-pid limits

If `stream` or `kernel` is exploding, tighten the per-pid caps:

```yaml
profi:
  cardinality:
    maxStreamsPerPid: 16      # from 32
    maxKernelsPerPid: 128     # from 512
```

This forces more aggressive collapse to `"default"`/`"other"`, giving up some
detail but restoring budget. `helm upgrade` — pods pick up the new values on
restart.

### Better: drop to `kernel-mode=anonymous`

If the noisy dimension is `kernel` names:

```yaml
profi:
  kernelMode: anonymous   # from 'full'
```

`anonymous` aggregates kernels by `kernel_class` (attention/mlp/norm/…)
instead of by exact name — you get the useful signal at a fraction of the
cardinality.

### Raise the cap (only if you have Prometheus headroom)

```yaml
profi:
  cardinality:
    maxTimeSeries: 100000   # from 50000
```

**Check first** that your Prometheus can swallow it — each extra 50k series
adds ≈100–200 MB of RAM at typical label widths. Do not blindly raise this on
a shared Prometheus.

### Tighten scrape with `metricRelabelings`

If one label is noisy and dashboards don't use it, drop it at scrape time:

```yaml
serviceMonitor:
  metricRelabelings:
    - action: labeldrop
      regex: "stream"     # drop the stream label entirely
```

This reduces *Prometheus-side* cardinality but doesn't reduce profi's
internal tracking. Use it when you want a slim view even though profi sees
the full fan-out.

### Quarantine the noisy pod

If one pod is genuinely abusing (e.g., a debug workload compiling 100k
kernels) and others are fine, deploy a second profi instance with stricter
limits and target it at that namespace via node taints/tolerations, or
pre-filter the noisy pod via `metricRelabelings: drop`.

## References

- Cardinality control code: `src/cache.rs`, `src/metrics.rs`
- Flags: `--max-time-series`, `--max-streams-per-pid`, `--max-kernels-per-pid`, `--sample-rate`
- Prometheus sizing reference: [Operational Aspects](https://prometheus.io/docs/prometheus/latest/storage/#operational-aspects)
- Related alerts: `ProfiDroppedEvents` (ring buffer overflow under the same load)
