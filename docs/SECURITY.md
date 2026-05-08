# profi security model

This document describes how profi protects its data plane (eBPF) and
management plane (the `/metrics` HTTP endpoint and the outbound OTLP push
path).

## Threat model

profi runs as a privileged DaemonSet on GPU nodes. It has:

- `CAP_SYS_ADMIN` / `CAP_BPF` / `CAP_PERFMON` to attach eBPF uprobes.
- `CAP_SYS_PTRACE` to read `/proc/<pid>/maps` of other pods' containers.
- `hostPID: true`.
- Host mounts for `/proc`, `/sys/kernel/debug`, and `/sys/fs/bpf`.
- NVIDIA runtime access (`runtimeClassName: nvidia`,
  `NVIDIA_VISIBLE_DEVICES=all`, `NVIDIA_DRIVER_CAPABILITIES=compute,utility`)
  so NVML can read GPU inventory and health counters without reserving GPUs.
- RBAC to `list`, `watch` Pods cluster-wide (for label enrichment).
- Conditional RBAC to `create` TokenReviews when
  `profi.metricsAuth.mode != "off"` (`bearer` or `mtls-or-bearer`).

Assumed adversaries:

1. **Untrusted tenant in the cluster.** Should not be able to read
   profi's metrics (per-pod GPU/kernel telemetry is a side-channel into
   other tenants' workloads).
2. **Compromised sidecar in an observability-adjacent namespace.** Should
   not be able to impersonate Prometheus and scrape profi just by reaching
   `:9401` over the pod network.
3. **Passive network observer.** Should not be able to read metrics off
   the wire.

## `/metrics` protection matrix

Two independent axes, configured via Helm values
(`profi.metricsTls.mode` × `profi.metricsAuth.mode`) or the equivalent
`--metrics-tls-mode` / `--metrics-auth-mode` flags:

| TLS ↓ / Auth → | `off`                  | `bearer`                    | `mtls-or-bearer`            |
|----------------|------------------------|-----------------------------|-----------------------------|
| `off`          | Plain HTTP (default; **no protection**) | Plain HTTP + Bearer (OK on trusted pod net) | invalid          |
| `server`       | TLS only, any client   | TLS + Bearer (classic K8s ServiceMonitor) | invalid                  |
| `mtls`         | TLS + client cert (CA-pinned) | invalid (use `mtls-or-bearer`) | TLS + (client cert **or** Bearer) — **recommended** |

**`/health` and `/ready` always bypass L7 auth** so kubelet probes keep
working. When TLS is enabled, the probes use HTTPS (kubelet supports
`scheme: HTTPS`) — insecureSkipVerify is implicit since kubelet does not
check the cert.

## Bearer token validation

profi validates Bearer tokens via the Kubernetes `TokenReview` API.

- **Audience binding** (`--metrics-auth-audience`): when set, only tokens
  projected with a matching `audience` are accepted. This is defense in
  depth: it prevents reuse of a default API-server token in case one
  leaks. Default: **disabled** for compatibility with default SA tokens.

- **Cache** (`--metrics-auth-cache-ttl`, `--metrics-auth-cache-size`):
  successful reviews are cached in-process, keyed by `SHA-256(token)`.
  Negative results are **never** cached so revoked tokens stop working
  within one request. The cache TTL should be well below the token
  lifetime — 60s is a safe default.

## Outbound OTLP mTLS

`--otlp-ca-cert` / `--otlp-client-cert` / `--otlp-client-key` drive
tonic's `ClientTlsConfig`. This is strictly outgoing: profi's OTLP
exporter authenticates itself to a collector, not the other way around.

All self-observability metrics live under the `profi_system_*` prefix
(e.g. `profi_system_http_auth_failures_total`,
`profi_system_prometheus_encode_duration_seconds`,
`profi_system_event_loop_process_duration_seconds`) and are **filtered**
from the OTLP stream by a single `starts_with("profi_system_")` check
(`should_skip_metric` / `SELF_OBS_PREFIX` in `src/otlp.rs`).
They remain on the local `/metrics` endpoint for operator debugging only.

## Recommended production deployment

```yaml
profi:
  metricsTls:
    mode: mtls
    certManager:
      enabled: true
      issuerName: internal-ca
  metricsAuth:
    mode: mtls-or-bearer
    # Uncomment to require audience-bound tokens:
    # audience: profi-metrics
    cacheTtlSeconds: 60
```

Prometheus Operator should scrape with both a client cert (for mTLS) and
a ServiceAccount token (for audit trail). `mtls-or-bearer` will accept
whichever arrives.

## Per-pod full-mode promotion via annotation

profi honors the pod annotation `profi/mode: full` — any PIDs
belonging to such a pod are promoted to full kernel tracing (per-launch
kernel names, duration histograms) regardless of the cluster-wide
`--kernel-mode` setting.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: suspect-inference
  annotations:
    profi/mode: full
spec:
  containers:
    - ...
```

- Upgrade-only. The annotation can *promote* a pod from `anonymous` →
  `full`, but cannot downgrade it below the cluster-wide baseline, and
  it cannot turn profi off. Operators still control the minimum
  observability floor via `--kernel-mode`.
- Only the literal value `full` upgrades. Any other value (`anonymous`,
  `true`, `yes`, typos) is ignored — avoids accidental overhead from
  annotation drift.
- Non-annotated pods run on the cluster-wide `--kernel-mode` with no
  per-PID lookups beyond the existing enrichment cache.
- The eBPF `UPGRADED_PIDS` map is capped at 1024 entries; if you have
  more than 1024 concurrent full-mode pods, entries past the cap fall
  back to the cluster default.
- RBAC: the profi ServiceAccount must have `pods.get/list/watch` on all
  namespaces (it already does for enrichment). No new permissions needed.

Use case: an operator suspects one pod is responsible for an inference
regression, annotates it, and gets per-kernel timing without raising
overhead cluster-wide.

## Reporting vulnerabilities

Please report suspected vulnerabilities through GitHub Security Advisories:
https://github.com/Kaonael/profi/security/advisories/new
