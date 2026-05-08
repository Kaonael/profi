# OTLP export

An OpenTelemetry OTLP push exporter is always compiled in and auto-activates when an endpoint is configured. Prometheus `/metrics` stays on regardless — they read from the same registry, so values are atomically consistent.

The exporter is a periodic bridge: every `--otlp-interval-secs` it gathers the Prometheus registry, converts to OTLP (Counter→Sum, Gauge→Gauge, Histogram→Histogram with cumulative temporality) and pushes. Hot path is untouched.

### Flags

| Flag | Env var | Default | Description |
|---|---|---|---|
| `--otlp-endpoint` | `OTEL_EXPORTER_OTLP_ENDPOINT` | — | gRPC or HTTP URL. Empty = disabled |
| `--otlp-protocol` | `OTEL_EXPORTER_OTLP_PROTOCOL` | `grpc` | `grpc` or `http/protobuf` |
| `--otlp-interval-secs` | `OTEL_METRIC_EXPORT_INTERVAL` (ms) | `60` | Push period |
| `--otlp-timeout-secs` | `OTEL_EXPORTER_OTLP_TIMEOUT` | `10` | Per-export deadline |
| `--otlp-service-name` | `OTEL_SERVICE_NAME` | `profi` | `service.name` resource attr |
| `--otlp-headers` | `OTEL_EXPORTER_OTLP_HEADERS` | — | `k1=v1,k2=v2` for auth; mount via Secret |
| `--otlp-ca-cert` | `OTEL_EXPORTER_OTLP_CERTIFICATE` | — | Custom TLS CA |
| `--otlp-client-cert` | `OTEL_EXPORTER_OTLP_CLIENT_CERTIFICATE` | — | mTLS client cert |
| `--otlp-client-key` | `OTEL_EXPORTER_OTLP_CLIENT_KEY` | — | mTLS client key |
| `--otlp-insecure` | — | `false` | Disable TLS (localhost collectors only) |
| `--otlp-resource-attrs` | `OTEL_RESOURCE_ATTRIBUTES` | — | Extra `k=v,k=v` resource attrs |

### Label → OTel semconv mapping

Metric names (`profi_*`) stay unchanged. Label keys are renamed on the wire to follow OTel semantic conventions:

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

profi-specific labels (`operation`, `kernel`, `kernel_class`, `phase`, `direction`, `stream`, `error_code`, `clock_type`, `type`, `state`, `error_type`, `location`, `reason`, `library`) are prefixed with `profi.*`.

All self-observability metrics live under the `profi_system_*` prefix (`profi_system_uptime_seconds`, `profi_system_event_loop_*`, `profi_system_http_*`, …) and are filtered out of OTLP by a single prefix check — they stay on `/metrics` for local debugging. New self-obs metrics under that namespace are excluded automatically.

### Example: send to a local OTel Collector

```bash
# Local collector on default gRPC port
sudo ./target/release/profi \
  --otlp-endpoint http://localhost:4317 \
  --otlp-insecure \
  --otlp-interval-secs 10
```

### Example: Helm values for a collector sidecar / DaemonSet

```yaml
profi:
  otlp:
    endpoint: "http://otel-collector.observability:4317"
    protocol: grpc
    intervalSeconds: 30
    serviceName: profi
    resourceAttrs: "deployment.environment=prod,team=ml-infra"

extraEnv:
  # Auth header from a Secret — never commit tokens to values.yaml
  - name: OTEL_EXPORTER_OTLP_HEADERS
    valueFrom:
      secretKeyRef:
        name: otlp-auth
        key: headers
```

### Cardinality in managed backends

profi can emit up to `--max-time-series` (default 50k) unique label tuples. Backends like Datadog/New Relic bill on series count and may reject very high-cardinality labels (`pid`, `comm`, `stream`). If that's a concern, drop those labels in the OTel Collector `attributes` processor before export — see `deploy/profi/README.md` for a reference collector config.
