// SPDX-License-Identifier: Apache-2.0

//! OpenTelemetry OTLP metrics bridge for profi.
//!
//! A periodic task reads `registry.gather()` from the Prometheus registry,
//! converts each `MetricFamily` into OTLP `ResourceMetrics` and pushes the
//! batch via `opentelemetry-otlp`. The Prometheus `/metrics` endpoint keeps
//! working unchanged; the bridge sits alongside it without touching the
//! hot-path metric handle cache.
//!
//! The bridge is compiled into every build and auto-activates whenever an
//! endpoint is configured (via `--otlp-endpoint` or
//! `OTEL_EXPORTER_OTLP_ENDPOINT`). When nothing is configured the task is
//! not spawned — no connection attempts, no errors.

use std::borrow::Cow;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Context, Result};
use clap::Parser;
use opentelemetry::{InstrumentationScope, KeyValue};
use opentelemetry_otlp::{MetricExporter, WithExportConfig};
use opentelemetry_sdk::metrics::data::{
    DataPoint, Gauge, Histogram, HistogramDataPoint, Metric, ResourceMetrics, ScopeMetrics, Sum,
};
use opentelemetry_sdk::metrics::exporter::PushMetricExporter;
use opentelemetry_sdk::metrics::Temporality;
use opentelemetry_sdk::Resource;
use prometheus::proto::{LabelPair, MetricFamily, MetricType};
use tokio::task::JoinHandle;

use crate::metrics::Metrics;

/// OTLP transport protocol selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OtlpProtocol {
    Grpc,
    HttpProtobuf,
}

/// Runtime configuration for the OTLP bridge.
#[derive(Debug, Clone)]
pub struct OtlpConfig {
    pub endpoint: String,
    pub protocol: OtlpProtocol,
    pub interval: Duration,
    pub timeout: Duration,
    pub service_name: String,
    pub headers: Vec<(String, String)>,
    pub ca_cert_path: Option<String>,
    pub client_cert_path: Option<String>,
    pub client_key_path: Option<String>,
    pub insecure: bool,
    pub resource_attrs: Vec<(String, String)>,
}

/// CLI arguments for the OTLP bridge.
///
/// Designed to be composed into the top-level `Args` via `#[command(flatten)]`
/// so every OTLP flag stays colocated with the module that owns it. All flags
/// map to the standard `OTEL_*` environment variables so profi behaves like
/// the rest of the OpenTelemetry ecosystem out of the box.
#[derive(Parser, Debug, Clone)]
pub struct OtlpArgs {
    /// OTLP endpoint URL. Setting this activates the bridge; otherwise OTLP
    /// export is disabled (no task spawned, no connection attempts).
    /// Example: `http://otel-collector.monitoring:4317`
    #[arg(long = "otlp-endpoint", env = "OTEL_EXPORTER_OTLP_ENDPOINT")]
    pub otlp_endpoint: Option<String>,

    /// OTLP transport protocol: `grpc` (default) or `http/protobuf`.
    #[arg(
        long = "otlp-protocol",
        env = "OTEL_EXPORTER_OTLP_PROTOCOL",
        default_value = "grpc"
    )]
    pub otlp_protocol: String,

    /// Push interval in seconds. OTel ecosystem default is 60s; don't go
    /// below the Prometheus scrape interval (typically 15–30s).
    #[arg(long = "otlp-interval-secs", default_value_t = 60)]
    pub otlp_interval_secs: u64,

    /// Per-export timeout in seconds.
    #[arg(
        long = "otlp-timeout-secs",
        env = "OTEL_EXPORTER_OTLP_TIMEOUT",
        default_value_t = 10
    )]
    pub otlp_timeout_secs: u64,

    /// `service.name` resource attribute.
    #[arg(
        long = "otlp-service-name",
        env = "OTEL_SERVICE_NAME",
        default_value = "profi"
    )]
    pub otlp_service_name: String,

    /// Headers sent with every export, format `k1=v1,k2=v2`. Used for auth
    /// tokens (e.g. `Authorization=Bearer ...`). Never logged.
    #[arg(long = "otlp-headers", env = "OTEL_EXPORTER_OTLP_HEADERS")]
    pub otlp_headers: Option<String>,

    /// Path to PEM-encoded CA certificate used to verify the collector's TLS.
    #[arg(long = "otlp-ca-cert", env = "OTEL_EXPORTER_OTLP_CERTIFICATE")]
    pub otlp_ca_cert: Option<PathBuf>,

    /// Path to PEM-encoded client certificate for mTLS.
    #[arg(
        long = "otlp-client-cert",
        env = "OTEL_EXPORTER_OTLP_CLIENT_CERTIFICATE"
    )]
    pub otlp_client_cert: Option<PathBuf>,

    /// Path to PEM-encoded client key for mTLS.
    #[arg(long = "otlp-client-key", env = "OTEL_EXPORTER_OTLP_CLIENT_KEY")]
    pub otlp_client_key: Option<PathBuf>,

    /// Force-disable TLS for the gRPC endpoint. Only useful for local
    /// collectors on `localhost:4317`.
    #[arg(long = "otlp-insecure")]
    pub otlp_insecure: bool,

    /// Extra resource attributes, format `k1=v1,k2=v2`. Merged on top of
    /// the built-in defaults (`service.name`, `host.name`, `k8s.node.name`).
    #[arg(long = "otlp-resource-attrs", env = "OTEL_RESOURCE_ATTRIBUTES")]
    pub otlp_resource_attrs: Option<String>,
}

/// Name prefix reserved for profi self-observability metrics (the bridge's
/// own overhead, cache sizes, scrape latency, HTTP auth/TLS counters). Any
/// metric starting with this MUST NOT be shipped over OTLP — it would be
/// noise in customer observability stacks. Enforced by [`should_skip_metric`].
pub const SELF_OBS_PREFIX: &str = "profi_system_";

/// Prometheus label → OTel semconv attribute rename table.
/// Labels not listed here are emitted with a `profi.` prefix.
pub const LABEL_MAP: &[(&str, &str)] = &[
    ("pod", "k8s.pod.name"),
    ("namespace", "k8s.namespace.name"),
    ("container", "k8s.container.name"),
    ("gpu", "gpu.id"),
    ("gpu_uuid", "gpu.uuid"),
    ("gpu_model", "gpu.model"),
    ("pid", "process.pid"),
    ("comm", "process.command"),
    ("operation", "profi.operation"),
    ("kernel", "profi.kernel"),
    ("kernel_class", "profi.kernel.class"),
    ("phase", "profi.phase"),
    ("direction", "profi.memcpy.direction"),
    ("stream", "profi.cuda.stream"),
    ("error_code", "profi.error.code"),
];

impl OtlpConfig {
    /// Returns `Some(cfg)` when an endpoint is configured and the bridge
    /// should spawn, `None` when OTLP is effectively disabled.
    ///
    /// This is what makes OTLP "auto-enabled": caller just asks — if the
    /// user hasn't set `--otlp-endpoint` / `OTEL_EXPORTER_OTLP_ENDPOINT`,
    /// we return `None` and nothing gets spawned.
    pub fn resolve(args: &OtlpArgs) -> Result<Option<Self>> {
        let endpoint = match args
            .otlp_endpoint
            .as_ref()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
        {
            Some(e) => e,
            None => return Ok(None),
        };

        let protocol = match args.otlp_protocol.as_str() {
            "grpc" | "grpc/protobuf" => OtlpProtocol::Grpc,
            "http/protobuf" | "http" => OtlpProtocol::HttpProtobuf,
            other => anyhow::bail!(
                "invalid --otlp-protocol '{}', expected 'grpc' or 'http/protobuf'",
                other
            ),
        };

        // OTEL_METRIC_EXPORT_INTERVAL is specified in milliseconds; when set,
        // it overrides --otlp-interval-secs to keep behaviour predictable for
        // ecosystem users who set env vars without reading our flag docs.
        let interval = if let Ok(ms_str) = std::env::var("OTEL_METRIC_EXPORT_INTERVAL") {
            let ms: u64 = ms_str
                .parse()
                .with_context(|| format!("parse OTEL_METRIC_EXPORT_INTERVAL='{}'", ms_str))?;
            Duration::from_millis(ms)
        } else {
            Duration::from_secs(args.otlp_interval_secs)
        };

        let timeout = Duration::from_secs(args.otlp_timeout_secs);

        let headers = args
            .otlp_headers
            .as_deref()
            .map(parse_kv_list)
            .unwrap_or_default();

        let resource_attrs = args
            .otlp_resource_attrs
            .as_deref()
            .map(parse_kv_list)
            .unwrap_or_default();

        let path_to_string = |p: &PathBuf| p.to_string_lossy().into_owned();

        Ok(Some(Self {
            endpoint,
            protocol,
            interval,
            timeout,
            service_name: args.otlp_service_name.clone(),
            headers,
            ca_cert_path: args.otlp_ca_cert.as_ref().map(path_to_string),
            client_cert_path: args.otlp_client_cert.as_ref().map(path_to_string),
            client_key_path: args.otlp_client_key.as_ref().map(path_to_string),
            insecure: args.otlp_insecure,
            resource_attrs,
        }))
    }
}

/// Parse `k1=v1,k2=v2` into key-value pairs. Empty keys and malformed
/// segments are silently dropped. Whitespace around keys/values is trimmed
/// because users often format these for readability (`Auth = Bearer xyz`).
fn parse_kv_list(raw: &str) -> Vec<(String, String)> {
    raw.split(',')
        .filter_map(|kv| {
            let mut parts = kv.splitn(2, '=');
            let k = parts.next()?.trim();
            let v = parts.next()?.trim();
            if k.is_empty() {
                None
            } else {
                Some((k.to_string(), v.to_string()))
            }
        })
        .collect()
}

/// Whether a metric family should be dropped before OTLP export.
///
/// profi's self-observability metrics (cache sizes, scrape duration,
/// event-loop timing, HTTP auth/TLS counters) all live under the
/// [`SELF_OBS_PREFIX`] namespace. Shipping them over OTLP would be noise in
/// customer observability stacks, and the prefix convention means a new
/// self-obs metric is excluded automatically — no manual whitelist to sync.
pub fn should_skip_metric(name: &str) -> bool {
    name.starts_with(SELF_OBS_PREFIX)
}

/// Build the OTLP `Resource` describing this profi instance.
///
/// Always includes `service.name` and `service.version`; adds
/// `service.instance.id` / `host.name` from `$HOSTNAME` and
/// `k8s.node.name` from `$NODE_NAME` when set. Extra K=V pairs from
/// `cfg.resource_attrs` win over the defaults.
pub fn build_resource(cfg: &OtlpConfig) -> Resource {
    let mut kvs: Vec<KeyValue> = vec![
        KeyValue::new("service.name", cfg.service_name.clone()),
        KeyValue::new("service.version", env!("CARGO_PKG_VERSION")),
    ];

    if let Ok(hostname) = std::env::var("HOSTNAME") {
        if !hostname.is_empty() {
            kvs.push(KeyValue::new("service.instance.id", hostname.clone()));
            kvs.push(KeyValue::new("host.name", hostname));
        }
    }

    if let Ok(node) = std::env::var("NODE_NAME") {
        if !node.is_empty() {
            kvs.push(KeyValue::new("k8s.node.name", node));
        }
    }

    // User-provided overrides come last so they win dedup.
    for (k, v) in &cfg.resource_attrs {
        kvs.push(KeyValue::new(k.clone(), v.clone()));
    }

    Resource::new(kvs)
}

/// Translate Prometheus labels to OTel `KeyValue` attributes.
///
/// Labels listed in [`LABEL_MAP`] are renamed to their OTel semconv name
/// (e.g. `pod` → `k8s.pod.name`). Unknown labels are prefixed with `profi.`
/// to avoid colliding with semconv names on the collector side.
pub fn translate_labels(labels: &[LabelPair]) -> Vec<KeyValue> {
    labels
        .iter()
        .map(|lp| {
            let name = lp.get_name();
            let otel_key = LABEL_MAP
                .iter()
                .find_map(|(prom, otel)| (*prom == name).then_some(*otel))
                .map(Cow::Borrowed)
                .unwrap_or_else(|| Cow::Owned(format!("profi.{}", name)));
            KeyValue::new(otel_key.into_owned(), lp.get_value().to_string())
        })
        .collect()
}

/// Convert a Prometheus `Counter`-typed `MetricFamily` into an OTLP `Sum<f64>`
/// aggregation wrapped in a `Metric`.
///
/// Returns `None` if `family` isn't a counter or has no data points — the
/// caller is a pure dispatcher and shouldn't special-case that.
///
/// The Prometheus `Counter` is process-cumulative monotonic f64, which maps
/// exactly onto `Sum { is_monotonic: true, temporality: Cumulative }`.
/// `start_time` is the fixed process start (same for every export), `time`
/// is the current export instant.
pub fn counter_family_to_otlp(
    family: &MetricFamily,
    start_time: SystemTime,
    now: SystemTime,
) -> Option<Metric> {
    if family.get_field_type() != MetricType::COUNTER {
        return None;
    }
    let data_points = family
        .get_metric()
        .iter()
        .map(|m| DataPoint {
            attributes: translate_labels(m.get_label()),
            start_time: Some(start_time),
            time: Some(now),
            value: m.get_counter().get_value(),
            exemplars: Vec::new(),
        })
        .collect::<Vec<_>>();

    if data_points.is_empty() {
        return None;
    }

    let data = Sum {
        data_points,
        temporality: Temporality::Cumulative,
        is_monotonic: true,
    };

    Some(Metric {
        name: Cow::Owned(family.get_name().to_string()),
        description: Cow::Owned(family.get_help().to_string()),
        unit: Cow::Borrowed(""),
        data: Box::new(data),
    })
}

/// Convert a Prometheus `Gauge`-typed `MetricFamily` into an OTLP `Gauge<f64>`
/// aggregation.
///
/// Gauges are non-monotonic point-in-time values — no temporality applies.
/// We still set `start_time` for consistency with Counter/Histogram so
/// collectors that join metrics by resource+start_time line things up.
pub fn gauge_family_to_otlp(
    family: &MetricFamily,
    start_time: SystemTime,
    now: SystemTime,
) -> Option<Metric> {
    if family.get_field_type() != MetricType::GAUGE {
        return None;
    }
    let data_points = family
        .get_metric()
        .iter()
        .map(|m| DataPoint {
            attributes: translate_labels(m.get_label()),
            start_time: Some(start_time),
            time: Some(now),
            value: m.get_gauge().get_value(),
            exemplars: Vec::new(),
        })
        .collect::<Vec<_>>();

    if data_points.is_empty() {
        return None;
    }

    Some(Metric {
        name: Cow::Owned(family.get_name().to_string()),
        description: Cow::Owned(family.get_help().to_string()),
        unit: Cow::Borrowed(""),
        data: Box::new(Gauge { data_points }),
    })
}

/// Convert a single Prometheus `Histogram` to an OTLP `HistogramDataPoint`.
///
/// Prometheus stores bucket counts as **cumulative** by `upper_bound`, in
/// ascending order, with an implicit `+Inf` bucket tracked through
/// `sample_count`. OTLP stores **non-cumulative** counts with
/// `bucket_counts.len() == bounds.len() + 1` (the trailing element is the
/// `+Inf` catch-all).
///
/// Conversion:
///   bounds[i]         = finite[i].upper_bound
///   bucket_counts[0]  = finite[0].cumulative_count
///   bucket_counts[i]  = finite[i].cumulative_count − finite[i-1].cumulative_count
///   bucket_counts[N]  = sample_count − finite[last].cumulative_count   // +Inf
///
/// We also tolerate a Prometheus source that emits an explicit `+Inf`
/// bucket (some libraries do) — it's filtered out here and re-implied by
/// the trailing `bucket_counts` element.
pub fn histogram_to_otlp(
    h: &prometheus::proto::Histogram,
    attributes: Vec<KeyValue>,
    start_time: SystemTime,
    now: SystemTime,
) -> HistogramDataPoint<f64> {
    let finite: Vec<&prometheus::proto::Bucket> = h
        .get_bucket()
        .iter()
        .filter(|b| b.get_upper_bound().is_finite())
        .collect();

    let sample_count = h.get_sample_count();
    let sample_sum = h.get_sample_sum();

    let bounds: Vec<f64> = finite.iter().map(|b| b.get_upper_bound()).collect();

    let mut bucket_counts: Vec<u64> = Vec::with_capacity(finite.len() + 1);
    let mut prev = 0u64;
    for b in &finite {
        let cum = b.get_cumulative_count();
        bucket_counts.push(cum.saturating_sub(prev));
        prev = cum;
    }
    bucket_counts.push(sample_count.saturating_sub(prev));

    HistogramDataPoint {
        attributes,
        start_time,
        time: now,
        count: sample_count,
        sum: sample_sum,
        bounds,
        bucket_counts,
        min: None,
        max: None,
        exemplars: Vec::new(),
    }
}

/// Convert a Prometheus `Histogram`-typed `MetricFamily` into an OTLP
/// `Histogram<f64>` aggregation with cumulative temporality.
pub fn histogram_family_to_otlp(
    family: &MetricFamily,
    start_time: SystemTime,
    now: SystemTime,
) -> Option<Metric> {
    if family.get_field_type() != MetricType::HISTOGRAM {
        return None;
    }
    let data_points: Vec<HistogramDataPoint<f64>> = family
        .get_metric()
        .iter()
        .map(|m| {
            histogram_to_otlp(
                m.get_histogram(),
                translate_labels(m.get_label()),
                start_time,
                now,
            )
        })
        .collect();

    if data_points.is_empty() {
        return None;
    }

    Some(Metric {
        name: Cow::Owned(family.get_name().to_string()),
        description: Cow::Owned(family.get_help().to_string()),
        unit: Cow::Borrowed(""),
        data: Box::new(Histogram {
            data_points,
            temporality: Temporality::Cumulative,
        }),
    })
}

/// Dispatch a Prometheus `MetricFamily` to the correct OTLP converter.
/// Summary/Untyped families are dropped — profi doesn't produce them.
pub fn convert_metric_family(
    family: &MetricFamily,
    start_time: SystemTime,
    now: SystemTime,
) -> Option<Metric> {
    match family.get_field_type() {
        MetricType::COUNTER => counter_family_to_otlp(family, start_time, now),
        MetricType::GAUGE => gauge_family_to_otlp(family, start_time, now),
        MetricType::HISTOGRAM => histogram_family_to_otlp(family, start_time, now),
        _ => None,
    }
}

fn build_scope() -> InstrumentationScope {
    InstrumentationScope::builder("profi")
        .with_version(env!("CARGO_PKG_VERSION"))
        .build()
}

fn build_exporter(cfg: &OtlpConfig) -> Result<MetricExporter> {
    match cfg.protocol {
        OtlpProtocol::Grpc => {
            use opentelemetry_otlp::WithTonicConfig;
            use tonic::metadata::{MetadataKey, MetadataMap, MetadataValue};

            let mut builder = MetricExporter::builder()
                .with_tonic()
                .with_endpoint(cfg.endpoint.clone())
                .with_timeout(cfg.timeout);

            if !cfg.headers.is_empty() {
                let mut md = MetadataMap::new();
                for (k, v) in &cfg.headers {
                    let key = MetadataKey::from_bytes(k.as_bytes())
                        .map_err(|e| anyhow::anyhow!("invalid OTLP header key '{}': {}", k, e))?;
                    let val = MetadataValue::try_from(v.as_str()).map_err(|e| {
                        anyhow::anyhow!("invalid OTLP header value for key '{}': {}", k, e)
                    })?;
                    md.insert(key, val);
                }
                builder = builder.with_metadata(md);
            }

            let need_tls = !cfg.insecure
                && (cfg.endpoint.starts_with("https://")
                    || cfg.ca_cert_path.is_some()
                    || cfg.client_cert_path.is_some());
            if need_tls {
                builder = builder.with_tls_config(build_tls_config(cfg)?);
            }

            builder.build().context("build OTLP gRPC metrics exporter")
        }
        OtlpProtocol::HttpProtobuf => {
            use opentelemetry_otlp::WithHttpConfig;
            use std::collections::HashMap;

            let mut builder = MetricExporter::builder()
                .with_http()
                .with_endpoint(cfg.endpoint.clone())
                .with_timeout(cfg.timeout);

            if !cfg.headers.is_empty() {
                let hmap: HashMap<String, String> = cfg.headers.iter().cloned().collect();
                builder = builder.with_headers(hmap);
            }

            builder.build().context("build OTLP HTTP metrics exporter")
        }
    }
}

fn build_tls_config(cfg: &OtlpConfig) -> Result<tonic::transport::ClientTlsConfig> {
    use crate::pem::read_pem_file;
    use tonic::transport::{Certificate, ClientTlsConfig, Identity};

    let mut tls = ClientTlsConfig::new().with_native_roots();

    if let Some(ca) = &cfg.ca_cert_path {
        let pem = read_pem_file(ca).with_context(|| format!("read OTLP CA cert {}", ca))?;
        tls = tls.ca_certificate(Certificate::from_pem(pem));
    }

    if let (Some(cert), Some(key)) = (&cfg.client_cert_path, &cfg.client_key_path) {
        let cert_pem =
            read_pem_file(cert).with_context(|| format!("read OTLP client cert {}", cert))?;
        let key_pem =
            read_pem_file(key).with_context(|| format!("read OTLP client key {}", key))?;
        tls = tls.identity(Identity::from_pem(cert_pem, key_pem));
    }

    Ok(tls)
}

/// Bridge handle. Owns the background task that periodically pushes
/// Prometheus-gathered metrics to an OTLP endpoint.
pub struct OtlpBridge;

impl OtlpBridge {
    /// Spawn the periodic OTLP push task.
    ///
    /// The exporter, resource and scope are built eagerly so config
    /// errors (bad endpoint, unreadable cert) surface at startup rather
    /// than silently every export cycle.
    pub fn start(cfg: OtlpConfig, metrics: Metrics, start_time: Instant) -> Result<JoinHandle<()>> {
        let exporter = build_exporter(&cfg)?;
        let resource = build_resource(&cfg);
        let scope = build_scope();
        let interval = cfg.interval;

        // Pin the export `start_time` to the *wall-clock* moment this process
        // came up. This is what OTel backends use to detect counter resets:
        // cumulative sums with a new start_time mean "fresh series, don't
        // flag this as a decrease." Using `Instant::elapsed()` here means the
        // value is stable for the process lifetime even if wall-clock skews.
        let start_wallclock = SystemTime::now()
            .checked_sub(start_time.elapsed())
            .unwrap_or_else(SystemTime::now);

        log::info!(
            "OTLP bridge started: endpoint={}, protocol={:?}, interval={:?}",
            cfg.endpoint,
            cfg.protocol,
            interval,
        );

        let handle = tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            // The first tick fires immediately; skip it so the first push
            // waits a full interval after startup (gives the scraper-side
            // stack a chance to come up cleanly).
            ticker.tick().await;

            loop {
                ticker.tick().await;
                let now = SystemTime::now();
                let families = metrics.registry.gather();
                let otlp_metrics: Vec<Metric> = families
                    .iter()
                    .filter(|f| !should_skip_metric(f.get_name()))
                    .filter_map(|f| convert_metric_family(f, start_wallclock, now))
                    .collect();

                if otlp_metrics.is_empty() {
                    continue;
                }

                let mut rm = ResourceMetrics {
                    resource: resource.clone(),
                    scope_metrics: vec![ScopeMetrics {
                        scope: scope.clone(),
                        metrics: otlp_metrics,
                    }],
                };

                if let Err(e) = exporter.export(&mut rm).await {
                    log::warn!("OTLP export failed: {}", e);
                }
            }
        });

        Ok(handle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prometheus::proto::{
        Bucket, Counter, Gauge as PromGauge, Histogram as PromHistogram, LabelPair,
        Metric as PromMetric, MetricFamily, MetricType,
    };

    fn label(name: &str, value: &str) -> LabelPair {
        let mut l = LabelPair::default();
        l.set_name(name.to_string());
        l.set_value(value.to_string());
        l
    }

    fn counter_family(name: &str, help: &str, points: &[(Vec<LabelPair>, f64)]) -> MetricFamily {
        let mut fam = MetricFamily::default();
        fam.set_name(name.to_string());
        fam.set_help(help.to_string());
        fam.set_field_type(MetricType::COUNTER);
        let metrics: Vec<PromMetric> = points
            .iter()
            .map(|(labels, value)| {
                let mut m = PromMetric::default();
                m.set_label(labels.clone().into());
                let mut c = Counter::default();
                c.set_value(*value);
                m.set_counter(c);
                m
            })
            .collect();
        fam.set_metric(metrics.into());
        fam
    }

    #[test]
    fn translate_labels_renames_known_and_prefixes_unknown() {
        let labels = vec![
            label("pod", "demo-0"),
            label("gpu", "0"),
            label("library", "libcudart.so"),
        ];
        let kv = translate_labels(&labels);
        let keys: Vec<String> = kv.iter().map(|k| k.key.as_str().to_string()).collect();
        assert_eq!(keys, vec!["k8s.pod.name", "gpu.id", "profi.library"]);
    }

    #[test]
    fn counter_family_to_otlp_produces_monotonic_cumulative_sum() {
        let start = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000);
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(2_000);
        let fam = counter_family(
            "profi_cuda_calls_total",
            "Total CUDA API calls intercepted",
            &[
                (vec![label("pod", "a"), label("operation", "launch")], 3.0),
                (vec![label("pod", "b"), label("operation", "launch")], 7.5),
            ],
        );

        let metric = counter_family_to_otlp(&fam, start, now).expect("counter converted");
        assert_eq!(metric.name.as_ref(), "profi_cuda_calls_total");
        assert_eq!(
            metric.description.as_ref(),
            "Total CUDA API calls intercepted"
        );

        let sum = metric
            .data
            .as_any()
            .downcast_ref::<Sum<f64>>()
            .expect("Sum<f64>");
        assert!(sum.is_monotonic);
        assert_eq!(sum.temporality, Temporality::Cumulative);
        assert_eq!(sum.data_points.len(), 2);

        let dp = &sum.data_points[0];
        assert_eq!(dp.start_time, Some(start));
        assert_eq!(dp.time, Some(now));
        assert_eq!(dp.value, 3.0);

        let pod_attr = dp
            .attributes
            .iter()
            .find(|kv| kv.key.as_str() == "k8s.pod.name")
            .expect("pod label renamed");
        assert_eq!(pod_attr.value.as_str(), "a");

        let op_attr = dp
            .attributes
            .iter()
            .find(|kv| kv.key.as_str() == "profi.operation")
            .expect("operation label mapped");
        assert_eq!(op_attr.value.as_str(), "launch");
    }

    #[test]
    fn non_counter_family_returns_none() {
        let mut fam = MetricFamily::default();
        fam.set_name("some_gauge".to_string());
        fam.set_field_type(MetricType::GAUGE);
        let now = SystemTime::now();
        assert!(counter_family_to_otlp(&fam, now, now).is_none());
    }

    #[test]
    fn counter_family_with_no_points_returns_none() {
        let fam = counter_family("empty", "none", &[]);
        let now = SystemTime::now();
        assert!(counter_family_to_otlp(&fam, now, now).is_none());
    }

    fn gauge_family(name: &str, help: &str, points: &[(Vec<LabelPair>, f64)]) -> MetricFamily {
        let mut fam = MetricFamily::default();
        fam.set_name(name.to_string());
        fam.set_help(help.to_string());
        fam.set_field_type(MetricType::GAUGE);
        let metrics: Vec<PromMetric> = points
            .iter()
            .map(|(labels, value)| {
                let mut m = PromMetric::default();
                m.set_label(labels.clone().into());
                let mut g = PromGauge::default();
                g.set_value(*value);
                m.set_gauge(g);
                m
            })
            .collect();
        fam.set_metric(metrics.into());
        fam
    }

    #[test]
    fn gauge_family_to_otlp_produces_non_monotonic_gauge() {
        let start = SystemTime::UNIX_EPOCH + Duration::from_secs(10);
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(20);
        let fam = gauge_family(
            "profi_gpu_utilization_percent",
            "GPU SM utilization",
            &[
                (vec![label("gpu", "0")], 42.0),
                (vec![label("gpu", "1")], 88.5),
            ],
        );

        let metric = gauge_family_to_otlp(&fam, start, now).expect("gauge converted");
        assert_eq!(metric.name.as_ref(), "profi_gpu_utilization_percent");

        let gauge = metric
            .data
            .as_any()
            .downcast_ref::<Gauge<f64>>()
            .expect("Gauge<f64>");
        assert_eq!(gauge.data_points.len(), 2);
        let dp = &gauge.data_points[1];
        assert_eq!(dp.value, 88.5);
        assert_eq!(
            dp.attributes
                .iter()
                .find(|kv| kv.key.as_str() == "gpu.id")
                .unwrap()
                .value
                .as_str(),
            "1"
        );
    }

    #[test]
    fn non_gauge_family_returns_none_for_gauge_fn() {
        let fam = counter_family("some_counter", "help", &[(vec![], 1.0)]);
        let now = SystemTime::now();
        assert!(gauge_family_to_otlp(&fam, now, now).is_none());
    }

    fn bucket(upper: f64, cum: u64) -> Bucket {
        let mut b = Bucket::default();
        b.set_upper_bound(upper);
        b.set_cumulative_count(cum);
        b
    }

    fn hist(buckets: Vec<Bucket>, sample_count: u64, sample_sum: f64) -> PromHistogram {
        let mut h = PromHistogram::default();
        h.set_bucket(buckets.into());
        h.set_sample_count(sample_count);
        h.set_sample_sum(sample_sum);
        h
    }

    #[test]
    fn histogram_converts_cumulative_to_delta_with_implicit_plus_inf() {
        // Observations: 0.05, 0.5, 2.0, 2.0, 15.0 → sum = 19.55, count = 5
        // Buckets (cumulative): ≤0.1 → 1, ≤1.0 → 2, ≤10.0 → 4, (+Inf) → 5 implicit
        let h = hist(
            vec![bucket(0.1, 1), bucket(1.0, 2), bucket(10.0, 4)],
            5,
            19.55,
        );
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1);
        let dp = histogram_to_otlp(&h, vec![], now, now);

        assert_eq!(dp.bounds, vec![0.1, 1.0, 10.0]);
        assert_eq!(dp.bucket_counts, vec![1, 1, 2, 1]);
        assert_eq!(dp.bucket_counts.iter().sum::<u64>(), dp.count);
        assert_eq!(dp.count, 5);
        assert!((dp.sum - 19.55).abs() < 1e-9);
        assert!(dp.min.is_none() && dp.max.is_none());
    }

    #[test]
    fn histogram_drops_explicit_plus_inf_bucket() {
        // Same as above but the source includes an explicit +Inf bucket.
        // Result must be identical — +Inf is filtered and re-implied by the
        // trailing bucket_counts entry.
        let h = hist(
            vec![
                bucket(0.1, 1),
                bucket(1.0, 2),
                bucket(10.0, 4),
                bucket(f64::INFINITY, 5),
            ],
            5,
            19.55,
        );
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1);
        let dp = histogram_to_otlp(&h, vec![], now, now);

        assert_eq!(dp.bounds, vec![0.1, 1.0, 10.0]);
        assert_eq!(dp.bucket_counts, vec![1, 1, 2, 1]);
    }

    #[test]
    fn histogram_with_overflow_into_plus_inf() {
        // Last finite bucket has cum=3 but sample_count=10 — 7 obs > 10.0.
        let h = hist(vec![bucket(10.0, 3)], 10, 0.0);
        let now = SystemTime::UNIX_EPOCH;
        let dp = histogram_to_otlp(&h, vec![], now, now);

        assert_eq!(dp.bounds, vec![10.0]);
        assert_eq!(dp.bucket_counts, vec![3, 7]);
        assert_eq!(dp.bucket_counts.iter().sum::<u64>(), 10);
    }

    #[test]
    fn histogram_family_to_otlp_wraps_data_points() {
        let mut fam = MetricFamily::default();
        fam.set_name("profi_cuda_duration_seconds".to_string());
        fam.set_help("CUDA API call duration".to_string());
        fam.set_field_type(MetricType::HISTOGRAM);

        let mut m = PromMetric::default();
        m.set_label(vec![label("gpu", "0")].into());
        m.set_histogram(hist(vec![bucket(0.001, 2), bucket(0.01, 5)], 6, 0.015));
        fam.set_metric(vec![m].into());

        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(42);
        let metric = histogram_family_to_otlp(&fam, now, now).expect("histogram");
        let hist_data = metric
            .data
            .as_any()
            .downcast_ref::<Histogram<f64>>()
            .expect("Histogram<f64>");
        assert_eq!(hist_data.temporality, Temporality::Cumulative);
        assert_eq!(hist_data.data_points.len(), 1);
        let dp = &hist_data.data_points[0];
        assert_eq!(dp.bounds, vec![0.001, 0.01]);
        assert_eq!(dp.bucket_counts, vec![2, 3, 1]);
        assert_eq!(
            dp.attributes
                .iter()
                .find(|kv| kv.key.as_str() == "gpu.id")
                .unwrap()
                .value
                .as_str(),
            "0"
        );
    }

    #[test]
    fn non_histogram_returns_none() {
        let fam = counter_family("x", "y", &[(vec![], 1.0)]);
        let now = SystemTime::now();
        assert!(histogram_family_to_otlp(&fam, now, now).is_none());
    }

    #[test]
    fn skip_prefix_matches_self_observability_metrics() {
        assert!(should_skip_metric(
            "profi_system_prometheus_encode_duration_seconds"
        ));
        assert!(should_skip_metric("profi_system_uptime_seconds"));
        assert!(should_skip_metric("profi_system_metric_handle_cache_size"));
        assert!(should_skip_metric("profi_system_http_auth_failures_total"));
        // Any future metric under the prefix is excluded automatically.
        assert!(should_skip_metric("profi_system_some_future_metric"));
        assert!(!should_skip_metric("profi_cuda_calls_total"));
        assert!(!should_skip_metric("profi_nccl_straggler_seconds"));
        assert!(!should_skip_metric("profi_tracked_pids"));
        assert!(!should_skip_metric("profi_dropped_events_total"));
        assert!(!should_skip_metric(""));
    }

    #[test]
    fn build_resource_has_service_name_and_version() {
        let cfg = OtlpConfig {
            endpoint: "http://x".to_string(),
            protocol: OtlpProtocol::Grpc,
            interval: Duration::from_secs(60),
            timeout: Duration::from_secs(10),
            service_name: "test-profi".to_string(),
            headers: vec![],
            ca_cert_path: None,
            client_cert_path: None,
            client_key_path: None,
            insecure: false,
            resource_attrs: vec![],
        };
        let resource = build_resource(&cfg);
        let svc = resource
            .get(opentelemetry::Key::new("service.name"))
            .expect("service.name present");
        assert_eq!(svc.as_str(), "test-profi");
        assert!(resource
            .get(opentelemetry::Key::new("service.version"))
            .is_some());
    }

    #[test]
    fn parse_kv_list_handles_whitespace_and_empty_segments() {
        let kv = parse_kv_list("a=1, b = 2 ,, c=x=y,=bad");
        assert_eq!(
            kv,
            vec![
                ("a".to_string(), "1".to_string()),
                ("b".to_string(), "2".to_string()),
                ("c".to_string(), "x=y".to_string()),
            ]
        );
    }

    #[test]
    fn resolve_returns_none_when_endpoint_empty() {
        let args = OtlpArgs {
            otlp_endpoint: None,
            otlp_protocol: "grpc".to_string(),
            otlp_interval_secs: 60,
            otlp_timeout_secs: 10,
            otlp_service_name: "profi".to_string(),
            otlp_headers: None,
            otlp_ca_cert: None,
            otlp_client_cert: None,
            otlp_client_key: None,
            otlp_insecure: false,
            otlp_resource_attrs: None,
        };
        assert!(OtlpConfig::resolve(&args).unwrap().is_none());

        let mut args = args.clone();
        args.otlp_endpoint = Some("   ".to_string());
        assert!(OtlpConfig::resolve(&args).unwrap().is_none());
    }

    #[test]
    fn resolve_picks_http_protocol_and_parses_headers() {
        let args = OtlpArgs {
            otlp_endpoint: Some("http://collector:4318".to_string()),
            otlp_protocol: "http/protobuf".to_string(),
            otlp_interval_secs: 30,
            otlp_timeout_secs: 5,
            otlp_service_name: "svc".to_string(),
            otlp_headers: Some("Authorization=Bearer tok,X-Team=ml".to_string()),
            otlp_ca_cert: None,
            otlp_client_cert: None,
            otlp_client_key: None,
            otlp_insecure: false,
            otlp_resource_attrs: Some("deployment.environment=prod".to_string()),
        };
        let cfg = OtlpConfig::resolve(&args).unwrap().unwrap();
        assert_eq!(cfg.protocol, OtlpProtocol::HttpProtobuf);
        assert_eq!(cfg.interval, Duration::from_secs(30));
        assert_eq!(cfg.timeout, Duration::from_secs(5));
        assert_eq!(cfg.headers.len(), 2);
        assert_eq!(
            cfg.headers[0],
            ("Authorization".into(), "Bearer tok".into())
        );
        assert_eq!(cfg.resource_attrs.len(), 1);
    }

    #[test]
    fn resolve_rejects_unknown_protocol() {
        let args = OtlpArgs {
            otlp_endpoint: Some("http://x".to_string()),
            otlp_protocol: "thrift".to_string(),
            otlp_interval_secs: 60,
            otlp_timeout_secs: 10,
            otlp_service_name: "profi".to_string(),
            otlp_headers: None,
            otlp_ca_cert: None,
            otlp_client_cert: None,
            otlp_client_key: None,
            otlp_insecure: false,
            otlp_resource_attrs: None,
        };
        assert!(OtlpConfig::resolve(&args).is_err());
    }

    #[test]
    fn build_resource_user_attrs_override_defaults() {
        let cfg = OtlpConfig {
            endpoint: "http://x".to_string(),
            protocol: OtlpProtocol::Grpc,
            interval: Duration::from_secs(60),
            timeout: Duration::from_secs(10),
            service_name: "default".to_string(),
            headers: vec![],
            ca_cert_path: None,
            client_cert_path: None,
            client_key_path: None,
            insecure: false,
            resource_attrs: vec![
                ("service.name".to_string(), "override".to_string()),
                ("deployment.environment".to_string(), "prod".to_string()),
            ],
        };
        let resource = build_resource(&cfg);
        assert_eq!(
            resource
                .get(opentelemetry::Key::new("service.name"))
                .unwrap()
                .as_str(),
            "override"
        );
        assert_eq!(
            resource
                .get(opentelemetry::Key::new("deployment.environment"))
                .unwrap()
                .as_str(),
            "prod"
        );
    }
}
