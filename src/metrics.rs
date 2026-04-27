// SPDX-License-Identifier: Apache-2.0

use std::borrow::Cow;
use std::sync::Arc;

use anyhow::Result;
use arrayvec::ArrayVec;
use lasso::Spur;
use prometheus::{
    opts, CounterVec, Encoder, GaugeVec, HistogramOpts, HistogramVec, IntGauge, Registry,
    TextEncoder,
};

use crate::enrichment::GpuDevice;
use crate::events::*;

// ── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, clap::ValueEnum)]
pub enum KernelMode {
    /// Full kernel tracing with names, histograms, classification (highest overhead)
    Full,
    /// Anonymous aggregated kernel count + duration (low overhead)
    Anonymous,
    /// No kernel tracing at all (zero overhead)
    Off,
}

// ── Label sets ──────────────────────────────────────────────────────────────

pub const COUNTER_LABELS: &[&str] = &[
    "operation",
    "pid",
    "comm",
    "namespace",
    "pod",
    "container",
    "gpu",
    "gpu_uuid",
    "stream",
];
pub const HIST_LABELS: &[&str] = &["operation", "namespace", "pod", "gpu"];
pub const HIST_BUCKET_LABELS: &[&str] = &["operation", "namespace", "pod", "gpu", "le"];
pub const MEMCPY_COUNTER_LABELS: &[&str] = &[
    "direction",
    "pid",
    "comm",
    "namespace",
    "pod",
    "container",
    "gpu",
    "gpu_uuid",
    "stream",
];
pub const MALLOC_LABELS: &[&str] = &[
    "pid",
    "comm",
    "namespace",
    "pod",
    "container",
    "gpu",
    "gpu_uuid",
];

pub const NCCL_COUNTER_LABELS: &[&str] = &[
    "operation",
    "pid",
    "comm",
    "namespace",
    "pod",
    "container",
    "gpu",
    "gpu_uuid",
];
pub const NCCL_HIST_LABELS: &[&str] = &["operation", "namespace", "pod", "gpu"];
pub const NCCL_HIST_BUCKET_LABELS: &[&str] = &["operation", "namespace", "pod", "gpu", "le"];
pub const NCCL_BYTES_LABELS: &[&str] = &[
    "operation",
    "pid",
    "comm",
    "namespace",
    "pod",
    "container",
    "gpu",
    "gpu_uuid",
];

pub const KERNEL_COUNTER_LABELS: &[&str] = &[
    "kernel",
    "pid",
    "comm",
    "namespace",
    "pod",
    "container",
    "gpu",
    "gpu_uuid",
    "kernel_class",
    "phase",
];
pub const KERNEL_HIST_LABELS: &[&str] =
    &["kernel", "namespace", "pod", "gpu", "kernel_class", "phase"];
pub const KERNEL_HIST_BUCKET_LABELS: &[&str] = &[
    "kernel",
    "namespace",
    "pod",
    "gpu",
    "kernel_class",
    "phase",
    "le",
];
pub const CUDA_DURATION_BUCKET_LE: [&str; 14] = [
    "0.000001", "0.000005", "0.00001", "0.00005", "0.0001", "0.0005", "0.001", "0.005", "0.01",
    "0.05", "0.1", "0.5", "1", "+Inf",
];
pub const NCCL_DURATION_BUCKET_LE: [&str; 12] = [
    "0.00001", "0.00005", "0.0001", "0.0005", "0.001", "0.005", "0.01", "0.05", "0.1", "0.5", "1",
    "+Inf",
];
pub const CUDA_KERNEL_DURATION_BUCKET_LE: [&str; 9] = [
    "0.000001", "0.000005", "0.00001", "0.00005", "0.0001", "0.0005", "0.001", "0.01", "+Inf",
];
pub const ERROR_LABELS: &[&str] = &[
    "operation",
    "pid",
    "comm",
    "namespace",
    "pod",
    "container",
    "gpu",
    "gpu_uuid",
    "error_code",
];
pub const ACTIVE_MEM_LABELS: &[&str] = &[
    "pid",
    "comm",
    "namespace",
    "pod",
    "container",
    "gpu",
    "gpu_uuid",
];

// ── Metrics struct ──────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct Metrics {
    pub cuda_calls: CounterVec,
    pub cuda_duration: HistogramVec,
    pub cuda_duration_bucket_total: CounterVec,
    pub cuda_duration_sum_seconds_total: CounterVec,
    pub cuda_duration_count_total: CounterVec,
    pub cuda_memcpy_bytes: CounterVec,
    pub cuda_malloc_bytes: CounterVec,
    pub cuda_errors: CounterVec,
    pub cuda_active_memory: prometheus::IntGaugeVec,
    pub nccl_calls: CounterVec,
    pub nccl_duration: HistogramVec,
    pub nccl_duration_bucket_total: CounterVec,
    pub nccl_duration_sum_seconds_total: CounterVec,
    pub nccl_duration_count_total: CounterVec,
    pub nccl_bytes: CounterVec,
    pub cuda_kernel_launches: CounterVec,
    pub cuda_kernel_duration: Option<HistogramVec>,
    pub cuda_kernel_duration_bucket_total: Option<CounterVec>,
    pub cuda_kernel_duration_sum_seconds_total: Option<CounterVec>,
    pub cuda_kernel_duration_count_total: Option<CounterVec>,
    pub gpu_info: GaugeVec,
    pub tracked_pids: IntGauge,
    pub dropped_events: prometheus::IntCounter,
    // Self-observability metrics
    pub discovery_scan_duration: prometheus::Histogram,
    pub discovery_attached_libs: prometheus::IntGaugeVec,
    pub event_loop_duration: prometheus::Histogram,
    pub agg_drain_duration: prometheus::Histogram,
    pub handle_cache_size: IntGauge,
    pub encode_duration: prometheus::Histogram,
    pub uptime: prometheus::Gauge,
    pub cardinality_drops: prometheus::IntCounter,
    pub ring_buffer_drops_rate: prometheus::Gauge,
    pub kernel_name_resolve_failures: prometheus::IntCounterVec,
    pub launch_agg_drops: prometheus::IntCounter,
    // GPU hardware health metrics (NVML)
    pub gpu_temperature: prometheus::GaugeVec,
    pub gpu_power: prometheus::GaugeVec,
    pub gpu_clock: prometheus::GaugeVec,
    pub gpu_utilization: prometheus::GaugeVec,
    pub gpu_memory: prometheus::GaugeVec,
    pub gpu_ecc_errors: prometheus::IntCounterVec,
    pub gpu_throttle: prometheus::GaugeVec,
    // NCCL hang detection
    pub nccl_hang_detected: prometheus::IntCounterVec,
    pub nccl_stale_entries: IntGauge,
    // Straggler detection
    pub nccl_straggler_ratio: prometheus::GaugeVec,
    // HTTP server TLS / auth (self-obs)
    pub http_auth_success: prometheus::IntCounterVec,
    pub http_auth_failures: prometheus::IntCounterVec,
    pub http_tokenreview_cache: prometheus::IntCounterVec,
    pub http_tokenreview_latency: prometheus::Histogram,
    pub http_tls_handshakes: prometheus::IntCounterVec,
    pub registry: Registry,
    /// Pre-allocated buffer reused across Prometheus scrapes (Fix 4).
    pub encode_buf: Arc<std::sync::Mutex<Vec<u8>>>,
}

impl Metrics {
    pub fn new(kernel_mode: KernelMode) -> Result<Self> {
        let registry = Registry::new();

        let cuda_calls = CounterVec::new(
            opts!("profi_cuda_calls_total", "Total CUDA Runtime API calls"),
            COUNTER_LABELS,
        )?;
        let cuda_duration = HistogramVec::new(
            HistogramOpts::new("profi_cuda_duration_seconds", "CUDA API call latency").buckets(
                vec![
                    1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1, 0.5, 1.0,
                ],
            ),
            HIST_LABELS,
        )?;
        let cuda_duration_bucket_total = CounterVec::new(
            opts!(
                "profi_cuda_duration_bucket_total",
                "Cumulative CUDA API latency bucket counts from aggregate eBPF maps"
            ),
            HIST_BUCKET_LABELS,
        )?;
        let cuda_duration_sum_seconds_total = CounterVec::new(
            opts!(
                "profi_cuda_duration_sum_seconds_total",
                "Cumulative CUDA API latency sum from aggregate eBPF maps"
            ),
            HIST_LABELS,
        )?;
        let cuda_duration_count_total = CounterVec::new(
            opts!(
                "profi_cuda_duration_count_total",
                "Cumulative CUDA API latency sample count from aggregate eBPF maps"
            ),
            HIST_LABELS,
        )?;
        let cuda_memcpy_bytes = CounterVec::new(
            opts!(
                "profi_cuda_memcpy_bytes_total",
                "Bytes transferred via cudaMemcpy/cudaMemcpyAsync"
            ),
            MEMCPY_COUNTER_LABELS,
        )?;
        let cuda_malloc_bytes = CounterVec::new(
            opts!(
                "profi_cuda_malloc_bytes_total",
                "Bytes allocated via cudaMalloc"
            ),
            MALLOC_LABELS,
        )?;
        let cuda_errors = CounterVec::new(
            opts!(
                "profi_cuda_errors_total",
                "CUDA/NCCL API calls returning non-zero error codes"
            ),
            ERROR_LABELS,
        )?;
        let cuda_active_memory = prometheus::IntGaugeVec::new(
            opts!(
                "profi_cuda_active_memory_bytes",
                "Net GPU memory allocated (cudaMalloc minus cudaFree) since profiler start; \
                 may be negative if allocations preceded profiler attach"
            ),
            ACTIVE_MEM_LABELS,
        )?;
        let nccl_calls = CounterVec::new(
            opts!("profi_nccl_calls_total", "Total NCCL collective calls"),
            NCCL_COUNTER_LABELS,
        )?;
        let nccl_duration = HistogramVec::new(
            HistogramOpts::new(
                "profi_nccl_duration_seconds",
                "NCCL collective call latency",
            )
            .buckets(vec![
                1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1, 0.5, 1.0,
            ]),
            NCCL_HIST_LABELS,
        )?;
        let nccl_duration_bucket_total = CounterVec::new(
            opts!(
                "profi_nccl_duration_bucket_total",
                "Cumulative NCCL latency bucket counts from aggregate eBPF maps"
            ),
            NCCL_HIST_BUCKET_LABELS,
        )?;
        let nccl_duration_sum_seconds_total = CounterVec::new(
            opts!(
                "profi_nccl_duration_sum_seconds_total",
                "Cumulative NCCL latency sum from aggregate eBPF maps"
            ),
            NCCL_HIST_LABELS,
        )?;
        let nccl_duration_count_total = CounterVec::new(
            opts!(
                "profi_nccl_duration_count_total",
                "Cumulative NCCL latency sample count from aggregate eBPF maps"
            ),
            NCCL_HIST_LABELS,
        )?;
        let nccl_bytes = CounterVec::new(
            opts!(
                "profi_nccl_bytes_total",
                "Bytes transferred via NCCL collectives"
            ),
            NCCL_BYTES_LABELS,
        )?;
        let cuda_kernel_launches = CounterVec::new(
            opts!(
                "profi_cuda_kernel_launches_total",
                "CUDA kernel launches by kernel name"
            ),
            KERNEL_COUNTER_LABELS,
        )?;
        let (
            cuda_kernel_duration,
            cuda_kernel_duration_bucket_total,
            cuda_kernel_duration_sum_seconds_total,
            cuda_kernel_duration_count_total,
        ) = if kernel_mode == KernelMode::Full {
            let h = HistogramVec::new(
                HistogramOpts::new(
                    "profi_cuda_kernel_duration_seconds",
                    "CUDA kernel launch latency by kernel name",
                )
                .buckets(vec![1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 1e-2]),
                KERNEL_HIST_LABELS,
            )?;
            let buckets = CounterVec::new(
                opts!(
                    "profi_cuda_kernel_duration_bucket_total",
                    "Cumulative CUDA kernel launch latency bucket counts from aggregate eBPF maps"
                ),
                KERNEL_HIST_BUCKET_LABELS,
            )?;
            let sum = CounterVec::new(
                opts!(
                    "profi_cuda_kernel_duration_sum_seconds_total",
                    "Cumulative CUDA kernel launch latency sum from aggregate eBPF maps"
                ),
                KERNEL_HIST_LABELS,
            )?;
            let count = CounterVec::new(
                opts!(
                    "profi_cuda_kernel_duration_count_total",
                    "Cumulative CUDA kernel launch latency sample count from aggregate eBPF maps"
                ),
                KERNEL_HIST_LABELS,
            )?;
            (Some(h), Some(buckets), Some(sum), Some(count))
        } else {
            (None, None, None, None)
        };

        let gpu_info = GaugeVec::new(
            opts!("profi_gpu_info", "GPU device information (always 1)"),
            &["gpu", "gpu_uuid", "gpu_model"],
        )?;
        let tracked_pids = IntGauge::new(
            "profi_tracked_pids",
            "Number of distinct CUDA processes observed",
        )?;
        let dropped_events = prometheus::IntCounter::new(
            "profi_dropped_events_total",
            "Events dropped due to eBPF RingBuf overflow",
        )?;

        // Self-observability metrics
        let discovery_scan_duration = prometheus::Histogram::with_opts(
            HistogramOpts::new(
                "profi_system_discovery_scan_duration_seconds",
                "Time spent scanning /proc for CUDA libraries",
            )
            .buckets(vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]),
        )?;
        let discovery_attached_libs = prometheus::IntGaugeVec::new(
            opts!(
                "profi_system_discovery_attached_libraries",
                "Number of attached library instances by library type"
            ),
            &["library"],
        )?;
        let event_loop_duration = prometheus::Histogram::with_opts(
            HistogramOpts::new(
                "profi_system_event_loop_process_duration_seconds",
                "Time spent processing events in one ring buffer batch",
            )
            .buckets(vec![1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]),
        )?;
        let agg_drain_duration = prometheus::Histogram::with_opts(
            HistogramOpts::new(
                "profi_system_aggregated_map_drain_duration_seconds",
                "Time spent draining the eBPF aggregated PerCpuHashMap",
            )
            .buckets(vec![1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 0.01]),
        )?;
        let handle_cache_size = IntGauge::new(
            "profi_system_metric_handle_cache_size",
            "Number of entries in the metric handle cache",
        )?;
        let encode_duration = prometheus::Histogram::with_opts(
            HistogramOpts::new(
                "profi_system_prometheus_encode_duration_seconds",
                "Time spent encoding Prometheus metrics text format",
            )
            .buckets(vec![1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1]),
        )?;
        let uptime = prometheus::Gauge::new(
            "profi_system_uptime_seconds",
            "Seconds since profi process started",
        )?;
        let cardinality_drops = prometheus::IntCounter::new(
            "profi_cardinality_limit_drops_total",
            "Events dropped due to cardinality limit (--max-time-series exceeded)",
        )?;
        let ring_buffer_drops_rate = prometheus::Gauge::new(
            "profi_system_ring_buffer_drops_rate",
            "Ring buffer event drops per second over the last drain interval",
        )?;
        let kernel_name_resolve_failures = prometheus::IntCounterVec::new(
            opts!(
                "profi_system_kernel_name_resolve_failures_total",
                "Lazy kernel-name reads from /proc/<pid>/mem that failed, by reason"
            ),
            &["reason"],
        )?;
        let launch_agg_drops = prometheus::IntCounter::new(
            "profi_system_launch_agg_drops_total",
            "cuLaunchKernel events dropped because LAUNCH_AGG eBPF map was full",
        )?;

        // GPU hardware health metrics (populated by NVML poller)
        let gpu_temperature = prometheus::GaugeVec::new(
            opts!("profi_gpu_temperature_celsius", "GPU die temperature"),
            &["gpu", "gpu_uuid"],
        )?;
        let gpu_power = prometheus::GaugeVec::new(
            opts!("profi_gpu_power_watts", "GPU power draw"),
            &["gpu", "gpu_uuid"],
        )?;
        let gpu_clock = prometheus::GaugeVec::new(
            opts!("profi_gpu_clock_mhz", "GPU clock speed"),
            &["gpu", "gpu_uuid", "clock_type"],
        )?;
        let gpu_utilization = prometheus::GaugeVec::new(
            opts!("profi_gpu_utilization_ratio", "GPU utilization (0-1)"),
            &["gpu", "gpu_uuid", "type"],
        )?;
        let gpu_memory = prometheus::GaugeVec::new(
            opts!("profi_gpu_memory_bytes", "GPU VRAM usage"),
            &["gpu", "gpu_uuid", "state"],
        )?;
        let gpu_ecc_errors = prometheus::IntCounterVec::new(
            opts!("profi_gpu_ecc_errors_total", "GPU ECC memory errors"),
            &["gpu", "gpu_uuid", "error_type", "location"],
        )?;
        let gpu_throttle = prometheus::GaugeVec::new(
            opts!(
                "profi_gpu_throttle_active",
                "GPU throttle reasons currently active (1=active)"
            ),
            &["gpu", "gpu_uuid", "reason"],
        )?;

        // NCCL hang detection
        let nccl_hang_detected = prometheus::IntCounterVec::new(
            opts!(
                "profi_nccl_hang_detected_total",
                "NCCL collective operations exceeding hang timeout"
            ),
            &["operation", "pid"],
        )?;
        let nccl_stale_entries = IntGauge::new(
            "profi_nccl_stale_entries",
            "Number of in-flight NCCL entries currently exceeding hang timeout",
        )?;

        // Straggler detection
        let nccl_straggler_ratio = prometheus::GaugeVec::new(
            opts!(
                "profi_nccl_straggler_ratio",
                "Ratio of this GPU NCCL latency to group median (>1.5 = straggler)"
            ),
            &["pid", "gpu", "gpu_uuid", "operation"],
        )?;

        // HTTP server TLS / auth self-observability
        let http_auth_success = prometheus::IntCounterVec::new(
            opts!(
                "profi_system_http_auth_success_total",
                "Successful /metrics authentications by method"
            ),
            &["method"],
        )?;
        let http_auth_failures = prometheus::IntCounterVec::new(
            opts!(
                "profi_system_http_auth_failures_total",
                "Rejected /metrics requests by reason"
            ),
            &["reason"],
        )?;
        let http_tokenreview_cache = prometheus::IntCounterVec::new(
            opts!(
                "profi_system_http_tokenreview_cache_total",
                "Kubernetes TokenReview cache outcomes"
            ),
            &["result"],
        )?;
        let http_tokenreview_latency = prometheus::Histogram::with_opts(
            HistogramOpts::new(
                "profi_system_http_tokenreview_latency_seconds",
                "Latency of TokenReview calls to the Kubernetes API",
            )
            .buckets(vec![1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5, 1.0, 5.0]),
        )?;
        let http_tls_handshakes = prometheus::IntCounterVec::new(
            opts!(
                "profi_system_http_tls_handshakes_total",
                "TLS handshakes on the /metrics endpoint"
            ),
            &["result"],
        )?;

        registry.register(Box::new(cuda_calls.clone()))?;
        registry.register(Box::new(cuda_duration.clone()))?;
        registry.register(Box::new(cuda_duration_bucket_total.clone()))?;
        registry.register(Box::new(cuda_duration_sum_seconds_total.clone()))?;
        registry.register(Box::new(cuda_duration_count_total.clone()))?;
        registry.register(Box::new(cuda_memcpy_bytes.clone()))?;
        registry.register(Box::new(cuda_malloc_bytes.clone()))?;
        registry.register(Box::new(cuda_errors.clone()))?;
        registry.register(Box::new(cuda_active_memory.clone()))?;
        registry.register(Box::new(nccl_calls.clone()))?;
        registry.register(Box::new(nccl_duration.clone()))?;
        registry.register(Box::new(nccl_duration_bucket_total.clone()))?;
        registry.register(Box::new(nccl_duration_sum_seconds_total.clone()))?;
        registry.register(Box::new(nccl_duration_count_total.clone()))?;
        registry.register(Box::new(nccl_bytes.clone()))?;
        registry.register(Box::new(cuda_kernel_launches.clone()))?;
        if let Some(ref h) = cuda_kernel_duration {
            registry.register(Box::new(h.clone()))?;
        }
        if let Some(ref h) = cuda_kernel_duration_bucket_total {
            registry.register(Box::new(h.clone()))?;
        }
        if let Some(ref h) = cuda_kernel_duration_sum_seconds_total {
            registry.register(Box::new(h.clone()))?;
        }
        if let Some(ref h) = cuda_kernel_duration_count_total {
            registry.register(Box::new(h.clone()))?;
        }
        registry.register(Box::new(gpu_info.clone()))?;
        registry.register(Box::new(tracked_pids.clone()))?;
        registry.register(Box::new(dropped_events.clone()))?;
        registry.register(Box::new(discovery_scan_duration.clone()))?;
        registry.register(Box::new(discovery_attached_libs.clone()))?;
        registry.register(Box::new(event_loop_duration.clone()))?;
        registry.register(Box::new(agg_drain_duration.clone()))?;
        registry.register(Box::new(handle_cache_size.clone()))?;
        registry.register(Box::new(encode_duration.clone()))?;
        registry.register(Box::new(uptime.clone()))?;
        registry.register(Box::new(cardinality_drops.clone()))?;
        registry.register(Box::new(ring_buffer_drops_rate.clone()))?;
        registry.register(Box::new(kernel_name_resolve_failures.clone()))?;
        registry.register(Box::new(launch_agg_drops.clone()))?;
        registry.register(Box::new(gpu_temperature.clone()))?;
        registry.register(Box::new(gpu_power.clone()))?;
        registry.register(Box::new(gpu_clock.clone()))?;
        registry.register(Box::new(gpu_utilization.clone()))?;
        registry.register(Box::new(gpu_memory.clone()))?;
        registry.register(Box::new(gpu_ecc_errors.clone()))?;
        registry.register(Box::new(gpu_throttle.clone()))?;
        registry.register(Box::new(nccl_hang_detected.clone()))?;
        registry.register(Box::new(nccl_stale_entries.clone()))?;
        registry.register(Box::new(nccl_straggler_ratio.clone()))?;
        registry.register(Box::new(http_auth_success.clone()))?;
        registry.register(Box::new(http_auth_failures.clone()))?;
        registry.register(Box::new(http_tokenreview_cache.clone()))?;
        registry.register(Box::new(http_tokenreview_latency.clone()))?;
        registry.register(Box::new(http_tls_handshakes.clone()))?;

        Ok(Self {
            cuda_calls,
            cuda_duration,
            cuda_duration_bucket_total,
            cuda_duration_sum_seconds_total,
            cuda_duration_count_total,
            cuda_memcpy_bytes,
            cuda_malloc_bytes,
            cuda_errors,
            cuda_active_memory,
            nccl_calls,
            nccl_duration,
            nccl_duration_bucket_total,
            nccl_duration_sum_seconds_total,
            nccl_duration_count_total,
            nccl_bytes,
            cuda_kernel_launches,
            cuda_kernel_duration,
            cuda_kernel_duration_bucket_total,
            cuda_kernel_duration_sum_seconds_total,
            cuda_kernel_duration_count_total,
            gpu_info,
            tracked_pids,
            dropped_events,
            discovery_scan_duration,
            discovery_attached_libs,
            event_loop_duration,
            agg_drain_duration,
            handle_cache_size,
            encode_duration,
            uptime,
            cardinality_drops,
            ring_buffer_drops_rate,
            kernel_name_resolve_failures,
            launch_agg_drops,
            gpu_temperature,
            gpu_power,
            gpu_clock,
            gpu_utilization,
            gpu_memory,
            gpu_ecc_errors,
            gpu_throttle,
            nccl_hang_detected,
            nccl_stale_entries,
            nccl_straggler_ratio,
            http_auth_success,
            http_auth_failures,
            http_tokenreview_cache,
            http_tokenreview_latency,
            http_tls_handshakes,
            registry,
            encode_buf: Arc::new(std::sync::Mutex::new(Vec::with_capacity(64 * 1024))),
        })
    }

    pub fn encode_bytes(&self) -> Vec<u8> {
        let start = std::time::Instant::now();
        let mut buf = self.encode_buf.lock().unwrap();
        buf.clear();
        TextEncoder::new()
            .encode(&self.registry.gather(), &mut *buf)
            .unwrap_or(());
        let result = buf.clone();
        drop(buf);
        self.encode_duration.observe(start.elapsed().as_secs_f64());
        result
    }

    pub fn publish_gpu_info(&self, devices: &[GpuDevice]) {
        for dev in devices {
            self.gpu_info
                .with_label_values(&[&dev.index.to_string(), &dev.uuid, &dev.name])
                .set(1.0);
        }
    }
}

// ── Utility functions ───────────────────────────────────────────────────────

pub fn operation_name(event_type: u32) -> &'static str {
    match event_type {
        EVENT_CUDA_MALLOC => "cudaMalloc",
        EVENT_CUDA_FREE => "cudaFree",
        EVENT_CUDA_MEMCPY => "cudaMemcpy",
        EVENT_CUDA_MEMCPY_ASYNC => "cudaMemcpyAsync",
        EVENT_CUDA_LAUNCH_KERNEL => "cudaLaunchKernel",
        EVENT_CUDA_MALLOC_ASYNC => "cudaMallocAsync",
        EVENT_CUDA_FREE_ASYNC => "cudaFreeAsync",
        EVENT_NCCL_ALL_REDUCE => "ncclAllReduce",
        EVENT_NCCL_ALL_GATHER => "ncclAllGather",
        EVENT_NCCL_REDUCE_SCATTER => "ncclReduceScatter",
        EVENT_NCCL_BROADCAST => "ncclBroadcast",
        EVENT_NCCL_SEND => "ncclSend",
        EVENT_NCCL_RECV => "ncclRecv",
        EVENT_CUDA_STREAM_SYNC => "cudaStreamSync",
        EVENT_CUDA_EVENT_SYNC => "cudaEventSync",
        EVENT_CUDA_MALLOC_HOST => "cudaMallocHost",
        EVENT_CUDA_FREE_HOST => "cudaFreeHost",
        EVENT_CUDA_MEMSET => "cudaMemset",
        EVENT_CUDA_MEMSET_ASYNC => "cudaMemsetAsync",
        EVENT_CUDA_GRAPH_LAUNCH => "cudaGraphLaunch",
        EVENT_CUDA_GRAPH_INSTANTIATE => "cudaGraphInstantiate",
        EVENT_CUDA_MODULE_LOAD => "cuModuleLoadData",
        _ => "unknown",
    }
}

// Shared with the BPF C header to keep NCCL routing semantics identical.
pub use crate::events::is_nccl_event;

pub fn nccl_event_bytes(event: &CudaEvent) -> u64 {
    let count = event.size;
    let dtype = event.memcpy_kind as usize;
    let dtype_size = NCCL_DTYPE_SIZES.get(dtype).copied().unwrap_or(1);
    count * dtype_size
}

pub fn stream_label(stream: u64) -> Cow<'static, str> {
    if stream == 0 {
        Cow::Borrowed("default")
    } else {
        Cow::Owned(format!("0x{:x}", stream))
    }
}

pub fn memcpy_direction(kind: u32) -> &'static str {
    match kind {
        MEMCPY_H2H => "h2h",
        MEMCPY_H2D => "h2d",
        MEMCPY_D2H => "d2h",
        MEMCPY_D2D => "d2d",
        _ if kind > 255 => "p2p", // packed (src_dev<<16)|dst_dev from cudaMemcpyPeer
        _ => "unknown",
    }
}

pub fn nvtx_hash(marker: &[u8; 16]) -> u64 {
    let mut h: u64 = 0;
    for &b in marker {
        h = h.wrapping_mul(31).wrapping_add(b as u64);
    }
    h
}

pub fn resolve_spurs<'a, const N: usize>(
    spurs: &ArrayVec<Spur, N>,
    interner: &'a lasso::ThreadedRodeo,
) -> ArrayVec<&'a str, N> {
    spurs.iter().map(|s| interner.resolve(s)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── operation_name ──────────────────────────────────────────────────

    #[test]
    fn operation_name_cuda_malloc() {
        assert_eq!(operation_name(EVENT_CUDA_MALLOC), "cudaMalloc");
    }

    #[test]
    fn operation_name_nccl() {
        assert_eq!(operation_name(EVENT_NCCL_ALL_REDUCE), "ncclAllReduce");
    }

    #[test]
    fn operation_name_unknown() {
        assert_eq!(operation_name(9999), "unknown");
    }

    #[test]
    fn operation_name_module_load() {
        assert_eq!(operation_name(EVENT_CUDA_MODULE_LOAD), "cuModuleLoadData");
    }

    // ── memcpy_direction ────────────────────────────────────────────────

    #[test]
    fn memcpy_h2h() {
        assert_eq!(memcpy_direction(MEMCPY_H2H), "h2h");
    }

    #[test]
    fn memcpy_h2d() {
        assert_eq!(memcpy_direction(MEMCPY_H2D), "h2d");
    }

    #[test]
    fn memcpy_d2h() {
        assert_eq!(memcpy_direction(MEMCPY_D2H), "d2h");
    }

    #[test]
    fn memcpy_d2d() {
        assert_eq!(memcpy_direction(MEMCPY_D2D), "d2d");
    }

    #[test]
    fn memcpy_p2p() {
        assert_eq!(memcpy_direction(256), "p2p");
    }

    #[test]
    fn memcpy_unknown() {
        assert_eq!(memcpy_direction(100), "unknown");
    }

    // ── is_nccl_event ───────────────────────────────────────────────────

    #[test]
    fn is_nccl_all_reduce() {
        assert!(is_nccl_event(EVENT_NCCL_ALL_REDUCE));
    }

    #[test]
    fn is_nccl_recv() {
        assert!(is_nccl_event(EVENT_NCCL_RECV));
    }

    #[test]
    fn is_not_nccl_malloc() {
        assert!(!is_nccl_event(EVENT_CUDA_MALLOC));
    }

    // ── nvtx_hash ───────────────────────────────────────────────────────

    #[test]
    fn nvtx_hash_zeros() {
        let h1 = nvtx_hash(&[0; 16]);
        let h2 = nvtx_hash(&[0; 16]);
        assert_eq!(h1, h2);
    }

    #[test]
    fn nvtx_hash_different() {
        let h1 = nvtx_hash(&[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let h2 = nvtx_hash(&[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_ne!(h1, h2);
    }

    // ── stream_label ────────────────────────────────────────────────────

    #[test]
    fn stream_label_default() {
        assert_eq!(stream_label(0).as_ref(), "default");
    }

    #[test]
    fn stream_label_hex() {
        assert_eq!(stream_label(0x1234).as_ref(), "0x1234");
    }

    // ── Metrics::new ────────────────────────────────────────────────────

    #[test]
    fn metrics_new_anonymous() {
        let m = Metrics::new(KernelMode::Anonymous).unwrap();
        assert!(m.cuda_kernel_duration.is_none());
    }

    #[test]
    fn metrics_new_full() {
        let m = Metrics::new(KernelMode::Full).unwrap();
        assert!(m.cuda_kernel_duration.is_some());
    }
}
