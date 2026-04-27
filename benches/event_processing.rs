// SPDX-License-Identifier: Apache-2.0

//! Benchmarks for Profi's hot paths.
//!
//! Run: cargo bench --package profi
//! Compare after changes: cargo bench --package profi -- --save-baseline before
//!                        <make changes>
//!                        cargo bench --package profi -- --baseline before

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use arrayvec::ArrayVec;
use lasso::ThreadedRodeo;
use prometheus::{opts, CounterVec, Encoder, HistogramOpts, HistogramVec, Registry};
use rustc_hash::FxHashMap;
use std::borrow::Cow;
use std::collections::HashMap;

// ── Simulate core data structures ──────────────────────────────────────────

const COUNTER_LABELS: &[&str] = &[
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
const HIST_LABELS: &[&str] = &["operation", "namespace", "pod", "gpu"];

fn setup_metrics() -> (Registry, CounterVec, HistogramVec) {
    let registry = Registry::new();
    let counter =
        CounterVec::new(opts!("bench_cuda_calls_total", "bench"), COUNTER_LABELS).unwrap();
    let histogram = HistogramVec::new(
        HistogramOpts::new("bench_cuda_duration_seconds", "bench").buckets(vec![
            1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1, 0.5, 1.0,
        ]),
        HIST_LABELS,
    )
    .unwrap();
    registry.register(Box::new(counter.clone())).unwrap();
    registry.register(Box::new(histogram.clone())).unwrap();
    (registry, counter, histogram)
}

fn setup_interner() -> ThreadedRodeo {
    let interner = ThreadedRodeo::default();
    // Pre-intern common values
    interner.get_or_intern("cudaLaunchKernel");
    interner.get_or_intern("cudaMemcpyAsync");
    interner.get_or_intern("default");
    interner.get_or_intern("");
    interner.get_or_intern("my-namespace");
    interner.get_or_intern("my-pod-abc123");
    interner.get_or_intern("vllm");
    interner.get_or_intern("7");
    interner.get_or_intern("GPU-abc-def-123");
    interner.get_or_intern("12345");
    interner
}

// ── Benchmarks ─────────────────────────────────────────────────────────────

/// Benchmark: cache hit path (HashMap lookup + Counter::inc + Histogram::observe)
fn bench_cache_hit(c: &mut Criterion) {
    let (_registry, counter, histogram) = setup_metrics();

    // Pre-create a cached handle (simulates cache hit)
    let cached_counter = counter.with_label_values(&[
        "cudaLaunchKernel",
        "12345",
        "vllm",
        "default",
        "my-pod",
        "main",
        "7",
        "GPU-abc",
        "default",
    ]);
    let cached_histogram =
        histogram.with_label_values(&["cudaLaunchKernel", "default", "my-pod", "7"]);

    // Simulate cache key lookup
    let mut handles: HashMap<(u32, u32, u32, u64, u64), usize> = HashMap::new();
    handles.insert((4, 12345, 0, 0, 0x1234), 0);

    c.bench_function("cache_hit_full", |b| {
        b.iter(|| {
            // 1. HashMap lookup (cache hit)
            let _idx = handles.get(&black_box((4, 12345, 0, 0, 0x1234)));
            // 2. Counter inc (atomic)
            cached_counter.inc();
            // 3. Histogram observe (atomic)
            cached_histogram.observe(black_box(0.000005));
        })
    });
}

/// Benchmark: Spur interning (get_or_intern for existing string)
fn bench_spur_intern_hit(c: &mut Criterion) {
    let interner = setup_interner();

    c.bench_function("spur_intern_hit", |b| {
        b.iter(|| {
            let _s = interner.get_or_intern(black_box("my-namespace"));
        })
    });
}

/// Benchmark: Spur resolve (Spur → &str)
fn bench_spur_resolve(c: &mut Criterion) {
    let interner = setup_interner();
    let spur = interner.get_or_intern("my-namespace");

    c.bench_function("spur_resolve", |b| {
        b.iter(|| {
            let _s = interner.resolve(&black_box(spur));
        })
    });
}

/// Benchmark: cache miss path — label construction with Spur (current optimized)
fn bench_cache_miss_spur(c: &mut Criterion) {
    let (_registry, counter, histogram) = setup_metrics();
    let interner = setup_interner();

    let op = interner.get_or_intern("cudaLaunchKernel");
    let pid = interner.get_or_intern("12345");
    let comm = interner.get_or_intern("vllm");
    let ns = interner.get_or_intern("my-namespace");
    let pod = interner.get_or_intern("my-pod-abc123");
    let container = interner.get_or_intern("vllm");
    let gpu = interner.get_or_intern("7");
    let gpu_uuid = interner.get_or_intern("GPU-abc-def-123");
    let stream = interner.get_or_intern("default");

    c.bench_function("cache_miss_spur_labels", |b| {
        b.iter(|| {
            // Resolve spurs to &str (what happens on cache miss)
            let labels: [&str; 9] = [
                interner.resolve(&op),
                interner.resolve(&pid),
                interner.resolve(&comm),
                interner.resolve(&ns),
                interner.resolve(&pod),
                interner.resolve(&container),
                interner.resolve(&gpu),
                interner.resolve(&gpu_uuid),
                interner.resolve(&stream),
            ];
            let _c = counter.with_label_values(black_box(&labels));

            let hist_labels: [&str; 4] = [
                interner.resolve(&op),
                interner.resolve(&ns),
                interner.resolve(&pod),
                interner.resolve(&gpu),
            ];
            let _h = histogram.with_label_values(black_box(&hist_labels));
        })
    });
}

/// Benchmark: cache miss path — label construction with String (old approach, for comparison)
fn bench_cache_miss_string(c: &mut Criterion) {
    let (_registry, counter, histogram) = setup_metrics();

    c.bench_function("cache_miss_string_labels", |b| {
        b.iter(|| {
            // Old approach: allocate Strings for every label
            let op = "cudaLaunchKernel".to_string();
            let pid = 12345u32.to_string();
            let comm = "vllm".to_string();
            let ns = "my-namespace".to_string();
            let pod = "my-pod-abc123".to_string();
            let container = "vllm".to_string();
            let gpu = "7".to_string();
            let gpu_uuid = "GPU-abc-def-123".to_string();
            let stream = "default".to_string();

            let refs: Vec<&str> = [
                &op, &pid, &comm, &ns, &pod, &container, &gpu, &gpu_uuid, &stream,
            ]
            .iter()
            .map(|s| s.as_str())
            .collect();
            let _c = counter.with_label_values(black_box(&refs));

            let hist_refs: Vec<&str> = [&op, &ns, &pod, &gpu].iter().map(|s| s.as_str()).collect();
            let _h = histogram.with_label_values(black_box(&hist_refs));
        })
    });
}

/// Benchmark: stream_label() — Cow vs String
fn bench_stream_label(c: &mut Criterion) {
    let mut group = c.benchmark_group("stream_label");

    group.bench_function("cow_default", |b| {
        b.iter(|| -> Cow<'static, str> {
            if black_box(0u64) == 0 {
                Cow::Borrowed("default")
            } else {
                Cow::Owned(format!("0x{:x}", 0u64))
            }
        })
    });

    group.bench_function("cow_hex", |b| {
        b.iter(|| -> Cow<'static, str> {
            let s = black_box(0x322849c0u64);
            if s == 0 {
                Cow::Borrowed("default")
            } else {
                Cow::Owned(format!("0x{:x}", s))
            }
        })
    });

    group.bench_function("string_default", |b| {
        b.iter(|| -> String {
            if black_box(0u64) == 0 {
                "default".to_string()
            } else {
                format!("0x{:x}", 0u64)
            }
        })
    });

    group.finish();
}

/// Benchmark: ArrayVec vs Vec for label refs
fn bench_label_refs(c: &mut Criterion) {
    let labels = [
        "cudaLaunchKernel",
        "12345",
        "vllm",
        "default",
        "my-pod",
        "main",
        "7",
        "GPU-abc",
        "default",
    ];

    let mut group = c.benchmark_group("label_refs");

    group.bench_function("arrayvec", |b| {
        b.iter(|| {
            let refs: ArrayVec<&str, 9> = black_box(&labels).iter().copied().collect();
            black_box(refs);
        })
    });

    group.bench_function("vec", |b| {
        b.iter(|| {
            let refs: Vec<&str> = black_box(&labels).to_vec();
            black_box(refs);
        })
    });

    group.finish();
}

/// Benchmark: HashMap lookup (simulates cache key lookup)
fn bench_hashmap_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_key_lookup");

    let mut std_map: HashMap<(u32, u32, u32, u64, u64), usize> = HashMap::new();
    for i in 0..1000 {
        std_map.insert((4, i, 0, 0, 0x1000 + i as u64), i as usize);
    }
    group.bench_function("std_HashMap", |b| {
        b.iter(|| std_map.get(&black_box((4, 500, 0, 0, 0x11f4))))
    });

    let mut fx_map: FxHashMap<(u32, u32, u32, u64, u64), usize> = FxHashMap::default();
    for i in 0..1000 {
        fx_map.insert((4, i, 0, 0, 0x1000 + i as u64), i as usize);
    }
    group.bench_function("FxHashMap", |b| {
        b.iter(|| fx_map.get(&black_box((4, 500, 0, 0, 0x11f4))))
    });

    group.finish();
}

/// Benchmark: throughput — events per second simulation
fn bench_throughput(c: &mut Criterion) {
    let (_registry, counter, histogram) = setup_metrics();

    let cached_counter = counter.with_label_values(&[
        "cudaLaunchKernel",
        "12345",
        "vllm",
        "default",
        "my-pod",
        "main",
        "7",
        "GPU-abc",
        "default",
    ]);
    let cached_histogram =
        histogram.with_label_values(&["cudaLaunchKernel", "default", "my-pod", "7"]);

    let mut handles: HashMap<(u32, u32, u32, u64, u64), usize> = HashMap::new();
    handles.insert((4, 12345, 0, 0, 0x1234), 0);

    let mut group = c.benchmark_group("throughput");
    for batch_size in [100, 1000, 10000] {
        group.throughput(Throughput::Elements(batch_size));
        group.bench_with_input(
            BenchmarkId::new("events", batch_size),
            &batch_size,
            |b, &size| {
                b.iter(|| {
                    for _ in 0..size {
                        let _idx = handles.get(&(4, 12345, 0, 0, 0x1234));
                        cached_counter.inc();
                        cached_histogram.observe(0.000005);
                    }
                })
            },
        );
    }
    group.finish();
}

/// Builds a realistic fixture: `series` distinct counter data points and
/// `series` histogram observations across `series/10` histogram series. Used
/// by both Prometheus and OTLP encoders so head-to-head comparisons are apples
/// to apples.
fn populate_registry(series: usize) -> (Registry, CounterVec, HistogramVec) {
    let (registry, counter, histogram) = setup_metrics();
    for i in 0..series {
        let pid = format!("{}", 10000 + i);
        let pod = format!("my-pod-{}", i % 10);
        let labels = [
            "cudaLaunchKernel",
            &pid,
            "vllm",
            "default",
            &pod,
            "main",
            "7",
            "GPU-abc",
            "default",
        ];
        counter.with_label_values(&labels).inc_by(1000.0);
        histogram
            .with_label_values(&["cudaLaunchKernel", "default", &pod, "7"])
            .observe(0.0001 + (i as f64) * 1e-5);
    }
    (registry, counter, histogram)
}

/// Benchmark: Prometheus encode (metrics scrape)
fn bench_encode(c: &mut Criterion) {
    let (registry, _counter, _histogram) = populate_registry(50);

    let mut reuse_buf = Vec::with_capacity(64 * 1024);

    let mut group = c.benchmark_group("encode");

    group.bench_function("new_vec", |b| {
        b.iter(|| {
            let mut buf = Vec::new();
            TextEncoder::new()
                .encode(&registry.gather(), &mut buf)
                .unwrap();
            black_box(String::from_utf8(buf).unwrap());
        })
    });

    group.bench_function("reuse_vec", |b| {
        b.iter(|| {
            reuse_buf.clear();
            TextEncoder::new()
                .encode(&registry.gather(), &mut reuse_buf)
                .unwrap();
            black_box(String::from_utf8_lossy(&reuse_buf).into_owned());
        })
    });

    group.finish();
}

/// Benchmark: OTLP conversion (one `registry.gather()` → `Vec<otlp::Metric>`).
///
/// Same fixture as `bench_encode` so the two encode paths can be compared
/// head-to-head. We don't benchmark the network send — it's async I/O and
/// dominated by the collector RTT, not by conversion cost.
fn bench_otlp_convert(c: &mut Criterion) {
    use profi::otlp::{convert_metric_family, should_skip_metric};
    use std::time::SystemTime;

    let start = SystemTime::now();

    let mut group = c.benchmark_group("otlp_convert");

    for &series in &[10usize, 50, 200, 1000] {
        let (registry, _counter, _histogram) = populate_registry(series);
        group.throughput(Throughput::Elements(series as u64));

        // Full scrape: gather + filter profi_system_* self-obs + convert every family.
        group.bench_with_input(
            BenchmarkId::new("gather_filter_convert", series),
            &registry,
            |b, registry| {
                b.iter(|| {
                    let families = registry.gather();
                    let now = SystemTime::now();
                    let out: Vec<_> = families
                        .iter()
                        .filter(|f| !should_skip_metric(f.get_name()))
                        .filter_map(|f| convert_metric_family(f, start, now))
                        .collect();
                    black_box(out);
                })
            },
        );

        // gather() alone — subtract from the above to isolate conversion cost.
        group.bench_with_input(
            BenchmarkId::new("gather_only", series),
            &registry,
            |b, registry| {
                b.iter(|| {
                    black_box(registry.gather());
                })
            },
        );
    }

    group.finish();
}

/// Per-family micro-bench: counter vs histogram conversion cost. Counters are
/// one data point per series, histograms carry N+1 bucket counts + sum + count
/// plus the bucket arithmetic — typically 2–3× a counter per series.
fn bench_otlp_convert_family(c: &mut Criterion) {
    use profi::otlp::convert_metric_family;
    use std::time::SystemTime;

    let start = SystemTime::now();

    let mut group = c.benchmark_group("otlp_convert_family");

    for &series in &[10usize, 50, 200] {
        let (registry, _c, _h) = populate_registry(series);
        let families = registry.gather();
        let counter_family = families
            .iter()
            .find(|f| f.get_name() == "bench_cuda_calls_total")
            .expect("counter family");
        let hist_family = families
            .iter()
            .find(|f| f.get_name() == "bench_cuda_duration_seconds")
            .expect("histogram family");

        group.throughput(Throughput::Elements(series as u64));

        group.bench_with_input(
            BenchmarkId::new("counter", series),
            counter_family,
            |b, fam| {
                b.iter(|| {
                    let now = SystemTime::now();
                    black_box(convert_metric_family(fam, start, now));
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("histogram", series),
            hist_family,
            |b, fam| {
                b.iter(|| {
                    let now = SystemTime::now();
                    black_box(convert_metric_family(fam, start, now));
                })
            },
        );
    }

    group.finish();
}

use prometheus::TextEncoder;

criterion_group!(
    benches,
    bench_cache_hit,
    bench_spur_intern_hit,
    bench_spur_resolve,
    bench_cache_miss_spur,
    bench_cache_miss_string,
    bench_stream_label,
    bench_label_refs,
    bench_hashmap_lookup,
    bench_throughput,
    bench_encode,
    bench_otlp_convert,
    bench_otlp_convert_family,
);
criterion_main!(benches);
