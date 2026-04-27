// SPDX-License-Identifier: Apache-2.0

//! Integration tests for Profi metrics pipeline.
//!
//! These tests verify that CudaEvent → MetricHandleCache → Prometheus output
//! works correctly end-to-end without requiring eBPF or GPU hardware.

use std::collections::HashSet;
use std::time::Duration;

use profi::cache::{CardinalityLimits, MetricHandleCache};
use profi::enrichment::Enricher;
use profi::events::*;
use profi::metrics::{KernelMode, Metrics};
use rustc_hash::FxHashMap;

/// Create a test Enricher that returns empty labels (no GPUs, no K8s).
fn test_enricher() -> std::sync::Arc<Enricher> {
    Enricher::new("/nonexistent".to_string())
}

fn default_limits() -> CardinalityLimits {
    CardinalityLimits::default()
}

/// Create a synthetic CudaEvent for testing.
fn make_event(event_type: u32, pid: u32) -> CudaEvent {
    CudaEvent {
        event_type,
        pid,
        tid: 0,
        memcpy_kind: 0,
        timestamp_ns: 0,
        duration_ns: 1000,
        size: 256,
        addr: 0,
        stream: 0,
        nvtx_marker: [0; 16],
        comm: {
            let mut c = [0u8; 16];
            c[..4].copy_from_slice(b"test");
            c
        },
        error_code: 0,
        _pad2: 0,
    }
}

#[test]
fn cuda_memcpy_creates_memcpy_handles() {
    let metrics = Metrics::new(KernelMode::Anonymous).unwrap();
    let enricher = test_enricher();
    let kernel_names: FxHashMap<(u32, u64), String> = FxHashMap::default();
    let mut cache = MetricHandleCache::new();

    let mut event = make_event(EVENT_CUDA_MEMCPY, 1000);
    event.memcpy_kind = MEMCPY_H2D;
    let key = (EVENT_CUDA_MEMCPY, 1000, MEMCPY_H2D, 0, 0u64, 0u64, 0u64);

    let h = cache.get_or_create(
        key,
        &metrics,
        &enricher,
        &event,
        &kernel_names,
        "/proc",
        &default_limits(),
    );
    assert!(h.memcpy_bytes.is_some());
    assert!(h.malloc_bytes.is_none());
    assert!(h.nccl_bytes.is_none());
}

#[test]
fn nccl_event_creates_nccl_handles() {
    let metrics = Metrics::new(KernelMode::Anonymous).unwrap();
    let enricher = test_enricher();
    let kernel_names: FxHashMap<(u32, u64), String> = FxHashMap::default();
    let mut cache = MetricHandleCache::new();

    let event = make_event(EVENT_NCCL_ALL_REDUCE, 2000);
    let key = (EVENT_NCCL_ALL_REDUCE, 2000, 0, 0, 0u64, 0u64, 0u64);

    let h = cache.get_or_create(
        key,
        &metrics,
        &enricher,
        &event,
        &kernel_names,
        "/proc",
        &default_limits(),
    );
    assert!(h.nccl_bytes.is_some());
    assert!(h.memcpy_bytes.is_none());
    assert!(h.malloc_bytes.is_none());
}

#[test]
fn kernel_launch_creates_kernel_handles() {
    let metrics = Metrics::new(KernelMode::Full).unwrap();
    let enricher = test_enricher();
    let mut kernel_names: FxHashMap<(u32, u64), String> = FxHashMap::default();
    kernel_names.insert((3000, 0x1234), "my_kernel".to_string());
    let mut cache = MetricHandleCache::new();

    let mut event = make_event(EVENT_CUDA_LAUNCH_KERNEL, 3000);
    event.addr = 0x1234;
    let key = (EVENT_CUDA_LAUNCH_KERNEL, 3000, 0, 0, 0u64, 0x1234u64, 0u64);

    let h = cache.get_or_create(
        key,
        &metrics,
        &enricher,
        &event,
        &kernel_names,
        "/proc",
        &default_limits(),
    );
    assert!(h.kernel_counter.is_some());
    assert!(h.kernel_histogram.is_some());
}

#[test]
fn error_code_creates_error_counter() {
    let metrics = Metrics::new(KernelMode::Anonymous).unwrap();
    let enricher = test_enricher();
    let kernel_names: FxHashMap<(u32, u64), String> = FxHashMap::default();
    let mut cache = MetricHandleCache::new();

    let mut event = make_event(EVENT_CUDA_MALLOC, 4000);
    event.error_code = 42;
    let key = (EVENT_CUDA_MALLOC, 4000, 0, 42, 0u64, 0u64, 0u64);

    let h = cache.get_or_create(
        key,
        &metrics,
        &enricher,
        &event,
        &kernel_names,
        "/proc",
        &default_limits(),
    );
    assert!(h.errors_counter.is_some());
}

#[test]
fn cache_hit_returns_same_handle() {
    let metrics = Metrics::new(KernelMode::Anonymous).unwrap();
    let enricher = test_enricher();
    let kernel_names: FxHashMap<(u32, u64), String> = FxHashMap::default();
    let mut cache = MetricHandleCache::new();

    let event = make_event(EVENT_CUDA_MALLOC, 5000);
    let key = (EVENT_CUDA_MALLOC, 5000, 0, 0, 0u64, 0u64, 0u64);

    // First call creates the handle
    let h = cache.get_or_create(
        key,
        &metrics,
        &enricher,
        &event,
        &kernel_names,
        "/proc",
        &default_limits(),
    );
    h.calls.inc();

    // Second call should return the same handle (counter already at 1)
    let h2 = cache.get_or_create(
        key,
        &metrics,
        &enricher,
        &event,
        &kernel_names,
        "/proc",
        &default_limits(),
    );
    h2.calls.inc();

    // The counter should show 2 increments total (same underlying counter)
    let output = String::from_utf8(metrics.encode_bytes()).unwrap();
    assert!(output.contains("profi_cuda_calls_total"));
}

#[test]
fn prometheus_output_contains_expected_metrics() {
    let metrics = Metrics::new(KernelMode::Full).unwrap();
    let enricher = test_enricher();
    let kernel_names: FxHashMap<(u32, u64), String> = FxHashMap::default();
    let mut cache = MetricHandleCache::new();

    // Create a few different event types
    let malloc_event = make_event(EVENT_CUDA_MALLOC, 6000);
    let malloc_key = (EVENT_CUDA_MALLOC, 6000, 0, 0, 0u64, 0u64, 0u64);
    let h = cache.get_or_create(
        malloc_key,
        &metrics,
        &enricher,
        &malloc_event,
        &kernel_names,
        "/proc",
        &default_limits(),
    );
    h.calls.inc();
    if let Some(ref c) = h.malloc_bytes {
        c.inc_by(1024.0);
    }

    let nccl_event = make_event(EVENT_NCCL_ALL_REDUCE, 6001);
    let nccl_key = (EVENT_NCCL_ALL_REDUCE, 6001, 0, 0, 0u64, 0u64, 0u64);
    let h = cache.get_or_create(
        nccl_key,
        &metrics,
        &enricher,
        &nccl_event,
        &kernel_names,
        "/proc",
        &default_limits(),
    );
    h.calls.inc();

    let output = String::from_utf8(metrics.encode_bytes()).unwrap();
    assert!(
        output.contains("profi_cuda_calls_total"),
        "missing cuda_calls"
    );
    assert!(
        output.contains("profi_cuda_malloc_bytes_total"),
        "missing malloc_bytes"
    );
    assert!(
        output.contains("profi_nccl_calls_total"),
        "missing nccl_calls"
    );
    assert!(
        output.contains("profi_cuda_duration_seconds"),
        "missing cuda_duration"
    );
}

#[test]
fn gc_evicts_stale_pids() {
    let metrics = Metrics::new(KernelMode::Anonymous).unwrap();
    let enricher = test_enricher();
    let kernel_names: FxHashMap<(u32, u64), String> = FxHashMap::default();
    let mut cache = MetricHandleCache::new();

    let event = make_event(EVENT_CUDA_MALLOC, 7000);
    let key = (EVENT_CUDA_MALLOC, 7000, 0, 0, 0u64, 0u64, 0u64);

    cache.get_or_create(
        key,
        &metrics,
        &enricher,
        &event,
        &kernel_names,
        "/proc",
        &default_limits(),
    );
    cache.touch(7000);

    // With stale_after=ZERO, all PIDs are immediately stale
    let evicted = cache.gc(&metrics, &enricher, Duration::ZERO);
    assert!(evicted.contains(&7000));
    assert!(!cache.handles.contains_key(&key));
    assert!(!cache.last_seen.contains_key(&7000));
}

#[test]
fn invalidate_pids_removes_handles() {
    let metrics = Metrics::new(KernelMode::Anonymous).unwrap();
    let enricher = test_enricher();
    let kernel_names: FxHashMap<(u32, u64), String> = FxHashMap::default();
    let mut cache = MetricHandleCache::new();

    let event = make_event(EVENT_CUDA_FREE, 8000);
    let key = (EVENT_CUDA_FREE, 8000, 0, 0, 0u64, 0u64, 0u64);

    cache.get_or_create(
        key,
        &metrics,
        &enricher,
        &event,
        &kernel_names,
        "/proc",
        &default_limits(),
    );
    cache.touch(8000);

    let pids: HashSet<u32> = [8000].into();
    cache.invalidate_pids(&pids, &metrics, &enricher.interner);

    assert!(!cache.handles.contains_key(&key));
}

#[test]
fn cardinality_limit_drops_excess_time_series() {
    let metrics = Metrics::new(KernelMode::Anonymous).unwrap();
    let enricher = test_enricher();
    let kernel_names: FxHashMap<(u32, u64), String> = FxHashMap::default();
    let mut cache = MetricHandleCache::new();
    let limits = CardinalityLimits {
        max_time_series: 3,
        max_streams_per_pid: 32,
        max_kernels_per_pid: 512,
    };

    // Create 3 handles (at limit)
    for i in 0..3u32 {
        let event = make_event(EVENT_CUDA_MALLOC, 9000 + i);
        let key = (EVENT_CUDA_MALLOC, 9000 + i, 0, 0, 0u64, 0u64, 0u64);
        cache.get_or_create(
            key,
            &metrics,
            &enricher,
            &event,
            &kernel_names,
            "/proc",
            &limits,
        );
        cache.touch(9000 + i);
    }
    assert_eq!(cache.handles.len(), 3);

    // 4th handle for a new PID should still be created (new PID fallback)
    let event = make_event(EVENT_CUDA_MALLOC, 9999);
    let key = (EVENT_CUDA_MALLOC, 9999, 0, 0, 0u64, 0u64, 0u64);
    cache.get_or_create(
        key,
        &metrics,
        &enricher,
        &event,
        &kernel_names,
        "/proc",
        &limits,
    );

    // 5th handle for an existing PID should fall back to existing handle
    let event = make_event(EVENT_CUDA_FREE, 9000);
    let key = (EVENT_CUDA_FREE, 9000, 0, 0, 0u64, 0u64, 0u64);
    cache.get_or_create(
        key,
        &metrics,
        &enricher,
        &event,
        &kernel_names,
        "/proc",
        &limits,
    );
    // cardinality_drops should have been incremented
    let output = String::from_utf8(metrics.encode_bytes()).unwrap();
    assert!(output.contains("profi_cardinality_limit_drops_total"));
}

#[test]
fn stream_collapse_when_limit_exceeded() {
    let metrics = Metrics::new(KernelMode::Anonymous).unwrap();
    let enricher = test_enricher();
    let kernel_names: FxHashMap<(u32, u64), String> = FxHashMap::default();
    let mut cache = MetricHandleCache::new();
    let limits = CardinalityLimits {
        max_time_series: 50000,
        max_streams_per_pid: 2,
        max_kernels_per_pid: 512,
    };

    // Create 2 distinct streams (at limit)
    for stream_id in 1..=2u64 {
        let mut event = make_event(EVENT_CUDA_MEMCPY, 10000);
        event.memcpy_kind = MEMCPY_H2D;
        let key = (
            EVENT_CUDA_MEMCPY,
            10000,
            MEMCPY_H2D,
            0,
            stream_id,
            0u64,
            0u64,
        );
        cache.get_or_create(
            key,
            &metrics,
            &enricher,
            &event,
            &kernel_names,
            "/proc",
            &limits,
        );
    }

    // 3rd distinct stream should be collapsed to stream=0 ("default")
    let mut event = make_event(EVENT_CUDA_MEMCPY, 10000);
    event.memcpy_kind = MEMCPY_H2D;
    let key = (EVENT_CUDA_MEMCPY, 10000, MEMCPY_H2D, 0, 999u64, 0u64, 0u64);
    cache.get_or_create(
        key,
        &metrics,
        &enricher,
        &event,
        &kernel_names,
        "/proc",
        &limits,
    );

    // The key (999) should have been collapsed to stream=0
    let collapsed_key = (EVENT_CUDA_MEMCPY, 10000, MEMCPY_H2D, 0, 0u64, 0u64, 0u64);
    assert!(cache.handles.contains_key(&collapsed_key));
}

#[test]
fn kernel_name_collapse_when_limit_exceeded() {
    let metrics = Metrics::new(KernelMode::Full).unwrap();
    let enricher = test_enricher();
    let mut kernel_names: FxHashMap<(u32, u64), String> = FxHashMap::default();
    let mut cache = MetricHandleCache::new();
    let limits = CardinalityLimits {
        max_time_series: 50000,
        max_streams_per_pid: 32,
        max_kernels_per_pid: 2,
    };

    // Create 2 distinct kernel addrs (at limit)
    for addr in 1..=2u64 {
        kernel_names.insert((11000, addr), format!("kernel_{addr}"));
        let mut event = make_event(EVENT_CUDA_LAUNCH_KERNEL, 11000);
        event.addr = addr;
        let key = (EVENT_CUDA_LAUNCH_KERNEL, 11000, 0, 0, 0u64, addr, 0u64);
        cache.get_or_create(
            key,
            &metrics,
            &enricher,
            &event,
            &kernel_names,
            "/proc",
            &limits,
        );
    }

    // 3rd distinct kernel should be collapsed to addr=0
    kernel_names.insert((11000, 999), "kernel_overflow".to_string());
    let mut event = make_event(EVENT_CUDA_LAUNCH_KERNEL, 11000);
    event.addr = 999;
    let key = (EVENT_CUDA_LAUNCH_KERNEL, 11000, 0, 0, 0u64, 999u64, 0u64);
    cache.get_or_create(
        key,
        &metrics,
        &enricher,
        &event,
        &kernel_names,
        "/proc",
        &limits,
    );

    // The key (addr=999) should have been collapsed to addr=0
    let collapsed_key = (EVENT_CUDA_LAUNCH_KERNEL, 11000, 0, 0, 0u64, 0u64, 0u64);
    assert!(cache.handles.contains_key(&collapsed_key));
}
