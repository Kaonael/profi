// SPDX-License-Identifier: Apache-2.0

use std::borrow::Cow;
use std::collections::HashSet;
use std::time::{Duration, Instant};

use arrayvec::ArrayVec;
use lasso::Spur;
use log::info;
use rustc_hash::FxHashMap;

use crate::enrichment::Enricher;
use crate::events::*;
use crate::kernel::{classify_kernel, sanitize_phase};
use crate::metrics::{
    is_nccl_event, memcpy_direction, operation_name, resolve_spurs, stream_label, Metrics,
    CUDA_DURATION_BUCKET_LE, CUDA_KERNEL_DURATION_BUCKET_LE, NCCL_DURATION_BUCKET_LE,
};

pub type MetricKey = (u32, u32, u32, u32, u64, u64, u64);

// ── Cardinality Limits ──────────────────────────────────────────────────────

pub struct CardinalityLimits {
    pub max_time_series: usize,
    pub max_streams_per_pid: usize,
    pub max_kernels_per_pid: usize,
}

impl Default for CardinalityLimits {
    fn default() -> Self {
        Self {
            max_time_series: 50_000,
            max_streams_per_pid: 32,
            max_kernels_per_pid: 512,
        }
    }
}

// ── Metric Handle Cache ─────────────────────────────────────────────────────
// Caches prometheus Counter/Histogram handles to avoid per-event
// with_label_values() overhead (hash + RwLock + HashMap lookup).

pub struct CachedHandles {
    pub calls: prometheus::Counter,
    pub duration: prometheus::Histogram,
    pub memcpy_bytes: Option<prometheus::Counter>,
    pub malloc_bytes: Option<prometheus::Counter>,
    pub nccl_bytes: Option<prometheus::Counter>,
    pub kernel_counter: Option<prometheus::Counter>,
    pub kernel_histogram: Option<prometheus::Histogram>,
    pub errors_counter: Option<prometheus::Counter>,
    // Stored label values as interned Spurs — resolved only during GC/invalidation (Fix 1)
    pub calls_labels: ArrayVec<Spur, 9>,
    pub hist_labels: ArrayVec<Spur, 9>,
    pub memcpy_labels: Option<ArrayVec<Spur, 9>>,
    pub malloc_labels: Option<ArrayVec<Spur, 9>>,
    pub nccl_bytes_labels: Option<ArrayVec<Spur, 9>>,
    pub kernel_counter_labels: Option<ArrayVec<Spur, 11>>,
    pub kernel_hist_labels: Option<ArrayVec<Spur, 11>>,
    pub errors_labels: Option<ArrayVec<Spur, 9>>,
}

pub struct MetricHandleCache {
    /// Key: (event_type, pid, memcpy_kind, error_code, stream, addr, nvtx_hash)
    /// addr is non-zero only for EVENT_CUDA_LAUNCH_KERNEL (per-kernel cached handles)
    /// nvtx_hash distinguishes NVTX phases for kernel metrics
    pub handles: FxHashMap<MetricKey, CachedHandles>,
    /// PID -> last event timestamp (for GC)
    pub last_seen: FxHashMap<u32, Instant>,
    /// PID -> number of distinct streams seen (for cardinality limiting)
    streams_per_pid: FxHashMap<u32, FxHashMap<u64, ()>>,
    /// PID -> number of distinct kernel addresses seen (for cardinality limiting)
    kernels_per_pid: FxHashMap<u32, FxHashMap<u64, ()>>,
}

impl Default for MetricHandleCache {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricHandleCache {
    pub fn new() -> Self {
        Self {
            handles: FxHashMap::default(),
            last_seen: FxHashMap::default(),
            streams_per_pid: FxHashMap::default(),
            kernels_per_pid: FxHashMap::default(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn get_or_create(
        &mut self,
        key: MetricKey,
        metrics: &Metrics,
        enricher: &Enricher,
        event: &CudaEvent,
        kernel_names: &FxHashMap<(u32, u64), String>,
        proc_path: &str,
        limits: &CardinalityLimits,
    ) -> &CachedHandles {
        // Apply cardinality limits by collapsing stream/kernel to "other" bucket
        let key = {
            let (event_type, pid, memcpy_kind, error_code, stream, addr, nvtx_key) = key;

            // Collapse streams: if PID has too many distinct streams, map to stream=0 ("default")
            let stream = if stream != 0 {
                let pid_streams = self.streams_per_pid.entry(pid).or_default();
                if pid_streams.contains_key(&stream) {
                    stream // already tracked
                } else if pid_streams.len() < limits.max_streams_per_pid {
                    pid_streams.insert(stream, ());
                    stream
                } else {
                    0 // collapse to "default"
                }
            } else {
                stream
            };

            // Collapse kernel addrs: if PID has too many distinct kernels, map addr to 0
            let addr = if addr != 0 && event_type == EVENT_CUDA_LAUNCH_KERNEL {
                let pid_kernels = self.kernels_per_pid.entry(pid).or_default();
                if pid_kernels.contains_key(&addr) {
                    addr
                } else if pid_kernels.len() < limits.max_kernels_per_pid {
                    pid_kernels.insert(addr, ());
                    addr
                } else {
                    0 // collapse to addr=0 ("unknown" kernel)
                }
            } else {
                addr
            };

            (
                event_type,
                pid,
                memcpy_kind,
                error_code,
                stream,
                addr,
                nvtx_key,
            )
        };

        if !self.handles.contains_key(&key) {
            // Global cardinality limit: refuse new handles beyond max
            if self.handles.len() >= limits.max_time_series {
                metrics.cardinality_drops.inc();
                // Return existing "overflow" entry if any, otherwise we must still create one
                // to avoid returning nothing. Use a degenerate key lookup for this PID.
                if let Some(fallback_key) = self
                    .handles
                    .keys()
                    .find(|(_, pid, _, _, _, _, _)| *pid == key.1)
                    .copied()
                {
                    return &self.handles[&fallback_key];
                }
                // No existing handle for this PID at all — fall through to create one
                // (this is rare: a brand-new PID when we're at the limit)
            }

            let (event_type, _pid, memcpy_kind, error_code, stream, addr, _nvtx_key) = key;

            // Parse comm from [u8; 16] — zero-alloc &str, resolve via interner
            let comm_len = event.comm.iter().position(|&c| c == 0).unwrap_or(16);
            let mut comm_str = std::str::from_utf8(&event.comm[..comm_len]).unwrap_or("unknown");

            // Fallback: read from /proc if eBPF sent empty comm (aggregated events)
            let proc_comm;
            if comm_str.is_empty() {
                let path = format!("{}/{}/comm", proc_path, event.pid);
                if let Ok(c) = std::fs::read_to_string(&path) {
                    proc_comm = c.trim_end().to_string();
                    comm_str = &proc_comm;
                } else {
                    comm_str = "unknown";
                }
            }

            let labels = enricher.lookup(event.pid, comm_str);
            let r = &enricher.interner;

            // Intern pid and stream too — everything becomes Spur (Fix 1)
            let pid_s = event.pid.to_string();
            let pid_spur = r.get_or_intern(&pid_s);
            let op: Cow<'static, str> = Cow::Borrowed(operation_name(event_type));
            let op_spur = r.get_or_intern(op.as_ref());
            let stream_s = stream_label(stream);
            let stream_spur = r.get_or_intern(stream_s.as_ref());

            // Resolve Spurs to &str for with_label_values() (lock-free concurrent read)
            let comm = r.resolve(&labels.comm);
            let namespace = r.resolve(&labels.namespace);
            let pod = r.resolve(&labels.pod);
            let container = r.resolve(&labels.container);
            let gpu = r.resolve(&labels.gpu);
            let gpu_uuid = r.resolve(&labels.gpu_uuid);

            if is_nccl_event(event_type) {
                // NCCL counter: full labels (no stream)
                let calls_refs = [
                    op.as_ref(),
                    pid_s.as_str(),
                    comm,
                    namespace,
                    pod,
                    container,
                    gpu,
                    gpu_uuid,
                ];
                let calls = metrics.nccl_calls.with_label_values(&calls_refs);
                let calls_labels: ArrayVec<Spur, 9> = ArrayVec::from_iter([
                    op_spur,
                    pid_spur,
                    labels.comm,
                    labels.namespace,
                    labels.pod,
                    labels.container,
                    labels.gpu,
                    labels.gpu_uuid,
                ]);

                // NCCL histogram: reduced labels
                let hist_refs = [op.as_ref(), namespace, pod, gpu];
                let duration = metrics.nccl_duration.with_label_values(&hist_refs);
                let hist_labels: ArrayVec<Spur, 9> =
                    ArrayVec::from_iter([op_spur, labels.namespace, labels.pod, labels.gpu]);

                // NCCL bytes counter (same labels as calls)
                let nccl_bytes = metrics.nccl_bytes.with_label_values(&calls_refs);
                let nccl_bytes_labels = calls_labels.clone();

                // Error counter (only when error_code != 0)
                let (errors_counter, errors_labels) = if error_code != 0 {
                    let ec_s = error_code.to_string();
                    let ec_spur = r.get_or_intern(&ec_s);
                    let el_refs = [
                        op.as_ref(),
                        pid_s.as_str(),
                        comm,
                        namespace,
                        pod,
                        container,
                        gpu,
                        gpu_uuid,
                        ec_s.as_str(),
                    ];
                    let el: ArrayVec<Spur, 9> = ArrayVec::from_iter([
                        op_spur,
                        pid_spur,
                        labels.comm,
                        labels.namespace,
                        labels.pod,
                        labels.container,
                        labels.gpu,
                        labels.gpu_uuid,
                        ec_spur,
                    ]);
                    (
                        Some(metrics.cuda_errors.with_label_values(&el_refs)),
                        Some(el),
                    )
                } else {
                    (None, None)
                };

                self.handles.insert(
                    key,
                    CachedHandles {
                        calls,
                        duration,
                        memcpy_bytes: None,
                        malloc_bytes: None,
                        nccl_bytes: Some(nccl_bytes),
                        kernel_counter: None,
                        kernel_histogram: None,
                        errors_counter,
                        calls_labels,
                        hist_labels,
                        memcpy_labels: None,
                        malloc_labels: None,
                        nccl_bytes_labels: Some(nccl_bytes_labels),
                        kernel_counter_labels: None,
                        kernel_hist_labels: None,
                        errors_labels,
                    },
                );
            } else {
                // CUDA counter: full labels (with stream)
                let calls_refs = [
                    op.as_ref(),
                    pid_s.as_str(),
                    comm,
                    namespace,
                    pod,
                    container,
                    gpu,
                    gpu_uuid,
                    stream_s.as_ref(),
                ];
                let calls = metrics.cuda_calls.with_label_values(&calls_refs);
                let calls_labels: ArrayVec<Spur, 9> = ArrayVec::from_iter([
                    op_spur,
                    pid_spur,
                    labels.comm,
                    labels.namespace,
                    labels.pod,
                    labels.container,
                    labels.gpu,
                    labels.gpu_uuid,
                    stream_spur,
                ]);

                // Histogram: reduced labels (no stream)
                let hist_refs = [op.as_ref(), namespace, pod, gpu];
                let duration = metrics.cuda_duration.with_label_values(&hist_refs);
                let hist_labels: ArrayVec<Spur, 9> =
                    ArrayVec::from_iter([op_spur, labels.namespace, labels.pod, labels.gpu]);

                // Memcpy/Memset counter (bytes transferred, with stream)
                let (memcpy_bytes, memcpy_labels) = if event_type == EVENT_CUDA_MEMCPY
                    || event_type == EVENT_CUDA_MEMCPY_ASYNC
                    || event_type == EVENT_CUDA_MEMSET
                    || event_type == EVENT_CUDA_MEMSET_ASYNC
                {
                    let dir: &'static str = memcpy_direction(memcpy_kind);
                    let dir_spur = r.get_or_intern(dir);
                    let ml_refs = [
                        dir,
                        pid_s.as_str(),
                        comm,
                        namespace,
                        pod,
                        container,
                        gpu,
                        gpu_uuid,
                        stream_s.as_ref(),
                    ];
                    let ml: ArrayVec<Spur, 9> = ArrayVec::from_iter([
                        dir_spur,
                        pid_spur,
                        labels.comm,
                        labels.namespace,
                        labels.pod,
                        labels.container,
                        labels.gpu,
                        labels.gpu_uuid,
                        stream_spur,
                    ]);
                    (
                        Some(metrics.cuda_memcpy_bytes.with_label_values(&ml_refs)),
                        Some(ml),
                    )
                } else {
                    (None, None)
                };

                // Malloc counter (no stream — not relevant)
                let (malloc_bytes, malloc_labels) =
                    if event_type == EVENT_CUDA_MALLOC || event_type == EVENT_CUDA_MALLOC_HOST {
                        let ml_refs = [
                            pid_s.as_str(),
                            comm,
                            namespace,
                            pod,
                            container,
                            gpu,
                            gpu_uuid,
                        ];
                        let ml: ArrayVec<Spur, 9> = ArrayVec::from_iter([
                            pid_spur,
                            labels.comm,
                            labels.namespace,
                            labels.pod,
                            labels.container,
                            labels.gpu,
                            labels.gpu_uuid,
                        ]);
                        (
                            Some(metrics.cuda_malloc_bytes.with_label_values(&ml_refs)),
                            Some(ml),
                        )
                    } else {
                        (None, None)
                    };

                // Kernel metrics for cudaLaunchKernel. addr==0 occurs in anonymous-mode
                // AGGREGATED drain path (no per-kernel identity): emit under kernel_name="aggregated"
                // so dashboards still see non-empty profi_cuda_kernel_launches_total.
                let (kernel_counter, kernel_histogram, kernel_counter_labels, kernel_hist_labels) =
                    if event_type == EVENT_CUDA_LAUNCH_KERNEL {
                        let kernel_name: Cow<'_, str> = if addr != 0 {
                            kernel_names
                                .get(&(event.pid, addr))
                                .map(|s| Cow::Borrowed(s.as_str()))
                                .unwrap_or_else(|| Cow::Owned(format!("unknown_0x{:x}", addr)))
                        } else {
                            Cow::Borrowed("aggregated")
                        };
                        let kernel_spur = r.get_or_intern(kernel_name.as_ref());
                        let kernel_class = classify_kernel(kernel_name.as_ref());
                        let class_spur = r.get_or_intern_static(kernel_class);

                        // Extract and sanitize NVTX phase from event
                        let phase_len =
                            event.nvtx_marker.iter().position(|&c| c == 0).unwrap_or(16);
                        let phase_raw =
                            std::str::from_utf8(&event.nvtx_marker[..phase_len]).unwrap_or("");
                        let phase = sanitize_phase(phase_raw);
                        let phase_spur = r.get_or_intern_static(phase);

                        let kcl_refs = [
                            kernel_name.as_ref(),
                            pid_s.as_str(),
                            comm,
                            namespace,
                            pod,
                            container,
                            gpu,
                            gpu_uuid,
                            kernel_class,
                            phase,
                        ];
                        let kcl: ArrayVec<Spur, 11> = ArrayVec::from_iter([
                            kernel_spur,
                            pid_spur,
                            labels.comm,
                            labels.namespace,
                            labels.pod,
                            labels.container,
                            labels.gpu,
                            labels.gpu_uuid,
                            class_spur,
                            phase_spur,
                        ]);
                        let kc = metrics.cuda_kernel_launches.with_label_values(&kcl_refs);

                        let (kh, khl) = if let Some(ref hist_vec) = metrics.cuda_kernel_duration {
                            let khl_refs = [
                                kernel_name.as_ref(),
                                namespace,
                                pod,
                                gpu,
                                kernel_class,
                                phase,
                            ];
                            let khl: ArrayVec<Spur, 11> = ArrayVec::from_iter([
                                kernel_spur,
                                labels.namespace,
                                labels.pod,
                                labels.gpu,
                                class_spur,
                                phase_spur,
                            ]);
                            (Some(hist_vec.with_label_values(&khl_refs)), Some(khl))
                        } else {
                            (None, None)
                        };

                        (Some(kc), kh, Some(kcl), khl)
                    } else {
                        (None, None, None, None)
                    };

                // Error counter (only when error_code != 0)
                let (errors_counter, errors_labels) = if error_code != 0 {
                    let ec_s = error_code.to_string();
                    let ec_spur = r.get_or_intern(&ec_s);
                    let el_refs = [
                        op.as_ref(),
                        pid_s.as_str(),
                        comm,
                        namespace,
                        pod,
                        container,
                        gpu,
                        gpu_uuid,
                        ec_s.as_str(),
                    ];
                    let el: ArrayVec<Spur, 9> = ArrayVec::from_iter([
                        op_spur,
                        pid_spur,
                        labels.comm,
                        labels.namespace,
                        labels.pod,
                        labels.container,
                        labels.gpu,
                        labels.gpu_uuid,
                        ec_spur,
                    ]);
                    (
                        Some(metrics.cuda_errors.with_label_values(&el_refs)),
                        Some(el),
                    )
                } else {
                    (None, None)
                };

                self.handles.insert(
                    key,
                    CachedHandles {
                        calls,
                        duration,
                        memcpy_bytes,
                        malloc_bytes,
                        nccl_bytes: None,
                        kernel_counter,
                        kernel_histogram,
                        errors_counter,
                        calls_labels,
                        hist_labels,
                        memcpy_labels,
                        malloc_labels,
                        nccl_bytes_labels: None,
                        kernel_counter_labels,
                        kernel_hist_labels,
                        errors_labels,
                    },
                );
            }
        }
        &self.handles[&key]
    }

    pub fn touch(&mut self, pid: u32) {
        self.last_seen.insert(pid, Instant::now());
    }

    /// Remove cached handles for specific PIDs and their prometheus time series.
    pub fn invalidate_pids(
        &mut self,
        pids: &HashSet<u32>,
        metrics: &Metrics,
        interner: &lasso::ThreadedRodeo,
    ) {
        let keys_to_remove: Vec<_> = self
            .handles
            .keys()
            .filter(|(_, pid, _, _, _, _, _)| pids.contains(pid))
            .copied()
            .collect();

        for key in keys_to_remove {
            let event_type = key.0;
            if let Some(h) = self.handles.remove(&key) {
                remove_handle_metrics(metrics, event_type, &h, interner);
            }
        }
    }

    /// Remove stale PIDs that are no longer running. Returns evicted PIDs.
    pub fn gc(
        &mut self,
        metrics: &Metrics,
        enricher: &Enricher,
        stale_after: Duration,
    ) -> Vec<u32> {
        let now = Instant::now();
        let stale_pids: Vec<u32> = self
            .last_seen
            .iter()
            .filter(|(_, &ts)| now.duration_since(ts) > stale_after)
            .map(|(&pid, _)| pid)
            .collect();

        for pid in &stale_pids {
            // Remove all handle entries for this PID
            let keys_to_remove: Vec<_> = self
                .handles
                .keys()
                .filter(|(_, p, _, _, _, _, _)| p == pid)
                .copied()
                .collect();

            for key in keys_to_remove {
                let event_type = key.0;
                if let Some(h) = self.handles.remove(&key) {
                    remove_handle_metrics(metrics, event_type, &h, &enricher.interner);
                }
            }

            self.last_seen.remove(pid);
            self.streams_per_pid.remove(pid);
            self.kernels_per_pid.remove(pid);
            enricher.evict_pid(*pid);
        }

        if !stale_pids.is_empty() {
            info!("GC: evicted {} stale PIDs", stale_pids.len());
            metrics.tracked_pids.set(self.last_seen.len() as i64);
        }

        stale_pids
    }
}

/// Remove all prometheus time series associated with a cached handle entry.
/// Resolves Spur->str via the interner only here (GC/invalidation path, not hot path).
fn remove_handle_metrics(
    metrics: &Metrics,
    event_type: u32,
    h: &CachedHandles,
    interner: &lasso::ThreadedRodeo,
) {
    if is_nccl_event(event_type) {
        let calls_labels = resolve_spurs(&h.calls_labels, interner);
        let hist_labels = resolve_spurs(&h.hist_labels, interner);
        let _ = metrics.nccl_calls.remove_label_values(&calls_labels);
        let _ = metrics.nccl_duration.remove_label_values(&hist_labels);
        remove_bucket_counter_values(
            &metrics.nccl_duration_bucket_total,
            &hist_labels,
            &NCCL_DURATION_BUCKET_LE,
        );
        let _ = metrics
            .nccl_duration_sum_seconds_total
            .remove_label_values(&hist_labels);
        let _ = metrics
            .nccl_duration_count_total
            .remove_label_values(&hist_labels);
    } else {
        let calls_labels = resolve_spurs(&h.calls_labels, interner);
        let hist_labels = resolve_spurs(&h.hist_labels, interner);
        let _ = metrics.cuda_calls.remove_label_values(&calls_labels);
        let _ = metrics.cuda_duration.remove_label_values(&hist_labels);
        remove_bucket_counter_values(
            &metrics.cuda_duration_bucket_total,
            &hist_labels,
            &CUDA_DURATION_BUCKET_LE,
        );
        let _ = metrics
            .cuda_duration_sum_seconds_total
            .remove_label_values(&hist_labels);
        let _ = metrics
            .cuda_duration_count_total
            .remove_label_values(&hist_labels);
    }
    if let Some(ml) = &h.memcpy_labels {
        let _ = metrics
            .cuda_memcpy_bytes
            .remove_label_values(&resolve_spurs(ml, interner));
    }
    if let Some(ml) = &h.malloc_labels {
        let _ = metrics
            .cuda_malloc_bytes
            .remove_label_values(&resolve_spurs(ml, interner));
    }
    if let Some(ml) = &h.nccl_bytes_labels {
        let _ = metrics
            .nccl_bytes
            .remove_label_values(&resolve_spurs(ml, interner));
    }
    if let Some(ml) = &h.kernel_counter_labels {
        let _ = metrics
            .cuda_kernel_launches
            .remove_label_values(&resolve_spurs(ml, interner));
    }
    if let Some(ml) = &h.kernel_hist_labels {
        if let Some(ref hist_vec) = metrics.cuda_kernel_duration {
            let labels = resolve_spurs(ml, interner);
            let _ = hist_vec.remove_label_values(&labels);
            if let Some(ref buckets) = metrics.cuda_kernel_duration_bucket_total {
                remove_bucket_counter_values(buckets, &labels, &CUDA_KERNEL_DURATION_BUCKET_LE);
            }
            if let Some(ref sum) = metrics.cuda_kernel_duration_sum_seconds_total {
                let _ = sum.remove_label_values(&labels);
            }
            if let Some(ref count) = metrics.cuda_kernel_duration_count_total {
                let _ = count.remove_label_values(&labels);
            }
        }
    }
    if let Some(ml) = &h.errors_labels {
        let _ = metrics
            .cuda_errors
            .remove_label_values(&resolve_spurs(ml, interner));
    }
}

fn remove_bucket_counter_values<const N: usize>(
    metric: &prometheus::CounterVec,
    base_labels: &ArrayVec<&str, N>,
    le_values: &[&str],
) {
    for le in le_values {
        let mut labels: ArrayVec<&str, 12> = ArrayVec::new();
        labels.extend(base_labels.iter().copied());
        labels.push(le);
        let _ = metric.remove_label_values(labels.as_slice());
    }
}
