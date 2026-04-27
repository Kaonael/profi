// SPDX-License-Identifier: Apache-2.0

use arrayvec::ArrayVec;
use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::mem::MaybeUninit;
use std::os::fd::BorrowedFd;
use std::os::unix::fs::MetadataExt;
use std::rc::Rc;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use libbpf_rs::skel::{OpenSkel, SkelBuilder};
use libbpf_rs::{MapCore, MapFlags, OpenObject, RingBufferBuilder, UprobeMultiOpts, UprobeOpts};
use log::{info, warn};
use tokio::io::unix::AsyncFd;
use tokio::io::Interest;
use tokio::sync::Mutex;

use profi::bpf::{OpenProfiSkel, ProfiSkel, ProfiSkelBuilder};
use profi::cache::{CardinalityLimits, MetricHandleCache};
use profi::discovery::{scan_proc_for_libs, DevInode};
use profi::enrichment::Enricher;
use profi::events::*;
use profi::http_security::{MetricsSecurityArgs, TokenReviewer};
use profi::kernel::normalize_kernel_name;
use profi::metrics::{
    nccl_event_bytes, nvtx_hash, operation_name, resolve_spurs, KernelMode, Metrics,
    CUDA_DURATION_BUCKET_LE, CUDA_KERNEL_DURATION_BUCKET_LE, NCCL_DURATION_BUCKET_LE,
};
use profi::otlp::{OtlpArgs, OtlpBridge, OtlpConfig};
use profi::report::{print_report, TermStats};
use profi::server::{serve_http, AppState};
use profi::symbols;

// ── CLI ─────────────────────────────────────────────────────────────────────

/// Which CUDA Runtime API surface profi attaches uprobes to.
/// `lean` (default) skips uprobes for service-y calls that fire millions of
/// times on the inference hot path but contribute little to dashboards.
/// `full` attaches everything — intended as an operator escape hatch when
/// diagnosing stream-sync, memset, or pinned-memory behavior specifically.
#[derive(Clone, Copy, Debug, PartialEq, clap::ValueEnum)]
enum ProbeProfile {
    /// Production default: skips sync / memset / pinned-memory runtime probes.
    /// Keeps malloc/free/memcpy/launch/NCCL/graph coverage intact.
    Lean,
    /// Diagnostic mode: adds stream/event syncs, memset, malloc_host/free_host
    /// uprobes. Useful when investigating CPU idle-on-sync or pinned-memory
    /// churn; adds noticeable per-fire overhead on inference-heavy workloads.
    Full,
}

/// Probe (entry, ret) name pairs attached only when `--probe-profile=full`.
/// Each name here must exactly match an `fn_name` in `probes_base`.
const DIAGNOSTIC_ONLY_FN_NAMES: &[&str] = &[
    "cudaStreamSynchronize",
    "cudaEventSynchronize",
    "cudaMemsetAsync",
    "cudaMemset",
    "cudaMallocHost",
    "cudaFreeHost",
];

const RINGBUF_CONSUME_BUDGET: usize = 4096;
const RINGBUF_EVENTS_QUEUE_CAPACITY: usize = 4096;
const RINGBUF_KERNEL_REG_QUEUE_CAPACITY: usize = 512;
const PERCPU_HASH_DRAIN_KEY_BUDGET: usize = 8192;

#[derive(Parser)]
#[command(
    name = "profi",
    about = "NVIDIA CUDA profiler — eBPF-based Prometheus exporter"
)]
struct Args {
    #[arg(short, long, default_value_t = 0)]
    pid: i32,

    #[arg(long, default_value = "/usr/local/cuda/lib64/libcudart.so")]
    cudart: String,

    #[arg(long, default_value = "0.0.0.0:9401")]
    listen: String,

    #[arg(long, default_value_t = 0)]
    report_interval: u64,

    #[arg(long, default_value = "/proc")]
    proc_path: String,

    #[arg(long, env = "NODE_NAME")]
    node_name: Option<String>,

    #[arg(long, default_value_t = 10)]
    refresh_interval: u64,

    /// GC interval for stale PID cleanup (seconds)
    #[arg(long, default_value_t = 60)]
    gc_interval: u64,

    /// Kernel tracing mode: full (names+histograms), anonymous (count+duration only), off
    #[arg(long, value_enum, default_value = "anonymous")]
    kernel_mode: KernelMode,

    /// CUDA Runtime probe profile. Default `lean` = production-ready overhead
    /// (skips stream/event syncs, memset, pinned-memory probes).
    #[arg(long, value_enum, default_value = "lean")]
    probe_profile: ProbeProfile,

    /// Enable NVTX range tracing (higher overhead, use for debugging)
    #[arg(long)]
    enable_nvtx_tracing: bool,

    /// Maximum total time series before dropping new ones (cardinality protection)
    #[arg(long, default_value_t = 50000)]
    max_time_series: usize,

    /// Maximum distinct streams per PID before collapsing to stream="default"
    #[arg(long, default_value_t = 32)]
    max_streams_per_pid: usize,

    /// Maximum unique kernel names per PID before collapsing to kernel="other"
    #[arg(long, default_value_t = 512)]
    max_kernels_per_pid: usize,

    /// INFLIGHT LruHashMap max entries (eBPF map for in-flight probes)
    #[arg(long, default_value_t = 10240)]
    entries_size: u32,

    /// AGGREGATED PerCpuHashMap max entries (eBPF map for in-kernel aggregation)
    #[arg(long, default_value_t = 2048)]
    aggregated_size: u32,

    /// LAUNCH_AGG PerCpuHashMap max entries (cuLaunchKernel aggregation, keyed by pid+host_fun+stream)
    #[arg(long, default_value_t = 8192)]
    launch_agg_size: u32,

    /// Also emit a ringbuf event on every cuLaunchKernel (in full mode) for Prometheus
    /// histogram precision and OTel exemplar correlation.
    #[arg(long)]
    detailed_launches: bool,

    /// MALLOC_SIZES LruHashMap max entries (eBPF map for active memory tracking, increase for PyTorch)
    #[arg(long, default_value_t = 131072)]
    malloc_sizes_size: u32,

    /// Sampling rate for aggregatable events (1=no sampling, N=sample 1 in N events)
    #[arg(long, default_value_t = 1)]
    sample_rate: u32,

    /// Disable NVML GPU hardware monitoring
    #[arg(long)]
    disable_nvml: bool,

    /// NVML polling interval in seconds
    #[arg(long, default_value_t = 5)]
    nvml_interval: u64,

    /// NCCL hang detection timeout in seconds (0=disabled)
    #[arg(long, default_value_t = 60)]
    nccl_hang_timeout: u64,

    #[command(flatten)]
    otlp: OtlpArgs,

    #[command(flatten)]
    metrics_security: MetricsSecurityArgs,
}

/// Attach request from the discovery thread to the main task.
struct AttachRequest {
    lib: String, // "libcudart.so" | "libnccl.so" | "libcuda.so" | "libnvtx3interop.so"
    host_path: String,
    devinode: DevInode,
}

// ── Attach helper ─────────────────────────────────────────────────────────
//
// Look up a program by name in the loaded skeleton and attach it as a
// u(ret)probe to (binary_path, func_name). Soft-fail per symbol: callers
// that want best-effort attach log the error and continue.

fn attach_uprobe(
    skel: &mut ProfiSkel<'static>,
    prog_name: &str,
    fn_name: &str,
    binary_path: &str,
    pid_opt: Option<i32>,
) -> Result<libbpf_rs::Link> {
    let retprobe = prog_name.ends_with("_ret");
    let pid = pid_opt.unwrap_or(-1);
    let mk = || UprobeOpts {
        func_name: Some(fn_name.to_string()),
        retprobe,
        ..Default::default()
    };
    // Match against every known probe. A big table but purely mechanical.
    let link_res: libbpf_rs::Result<libbpf_rs::Link> = match prog_name {
        "cuda_malloc" => skel
            .progs
            .cuda_malloc
            .attach_uprobe_with_opts(pid, binary_path, 0, mk()),
        "cuda_malloc_ret" => {
            skel.progs
                .cuda_malloc_ret
                .attach_uprobe_with_opts(pid, binary_path, 0, mk())
        }
        "cuda_free" => skel
            .progs
            .cuda_free
            .attach_uprobe_with_opts(pid, binary_path, 0, mk()),
        "cuda_free_ret" => {
            skel.progs
                .cuda_free_ret
                .attach_uprobe_with_opts(pid, binary_path, 0, mk())
        }
        "cuda_memcpy" => skel
            .progs
            .cuda_memcpy
            .attach_uprobe_with_opts(pid, binary_path, 0, mk()),
        "cuda_memcpy_ret" => {
            skel.progs
                .cuda_memcpy_ret
                .attach_uprobe_with_opts(pid, binary_path, 0, mk())
        }
        "cuda_memcpy_async" => {
            skel.progs
                .cuda_memcpy_async
                .attach_uprobe_with_opts(pid, binary_path, 0, mk())
        }
        "cuda_memcpy_async_ret" => {
            skel.progs
                .cuda_memcpy_async_ret
                .attach_uprobe_with_opts(pid, binary_path, 0, mk())
        }
        "cuda_launch_kernel" => {
            skel.progs
                .cuda_launch_kernel
                .attach_uprobe_with_opts(pid, binary_path, 0, mk())
        }
        "cuda_launch_kernel_ret" => {
            skel.progs
                .cuda_launch_kernel_ret
                .attach_uprobe_with_opts(pid, binary_path, 0, mk())
        }
        "cuda_malloc_async" => {
            skel.progs
                .cuda_malloc_async
                .attach_uprobe_with_opts(pid, binary_path, 0, mk())
        }
        "cuda_malloc_async_ret" => {
            skel.progs
                .cuda_malloc_async_ret
                .attach_uprobe_with_opts(pid, binary_path, 0, mk())
        }
        "cuda_free_async" => {
            skel.progs
                .cuda_free_async
                .attach_uprobe_with_opts(pid, binary_path, 0, mk())
        }
        "cuda_free_async_ret" => {
            skel.progs
                .cuda_free_async_ret
                .attach_uprobe_with_opts(pid, binary_path, 0, mk())
        }
        "cuda_register_function" => {
            skel.progs
                .cuda_register_function
                .attach_uprobe_with_opts(pid, binary_path, 0, mk())
        }
        "cu_module_get_function" => {
            skel.progs
                .cu_module_get_function
                .attach_uprobe_with_opts(pid, binary_path, 0, mk())
        }
        "cu_module_get_function_ret" => skel
            .progs
            .cu_module_get_function_ret
            .attach_uprobe_with_opts(pid, binary_path, 0, mk()),
        "cu_launch_kernel" => {
            skel.progs
                .cu_launch_kernel
                .attach_uprobe_with_opts(pid, binary_path, 0, mk())
        }
        "cu_launch_kernel_ret" => {
            skel.progs
                .cu_launch_kernel_ret
                .attach_uprobe_with_opts(pid, binary_path, 0, mk())
        }
        "cu_launch_kernel_ex" => {
            skel.progs
                .cu_launch_kernel_ex
                .attach_uprobe_with_opts(pid, binary_path, 0, mk())
        }
        "cu_launch_kernel_ex_ret" => {
            skel.progs
                .cu_launch_kernel_ex_ret
                .attach_uprobe_with_opts(pid, binary_path, 0, mk())
        }
        "cu_launch_cooperative_kernel" => skel
            .progs
            .cu_launch_cooperative_kernel
            .attach_uprobe_with_opts(pid, binary_path, 0, mk()),
        "cu_launch_cooperative_kernel_ret" => skel
            .progs
            .cu_launch_cooperative_kernel_ret
            .attach_uprobe_with_opts(pid, binary_path, 0, mk()),
        "cu_graph_launch" => {
            skel.progs
                .cu_graph_launch
                .attach_uprobe_with_opts(pid, binary_path, 0, mk())
        }
        "cu_graph_launch_ret" => {
            skel.progs
                .cu_graph_launch_ret
                .attach_uprobe_with_opts(pid, binary_path, 0, mk())
        }
        "cuda_stream_sync" => {
            skel.progs
                .cuda_stream_sync
                .attach_uprobe_with_opts(pid, binary_path, 0, mk())
        }
        "cuda_stream_sync_ret" => {
            skel.progs
                .cuda_stream_sync_ret
                .attach_uprobe_with_opts(pid, binary_path, 0, mk())
        }
        "cuda_event_sync" => {
            skel.progs
                .cuda_event_sync
                .attach_uprobe_with_opts(pid, binary_path, 0, mk())
        }
        "cuda_event_sync_ret" => {
            skel.progs
                .cuda_event_sync_ret
                .attach_uprobe_with_opts(pid, binary_path, 0, mk())
        }
        "cuda_malloc_host" => {
            skel.progs
                .cuda_malloc_host
                .attach_uprobe_with_opts(pid, binary_path, 0, mk())
        }
        "cuda_malloc_host_ret" => {
            skel.progs
                .cuda_malloc_host_ret
                .attach_uprobe_with_opts(pid, binary_path, 0, mk())
        }
        "cuda_free_host" => {
            skel.progs
                .cuda_free_host
                .attach_uprobe_with_opts(pid, binary_path, 0, mk())
        }
        "cuda_free_host_ret" => {
            skel.progs
                .cuda_free_host_ret
                .attach_uprobe_with_opts(pid, binary_path, 0, mk())
        }
        "cuda_memset" => skel
            .progs
            .cuda_memset
            .attach_uprobe_with_opts(pid, binary_path, 0, mk()),
        "cuda_memset_ret" => {
            skel.progs
                .cuda_memset_ret
                .attach_uprobe_with_opts(pid, binary_path, 0, mk())
        }
        "cuda_memset_async" => {
            skel.progs
                .cuda_memset_async
                .attach_uprobe_with_opts(pid, binary_path, 0, mk())
        }
        "cuda_memset_async_ret" => {
            skel.progs
                .cuda_memset_async_ret
                .attach_uprobe_with_opts(pid, binary_path, 0, mk())
        }
        "cuda_graph_launch" => {
            skel.progs
                .cuda_graph_launch
                .attach_uprobe_with_opts(pid, binary_path, 0, mk())
        }
        "cuda_graph_launch_ret" => {
            skel.progs
                .cuda_graph_launch_ret
                .attach_uprobe_with_opts(pid, binary_path, 0, mk())
        }
        "cuda_graph_instantiate" => {
            skel.progs
                .cuda_graph_instantiate
                .attach_uprobe_with_opts(pid, binary_path, 0, mk())
        }
        "cuda_graph_instantiate_ret" => skel
            .progs
            .cuda_graph_instantiate_ret
            .attach_uprobe_with_opts(pid, binary_path, 0, mk()),
        "cu_module_load_data" => {
            skel.progs
                .cu_module_load_data
                .attach_uprobe_with_opts(pid, binary_path, 0, mk())
        }
        "cu_module_load_data_ret" => {
            skel.progs
                .cu_module_load_data_ret
                .attach_uprobe_with_opts(pid, binary_path, 0, mk())
        }
        "nvtx_range_push" => {
            skel.progs
                .nvtx_range_push
                .attach_uprobe_with_opts(pid, binary_path, 0, mk())
        }
        "nvtx_range_pop" => {
            skel.progs
                .nvtx_range_pop
                .attach_uprobe_with_opts(pid, binary_path, 0, mk())
        }
        "nccl_all_reduce" => {
            skel.progs
                .nccl_all_reduce
                .attach_uprobe_with_opts(pid, binary_path, 0, mk())
        }
        "nccl_all_reduce_ret" => {
            skel.progs
                .nccl_all_reduce_ret
                .attach_uprobe_with_opts(pid, binary_path, 0, mk())
        }
        "nccl_all_gather" => {
            skel.progs
                .nccl_all_gather
                .attach_uprobe_with_opts(pid, binary_path, 0, mk())
        }
        "nccl_all_gather_ret" => {
            skel.progs
                .nccl_all_gather_ret
                .attach_uprobe_with_opts(pid, binary_path, 0, mk())
        }
        "nccl_reduce_scatter" => {
            skel.progs
                .nccl_reduce_scatter
                .attach_uprobe_with_opts(pid, binary_path, 0, mk())
        }
        "nccl_reduce_scatter_ret" => {
            skel.progs
                .nccl_reduce_scatter_ret
                .attach_uprobe_with_opts(pid, binary_path, 0, mk())
        }
        "nccl_broadcast" => {
            skel.progs
                .nccl_broadcast
                .attach_uprobe_with_opts(pid, binary_path, 0, mk())
        }
        "nccl_broadcast_ret" => {
            skel.progs
                .nccl_broadcast_ret
                .attach_uprobe_with_opts(pid, binary_path, 0, mk())
        }
        "nccl_send" => skel
            .progs
            .nccl_send
            .attach_uprobe_with_opts(pid, binary_path, 0, mk()),
        "nccl_send_ret" => {
            skel.progs
                .nccl_send_ret
                .attach_uprobe_with_opts(pid, binary_path, 0, mk())
        }
        "nccl_recv" => skel
            .progs
            .nccl_recv
            .attach_uprobe_with_opts(pid, binary_path, 0, mk()),
        "nccl_recv_ret" => {
            skel.progs
                .nccl_recv_ret
                .attach_uprobe_with_opts(pid, binary_path, 0, mk())
        }
        _ => anyhow::bail!("unknown BPF program: {prog_name}"),
    };
    link_res.with_context(|| format!("attach {prog_name} ({fn_name}) → {binary_path}"))
}

fn attach_probes(
    skel: &mut ProfiSkel<'static>,
    links: &mut Vec<libbpf_rs::Link>,
    probes: &[(&str, &str)],
    binary_path: &str,
    pid_opt: Option<i32>,
) -> Result<usize> {
    let start_len = links.len();
    for &(prog, func) in probes {
        let link = attach_uprobe(skel, prog, func, binary_path, pid_opt)?;
        links.push(link);
    }
    Ok(links.len() - start_len)
}

fn attach_nccl_probes_multi(
    skel: &mut ProfiSkel<'static>,
    links: &mut Vec<libbpf_rs::Link>,
    binary_path: &str,
    pid_opt: Option<i32>,
) -> Result<usize> {
    let pid = pid_opt.unwrap_or(-1);
    let start_len = links.len();

    let count_dtype_3_4 = UprobeMultiOpts {
        syms: vec![
            "ncclAllReduce".to_string(),
            "ncclAllGather".to_string(),
            "ncclReduceScatter".to_string(),
            "ncclBroadcast".to_string(),
        ],
        cookies: vec![
            EVENT_NCCL_ALL_REDUCE as u64,
            EVENT_NCCL_ALL_GATHER as u64,
            EVENT_NCCL_REDUCE_SCATTER as u64,
            EVENT_NCCL_BROADCAST as u64,
        ],
        ..Default::default()
    };
    links.push(
        skel.progs
            .nccl_count_dtype_3_4
            .attach_uprobe_multi_with_opts(pid, binary_path, "", count_dtype_3_4)
            .with_context(|| {
                format!(
                    "attach NCCL uprobe_multi count/datatype args 3/4 to {binary_path} \
                     (Linux >= 6.6 required)"
                )
            })?,
    );

    let count_dtype_2_3 = UprobeMultiOpts {
        syms: vec!["ncclSend".to_string(), "ncclRecv".to_string()],
        cookies: vec![EVENT_NCCL_SEND as u64, EVENT_NCCL_RECV as u64],
        ..Default::default()
    };
    links.push(
        skel.progs
            .nccl_count_dtype_2_3
            .attach_uprobe_multi_with_opts(pid, binary_path, "", count_dtype_2_3)
            .with_context(|| {
                format!(
                    "attach NCCL uprobe_multi count/datatype args 2/3 to {binary_path} \
                     (Linux >= 6.6 required)"
                )
            })?,
    );

    let retprobes = UprobeMultiOpts {
        retprobe: true,
        syms: vec![
            "ncclAllReduce".to_string(),
            "ncclAllGather".to_string(),
            "ncclReduceScatter".to_string(),
            "ncclBroadcast".to_string(),
            "ncclSend".to_string(),
            "ncclRecv".to_string(),
        ],
        ..Default::default()
    };
    links.push(
        skel.progs
            .nccl_multi_ret
            .attach_uprobe_multi_with_opts(pid, binary_path, "", retprobes)
            .with_context(|| {
                format!("attach NCCL return uprobe_multi to {binary_path} (Linux >= 6.6 required)")
            })?,
    );

    Ok(links.len() - start_len)
}

// ── Map access helpers ────────────────────────────────────────────────────

/// Per-CPU HASH map drain: processes each present key inline, removing it
/// afterwards so slots are freed for future inserts.
fn drain_percpu_hash<V, F>(map: &libbpf_rs::MapMut<'_>, mut process: F) -> Result<usize>
where
    V: Copy,
    F: FnMut(&[u8], &[V]) -> Result<()>,
{
    let keys: Vec<Vec<u8>> = map.keys().take(PERCPU_HASH_DRAIN_KEY_BUDGET).collect();
    let mut values = Vec::new();
    let mut drained = 0usize;
    for key in keys {
        let Ok(Some(percpu)) = map.lookup_percpu(&key, MapFlags::ANY) else {
            continue;
        };
        values.clear();
        values.reserve(percpu.len());
        for bytes in percpu {
            assert_eq!(bytes.len(), std::mem::size_of::<V>());
            values.push(unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const V) });
        }
        process(&key, &values)?;
        let _ = map.delete(&key);
        drained += 1;
    }
    Ok(drained)
}

fn inc_cumulative_bucket_counters<const N: usize>(
    metric: &prometheus::CounterVec,
    base_labels: &ArrayVec<&str, N>,
    le_values: &[&'static str],
    bucket_counts: &[u64],
) {
    let mut cumulative = 0u64;
    for (le, count) in le_values.iter().zip(bucket_counts.iter()) {
        cumulative += *count;
        if cumulative == 0 {
            continue;
        }
        let mut labels: ArrayVec<&str, 12> = ArrayVec::new();
        labels.extend(base_labels.iter().copied());
        labels.push(le);
        metric
            .with_label_values(labels.as_slice())
            .inc_by(cumulative as f64);
    }
}

/// Read a single-slot PerCpuArray<u64> and sum across CPUs.
fn read_percpu_counter(map: &libbpf_rs::MapMut<'_>) -> Option<u64> {
    let key = 0u32.to_ne_bytes();
    let percpu = map.lookup_percpu(&key, MapFlags::ANY).ok()??;
    let mut total: u64 = 0;
    for bytes in percpu {
        if bytes.len() == 8 {
            total += u64::from_ne_bytes(bytes.try_into().ok()?);
        }
    }
    Some(total)
}

// ── Main ──────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    env_logger::init();

    args.metrics_security
        .validate()
        .context("invalid --metrics-tls-* / --metrics-auth-* configuration")?;
    rustls::crypto::ring::default_provider()
        .install_default()
        .map_err(|_| anyhow::anyhow!("install rustls default crypto provider"))?;

    let kernel_mode = args.kernel_mode;
    info!("kernel tracing mode: {:?}", kernel_mode);

    // ── Load BPF skeleton ───────────────────────────────────────────────
    let open_object: &'static mut MaybeUninit<OpenObject> =
        Box::leak(Box::new(MaybeUninit::uninit()));
    let mut open_skel: OpenProfiSkel<'static> = ProfiSkelBuilder::default()
        .open(open_object)
        .context("failed to open BPF skeleton")?;

    open_skel
        .maps
        .INFLIGHT
        .set_max_entries(args.entries_size)
        .context("set INFLIGHT max_entries")?;
    open_skel
        .maps
        .AGGREGATED
        .set_max_entries(args.aggregated_size)
        .context("set AGGREGATED max_entries")?;
    open_skel
        .maps
        .MALLOC_SIZES
        .set_max_entries(args.malloc_sizes_size)
        .context("set MALLOC_SIZES max_entries")?;
    open_skel
        .maps
        .LAUNCH_AGG
        .set_max_entries(args.launch_agg_size)
        .context("set LAUNCH_AGG max_entries")?;

    let mut skel: ProfiSkel<'static> = open_skel.load().context("load BPF skeleton")?;
    let mut attached_links: Vec<libbpf_rs::Link> = Vec::new();

    // Write control maps. Index-0 single-slot ARRAY — key and value are u32.
    let zero_key = 0u32.to_ne_bytes();
    let mode_val: u32 = match kernel_mode {
        KernelMode::Anonymous => 1,
        _ => 0, // Full and Off both use 0 (Off doesn't attach anyway)
    };
    skel.maps
        .KERNEL_AGG_MODE
        .update(&zero_key, &mode_val.to_ne_bytes(), MapFlags::ANY)
        .context("init KERNEL_AGG_MODE")?;

    if args.sample_rate > 1 {
        skel.maps
            .SAMPLE_RATE
            .update(&zero_key, &args.sample_rate.to_ne_bytes(), MapFlags::ANY)
            .context("init SAMPLE_RATE")?;
        info!(
            "sampling rate: 1/{} for aggregatable events",
            args.sample_rate
        );
    }

    let dl: u32 = if args.detailed_launches { 1 } else { 0 };
    skel.maps
        .DETAILED_LAUNCHES
        .update(&zero_key, &dl.to_ne_bytes(), MapFlags::ANY)
        .context("init DETAILED_LAUNCHES")?;
    if args.detailed_launches {
        info!("detailed-launches: enabled — per-launch ringbuf events will be emitted");
    }

    let pid = if args.pid > 0 { Some(args.pid) } else { None };

    // ── Probe lists ────────────────────────────────────────────────────
    let probes_base: &[(&str, &str)] = &[
        ("cuda_malloc", "cudaMalloc"),
        ("cuda_malloc_ret", "cudaMalloc"),
        ("cuda_free", "cudaFree"),
        ("cuda_free_ret", "cudaFree"),
        ("cuda_memcpy", "cudaMemcpy"),
        ("cuda_memcpy_ret", "cudaMemcpy"),
        ("cuda_memcpy_async", "cudaMemcpyAsync"),
        ("cuda_memcpy_async_ret", "cudaMemcpyAsync"),
        ("cuda_launch_kernel", "cudaLaunchKernel"),
        ("cuda_launch_kernel_ret", "cudaLaunchKernel"),
        ("cuda_malloc_async", "cudaMallocAsync"),
        ("cuda_malloc_async_ret", "cudaMallocAsync"),
        ("cuda_free_async", "cudaFreeAsync"),
        ("cuda_free_async_ret", "cudaFreeAsync"),
        ("cuda_graph_launch", "cudaGraphLaunch"),
        ("cuda_graph_launch_ret", "cudaGraphLaunch"),
        ("cuda_graph_instantiate", "cudaGraphInstantiate"),
        ("cuda_graph_instantiate_ret", "cudaGraphInstantiate"),
    ];
    let mut probes_vec: Vec<(&str, &str)> = probes_base.to_vec();
    if kernel_mode == KernelMode::Full {
        probes_vec.push(("cuda_register_function", "__cudaRegisterFunction"));
    }
    if args.probe_profile == ProbeProfile::Full {
        for fn_name in DIAGNOSTIC_ONLY_FN_NAMES {
            let (entry, ret) = match *fn_name {
                "cudaStreamSynchronize" => ("cuda_stream_sync", "cuda_stream_sync_ret"),
                "cudaEventSynchronize" => ("cuda_event_sync", "cuda_event_sync_ret"),
                "cudaMemsetAsync" => ("cuda_memset_async", "cuda_memset_async_ret"),
                "cudaMemset" => ("cuda_memset", "cuda_memset_ret"),
                "cudaMallocHost" => ("cuda_malloc_host", "cuda_malloc_host_ret"),
                "cudaFreeHost" => ("cuda_free_host", "cuda_free_host_ret"),
                _ => continue,
            };
            probes_vec.push((entry, fn_name));
            probes_vec.push((ret, fn_name));
        }
        info!(
            "probe profile: full — appended {} diagnostic uprobes",
            DIAGNOSTIC_ONLY_FN_NAMES.len() * 2
        );
    } else {
        info!("probe profile: lean (production default)");
    }
    let probes: &[(&str, &str)] = &probes_vec[..];

    let driver_probes: &[(&str, &str)] = match kernel_mode {
        KernelMode::Full => &[
            ("cu_module_get_function", "cuModuleGetFunction"),
            ("cu_module_get_function_ret", "cuModuleGetFunction"),
            ("cu_launch_kernel", "cuLaunchKernel"),
            ("cu_launch_kernel_ret", "cuLaunchKernel"),
            ("cu_launch_kernel_ex", "cuLaunchKernelEx"),
            ("cu_launch_kernel_ex_ret", "cuLaunchKernelEx"),
            ("cu_launch_cooperative_kernel", "cuLaunchCooperativeKernel"),
            (
                "cu_launch_cooperative_kernel_ret",
                "cuLaunchCooperativeKernel",
            ),
            ("cu_graph_launch", "cuGraphLaunch"),
            ("cu_graph_launch_ret", "cuGraphLaunch"),
            ("cu_module_load_data", "cuModuleLoadData"),
            ("cu_module_load_data_ret", "cuModuleLoadData"),
        ],
        KernelMode::Anonymous => &[
            ("cu_launch_kernel", "cuLaunchKernel"),
            ("cu_launch_kernel_ret", "cuLaunchKernel"),
            ("cu_launch_kernel_ex", "cuLaunchKernelEx"),
            ("cu_launch_kernel_ex_ret", "cuLaunchKernelEx"),
            ("cu_launch_cooperative_kernel", "cuLaunchCooperativeKernel"),
            (
                "cu_launch_cooperative_kernel_ret",
                "cuLaunchCooperativeKernel",
            ),
            ("cu_graph_launch", "cuGraphLaunch"),
            ("cu_graph_launch_ret", "cuGraphLaunch"),
            ("cu_module_load_data", "cuModuleLoadData"),
            ("cu_module_load_data_ret", "cuModuleLoadData"),
        ],
        KernelMode::Off => &[],
    };

    let nvtx_probes: &[(&str, &str)] = &[
        ("nvtx_range_push", "nvtxRangePushA"),
        ("nvtx_range_pop", "nvtxRangePop"),
    ];

    // ── Initial attach (best-effort) ───────────────────────────────────
    let mut initial_attached_cuda: HashSet<DevInode> = HashSet::new();
    if std::path::Path::new(&args.cudart).exists() {
        match attach_probes(&mut skel, &mut attached_links, probes, &args.cudart, pid) {
            Ok(count) => {
                info!("attached {count} base probes to {}", args.cudart);
                if let Ok(meta) = std::fs::metadata(&args.cudart) {
                    initial_attached_cuda.insert(DevInode {
                        dev: meta.dev(),
                        ino: meta.ino(),
                    });
                }
            }
            Err(e) => {
                warn!(
                    "initial attach to {} failed: {e} — will discover via {}",
                    args.cudart, args.proc_path
                );
            }
        }
    } else {
        info!(
            "{} not found — will discover via {}",
            args.cudart, args.proc_path
        );
    }

    // ── Enrichment (K8s + GPU) ─────────────────────────────────────────
    let enricher = Enricher::new(args.proc_path.clone());
    let refresh = Duration::from_secs(args.refresh_interval);
    if let Some(node) = &args.node_name {
        enricher.start_k8s_refresh(node.clone(), refresh);
    } else {
        info!("NODE_NAME not set — K8s pod enrichment disabled");
    }

    // ── Prometheus metrics ──────────────────────────────────────────────
    let metrics = Metrics::new(kernel_mode).context("failed to create metrics registry")?;
    metrics.publish_gpu_info(&enricher.gpu_devices);

    // ── OTLP bridge ────────────────────────────────────────────────────
    let process_start = Instant::now();
    if let Some(otlp_cfg) = OtlpConfig::resolve(&args.otlp)? {
        OtlpBridge::start(otlp_cfg, metrics.clone(), process_start)
            .context("failed to start OTLP bridge")?;
    } else {
        info!("OTLP endpoint not configured — push exporter disabled (Prometheus /metrics still active)");
    }

    // ── Shared state ───────────────────────────────────────────────────
    let app_state = Arc::new(AppState {
        metrics: metrics.clone(),
        heartbeat_ns: std::sync::atomic::AtomicU64::new(0),
        ring_buffer_open: std::sync::atomic::AtomicBool::new(false),
        libs_attached: std::sync::atomic::AtomicUsize::new(
            if std::path::Path::new(&args.cudart).exists() {
                1
            } else {
                0
            },
        ),
        k8s_ready: std::sync::atomic::AtomicBool::new(true),
        events_processed: std::sync::atomic::AtomicU64::new(0),
        kernel_mode: format!("{:?}", kernel_mode).to_ascii_lowercase(),
        start_time: process_start,
    });

    let term_stats: Arc<Mutex<HashMap<(u32, u32), TermStats>>> = Arc::default();

    // ── Discovery thread: scans /proc/*/maps, sends AttachRequests ─────
    let (attach_tx, mut attach_rx) = tokio::sync::mpsc::unbounded_channel::<AttachRequest>();

    let known_pids: Arc<std::sync::RwLock<HashSet<u32>>> =
        Arc::new(std::sync::RwLock::new(HashSet::new()));
    let known_pids_discover = known_pids.clone();
    let known_pids_writer = known_pids.clone();

    let discover_proc_path = args.proc_path.clone();
    let discover_interval = Duration::from_secs(args.refresh_interval);
    let enable_nvtx_tracing = args.enable_nvtx_tracing;
    let discover_metrics = metrics.clone();

    std::thread::spawn(move || {
        let mut attached_cuda = initial_attached_cuda;
        let mut attached_nccl: HashSet<DevInode> = HashSet::new();
        let mut attached_driver: HashSet<DevInode> = HashSet::new();
        let mut attached_nvtx: HashSet<DevInode> = HashSet::new();
        let mut iteration: u64 = 0;
        loop {
            // Every 6th iteration (~60s with 10s interval) do a full /proc scan;
            // otherwise only scan known PIDs.
            let only_pids = if iteration.is_multiple_of(6) {
                None
            } else {
                Some(known_pids_discover.read().unwrap().clone())
            };
            iteration += 1;

            let mut libs_to_scan: Vec<(&str, &HashSet<DevInode>)> = vec![
                ("libcudart.so", &attached_cuda),
                ("libnccl.so", &attached_nccl),
                ("libcuda.so", &attached_driver),
            ];
            if enable_nvtx_tracing {
                libs_to_scan.push(("libnvtx3interop.so", &attached_nvtx));
            }
            let scan_start = Instant::now();
            let found = scan_proc_for_libs(&discover_proc_path, &libs_to_scan, only_pids.as_ref());
            discover_metrics
                .discovery_scan_duration
                .observe(scan_start.elapsed().as_secs_f64());

            for (lib_name, items) in found {
                for (host_path, di) in items {
                    let target = match lib_name.as_str() {
                        "libcudart.so" => &mut attached_cuda,
                        "libnccl.so" => &mut attached_nccl,
                        "libcuda.so" => &mut attached_driver,
                        "libnvtx3interop.so" => &mut attached_nvtx,
                        _ => continue,
                    };
                    if target.contains(&di) {
                        continue;
                    }
                    target.insert(di.clone());
                    if attach_tx
                        .send(AttachRequest {
                            lib: lib_name.clone(),
                            host_path,
                            devinode: di,
                        })
                        .is_err()
                    {
                        return; // main task ended
                    }
                }
            }

            discover_metrics
                .discovery_attached_libs
                .with_label_values(&["libcudart.so"])
                .set(attached_cuda.len() as i64);
            discover_metrics
                .discovery_attached_libs
                .with_label_values(&["libnccl.so"])
                .set(attached_nccl.len() as i64);
            discover_metrics
                .discovery_attached_libs
                .with_label_values(&["libcuda.so"])
                .set(attached_driver.len() as i64);
            if enable_nvtx_tracing {
                discover_metrics
                    .discovery_attached_libs
                    .with_label_values(&["libnvtx3interop.so"])
                    .set(attached_nvtx.len() as i64);
            }

            std::thread::sleep(discover_interval);
        }
    });

    // ── HTTP server ─────────────────────────────────────────────────────
    let reviewer = if args.metrics_security.needs_k8s_client() {
        match kube::Client::try_default().await {
            Ok(client) => {
                let ttl = Duration::from_secs(args.metrics_security.metrics_auth_cache_ttl);
                Some(Arc::new(TokenReviewer::new(
                    client,
                    args.metrics_security.metrics_auth_audience.clone(),
                    ttl,
                    args.metrics_security.metrics_auth_cache_size,
                    metrics.clone(),
                )))
            }
            Err(e) => {
                return Err(anyhow::anyhow!(
                    "TokenReview auth requested but K8s client init failed: {e}"
                ));
            }
        }
    } else {
        None
    };

    let listen = args.listen.clone();
    let http_state = app_state.clone();
    let security = args.metrics_security.clone();
    let reviewer_for_server = reviewer.clone();
    tokio::spawn(async move {
        if let Err(e) = serve_http(listen, http_state, security, reviewer_for_server).await {
            log::error!("HTTP server error: {e}");
        }
    });

    // ── NVML ────────────────────────────────────────────────────────────
    if !args.disable_nvml {
        if let Some(mut poller) = profi::nvml::NvmlPoller::new() {
            let nvml_metrics = metrics.clone();
            let nvml_interval_secs = args.nvml_interval;
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_secs(nvml_interval_secs));
                loop {
                    interval.tick().await;
                    poller.poll(&nvml_metrics);
                }
            });
        }
    } else {
        info!("NVML disabled via --disable-nvml");
    }

    // ── Symbol resolution thread (full mode only) ──────────────────────
    let (sym_tx, mut sym_result_rx) = if kernel_mode == KernelMode::Full {
        let (req_tx, mut req_rx) = tokio::sync::mpsc::unbounded_channel::<symbols::SymRequest>();
        let (res_tx, res_rx) = tokio::sync::mpsc::unbounded_channel::<(u32, u64, String)>();
        let sym_proc_path = args.proc_path.clone();
        std::thread::spawn(move || {
            let mut resolver = symbols::SymbolResolver::new(sym_proc_path);
            while let Some(req) = req_rx.blocking_recv() {
                match req {
                    symbols::SymRequest::Resolve(pid, addr) => {
                        let name = resolver.resolve(pid, addr);
                        let _ = res_tx.send((pid, addr, name));
                    }
                    symbols::SymRequest::EvictPids(pids) => {
                        resolver.evict_pids(&pids);
                    }
                }
            }
        });
        (Some(req_tx), Some(res_rx))
    } else {
        (None, None)
    };

    // ── RingBuffer ──────────────────────────────────────────────────────
    // One builder with up to 2 ringbuf maps registered; the returned Ring
    // exposes a single epoll fd we feed into tokio::io::unix::AsyncFd.
    let events_queue: Rc<RefCell<Vec<CudaEvent>>> = Rc::new(RefCell::new(Vec::with_capacity(
        RINGBUF_EVENTS_QUEUE_CAPACITY,
    )));
    let kreg_queue: Rc<RefCell<Vec<KernelRegEvent>>> = Rc::new(RefCell::new(Vec::with_capacity(
        RINGBUF_KERNEL_REG_QUEUE_CAPACITY,
    )));

    let mut rb_builder = RingBufferBuilder::new();
    {
        let eq = events_queue.clone();
        rb_builder
            .add(&skel.maps.EVENTS, move |data: &[u8]| {
                if data.len() >= std::mem::size_of::<CudaEvent>() {
                    let event = unsafe { (data.as_ptr() as *const CudaEvent).read_unaligned() };
                    eq.borrow_mut().push(event);
                }
                0
            })
            .context("register EVENTS ringbuf callback")?;
    }
    if kernel_mode == KernelMode::Full {
        let kq = kreg_queue.clone();
        rb_builder
            .add(&skel.maps.KERNEL_REGS, move |data: &[u8]| {
                if data.len() == std::mem::size_of::<KernelRegEvent>() {
                    let reg = unsafe { (data.as_ptr() as *const KernelRegEvent).read_unaligned() };
                    kq.borrow_mut().push(reg);
                }
                0
            })
            .context("register KERNEL_REGS ringbuf callback")?;
    }
    let ring = rb_builder.build().context("build RingBuffer")?;
    // SAFETY: ring owns the epoll fd and outlives the AsyncFd wrapper.
    let ring_fd: std::os::fd::RawFd = ring.epoll_fd();
    let borrowed = unsafe { BorrowedFd::borrow_raw(ring_fd) };
    let async_fd = AsyncFd::with_interest(borrowed, Interest::READABLE)
        .context("wrap ring epoll fd in AsyncFd")?;
    app_state.ring_buffer_open.store(true, Ordering::Relaxed);

    // ── Event loop state ────────────────────────────────────────────────
    let proc_path = args.proc_path.clone();
    let cardinality_limits = CardinalityLimits {
        max_time_series: args.max_time_series,
        max_streams_per_pid: args.max_streams_per_pid,
        max_kernels_per_pid: args.max_kernels_per_pid,
    };
    let metrics_reader = metrics.clone();
    let enricher_reader = enricher.clone();
    let term_stats_reader = term_stats.clone();
    let report = args.report_interval > 0;
    let gc_secs = args.gc_interval;
    let nccl_hang_timeout_ns = args.nccl_hang_timeout * 1_000_000_000;

    let mut cache = MetricHandleCache::new();
    let mut kernel_names: FxHashMap<(u32, u64), String> = FxHashMap::default();
    let mut pending_resolve: HashSet<(u32, u64)> = HashSet::new();
    let mut gc_interval = tokio::time::interval(Duration::from_secs(gc_secs));
    gc_interval.tick().await;
    let mut agg_interval = tokio::time::interval(Duration::from_secs(1));
    agg_interval.tick().await;
    let mut last_dropped_total: u64 = 0;
    let mut last_launch_dropped: u64 = 0;
    let mut consecutive_overflow_ticks: u32 = 0;
    let mut consecutive_clear_ticks: u32 = 0;
    let mut agg_interval_fast = false;
    let mut local_term_stats: FxHashMap<(u32, u32), TermStats> = FxHashMap::default();
    let stale_after = Duration::from_secs(gc_secs);
    let mut new_pids_batch: Vec<u32> = Vec::new();
    let mut active_memory: HashMap<u32, i64> = HashMap::new();
    let mut upgraded_pids_synced: HashSet<u32> = HashSet::new();
    let mut nccl_durations: FxHashMap<(u32, u32), std::collections::VecDeque<u64>> =
        FxHashMap::default();
    const NCCL_WINDOW_SIZE: usize = 100;
    const NCCL_BUCKET_MIDPOINTS_S: [f64; 12] = [
        5e-6, 3e-5, 7.5e-5, 3e-4, 7.5e-4, 3e-3, 7.5e-3, 0.03, 0.075, 0.3, 0.75, 2.0,
    ];

    println!(
        "profi: tracing CUDA calls{} on {}",
        pid.map_or(String::new(), |p| format!(" (pid {p})")),
        args.cudart,
    );
    println!("Prometheus: http://{}/metrics", args.listen);
    println!("Ctrl+C to stop.\n");

    if kernel_mode == KernelMode::Off {
        info!("kernel tracing disabled — driver API probes and kernel name resolution skipped");
    } else if kernel_mode == KernelMode::Anonymous {
        info!("anonymous kernel mode — kernel launches aggregated without names");
    }

    let mut report_interval = if report {
        Some(tokio::time::interval(Duration::from_secs(
            args.report_interval,
        )))
    } else {
        None
    };

    // ── Main event loop ────────────────────────────────────────────────
    loop {
        app_state.heartbeat_ns.store(
            app_state.start_time.elapsed().as_nanos() as u64,
            Ordering::Relaxed,
        );

        tokio::select! {
            // Ring buffer readable — consume a bounded batch, then process queued events.
            guard_result = async_fd.readable() => {
                let mut guard = match guard_result {
                    Ok(g) => g,
                    Err(e) => { warn!("ring buffer poll error: {e}"); break; }
                };

                // PID invalidation on enrichment changes
                if enricher_reader.has_changes.load(Ordering::Relaxed) {
                    let mut changed = enricher_reader.changed_pids.lock().unwrap();
                    if !changed.is_empty() {
                        let pids: HashSet<u32> = std::mem::take(&mut *changed);
                        enricher_reader.has_changes.store(false, Ordering::Relaxed);
                        drop(changed);
                        cache.invalidate_pids(&pids, &metrics_reader, &enricher_reader.interner);
                    }
                }

                let batch_start = Instant::now();
                let consumed = ring.consume_raw_n(RINGBUF_CONSUME_BUDGET);
                if consumed < 0 {
                    warn!("ring.consume_raw_n({RINGBUF_CONSUME_BUDGET}) failed: {consumed}");
                }

                for event in events_queue.borrow_mut().drain(..) {
                    let dur_s = event.duration_ns as f64 / 1e9;
                    let addr_key = if event.event_type == EVENT_CUDA_LAUNCH_KERNEL {
                        event.addr
                    } else {
                        0
                    };
                    let nvtx_key = if event.event_type == EVENT_CUDA_LAUNCH_KERNEL {
                        nvtx_hash(&event.nvtx_marker)
                    } else {
                        0
                    };
                    let key = (event.event_type, event.pid, event.memcpy_kind, event.error_code, event.stream, addr_key, nvtx_key);

                    let h = cache.get_or_create(
                        key, &metrics_reader, &enricher_reader, &event,
                        &kernel_names, &proc_path, &cardinality_limits,
                    );
                    app_state.events_processed.fetch_add(1, Ordering::Relaxed);
                    h.calls.inc();
                    h.duration.observe(dur_s);

                    if let Some(ref c) = h.memcpy_bytes {
                        if event.size > 0 { c.inc_by(event.size as f64); }
                    }
                    if let Some(ref c) = h.malloc_bytes {
                        if event.size > 0 { c.inc_by(event.size as f64); }
                    }
                    if let Some(ref c) = h.nccl_bytes {
                        let bytes = nccl_event_bytes(&event);
                        if bytes > 0 { c.inc_by(bytes as f64); }
                    }
                    if let Some(ref c) = h.kernel_counter { c.inc(); }
                    if let Some(ref hh) = h.kernel_histogram { hh.observe(dur_s); }
                    if let Some(ref c) = h.errors_counter { c.inc(); }

                    if matches!(event.event_type, EVENT_NCCL_ALL_REDUCE..=EVENT_NCCL_RECV)
                        && event.duration_ns > 0
                    {
                        let window = nccl_durations
                            .entry((event.pid, event.event_type))
                            .or_insert_with(|| std::collections::VecDeque::with_capacity(NCCL_WINDOW_SIZE));
                        if window.len() >= NCCL_WINDOW_SIZE {
                            window.pop_front();
                        }
                        window.push_back(event.duration_ns);
                    }

                    if kernel_mode == KernelMode::Full
                        && event.event_type == EVENT_CUDA_LAUNCH_KERNEL
                        && event.addr != 0
                        && !kernel_names.contains_key(&(event.pid, event.addr))
                        && !pending_resolve.contains(&(event.pid, event.addr))
                    {
                        pending_resolve.insert((event.pid, event.addr));
                        if let Some(ref tx) = sym_tx {
                            let _ = tx.send(symbols::SymRequest::Resolve(event.pid, event.addr));
                        }
                    }

                    if !cache.last_seen.contains_key(&event.pid) {
                        new_pids_batch.push(event.pid);
                        metrics_reader.tracked_pids.set(cache.last_seen.len() as i64 + 1);
                    }
                    cache.touch(event.pid);

                    if report {
                        local_term_stats
                            .entry((event.event_type, event.memcpy_kind))
                            .or_default()
                            .record(event.duration_ns, event.size);
                    }
                }

                // Drain KERNEL_REGS (full mode only)
                if kernel_mode == KernelMode::Full {
                    for reg in kreg_queue.borrow_mut().drain(..) {
                        if kernel_names.contains_key(&(reg.pid, reg.host_fun)) {
                            continue;
                        }
                        match symbols::read_kernel_name(&proc_path, reg.pid, reg.name_ptr) {
                            Ok(name) => {
                                kernel_names.insert(
                                    (reg.pid, reg.host_fun),
                                    normalize_kernel_name(&name),
                                );
                            }
                            Err(e) => {
                                metrics_reader
                                    .kernel_name_resolve_failures
                                    .with_label_values(&[e.as_label()])
                                    .inc();
                            }
                        }
                    }
                }

                metrics_reader.event_loop_duration.observe(batch_start.elapsed().as_secs_f64());
                guard.clear_ready();
            }

            // Discovery thread asked us to attach probes to a newly-found library.
            Some(req) = attach_rx.recv() => {
                if req.lib == "libnccl.so" {
                    match attach_nccl_probes_multi(&mut skel, &mut attached_links, &req.host_path, pid) {
                        Ok(count) => {
                            info!("discovered {}: {} ({} uprobe_multi links)", req.lib, req.host_path, count);
                            app_state.libs_attached.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(e) => {
                            warn!("failed to attach NCCL uprobe_multi to {}: {e}", req.host_path);
                            break;
                        }
                    }
                    let _ = req.devinode;
                    continue;
                }

                let probeset: &[(&str, &str)] = match req.lib.as_str() {
                    "libcudart.so" => probes,
                    "libcuda.so" => driver_probes,
                    "libnvtx3interop.so" if args.enable_nvtx_tracing => nvtx_probes,
                    _ => &[],
                };
                let mut all_ok = true;
                for &(prog, func) in probeset {
                    match attach_uprobe(&mut skel, prog, func, &req.host_path, pid) {
                        Ok(link) => attached_links.push(link),
                        Err(e) => {
                            warn!("failed to attach {prog} to {}: {e}", req.host_path);
                            all_ok = false;
                            break;
                        }
                    }
                }
                if all_ok {
                    info!("discovered {}: {}", req.lib, req.host_path);
                    app_state.libs_attached.fetch_add(1, Ordering::Relaxed);
                }
                let _ = req.devinode; // retained for future per-lib accounting
            }

            // Periodic: drain aggregation maps, update gauges.
            _ = agg_interval.tick() => {
                let agg_start = Instant::now();

                // UPGRADED_PIDS: full-replace sync when dirty.
                if enricher_reader.upgrade_dirty.swap(false, Ordering::AcqRel) {
                    let desired: HashSet<u32> = enricher_reader
                        .upgraded_pids
                        .read()
                        .unwrap()
                        .iter()
                        .copied()
                        .collect();

                    if upgraded_pids_synced.is_empty() {
                        upgraded_pids_synced = skel
                            .maps
                            .UPGRADED_PIDS
                            .keys()
                            .filter_map(|k| {
                                let bytes: [u8; 4] = k.as_slice().try_into().ok()?;
                                Some(u32::from_ne_bytes(bytes))
                            })
                            .collect();
                    }

                    for pid_k in upgraded_pids_synced.difference(&desired) {
                        let _ = skel.maps.UPGRADED_PIDS.delete(&pid_k.to_ne_bytes());
                    }
                    for pid_k in desired.difference(&upgraded_pids_synced) {
                        let _ = skel.maps.UPGRADED_PIDS.update(
                            &pid_k.to_ne_bytes(),
                            &1u8.to_ne_bytes(),
                            MapFlags::ANY,
                        );
                    }
                    upgraded_pids_synced = desired;
                    info!(
                        "UPGRADED_PIDS synced: {} entries (annotation-driven full-mode pods)",
                        upgraded_pids_synced.len()
                    );
                }

                // LAUNCH_DROPPED counter
                if let Some(total) = read_percpu_counter(&skel.maps.LAUNCH_DROPPED) {
                    let delta = total.saturating_sub(last_launch_dropped);
                    if delta > 0 {
                        metrics_reader.launch_agg_drops.inc_by(delta);
                        last_launch_dropped = total;
                    }
                }

                // DROPPED ringbuf-overflow counter + adaptive drain
                if let Some(total) = read_percpu_counter(&skel.maps.DROPPED) {
                    let delta = total.saturating_sub(last_dropped_total);
                    if delta > 0 {
                        metrics_reader.dropped_events.inc_by(delta);
                        let interval_secs = if agg_interval_fast { 0.1 } else { 1.0 };
                        let rate = delta as f64 / interval_secs;
                        metrics_reader.ring_buffer_drops_rate.set(rate);
                        warn!("RingBuf overflow: {delta} events dropped ({rate:.0}/s)");

                        consecutive_overflow_ticks += 1;
                        consecutive_clear_ticks = 0;

                        if consecutive_overflow_ticks >= 3 && !agg_interval_fast {
                            agg_interval_fast = true;
                            agg_interval = tokio::time::interval(Duration::from_millis(100));
                            agg_interval.tick().await;
                            warn!("adaptive drain: increased frequency to 100ms due to persistent overflow");
                        }
                        last_dropped_total = total;
                    } else {
                        metrics_reader.ring_buffer_drops_rate.set(0.0);
                        consecutive_overflow_ticks = 0;
                        consecutive_clear_ticks += 1;

                        if consecutive_clear_ticks >= 30 && agg_interval_fast {
                            agg_interval_fast = false;
                            agg_interval = tokio::time::interval(Duration::from_secs(1));
                            agg_interval.tick().await;
                            info!("adaptive drain: restored normal 1s frequency");
                        }
                    }
                }

                // Drain AGGREGATED (PerCpuHashMap<AggKey, AggValue>)
                let mut active_mem_delta: HashMap<u32, i64> = HashMap::new();
                if let Err(e) = drain_percpu_hash::<AggValue, _>(&skel.maps.AGGREGATED, |key_bytes, per_cpu| {
                    if key_bytes.len() != std::mem::size_of::<AggKey>() { return Ok(()); }
                    let key = unsafe { std::ptr::read_unaligned(key_bytes.as_ptr() as *const AggKey) };

                    let mut total_count: u64 = 0;
                    let mut total_duration: u64 = 0;
                    let mut total_size: u64 = 0;
                    let mut buckets = [0u64; 14];
                    for v in per_cpu {
                        total_count += v.count;
                        total_duration += v.duration_sum_ns;
                        total_size += v.size_sum;
                        for (i, c) in v.bucket_counts.iter().enumerate() {
                            buckets[i] += u64::from(*c);
                        }
                    }
                    if total_count == 0 { return Ok(()); }

                    let synthetic = CudaEvent {
                        event_type: key.event_type,
                        pid: key.pid,
                        tid: 0,
                        memcpy_kind: key.memcpy_kind,
                        timestamp_ns: 0,
                        duration_ns: 0,
                        size: 0,
                        addr: 0,
                        stream: key.stream,
                        nvtx_marker: [0; 16],
                        comm: [0; 16],
                        error_code: key.error_code,
                        _pad2: 0,
                    };
                    let cache_key = (key.event_type, key.pid, key.memcpy_kind, key.error_code, key.stream, 0u64, 0u64);
                    let h = cache.get_or_create(
                        cache_key, &metrics_reader, &enricher_reader, &synthetic,
                        &kernel_names, &proc_path, &cardinality_limits,
                    );
                    h.calls.inc_by(total_count as f64);
                    let hist_labels = resolve_spurs(&h.hist_labels, &enricher_reader.interner);
                    inc_cumulative_bucket_counters(
                        &metrics_reader.cuda_duration_bucket_total,
                        &hist_labels,
                        &CUDA_DURATION_BUCKET_LE,
                        &buckets,
                    );
                    metrics_reader
                        .cuda_duration_sum_seconds_total
                        .with_label_values(hist_labels.as_slice())
                        .inc_by(total_duration as f64 / 1e9);
                    metrics_reader
                        .cuda_duration_count_total
                        .with_label_values(hist_labels.as_slice())
                        .inc_by(total_count as f64);
                    if let Some(ref c) = h.kernel_counter { c.inc_by(total_count as f64); }
                    if let Some(ref c) = h.memcpy_bytes { if total_size > 0 { c.inc_by(total_size as f64); } }
                    if let Some(ref c) = h.malloc_bytes { if total_size > 0 { c.inc_by(total_size as f64); } }
                    if let Some(ref c) = h.errors_counter { c.inc_by(total_count as f64); }

                    if total_size > 0 {
                        match key.event_type {
                            EVENT_CUDA_MALLOC | EVENT_CUDA_MALLOC_ASYNC => {
                                *active_mem_delta.entry(key.pid).or_insert(0) += total_size as i64;
                            }
                            EVENT_CUDA_FREE | EVENT_CUDA_FREE_ASYNC => {
                                *active_mem_delta.entry(key.pid).or_insert(0) -= total_size as i64;
                            }
                            _ => {}
                        }
                    }

                    if !cache.last_seen.contains_key(&key.pid) {
                        new_pids_batch.push(key.pid);
                        metrics_reader.tracked_pids.set(cache.last_seen.len() as i64 + 1);
                    }
                    cache.touch(key.pid);

                    if report {
                        let ts = local_term_stats.entry((key.event_type, key.memcpy_kind)).or_default();
                        ts.count += total_count;
                        ts.total_ns += total_duration;
                        ts.total_bytes += total_size;
                    }
                    Ok(())
                }) {
                    warn!("drain AGGREGATED: {e}");
                }

                // Drain LAUNCH_AGG (skip when --detailed-launches, which ships via ringbuf)
                if !args.detailed_launches {
                    if let Err(e) = drain_percpu_hash::<LaunchAggValue, _>(&skel.maps.LAUNCH_AGG, |key_bytes, per_cpu| {
                        if key_bytes.len() != std::mem::size_of::<LaunchKey>() { return Ok(()); }
                        let key = unsafe { std::ptr::read_unaligned(key_bytes.as_ptr() as *const LaunchKey) };

                        let (mut count, mut total_dur, mut max_dur) = (0u64, 0u64, 0u64);
                        let mut buckets = [0u64; 9];
                        for v in per_cpu {
                            count += v.count;
                            total_dur += v.total_duration_ns;
                            if v.max_duration_ns > max_dur {
                                max_dur = v.max_duration_ns;
                            }
                            for (i, c) in v.bucket_counts.iter().enumerate() {
                                buckets[i] += u64::from(*c);
                            }
                        }
                        if count == 0 { return Ok(()); }

                        let synthetic = CudaEvent {
                            event_type: EVENT_CUDA_LAUNCH_KERNEL,
                            pid: key.pid,
                            tid: 0,
                            memcpy_kind: 0,
                            timestamp_ns: 0,
                            duration_ns: 0,
                            size: 0,
                            addr: key.host_fun,
                            stream: key.stream,
                            nvtx_marker: [0; 16],
                            comm: [0; 16],
                            error_code: 0,
                            _pad2: 0,
                        };
                        let cache_key = (
                            EVENT_CUDA_LAUNCH_KERNEL,
                            key.pid, 0u32, 0u32,
                            key.stream, key.host_fun, 0u64,
                        );
                        let h = cache.get_or_create(
                            cache_key, &metrics_reader, &enricher_reader, &synthetic,
                            &kernel_names, &proc_path, &cardinality_limits,
                        );
                        h.calls.inc_by(count as f64);
                        if let Some(ref c) = h.kernel_counter { c.inc_by(count as f64); }
                        if let (
                            Some(kernel_labels),
                            Some(bucket_metric),
                            Some(sum_metric),
                            Some(count_metric),
                        ) = (
                            h.kernel_hist_labels.as_ref(),
                            metrics_reader.cuda_kernel_duration_bucket_total.as_ref(),
                            metrics_reader.cuda_kernel_duration_sum_seconds_total.as_ref(),
                            metrics_reader.cuda_kernel_duration_count_total.as_ref(),
                        ) {
                            let labels = resolve_spurs(kernel_labels, &enricher_reader.interner);
                            inc_cumulative_bucket_counters(
                                bucket_metric,
                                &labels,
                                &CUDA_KERNEL_DURATION_BUCKET_LE,
                                &buckets,
                            );
                            sum_metric
                                .with_label_values(labels.as_slice())
                                .inc_by(total_dur as f64 / 1e9);
                            count_metric
                                .with_label_values(labels.as_slice())
                                .inc_by(count as f64);
                        }

                        if key.host_fun != 0
                            && !kernel_names.contains_key(&(key.pid, key.host_fun))
                            && !pending_resolve.contains(&(key.pid, key.host_fun))
                        {
                            pending_resolve.insert((key.pid, key.host_fun));
                            if let Some(ref tx) = sym_tx {
                                let _ = tx.send(symbols::SymRequest::Resolve(key.pid, key.host_fun));
                            }
                        }

                        if !cache.last_seen.contains_key(&key.pid) {
                            new_pids_batch.push(key.pid);
                            metrics_reader.tracked_pids.set(cache.last_seen.len() as i64 + 1);
                        }
                        cache.touch(key.pid);
                        let _ = max_dur;
                        Ok(())
                    }) {
                        warn!("drain LAUNCH_AGG: {e}");
                    }
                }

                // Drain NCCL_AGG
                if let Err(e) = drain_percpu_hash::<NcclAggValue, _>(&skel.maps.NCCL_AGG, |key_bytes, per_cpu| {
                    if key_bytes.len() != std::mem::size_of::<AggKey>() { return Ok(()); }
                    let key = unsafe { std::ptr::read_unaligned(key_bytes.as_ptr() as *const AggKey) };

                    let mut count = 0u64;
                    let mut dur_sum = 0u64;
                    let mut bytes_sum = 0u64;
                    let mut buckets = [0u64; 12];
                    for v in per_cpu {
                        count += v.count;
                        dur_sum += v.duration_sum_ns;
                        bytes_sum += v.bytes_sum;
                        for (i, c) in v.bucket_counts.iter().enumerate() {
                            buckets[i] += u64::from(*c);
                        }
                    }
                    if count == 0 { return Ok(()); }

                    let synthetic = CudaEvent {
                        event_type: key.event_type,
                        pid: key.pid,
                        tid: 0,
                        memcpy_kind: key.memcpy_kind,
                        timestamp_ns: 0,
                        duration_ns: 0,
                        size: 0,
                        addr: 0,
                        stream: key.stream,
                        nvtx_marker: [0; 16],
                        comm: [0; 16],
                        error_code: key.error_code,
                        _pad2: 0,
                    };
                    let cache_key = (key.event_type, key.pid, key.memcpy_kind, key.error_code, key.stream, 0u64, 0u64);
                    let h = cache.get_or_create(
                        cache_key, &metrics_reader, &enricher_reader, &synthetic,
                        &kernel_names, &proc_path, &cardinality_limits,
                    );
                    h.calls.inc_by(count as f64);
                    if let Some(ref c) = h.nccl_bytes { if bytes_sum > 0 { c.inc_by(bytes_sum as f64); } }
                    if let Some(ref c) = h.errors_counter { c.inc_by(count as f64); }
                    let hist_labels = resolve_spurs(&h.hist_labels, &enricher_reader.interner);
                    inc_cumulative_bucket_counters(
                        &metrics_reader.nccl_duration_bucket_total,
                        &hist_labels,
                        &NCCL_DURATION_BUCKET_LE,
                        &buckets,
                    );
                    metrics_reader
                        .nccl_duration_sum_seconds_total
                        .with_label_values(hist_labels.as_slice())
                        .inc_by(dur_sum as f64 / 1e9);
                    metrics_reader
                        .nccl_duration_count_total
                        .with_label_values(hist_labels.as_slice())
                        .inc_by(count as f64);

                    let window = nccl_durations
                        .entry((key.pid, key.event_type))
                        .or_insert_with(|| std::collections::VecDeque::with_capacity(NCCL_WINDOW_SIZE));
                    for (i, &cnt) in buckets.iter().enumerate() {
                        if cnt == 0 { continue; }
                        let dur_ns = (NCCL_BUCKET_MIDPOINTS_S[i] * 1e9) as u64;
                        if window.len() >= NCCL_WINDOW_SIZE {
                            window.pop_front();
                        }
                        window.push_back(dur_ns);
                    }

                    if !cache.last_seen.contains_key(&key.pid) {
                        new_pids_batch.push(key.pid);
                        metrics_reader.tracked_pids.set(cache.last_seen.len() as i64 + 1);
                    }
                    cache.touch(key.pid);
                    Ok(())
                }) {
                    warn!("drain NCCL_AGG: {e}");
                }

                // Active memory gauge
                for (pid_k, delta) in active_mem_delta {
                    let bytes = active_memory.entry(pid_k).or_insert(0);
                    *bytes += delta;
                    let pid_labels = enricher_reader.lookup(pid_k, "");
                    let r = &enricher_reader.interner;
                    let pid_s = pid_k.to_string();
                    let label_vals = [
                        pid_s.as_str(),
                        r.resolve(&pid_labels.comm),
                        r.resolve(&pid_labels.namespace),
                        r.resolve(&pid_labels.pod),
                        r.resolve(&pid_labels.container),
                        r.resolve(&pid_labels.gpu),
                        r.resolve(&pid_labels.gpu_uuid),
                    ];
                    metrics_reader.cuda_active_memory.with_label_values(&label_vals).set(*bytes);
                }

                metrics_reader.agg_drain_duration.observe(agg_start.elapsed().as_secs_f64());
                metrics_reader.uptime.set(app_state.start_time.elapsed().as_secs_f64());
            }

            // Periodic: GC stale PIDs, straggler detection, NCCL hang scan.
            _ = gc_interval.tick() => {
                if let Some(ref mut rx) = sym_result_rx {
                    while let Ok((pid_k, addr, name)) = rx.try_recv() {
                        pending_resolve.remove(&(pid_k, addr));
                        if !name.starts_with("unknown_") {
                            kernel_names.insert((pid_k, addr), normalize_kernel_name(&name));
                        }
                    }
                }

                if !new_pids_batch.is_empty() {
                    if let Ok(mut kp) = known_pids_writer.try_write() {
                        kp.extend(new_pids_batch.drain(..));
                    }
                }

                let stale_pids = cache.gc(&metrics_reader, &enricher_reader, stale_after);
                metrics_reader.handle_cache_size.set(cache.handles.len() as i64);

                if !stale_pids.is_empty() {
                    kernel_names.retain(|&(pid_k, _), _| !stale_pids.contains(&pid_k));
                    pending_resolve.retain(|&(pid_k, _)| !stale_pids.contains(&pid_k));
                    for pid_k in &stale_pids {
                        active_memory.remove(pid_k);
                    }
                    if let Some(ref tx) = sym_tx {
                        let _ = tx.send(symbols::SymRequest::EvictPids(stale_pids));
                    }
                }

                // NCCL hang detection via INFLIGHT scan
                if nccl_hang_timeout_ns > 0 {
                    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
                    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
                    let now_ns = ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64;
                    let mut stale_count = 0i64;
                    for key_bytes in skel.maps.INFLIGHT.keys() {
                        if key_bytes.len() != 8 { continue; }
                        let key = u64::from_ne_bytes(key_bytes.as_slice().try_into().unwrap());
                        let Ok(Some(val_bytes)) = skel.maps.INFLIGHT.lookup(&key_bytes, MapFlags::ANY) else {
                            continue;
                        };
                        if val_bytes.len() < std::mem::size_of::<EntryData>() { continue; }
                        let entry = unsafe {
                            std::ptr::read_unaligned(val_bytes.as_ptr() as *const EntryData)
                        };
                        if matches!(entry.event_type, EVENT_NCCL_ALL_REDUCE..=EVENT_NCCL_RECV) {
                            let age_ns = now_ns.saturating_sub(entry.timestamp_ns);
                            if age_ns > nccl_hang_timeout_ns {
                                stale_count += 1;
                                let pid_k = (key >> 32) as u32;
                                metrics_reader.nccl_hang_detected
                                    .with_label_values(&[
                                        operation_name(entry.event_type),
                                        &pid_k.to_string(),
                                    ]).inc();
                                warn!("NCCL hang: pid={pid_k} op={} stale {:.1}s",
                                    operation_name(entry.event_type),
                                    age_ns as f64 / 1e9);
                            }
                        }
                    }
                    metrics_reader.nccl_stale_entries.set(stale_count);
                }

                // Straggler detection
                if !nccl_durations.is_empty() {
                    let mut per_op: FxHashMap<u32, Vec<(u32, u64)>> = FxHashMap::default();
                    for (&(pid_k, event_type), window) in &nccl_durations {
                        if window.len() >= 10 {
                            let mut sorted: Vec<u64> = window.iter().copied().collect();
                            sorted.sort_unstable();
                            let p95_idx = (sorted.len() as f64 * 0.95) as usize;
                            let p95 = sorted[p95_idx.min(sorted.len() - 1)];
                            per_op.entry(event_type).or_default().push((pid_k, p95));
                        }
                    }
                    for (event_type, mut pid_p95s) in per_op {
                        if pid_p95s.len() < 2 { continue; }
                        pid_p95s.sort_by_key(|&(_, p95)| p95);
                        let median_p95 = pid_p95s[pid_p95s.len() / 2].1;
                        if median_p95 == 0 { continue; }
                        for &(pid_k, p95) in &pid_p95s {
                            let ratio = p95 as f64 / median_p95 as f64;
                            let pid_labels = enricher_reader.lookup(pid_k, "");
                            let r = &enricher_reader.interner;
                            metrics_reader.nccl_straggler_ratio
                                .with_label_values(&[
                                    &pid_k.to_string(),
                                    r.resolve(&pid_labels.gpu),
                                    r.resolve(&pid_labels.gpu_uuid),
                                    operation_name(event_type),
                                ])
                                .set(ratio);
                        }
                    }
                }

                if report && !local_term_stats.is_empty() {
                    let mut ts = term_stats_reader.lock().await;
                    for ((et, mk), local) in local_term_stats.drain() {
                        ts.entry((et, mk)).or_default().merge(&local);
                    }
                }
            }

            // Periodic report printout
            _ = async {
                match report_interval.as_mut() {
                    Some(i) => { i.tick().await; }
                    None => std::future::pending::<()>().await,
                }
            } => {
                let ts = term_stats.lock().await;
                if !ts.is_empty() {
                    print_report(&ts);
                }
            }

            _ = tokio::signal::ctrl_c() => {
                info!("shutting down");
                break;
            }
        }
    }

    Ok(())
}
