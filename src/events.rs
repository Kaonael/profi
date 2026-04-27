// SPDX-License-Identifier: Apache-2.0

// Event type constants
pub const EVENT_CUDA_MALLOC: u32 = 1;
pub const EVENT_CUDA_FREE: u32 = 2;
pub const EVENT_CUDA_MEMCPY: u32 = 3;
pub const EVENT_CUDA_LAUNCH_KERNEL: u32 = 4;
pub const EVENT_CUDA_MEMCPY_ASYNC: u32 = 5;
pub const EVENT_CUDA_MALLOC_ASYNC: u32 = 6;
pub const EVENT_CUDA_FREE_ASYNC: u32 = 7;
pub const EVENT_CUDA_REGISTER_FUNCTION: u32 = 8;

// NCCL collective event types
pub const EVENT_NCCL_ALL_REDUCE: u32 = 9;
pub const EVENT_NCCL_ALL_GATHER: u32 = 10;
pub const EVENT_NCCL_REDUCE_SCATTER: u32 = 11;
pub const EVENT_NCCL_BROADCAST: u32 = 12;
pub const EVENT_NCCL_SEND: u32 = 13;
pub const EVENT_NCCL_RECV: u32 = 14;

// Extended CUDA event types
pub const EVENT_CUDA_STREAM_SYNC: u32 = 15;
pub const EVENT_CUDA_EVENT_SYNC: u32 = 16;
pub const EVENT_CUDA_MALLOC_HOST: u32 = 17;
pub const EVENT_CUDA_FREE_HOST: u32 = 18;
pub const EVENT_CUDA_MEMSET: u32 = 19;
pub const EVENT_CUDA_MEMSET_ASYNC: u32 = 20;
pub const EVENT_CUDA_GRAPH_LAUNCH: u32 = 21;
pub const EVENT_CUDA_GRAPH_INSTANTIATE: u32 = 22;
pub const EVENT_CUDA_MODULE_LOAD: u32 = 23;

/// Byte offset of `hStream` inside the `CUlaunchConfig` struct passed to
/// `cuLaunchKernelEx`. Layout (CUDA 11.8+, stable ABI): 6×`unsigned int` for
/// grid/block dims (24B), `unsigned int sharedMemBytes` (4B) → offset 28, then
/// 4B pad to 8-byte align the `CUstream` pointer at offset 32.
pub const CULAUNCH_CONFIG_STREAM_OFFSET: usize = 32;

// ncclDataType_t element sizes in bytes
pub const NCCL_DTYPE_SIZES: [u64; 10] = [
    1, // ncclInt8/ncclChar
    1, // ncclUint8
    4, // ncclInt32
    4, // ncclUint32
    2, // ncclFloat16/ncclHalf
    4, // ncclFloat32
    8, // ncclFloat64
    8, // ncclInt64
    8, // ncclUint64
    2, // ncclBfloat16
];

// cudaMemcpyKind values (matches CUDA enum)
pub const MEMCPY_H2H: u32 = 0;
pub const MEMCPY_H2D: u32 = 1;
pub const MEMCPY_D2H: u32 = 2;
pub const MEMCPY_D2D: u32 = 3;

/// Event sent from eBPF to userspace via perf buffer.
/// All fields are naturally aligned for eBPF compatibility.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CudaEvent {
    pub event_type: u32,
    pub pid: u32,
    pub tid: u32,
    pub memcpy_kind: u32,
    pub timestamp_ns: u64,
    pub duration_ns: u64,
    pub size: u64,
    pub addr: u64,
    pub stream: u64,
    pub nvtx_marker: [u8; 16],
    pub comm: [u8; 16],
    pub error_code: u32,
    pub _pad2: u32,
}

/// Kernel registration event sent from __cudaRegisterFunction / cuModuleGetFunction
/// uprobes. Maps a host stub address to a kernel-name pointer in the target
/// process' address space. The name string itself is resolved lazily in
/// userspace via /proc/<pid>/mem — keeping this event tiny (24 bytes) and
/// avoiding bpf_probe_read_user_str on the hot path.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct KernelRegEvent {
    pub pid: u32,
    pub _pad: u32,
    pub host_fun: u64,
    pub name_ptr: u64,
}

/// Data stored in the eBPF hashmap between uprobe entry and uretprobe exit.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct EntryData {
    pub timestamp_ns: u64,
    pub arg0: u64,
    pub arg1: u64,
    pub arg2: u64,
    pub event_type: u32,
    pub _pad: u32,
}

/// Key for in-kernel aggregated metrics.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct AggKey {
    pub event_type: u32,
    pub pid: u32,
    pub memcpy_kind: u32,
    pub error_code: u32,
    pub stream: u64,
}

/// Aggregated metric values accumulated in eBPF per-CPU hash map.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct AggValue {
    pub count: u64,
    pub duration_sum_ns: u64,
    pub size_sum: u64,
    pub bucket_counts: [u32; 14],
}

/// Key for in-kernel aggregated launch metrics (LAUNCH_AGG map).
/// Separate from AggKey: launches carry host_fun (for per-kernel breakdown in
/// full mode) and stream, but not memcpy_kind/error_code.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LaunchKey {
    pub pid: u32,
    pub _pad: u32,
    pub host_fun: u64,
    pub stream: u64,
}

/// Aggregated launch-kernel values accumulated in eBPF per-CPU hash map.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LaunchAggValue {
    pub count: u64,
    pub total_duration_ns: u64,
    pub max_duration_ns: u64,
    pub bucket_counts: [u32; 9],
}

/// Bucket upper bounds for NCCL latency histograms, in nanoseconds.
/// Mirrors the Prometheus bucket layout declared for
/// `profi_nccl_duration_seconds` in `src/metrics.rs`:
/// \[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1, 0.5, 1.0\] seconds.
/// Values exceeding the last bound fall into the `+Inf` bucket (index 11).
pub const NCCL_BUCKET_BOUNDS_NS: [u64; 11] = [
    10_000,
    50_000,
    100_000,
    500_000,
    1_000_000,
    5_000_000,
    10_000_000,
    50_000_000,
    100_000_000,
    500_000_000,
    1_000_000_000,
];

/// Bucket upper bounds for aggregate CUDA API latency counters, in nanoseconds.
/// Mirrors `profi_cuda_duration_seconds` finite buckets.
pub const CUDA_BUCKET_BOUNDS_NS: [u64; 13] = [
    1_000,
    5_000,
    10_000,
    50_000,
    100_000,
    500_000,
    1_000_000,
    5_000_000,
    10_000_000,
    50_000_000,
    100_000_000,
    500_000_000,
    1_000_000_000,
];

/// Bucket upper bounds for aggregate kernel launch latency counters, in nanoseconds.
/// Mirrors `profi_cuda_kernel_duration_seconds` finite buckets.
pub const KERNEL_BUCKET_BOUNDS_NS: [u64; 8] = [
    1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 10_000_000,
];

/// NCCL collective aggregate accumulated per-CPU in the `NCCL_AGG` map.
/// Unlike `LaunchAggValue`, carries an inline bucket histogram — straggler
/// detection needs P95/P99 and NCCL call frequency (~1000/s) is low enough
/// for userspace to `observe()` once per bucket hit without eating the drain
/// budget.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NcclAggValue {
    pub count: u64,
    pub duration_sum_ns: u64,
    pub bytes_sum: u64,
    /// 11 finite buckets + 1 `+Inf` bucket = 12 entries; same order as
    /// `NCCL_BUCKET_BOUNDS_NS`.
    pub bucket_counts: [u32; 12],
}

/// Map a duration in nanoseconds to the `NcclAggValue::bucket_counts` index.
#[inline(always)]
pub fn nccl_bucket_idx(duration_ns: u64) -> usize {
    if duration_ns <= NCCL_BUCKET_BOUNDS_NS[0] {
        return 0;
    }
    if duration_ns <= NCCL_BUCKET_BOUNDS_NS[1] {
        return 1;
    }
    if duration_ns <= NCCL_BUCKET_BOUNDS_NS[2] {
        return 2;
    }
    if duration_ns <= NCCL_BUCKET_BOUNDS_NS[3] {
        return 3;
    }
    if duration_ns <= NCCL_BUCKET_BOUNDS_NS[4] {
        return 4;
    }
    if duration_ns <= NCCL_BUCKET_BOUNDS_NS[5] {
        return 5;
    }
    if duration_ns <= NCCL_BUCKET_BOUNDS_NS[6] {
        return 6;
    }
    if duration_ns <= NCCL_BUCKET_BOUNDS_NS[7] {
        return 7;
    }
    if duration_ns <= NCCL_BUCKET_BOUNDS_NS[8] {
        return 8;
    }
    if duration_ns <= NCCL_BUCKET_BOUNDS_NS[9] {
        return 9;
    }
    if duration_ns <= NCCL_BUCKET_BOUNDS_NS[10] {
        return 10;
    }
    11 // +Inf bucket
}

#[inline(always)]
pub fn cuda_bucket_idx(duration_ns: u64) -> usize {
    for (i, bound) in CUDA_BUCKET_BOUNDS_NS.iter().enumerate() {
        if duration_ns <= *bound {
            return i;
        }
    }
    CUDA_BUCKET_BOUNDS_NS.len()
}

#[inline(always)]
pub fn kernel_bucket_idx(duration_ns: u64) -> usize {
    for (i, bound) in KERNEL_BUCKET_BOUNDS_NS.iter().enumerate() {
        if duration_ns <= *bound {
            return i;
        }
    }
    KERNEL_BUCKET_BOUNDS_NS.len()
}

/// Check if an event type is an NCCL collective. Used by the eBPF emitter
/// to route into the dedicated NCCL_AGG map with bucket histogram.
pub fn is_nccl_event(event_type: u32) -> bool {
    matches!(event_type, EVENT_NCCL_ALL_REDUCE..=EVENT_NCCL_RECV)
}

/// Check if an event type should be aggregated in-kernel
pub fn is_aggregatable(event_type: u32) -> bool {
    matches!(
        event_type,
        EVENT_CUDA_MALLOC
            | EVENT_CUDA_FREE
            | EVENT_CUDA_MEMCPY
            | EVENT_CUDA_MEMCPY_ASYNC
            | EVENT_CUDA_MALLOC_ASYNC
            | EVENT_CUDA_FREE_ASYNC
            | EVENT_CUDA_MALLOC_HOST
            | EVENT_CUDA_FREE_HOST
            | EVENT_CUDA_MEMSET
            | EVENT_CUDA_MEMSET_ASYNC
            | EVENT_CUDA_STREAM_SYNC
            | EVENT_CUDA_EVENT_SYNC
            // Userspace uses only the call count for these; moving them off
            // the ringbuf hot path saves ~300 ringbuf events per 1000 prompts
            // for GRAPH_LAUNCH and trims init-path noise for the other two.
            | EVENT_CUDA_GRAPH_LAUNCH
            | EVENT_CUDA_GRAPH_INSTANTIATE
            | EVENT_CUDA_MODULE_LOAD
    )
}

// ── C↔Rust layout asserts ──────────────────────────────────────────────
//
// Catches any drift between src/bpf/profi_events.h (source of
// truth for the BPF C side) and these Rust mirrors. Every struct here must
// also appear in the header with the same field order, sizes, and offsets.

use core::mem::{offset_of, size_of};

const _: () = assert!(size_of::<CudaEvent>() == 96);
const _: () = assert!(offset_of!(CudaEvent, event_type) == 0);
const _: () = assert!(offset_of!(CudaEvent, pid) == 4);
const _: () = assert!(offset_of!(CudaEvent, tid) == 8);
const _: () = assert!(offset_of!(CudaEvent, memcpy_kind) == 12);
const _: () = assert!(offset_of!(CudaEvent, timestamp_ns) == 16);
const _: () = assert!(offset_of!(CudaEvent, duration_ns) == 24);
const _: () = assert!(offset_of!(CudaEvent, size) == 32);
const _: () = assert!(offset_of!(CudaEvent, addr) == 40);
const _: () = assert!(offset_of!(CudaEvent, stream) == 48);
const _: () = assert!(offset_of!(CudaEvent, nvtx_marker) == 56);
const _: () = assert!(offset_of!(CudaEvent, comm) == 72);
const _: () = assert!(offset_of!(CudaEvent, error_code) == 88);
const _: () = assert!(offset_of!(CudaEvent, _pad2) == 92);

const _: () = assert!(size_of::<KernelRegEvent>() == 24);
const _: () = assert!(offset_of!(KernelRegEvent, pid) == 0);
const _: () = assert!(offset_of!(KernelRegEvent, _pad) == 4);
const _: () = assert!(offset_of!(KernelRegEvent, host_fun) == 8);
const _: () = assert!(offset_of!(KernelRegEvent, name_ptr) == 16);

const _: () = assert!(size_of::<EntryData>() == 40);
const _: () = assert!(offset_of!(EntryData, timestamp_ns) == 0);
const _: () = assert!(offset_of!(EntryData, arg0) == 8);
const _: () = assert!(offset_of!(EntryData, arg1) == 16);
const _: () = assert!(offset_of!(EntryData, arg2) == 24);
const _: () = assert!(offset_of!(EntryData, event_type) == 32);
const _: () = assert!(offset_of!(EntryData, _pad) == 36);

const _: () = assert!(size_of::<AggKey>() == 24);
const _: () = assert!(offset_of!(AggKey, event_type) == 0);
const _: () = assert!(offset_of!(AggKey, pid) == 4);
const _: () = assert!(offset_of!(AggKey, memcpy_kind) == 8);
const _: () = assert!(offset_of!(AggKey, error_code) == 12);
const _: () = assert!(offset_of!(AggKey, stream) == 16);

const _: () = assert!(size_of::<AggValue>() == 80);
const _: () = assert!(offset_of!(AggValue, count) == 0);
const _: () = assert!(offset_of!(AggValue, duration_sum_ns) == 8);
const _: () = assert!(offset_of!(AggValue, size_sum) == 16);
const _: () = assert!(offset_of!(AggValue, bucket_counts) == 24);

const _: () = assert!(size_of::<LaunchKey>() == 24);
const _: () = assert!(offset_of!(LaunchKey, pid) == 0);
const _: () = assert!(offset_of!(LaunchKey, _pad) == 4);
const _: () = assert!(offset_of!(LaunchKey, host_fun) == 8);
const _: () = assert!(offset_of!(LaunchKey, stream) == 16);

const _: () = assert!(size_of::<LaunchAggValue>() == 64);
const _: () = assert!(offset_of!(LaunchAggValue, count) == 0);
const _: () = assert!(offset_of!(LaunchAggValue, total_duration_ns) == 8);
const _: () = assert!(offset_of!(LaunchAggValue, max_duration_ns) == 16);
const _: () = assert!(offset_of!(LaunchAggValue, bucket_counts) == 24);

const _: () = assert!(size_of::<NcclAggValue>() == 72);
const _: () = assert!(offset_of!(NcclAggValue, count) == 0);
const _: () = assert!(offset_of!(NcclAggValue, duration_sum_ns) == 8);
const _: () = assert!(offset_of!(NcclAggValue, bytes_sum) == 16);
const _: () = assert!(offset_of!(NcclAggValue, bucket_counts) == 24);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nccl_bucket_idx_covers_boundaries() {
        // Inclusive upper bounds — the boundary value belongs to its own bucket.
        assert_eq!(nccl_bucket_idx(0), 0);
        assert_eq!(nccl_bucket_idx(NCCL_BUCKET_BOUNDS_NS[0]), 0);
        assert_eq!(nccl_bucket_idx(NCCL_BUCKET_BOUNDS_NS[0] + 1), 1);
        assert_eq!(nccl_bucket_idx(NCCL_BUCKET_BOUNDS_NS[10]), 10);
        assert_eq!(nccl_bucket_idx(NCCL_BUCKET_BOUNDS_NS[10] + 1), 11);
        assert_eq!(nccl_bucket_idx(u64::MAX), 11);
    }

    #[test]
    fn cuda_bucket_idx_covers_boundaries() {
        assert_eq!(cuda_bucket_idx(0), 0);
        assert_eq!(cuda_bucket_idx(CUDA_BUCKET_BOUNDS_NS[0]), 0);
        assert_eq!(cuda_bucket_idx(CUDA_BUCKET_BOUNDS_NS[0] + 1), 1);
        assert_eq!(cuda_bucket_idx(CUDA_BUCKET_BOUNDS_NS[12]), 12);
        assert_eq!(cuda_bucket_idx(CUDA_BUCKET_BOUNDS_NS[12] + 1), 13);
        assert_eq!(cuda_bucket_idx(u64::MAX), 13);
    }

    #[test]
    fn kernel_bucket_idx_covers_boundaries() {
        assert_eq!(kernel_bucket_idx(0), 0);
        assert_eq!(kernel_bucket_idx(KERNEL_BUCKET_BOUNDS_NS[0]), 0);
        assert_eq!(kernel_bucket_idx(KERNEL_BUCKET_BOUNDS_NS[0] + 1), 1);
        assert_eq!(kernel_bucket_idx(KERNEL_BUCKET_BOUNDS_NS[7]), 7);
        assert_eq!(kernel_bucket_idx(KERNEL_BUCKET_BOUNDS_NS[7] + 1), 8);
        assert_eq!(kernel_bucket_idx(u64::MAX), 8);
    }
}
