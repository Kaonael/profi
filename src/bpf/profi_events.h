/* SPDX-License-Identifier: Apache-2.0 */

/*
 * profi_events.h — shared eBPF/userspace layout.
 *
 * Source of truth for struct layout and event-type constants. The Rust mirror
 * in src/events.rs carries compile-time size_of/offset_of asserts
 * that must match every field here.
 *
 * Included by:
 *   - src/bpf/profi.bpf.c (BPF C, compiled with clang -target bpf)
 */

#ifndef PROFI_EVENTS_H
#define PROFI_EVENTS_H

#ifdef __bpf__
/* BPF build: integer typedefs come from vmlinux.h/bpf_helpers.h.
 * Be conservative — include only what's needed for typedefs. */
typedef unsigned char __u8;
typedef unsigned int __u32;
typedef unsigned long long __u64;
#else
#include <stdint.h>
typedef uint8_t __u8;
typedef uint32_t __u32;
typedef uint64_t __u64;
#endif

#define EVENT_CUDA_MALLOC 1
#define EVENT_CUDA_FREE 2
#define EVENT_CUDA_MEMCPY 3
#define EVENT_CUDA_LAUNCH_KERNEL 4
#define EVENT_CUDA_MEMCPY_ASYNC 5
#define EVENT_CUDA_MALLOC_ASYNC 6
#define EVENT_CUDA_FREE_ASYNC 7
#define EVENT_CUDA_REGISTER_FUNCTION 8

#define EVENT_NCCL_ALL_REDUCE 9
#define EVENT_NCCL_ALL_GATHER 10
#define EVENT_NCCL_REDUCE_SCATTER 11
#define EVENT_NCCL_BROADCAST 12
#define EVENT_NCCL_SEND 13
#define EVENT_NCCL_RECV 14

#define EVENT_CUDA_STREAM_SYNC 15
#define EVENT_CUDA_EVENT_SYNC 16
#define EVENT_CUDA_MALLOC_HOST 17
#define EVENT_CUDA_FREE_HOST 18
#define EVENT_CUDA_MEMSET 19
#define EVENT_CUDA_MEMSET_ASYNC 20
#define EVENT_CUDA_GRAPH_LAUNCH 21
#define EVENT_CUDA_GRAPH_INSTANTIATE 22
#define EVENT_CUDA_MODULE_LOAD 23

#define MEMCPY_H2H 0
#define MEMCPY_H2D 1
#define MEMCPY_D2H 2
#define MEMCPY_D2D 3

static const __u64 NCCL_DTYPE_SIZES[10] = {
    1,
    1,
    4,
    4,
    2,
    4,
    8,
    8,
    8,
    2,
};

/* NCCL histogram bucket upper bounds (nanoseconds). Matches Prometheus
 * profi_nccl_duration_seconds layout. Values above the last bound land in
 * the +Inf bucket (index 11). */
static const __u64 NCCL_BUCKET_BOUNDS_NS[11] = {
    10000ULL,
    50000ULL,
    100000ULL,
    500000ULL,
    1000000ULL,
    5000000ULL,
    10000000ULL,
    50000000ULL,
    100000000ULL,
    500000000ULL,
    1000000000ULL,
};

/* CUDA aggregate histogram bucket upper bounds (nanoseconds). Matches
 * profi_cuda_duration_seconds finite buckets. Index 13 is +Inf. */
static const __u64 CUDA_BUCKET_BOUNDS_NS[13] = {
    1000ULL,
    5000ULL,
    10000ULL,
    50000ULL,
    100000ULL,
    500000ULL,
    1000000ULL,
    5000000ULL,
    10000000ULL,
    50000000ULL,
    100000000ULL,
    500000000ULL,
    1000000000ULL,
};

/* Kernel launch aggregate histogram bucket upper bounds (nanoseconds). Matches
 * profi_cuda_kernel_duration_seconds finite buckets. Index 8 is +Inf. */
static const __u64 KERNEL_BUCKET_BOUNDS_NS[8] = {
    1000ULL,
    5000ULL,
    10000ULL,
    50000ULL,
    100000ULL,
    500000ULL,
    1000000ULL,
    10000000ULL,
};

struct CudaEvent {
    __u32 event_type;
    __u32 pid;
    __u32 tid;
    __u32 memcpy_kind;
    __u64 timestamp_ns;
    __u64 duration_ns;
    __u64 size;
    __u64 addr;
    __u64 stream;
    __u8 nvtx_marker[16];
    __u8 comm[16];
    __u32 error_code;
    __u32 _pad2;
};

/* __cudaRegisterFunction / cuModuleGetFunction event. 24 bytes.
 * Maps host stub → kernel-name pointer in target process. String itself
 * is resolved lazily in userspace via /proc/<pid>/mem. */
struct KernelRegEvent {
    __u32 pid;
    __u32 _pad;
    __u64 host_fun;
    __u64 name_ptr;
};

struct EntryData {
    __u64 timestamp_ns;
    __u64 arg0;
    __u64 arg1;
    __u64 arg2;
    __u32 event_type;
    __u32 _pad;
};

struct AggKey {
    __u32 event_type;
    __u32 pid;
    __u32 memcpy_kind;
    __u32 error_code;
    __u64 stream;
};

struct AggValue {
    __u64 count;
    __u64 duration_sum_ns;
    __u64 size_sum;
    __u32 bucket_counts[14];
};

struct LaunchKey {
    __u32 pid;
    __u32 _pad;
    __u64 host_fun;
    __u64 stream;
};

struct LaunchAggValue {
    __u64 count;
    __u64 total_duration_ns;
    __u64 max_duration_ns;
    __u32 bucket_counts[9];
};

/* NCCL_AGG value with inline 12-bucket latency histogram. 72 bytes.
 * bucket_counts index 0..10 are finite buckets; index 11 is +Inf. */
struct NcclAggValue {
    __u64 count;
    __u64 duration_sum_ns;
    __u64 bytes_sum;
    __u32 bucket_counts[12];
};

static __inline int profi_is_nccl_event(__u32 event_type)
{
    return event_type >= EVENT_NCCL_ALL_REDUCE && event_type <= EVENT_NCCL_RECV;
}

static __inline int profi_is_aggregatable(__u32 event_type)
{
    switch (event_type) {
    case EVENT_CUDA_MALLOC:
    case EVENT_CUDA_FREE:
    case EVENT_CUDA_MEMCPY:
    case EVENT_CUDA_MEMCPY_ASYNC:
    case EVENT_CUDA_MALLOC_ASYNC:
    case EVENT_CUDA_FREE_ASYNC:
    case EVENT_CUDA_MALLOC_HOST:
    case EVENT_CUDA_FREE_HOST:
    case EVENT_CUDA_MEMSET:
    case EVENT_CUDA_MEMSET_ASYNC:
    case EVENT_CUDA_STREAM_SYNC:
    case EVENT_CUDA_EVENT_SYNC:
    case EVENT_CUDA_GRAPH_LAUNCH:
    case EVENT_CUDA_GRAPH_INSTANTIATE:
    case EVENT_CUDA_MODULE_LOAD:
        return 1;
    default:
        return 0;
    }
}

/* Map duration_ns to NcclAggValue.bucket_counts index (0..11 inclusive).
 * Inclusive upper bounds — boundary value belongs to its own bucket. */
static __inline int profi_nccl_bucket_idx(__u64 duration_ns)
{
    if (duration_ns <= NCCL_BUCKET_BOUNDS_NS[0]) return 0;
    if (duration_ns <= NCCL_BUCKET_BOUNDS_NS[1]) return 1;
    if (duration_ns <= NCCL_BUCKET_BOUNDS_NS[2]) return 2;
    if (duration_ns <= NCCL_BUCKET_BOUNDS_NS[3]) return 3;
    if (duration_ns <= NCCL_BUCKET_BOUNDS_NS[4]) return 4;
    if (duration_ns <= NCCL_BUCKET_BOUNDS_NS[5]) return 5;
    if (duration_ns <= NCCL_BUCKET_BOUNDS_NS[6]) return 6;
    if (duration_ns <= NCCL_BUCKET_BOUNDS_NS[7]) return 7;
    if (duration_ns <= NCCL_BUCKET_BOUNDS_NS[8]) return 8;
    if (duration_ns <= NCCL_BUCKET_BOUNDS_NS[9]) return 9;
    if (duration_ns <= NCCL_BUCKET_BOUNDS_NS[10]) return 10;
    return 11;
}

static __inline int profi_cuda_bucket_idx(__u64 duration_ns)
{
    if (duration_ns <= CUDA_BUCKET_BOUNDS_NS[0]) return 0;
    if (duration_ns <= CUDA_BUCKET_BOUNDS_NS[1]) return 1;
    if (duration_ns <= CUDA_BUCKET_BOUNDS_NS[2]) return 2;
    if (duration_ns <= CUDA_BUCKET_BOUNDS_NS[3]) return 3;
    if (duration_ns <= CUDA_BUCKET_BOUNDS_NS[4]) return 4;
    if (duration_ns <= CUDA_BUCKET_BOUNDS_NS[5]) return 5;
    if (duration_ns <= CUDA_BUCKET_BOUNDS_NS[6]) return 6;
    if (duration_ns <= CUDA_BUCKET_BOUNDS_NS[7]) return 7;
    if (duration_ns <= CUDA_BUCKET_BOUNDS_NS[8]) return 8;
    if (duration_ns <= CUDA_BUCKET_BOUNDS_NS[9]) return 9;
    if (duration_ns <= CUDA_BUCKET_BOUNDS_NS[10]) return 10;
    if (duration_ns <= CUDA_BUCKET_BOUNDS_NS[11]) return 11;
    if (duration_ns <= CUDA_BUCKET_BOUNDS_NS[12]) return 12;
    return 13;
}

static __inline int profi_kernel_bucket_idx(__u64 duration_ns)
{
    if (duration_ns <= KERNEL_BUCKET_BOUNDS_NS[0]) return 0;
    if (duration_ns <= KERNEL_BUCKET_BOUNDS_NS[1]) return 1;
    if (duration_ns <= KERNEL_BUCKET_BOUNDS_NS[2]) return 2;
    if (duration_ns <= KERNEL_BUCKET_BOUNDS_NS[3]) return 3;
    if (duration_ns <= KERNEL_BUCKET_BOUNDS_NS[4]) return 4;
    if (duration_ns <= KERNEL_BUCKET_BOUNDS_NS[5]) return 5;
    if (duration_ns <= KERNEL_BUCKET_BOUNDS_NS[6]) return 6;
    if (duration_ns <= KERNEL_BUCKET_BOUNDS_NS[7]) return 7;
    return 8;
}

#endif
