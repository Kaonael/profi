/* SPDX-License-Identifier: Apache-2.0 */

/*
 * profi.bpf.c — CUDA/NCCL uprobe instrumentation.
 *
 * Port of the old aya-ebpf program to libbpf-style C.
 * Struct layout and event constants come from profi_events.h; Rust mirror
 * in src/events.rs carries compile-time size/offset asserts.
 */

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include "profi_events.h"

char LICENSE[] SEC("license") = "Dual BSD/GPL";

/* ── Stack-arg helper ──────────────────────────────────────────────────
 * aya's ctx.arg(N) transparently pulls from stack when N >= 6 (x86_64 SysV).
 * In libbpf C we must read the user stack manually for that case.
 * `n` is the 0-based arg index. Only valid for n >= 6.
 * At uprobe fire time [sp] holds the return address, [sp+8] = 1st stack arg.
 */
#if defined(__TARGET_ARCH_x86) || defined(__x86_64__)
#define PT_REGS_PARM_STACK(ctx, n)                                                                 \
    ({                                                                                             \
        __u64 __v = 0;                                                                             \
        bpf_probe_read_user(&__v, sizeof(__v), (void *)(PT_REGS_SP(ctx) + (__u64)8 * ((n) - 5)));  \
        __v;                                                                                       \
    })
#else
#error "PT_REGS_PARM_STACK implemented only for x86_64"
#endif

/* ── Map definitions ───────────────────────────────────────────────────
 * max_entries defaults match the Rust source; userspace overrides three of
 * them (INFLIGHT, AGGREGATED, MALLOC_SIZES) via open-skeleton set_max_entries.
 */

struct MallocKey {
    __u32 pid;
    __u32 _pad;
    __u64 addr;
};

struct nvtx_name_t {
    __u8 name[64];
};

struct {
    __uint(type, BPF_MAP_TYPE_LRU_HASH);
    __uint(max_entries, 10240);
    __type(key, __u64);
    __type(value, struct EntryData);
} INFLIGHT SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 2 * 1024 * 1024);
} EVENTS SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024);
} KERNEL_REGS SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_HASH);
    __uint(max_entries, 2048);
    __type(key, struct AggKey);
    __type(value, struct AggValue);
} AGGREGATED SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_HASH);
    __uint(max_entries, 8192);
    __type(key, struct LaunchKey);
    __type(value, struct LaunchAggValue);
} LAUNCH_AGG SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, __u64);
} LAUNCH_DROPPED SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, __u32);
} DETAILED_LAUNCHES SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1024);
    __type(key, __u32);
    __type(value, __u8);
} UPGRADED_PIDS SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_HASH);
    __uint(max_entries, 1024);
    __type(key, struct AggKey);
    __type(value, struct NcclAggValue);
} NCCL_AGG SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, __u64);
} DROPPED SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_LRU_HASH);
    __uint(max_entries, 1024);
    __type(key, __u64);
    __type(value, struct nvtx_name_t);
} NVTX_STACK SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, __u32);
} KERNEL_AGG_MODE SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, __u32);
} SAMPLE_RATE SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, __u64);
} SAMPLE_COUNTER SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_LRU_HASH);
    __uint(max_entries, 131072);
    __type(key, struct MallocKey);
    __type(value, __u64);
} MALLOC_SIZES SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_LRU_HASH);
    __uint(max_entries, 512);
    __type(key, __u64);
    __type(value, __u64);
} MALLOC_PTRS SEC(".maps");

static __always_inline __u32 cfg_u32(void *map)
{
    __u32 zero = 0;
    __u32 *v = bpf_map_lookup_elem(map, &zero);
    return v ? *v : 0;
}

static __always_inline void save_entry(__u32 event_type, __u64 arg0, __u64 arg1, __u64 arg2)
{
    __u64 key = bpf_get_current_pid_tgid();

    /* Recursion guard: if an entry already exists, a wrapper probe fired
     * (e.g. cudaLaunchKernel → cuLaunchKernel). Skip inner. */
    if (bpf_map_lookup_elem(&INFLIGHT, &key)) return;

    if (profi_is_aggregatable(event_type)) {
        __u32 rate = cfg_u32(&SAMPLE_RATE);
        if (rate > 1) {
            __u32 zero = 0;
            __u64 *counter = bpf_map_lookup_elem(&SAMPLE_COUNTER, &zero);
            if (counter) {
                __u64 c = *counter;
                *counter = c + 1;
                if (c % (__u64)rate != 0) return;
            }
        }
    }

    struct EntryData entry = {
        .timestamp_ns = bpf_ktime_get_ns(),
        .arg0 = arg0,
        .arg1 = arg1,
        .arg2 = arg2,
        .event_type = event_type,
        ._pad = 0,
    };
    bpf_map_update_elem(&INFLIGHT, &key, &entry, BPF_ANY);
}

static __always_inline void bump_dropped(void *map)
{
    __u32 zero = 0;
    __u64 *cnt = bpf_map_lookup_elem(map, &zero);
    if (cnt) (*cnt)++;
}

static __always_inline int emit_exit(struct pt_regs *ctx)
{
    __u32 rc = (__u32)PT_REGS_RC(ctx);
    __u64 id = bpf_get_current_pid_tgid();
    __u32 pid = id >> 32;
    __u32 tid = (__u32)id;

    struct EntryData *e = bpf_map_lookup_elem(&INFLIGHT, &id);
    if (!e) return 0;
    struct EntryData entry = *e;

    __u64 now = bpf_ktime_get_ns();
    __u64 duration_ns = now > entry.timestamp_ns ? now - entry.timestamp_ns : 0;

    if (profi_is_aggregatable(entry.event_type)) {
        __u64 sample_mul = 1;
        __u32 rate = cfg_u32(&SAMPLE_RATE);
        if (rate > 1) sample_mul = rate;
        __u32 bucket_inc = (__u32)sample_mul;
        int idx = profi_cuda_bucket_idx(duration_ns);

        struct AggKey key = {
            .event_type = entry.event_type,
            .pid = pid,
            .memcpy_kind = (__u32)entry.arg1,
            .error_code = rc,
            .stream = entry.arg2,
        };

        struct AggValue *val = bpf_map_lookup_elem(&AGGREGATED, &key);
        if (val) {
            val->count += sample_mul;
            val->duration_sum_ns += duration_ns * sample_mul;
            val->size_sum += entry.arg0 * sample_mul;
            if (idx >= 0 && idx < 14) val->bucket_counts[idx] += bucket_inc;
        } else {
            struct AggValue new_val = {
                .count = sample_mul,
                .duration_sum_ns = duration_ns * sample_mul,
                .size_sum = entry.arg0 * sample_mul,
                .bucket_counts = {0},
            };
            if (idx >= 0 && idx < 14) new_val.bucket_counts[idx] = bucket_inc;
            bpf_map_update_elem(&AGGREGATED, &key, &new_val, BPF_ANY);
        }
        bpf_map_delete_elem(&INFLIGHT, &id);
        return 0;
    }

    if (profi_is_nccl_event(entry.event_type)) {
        struct AggKey key = {
            .event_type = entry.event_type,
            .pid = pid,
            .memcpy_kind = (__u32)entry.arg1,
            .error_code = rc,
            .stream = 0,
        };
        __u64 datatype = entry.arg1;
        __u64 dtype_size = datatype < 10 ? NCCL_DTYPE_SIZES[datatype] : 1;
        /* Wrap is intentional — saturating mul lowers to __multi3 which the
         * verifier rejects. NCCL payloads stay well under 2^64 anyway. */
        __u64 bytes = entry.arg0 * dtype_size;
        int idx = profi_nccl_bucket_idx(duration_ns);

        struct NcclAggValue *val = bpf_map_lookup_elem(&NCCL_AGG, &key);
        if (val) {
            val->count += 1;
            val->duration_sum_ns += duration_ns;
            val->bytes_sum += bytes;
            if (idx >= 0 && idx < 12) val->bucket_counts[idx] += 1;
        } else {
            struct NcclAggValue new_val = {
                .count = 1,
                .duration_sum_ns = duration_ns,
                .bytes_sum = bytes,
                .bucket_counts = {0},
            };
            if (idx >= 0 && idx < 12) new_val.bucket_counts[idx] = 1;
            bpf_map_update_elem(&NCCL_AGG, &key, &new_val, BPF_ANY);
        }
        bpf_map_delete_elem(&INFLIGHT, &id);
        return 0;
    }

    /* Kernel launches: LAUNCH_AGG always, ringbuf only if full-mode + detailed.
     * UPGRADED_PIDS forces full-mode semantics regardless of global flag. */
    if (entry.event_type == EVENT_CUDA_LAUNCH_KERNEL) {
        __u32 global_mode = cfg_u32(&KERNEL_AGG_MODE);
        __u8 *upgraded = bpf_map_lookup_elem(&UPGRADED_PIDS, &pid);
        __u32 mode = upgraded ? 0 : global_mode;
        __u64 host_fun = (mode == 1) ? 0 : entry.arg0;

        struct LaunchKey lkey = {
            .pid = pid,
            ._pad = 0,
            .host_fun = host_fun,
            .stream = entry.arg2,
        };
        struct LaunchAggValue *lval = bpf_map_lookup_elem(&LAUNCH_AGG, &lkey);
        int lidx = profi_kernel_bucket_idx(duration_ns);
        if (lval) {
            lval->count += 1;
            lval->total_duration_ns += duration_ns;
            if (duration_ns > lval->max_duration_ns) lval->max_duration_ns = duration_ns;
            if (lidx >= 0 && lidx < 9) lval->bucket_counts[lidx] += 1;
        } else {
            struct LaunchAggValue new_val = {
                .count = 1,
                .total_duration_ns = duration_ns,
                .max_duration_ns = duration_ns,
                .bucket_counts = {0},
            };
            if (lidx >= 0 && lidx < 9) new_val.bucket_counts[lidx] = 1;
            if (bpf_map_update_elem(&LAUNCH_AGG, &lkey, &new_val, BPF_ANY) < 0)
                bump_dropped(&LAUNCH_DROPPED);
        }

        __u32 detailed = cfg_u32(&DETAILED_LAUNCHES);
        if (mode == 1 || detailed != 1) {
            bpf_map_delete_elem(&INFLIGHT, &id);
            return 0;
        }
    }

    struct CudaEvent *evt = bpf_ringbuf_reserve(&EVENTS, sizeof(*evt), 0);
    if (!evt) {
        bump_dropped(&DROPPED);
        bpf_map_delete_elem(&INFLIGHT, &id);
        return 0;
    }

    evt->event_type = entry.event_type;
    evt->pid = pid;
    evt->tid = tid;
    evt->memcpy_kind = (__u32)entry.arg1;
    evt->timestamp_ns = now;
    evt->duration_ns = duration_ns;
    evt->size = entry.arg0;
    evt->addr = entry.arg0;
    evt->stream = entry.arg2;
    evt->error_code = rc;
    evt->_pad2 = 0;

    bpf_get_current_comm(&evt->comm, sizeof(evt->comm));

    __builtin_memset(&evt->nvtx_marker, 0, sizeof(evt->nvtx_marker));
    struct nvtx_name_t *marker = bpf_map_lookup_elem(&NVTX_STACK, &id);
    if (marker) {
        __builtin_memcpy(&evt->nvtx_marker, marker->name, sizeof(evt->nvtx_marker));
    }

    bpf_ringbuf_submit(evt, 0);
    bpf_map_delete_elem(&INFLIGHT, &id);
    return 0;
}

SEC("uprobe/cuda_malloc")
int BPF_KPROBE(cuda_malloc, void *ptr_to_ptr, __u64 size)
{
    __u64 id = bpf_get_current_pid_tgid();
    __u64 pp = (__u64)ptr_to_ptr;
    if (pp != 0) bpf_map_update_elem(&MALLOC_PTRS, &id, &pp, BPF_ANY);
    save_entry(EVENT_CUDA_MALLOC, size, 0, 0);
    return 0;
}

SEC("uretprobe/cuda_malloc")
int BPF_KRETPROBE(cuda_malloc_ret)
{
    __u64 id = bpf_get_current_pid_tgid();
    __u32 pid = id >> 32;
    __u64 *pp = bpf_map_lookup_elem(&MALLOC_PTRS, &id);
    if (pp) {
        __u64 ptr_to_ptr = *pp;
        bpf_map_delete_elem(&MALLOC_PTRS, &id);
        struct EntryData *entry = bpf_map_lookup_elem(&INFLIGHT, &id);
        if (entry) {
            __u64 dev_ptr = 0;
            if (bpf_probe_read_user(&dev_ptr, sizeof(dev_ptr), (void *)ptr_to_ptr) == 0 &&
                dev_ptr) {
                struct MallocKey mk = {.pid = pid, ._pad = 0, .addr = dev_ptr};
                __u64 size = entry->arg0;
                bpf_map_update_elem(&MALLOC_SIZES, &mk, &size, BPF_ANY);
            }
        }
    }
    return emit_exit(ctx);
}

SEC("uprobe/cuda_free")
int BPF_KPROBE(cuda_free, void *ptr)
{
    __u32 pid = bpf_get_current_pid_tgid() >> 32;
    __u64 freed_size = 0;
    __u64 pptr = (__u64)ptr;
    if (pptr) {
        struct MallocKey mk = {.pid = pid, ._pad = 0, .addr = pptr};
        __u64 *sz = bpf_map_lookup_elem(&MALLOC_SIZES, &mk);
        if (sz) {
            freed_size = *sz;
            bpf_map_delete_elem(&MALLOC_SIZES, &mk);
        }
    }
    save_entry(EVENT_CUDA_FREE, freed_size, 0, 0);
    return 0;
}

SEC("uretprobe/cuda_free")
int BPF_KRETPROBE(cuda_free_ret)
{
    return emit_exit(ctx);
}

SEC("uprobe/cuda_memcpy")
int BPF_KPROBE(cuda_memcpy, void *dst, void *src, __u64 count, __u64 kind)
{
    save_entry(EVENT_CUDA_MEMCPY, count, kind, 0);
    return 0;
}

SEC("uretprobe/cuda_memcpy")
int BPF_KRETPROBE(cuda_memcpy_ret)
{
    return emit_exit(ctx);
}

SEC("uprobe/cuda_memcpy_async")
int BPF_KPROBE(cuda_memcpy_async, void *dst, void *src, __u64 count, __u64 kind, __u64 stream)
{
    save_entry(EVENT_CUDA_MEMCPY_ASYNC, count, kind, stream);
    return 0;
}

SEC("uretprobe/cuda_memcpy_async")
int BPF_KRETPROBE(cuda_memcpy_async_ret)
{
    return emit_exit(ctx);
}

SEC("uprobe/cuda_launch_kernel")
int BPF_KPROBE(cuda_launch_kernel, void *func)
{
    save_entry(EVENT_CUDA_LAUNCH_KERNEL, (__u64)func, 0, 0);
    return 0;
}

SEC("uretprobe/cuda_launch_kernel")
int BPF_KRETPROBE(cuda_launch_kernel_ret)
{
    return emit_exit(ctx);
}

SEC("uprobe/cuda_malloc_async")
int BPF_KPROBE(cuda_malloc_async, void *ptr_to_ptr, __u64 size, __u64 stream)
{
    __u64 id = bpf_get_current_pid_tgid();
    __u64 pp = (__u64)ptr_to_ptr;
    if (pp != 0) bpf_map_update_elem(&MALLOC_PTRS, &id, &pp, BPF_ANY);
    save_entry(EVENT_CUDA_MALLOC_ASYNC, size, 0, stream);
    return 0;
}

SEC("uretprobe/cuda_malloc_async")
int BPF_KRETPROBE(cuda_malloc_async_ret)
{
    __u64 id = bpf_get_current_pid_tgid();
    __u32 pid = id >> 32;
    __u64 *pp = bpf_map_lookup_elem(&MALLOC_PTRS, &id);
    if (pp) {
        __u64 ptr_to_ptr = *pp;
        bpf_map_delete_elem(&MALLOC_PTRS, &id);
        struct EntryData *entry = bpf_map_lookup_elem(&INFLIGHT, &id);
        if (entry) {
            __u64 dev_ptr = 0;
            if (bpf_probe_read_user(&dev_ptr, sizeof(dev_ptr), (void *)ptr_to_ptr) == 0 &&
                dev_ptr) {
                struct MallocKey mk = {.pid = pid, ._pad = 0, .addr = dev_ptr};
                __u64 size = entry->arg0;
                bpf_map_update_elem(&MALLOC_SIZES, &mk, &size, BPF_ANY);
            }
        }
    }
    return emit_exit(ctx);
}

SEC("uprobe/cuda_free_async")
int BPF_KPROBE(cuda_free_async, void *ptr, __u64 stream)
{
    __u32 pid = bpf_get_current_pid_tgid() >> 32;
    __u64 freed_size = 0;
    __u64 pptr = (__u64)ptr;
    if (pptr) {
        struct MallocKey mk = {.pid = pid, ._pad = 0, .addr = pptr};
        __u64 *sz = bpf_map_lookup_elem(&MALLOC_SIZES, &mk);
        if (sz) {
            freed_size = *sz;
            bpf_map_delete_elem(&MALLOC_SIZES, &mk);
        }
    }
    save_entry(EVENT_CUDA_FREE_ASYNC, freed_size, 0, stream);
    return 0;
}

SEC("uretprobe/cuda_free_async")
int BPF_KRETPROBE(cuda_free_async_ret)
{
    return emit_exit(ctx);
}

/* __cudaRegisterFunction(fatCubin, hostFun, deviceFun, deviceName, ...) — emit
 * directly to KERNEL_REGS ringbuf; no entry/exit pairing. */
SEC("uprobe/cuda_register_function")
int BPF_KPROBE(cuda_register_function, void *fat, void *host_fun, void *dev_fun, void *dev_name)
{
    __u64 hf = (__u64)host_fun;
    __u64 np = (__u64)dev_name;
    if (!hf || !np) return 0;

    __u32 pid = bpf_get_current_pid_tgid() >> 32;
    struct KernelRegEvent *evt = bpf_ringbuf_reserve(&KERNEL_REGS, sizeof(*evt), 0);
    if (!evt) return 0;
    evt->pid = pid;
    evt->_pad = 0;
    evt->host_fun = hf;
    evt->name_ptr = np;
    bpf_ringbuf_submit(evt, 0);
    return 0;
}

/* cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) —
 * save pointers on entry, deref hfunc_ptr on exit, emit to KERNEL_REGS. */
SEC("uprobe/cu_module_get_function")
int BPF_KPROBE(cu_module_get_function, void *hfunc_ptr, void *hmod, void *name_ptr)
{
    __u64 hp = (__u64)hfunc_ptr;
    __u64 np = (__u64)name_ptr;
    if (!hp || !np) return 0;
    __u64 id = bpf_get_current_pid_tgid();
    struct EntryData entry = {
        .timestamp_ns = bpf_ktime_get_ns(),
        .arg0 = hp,
        .arg1 = np,
        .arg2 = 0,
        .event_type = EVENT_CUDA_REGISTER_FUNCTION,
        ._pad = 0,
    };
    bpf_map_update_elem(&INFLIGHT, &id, &entry, BPF_ANY);
    return 0;
}

SEC("uretprobe/cu_module_get_function")
int BPF_KRETPROBE(cu_module_get_function_ret)
{
    __u64 id = bpf_get_current_pid_tgid();
    __u32 pid = id >> 32;
    struct EntryData *e = bpf_map_lookup_elem(&INFLIGHT, &id);
    if (!e || e->event_type != EVENT_CUDA_REGISTER_FUNCTION) return 0;
    __u64 hfunc_ptr = e->arg0;
    __u64 name_ptr = e->arg1;

    __u64 hfunc = 0;
    if (bpf_probe_read_user(&hfunc, sizeof(hfunc), (void *)hfunc_ptr) != 0) {
        bpf_map_delete_elem(&INFLIGHT, &id);
        return 0;
    }

    struct KernelRegEvent *evt = bpf_ringbuf_reserve(&KERNEL_REGS, sizeof(*evt), 0);
    if (evt) {
        evt->pid = pid;
        evt->_pad = 0;
        evt->host_fun = hfunc;
        evt->name_ptr = name_ptr;
        bpf_ringbuf_submit(evt, 0);
    }
    bpf_map_delete_elem(&INFLIGHT, &id);
    return 0;
}

SEC("uprobe/cu_launch_kernel")
int BPF_KPROBE(cu_launch_kernel, void *func)
{
    save_entry(EVENT_CUDA_LAUNCH_KERNEL, (__u64)func, 0, 0);
    return 0;
}

SEC("uretprobe/cu_launch_kernel")
int BPF_KRETPROBE(cu_launch_kernel_ret)
{
    return emit_exit(ctx);
}

/* cuLaunchKernelEx(const CUlaunchConfig *config, CUfunction f, ...) —
 * func is 2nd arg (index 1). Stream lives inside config; we skip reading it. */
SEC("uprobe/cu_launch_kernel_ex")
int BPF_KPROBE(cu_launch_kernel_ex, void *config, void *func)
{
    save_entry(EVENT_CUDA_LAUNCH_KERNEL, (__u64)func, 0, 0);
    return 0;
}

SEC("uretprobe/cu_launch_kernel_ex")
int BPF_KRETPROBE(cu_launch_kernel_ex_ret)
{
    return emit_exit(ctx);
}

/* cuLaunchCooperativeKernel(CUfunction f, gridX, gridY, gridZ, blockX, blockY,
 * blockZ, sharedMem, CUstream hStream, void **params)
 * stream is arg index 8 (9th) — sits on the user stack on x86_64 SysV. */
SEC("uprobe/cu_launch_cooperative_kernel")
int BPF_KPROBE(cu_launch_cooperative_kernel, void *func)
{
    __u64 stream = PT_REGS_PARM_STACK(ctx, 8);
    save_entry(EVENT_CUDA_LAUNCH_KERNEL, (__u64)func, 0, stream);
    return 0;
}

SEC("uretprobe/cu_launch_cooperative_kernel")
int BPF_KRETPROBE(cu_launch_cooperative_kernel_ret)
{
    return emit_exit(ctx);
}

SEC("uprobe/cu_graph_launch")
int BPF_KPROBE(cu_graph_launch, void *graph, void *stream)
{
    save_entry(EVENT_CUDA_LAUNCH_KERNEL, (__u64)graph, 0, (__u64)stream);
    return 0;
}

SEC("uretprobe/cu_graph_launch")
int BPF_KRETPROBE(cu_graph_launch_ret)
{
    return emit_exit(ctx);
}

SEC("uprobe/cuda_stream_sync")
int BPF_KPROBE(cuda_stream_sync, void *stream)
{
    save_entry(EVENT_CUDA_STREAM_SYNC, 0, 0, (__u64)stream);
    return 0;
}

SEC("uretprobe/cuda_stream_sync")
int BPF_KRETPROBE(cuda_stream_sync_ret)
{
    return emit_exit(ctx);
}

SEC("uprobe/cuda_event_sync")
int BPF_KPROBE(cuda_event_sync, void *event)
{
    save_entry(EVENT_CUDA_EVENT_SYNC, (__u64)event, 0, 0);
    return 0;
}

SEC("uretprobe/cuda_event_sync")
int BPF_KRETPROBE(cuda_event_sync_ret)
{
    return emit_exit(ctx);
}

SEC("uprobe/cuda_malloc_host")
int BPF_KPROBE(cuda_malloc_host, void *ptr, __u64 size)
{
    save_entry(EVENT_CUDA_MALLOC_HOST, size, 0, 0);
    return 0;
}

SEC("uretprobe/cuda_malloc_host")
int BPF_KRETPROBE(cuda_malloc_host_ret)
{
    return emit_exit(ctx);
}

SEC("uprobe/cuda_free_host")
int BPF_KPROBE(cuda_free_host, void *ptr)
{
    save_entry(EVENT_CUDA_FREE_HOST, (__u64)ptr, 0, 0);
    return 0;
}

SEC("uretprobe/cuda_free_host")
int BPF_KRETPROBE(cuda_free_host_ret)
{
    return emit_exit(ctx);
}

SEC("uprobe/cuda_memset")
int BPF_KPROBE(cuda_memset, void *dst, int value, __u64 count)
{
    save_entry(EVENT_CUDA_MEMSET, count, 0, 0);
    return 0;
}

SEC("uretprobe/cuda_memset")
int BPF_KRETPROBE(cuda_memset_ret)
{
    return emit_exit(ctx);
}

SEC("uprobe/cuda_memset_async")
int BPF_KPROBE(cuda_memset_async, void *dst, int value, __u64 count, __u64 stream)
{
    save_entry(EVENT_CUDA_MEMSET_ASYNC, count, 0, stream);
    return 0;
}

SEC("uretprobe/cuda_memset_async")
int BPF_KRETPROBE(cuda_memset_async_ret)
{
    return emit_exit(ctx);
}

SEC("uprobe/cuda_graph_launch")
int BPF_KPROBE(cuda_graph_launch, void *graph, void *stream)
{
    save_entry(EVENT_CUDA_GRAPH_LAUNCH, (__u64)graph, 0, (__u64)stream);
    return 0;
}

SEC("uretprobe/cuda_graph_launch")
int BPF_KRETPROBE(cuda_graph_launch_ret)
{
    return emit_exit(ctx);
}

SEC("uprobe/cuda_graph_instantiate")
int BPF_KPROBE(cuda_graph_instantiate)
{
    save_entry(EVENT_CUDA_GRAPH_INSTANTIATE, 0, 0, 0);
    return 0;
}

SEC("uretprobe/cuda_graph_instantiate")
int BPF_KRETPROBE(cuda_graph_instantiate_ret)
{
    return emit_exit(ctx);
}

SEC("uprobe/cu_module_load_data")
int BPF_KPROBE(cu_module_load_data)
{
    save_entry(EVENT_CUDA_MODULE_LOAD, 0, 0, 0);
    return 0;
}

SEC("uretprobe/cu_module_load_data")
int BPF_KRETPROBE(cu_module_load_data_ret)
{
    return emit_exit(ctx);
}

SEC("uprobe/nvtx_range_push")
int BPF_KPROBE(nvtx_range_push, void *msg_ptr)
{
    __u64 mp = (__u64)msg_ptr;
    if (!mp) return 0;
    __u64 id = bpf_get_current_pid_tgid();
    struct nvtx_name_t name = {0};
    bpf_probe_read_user_str(&name.name, sizeof(name.name), (void *)mp);
    bpf_map_update_elem(&NVTX_STACK, &id, &name, BPF_ANY);
    return 0;
}

SEC("uprobe/nvtx_range_pop")
int BPF_KPROBE(nvtx_range_pop)
{
    __u64 id = bpf_get_current_pid_tgid();
    bpf_map_delete_elem(&NVTX_STACK, &id);
    return 0;
}

SEC("uprobe.multi/nccl_count_dtype_3_4")
int BPF_KPROBE(nccl_count_dtype_3_4, void *s, void *r, __u64 count, __u64 datatype)
{
    __u64 event_type = bpf_get_attach_cookie(ctx);
    save_entry((__u32)event_type, count, datatype, 0);
    return 0;
}

SEC("uprobe.multi/nccl_count_dtype_2_3")
int BPF_KPROBE(nccl_count_dtype_2_3, void *b, __u64 count, __u64 datatype)
{
    __u64 event_type = bpf_get_attach_cookie(ctx);
    save_entry((__u32)event_type, count, datatype, 0);
    return 0;
}

SEC("uretprobe.multi/nccl_multi_ret")
int BPF_KRETPROBE(nccl_multi_ret)
{
    return emit_exit(ctx);
}

SEC("uprobe/nccl_all_reduce")
int BPF_KPROBE(nccl_all_reduce, void *s, void *r, __u64 count, __u64 datatype)
{
    save_entry(EVENT_NCCL_ALL_REDUCE, count, datatype, 0);
    return 0;
}

SEC("uretprobe/nccl_all_reduce")
int BPF_KRETPROBE(nccl_all_reduce_ret)
{
    return emit_exit(ctx);
}

SEC("uprobe/nccl_all_gather")
int BPF_KPROBE(nccl_all_gather, void *s, void *r, __u64 count, __u64 datatype)
{
    save_entry(EVENT_NCCL_ALL_GATHER, count, datatype, 0);
    return 0;
}

SEC("uretprobe/nccl_all_gather")
int BPF_KRETPROBE(nccl_all_gather_ret)
{
    return emit_exit(ctx);
}

SEC("uprobe/nccl_reduce_scatter")
int BPF_KPROBE(nccl_reduce_scatter, void *s, void *r, __u64 count, __u64 datatype)
{
    save_entry(EVENT_NCCL_REDUCE_SCATTER, count, datatype, 0);
    return 0;
}

SEC("uretprobe/nccl_reduce_scatter")
int BPF_KRETPROBE(nccl_reduce_scatter_ret)
{
    return emit_exit(ctx);
}

SEC("uprobe/nccl_broadcast")
int BPF_KPROBE(nccl_broadcast, void *s, void *r, __u64 count, __u64 datatype)
{
    save_entry(EVENT_NCCL_BROADCAST, count, datatype, 0);
    return 0;
}

SEC("uretprobe/nccl_broadcast")
int BPF_KRETPROBE(nccl_broadcast_ret)
{
    return emit_exit(ctx);
}

SEC("uprobe/nccl_send")
int BPF_KPROBE(nccl_send, void *b, __u64 count, __u64 datatype)
{
    save_entry(EVENT_NCCL_SEND, count, datatype, 0);
    return 0;
}

SEC("uretprobe/nccl_send")
int BPF_KRETPROBE(nccl_send_ret)
{
    return emit_exit(ctx);
}

SEC("uprobe/nccl_recv")
int BPF_KPROBE(nccl_recv, void *b, __u64 count, __u64 datatype)
{
    save_entry(EVENT_NCCL_RECV, count, datatype, 0);
    return 0;
}

SEC("uretprobe/nccl_recv")
int BPF_KRETPROBE(nccl_recv_ret)
{
    return emit_exit(ctx);
}
