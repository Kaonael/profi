// SPDX-License-Identifier: Apache-2.0

use lru::LruCache;
use std::collections::HashMap;
use std::num::NonZeroUsize;

/// Why a lazy kernel-name resolve via /proc/<pid>/mem failed.
/// Used as the `reason` label on profi_system_kernel_name_resolve_failures_total.
#[derive(Clone, Copy, Debug)]
pub enum KernelNameReadError {
    /// /proc/<pid>/mem could not be opened (CAP_SYS_PTRACE missing or pid gone).
    OpenFailed,
    /// pread at the name pointer failed — typically the module was unloaded
    /// or the pointer lies outside mapped memory.
    ReadFailed,
    /// Pointer read succeeded but the bytes are not valid UTF-8.
    NotUtf8,
    /// Read succeeded but the string was empty.
    Empty,
}

impl KernelNameReadError {
    pub fn as_label(self) -> &'static str {
        match self {
            KernelNameReadError::OpenFailed => "open_failed",
            KernelNameReadError::ReadFailed => "read_failed",
            KernelNameReadError::NotUtf8 => "not_utf8",
            KernelNameReadError::Empty => "empty",
        }
    }
}

/// Read a NUL-terminated C string from another process' address space via
/// /proc/<pid>/mem. Used to resolve CUDA kernel names lazily — the eBPF probe
/// sends only the name pointer (to avoid bpf_probe_read_user_str_bytes cost),
/// and this helper materializes the string in userspace.
///
/// Requires CAP_SYS_PTRACE (which the profi DaemonSet already holds). Falls
/// back via KernelNameReadError variants if the target memory was unmapped
/// (e.g. cuModuleUnload raced the resolve).
pub fn read_kernel_name(
    proc_path: &str,
    pid: u32,
    name_ptr: u64,
) -> Result<String, KernelNameReadError> {
    use std::io::{Read, Seek, SeekFrom};
    if name_ptr == 0 {
        return Err(KernelNameReadError::ReadFailed);
    }
    let path = format!("{}/{}/mem", proc_path, pid);
    let mut f = std::fs::File::open(&path).map_err(|_| KernelNameReadError::OpenFailed)?;
    f.seek(SeekFrom::Start(name_ptr))
        .map_err(|_| KernelNameReadError::ReadFailed)?;

    // Kernel names are mangled C++ symbols — up to ~512 bytes for deeply
    // templated kernels (e.g. CUTLASS gemm). 1024 is a safe upper bound;
    // we stop at the first NUL anyway.
    let mut buf = [0u8; 1024];
    let n = f
        .read(&mut buf)
        .map_err(|_| KernelNameReadError::ReadFailed)?;
    if n == 0 {
        return Err(KernelNameReadError::ReadFailed);
    }
    let end = buf[..n].iter().position(|&c| c == 0).unwrap_or(n);
    if end == 0 {
        return Err(KernelNameReadError::Empty);
    }
    let s = std::str::from_utf8(&buf[..end]).map_err(|_| KernelNameReadError::NotUtf8)?;
    Ok(s.to_string())
}

struct MapEntry {
    start: u64,
    end: u64,
    offset: u64,
    host_path: String,
}

struct LibSymbols {
    symbols: Vec<(u64, u64, String)>, // (addr, size, name) sorted by addr
}

pub enum SymRequest {
    Resolve(u32, u64),
    EvictPids(Vec<u32>),
}

pub struct SymbolResolver {
    proc_path: String,
    addr_cache: LruCache<(u32, u64), String>,
    lib_cache: HashMap<String, Option<LibSymbols>>,
    maps_cache: LruCache<u32, Vec<MapEntry>>,
}

impl SymbolResolver {
    pub fn new(proc_path: String) -> Self {
        Self {
            proc_path,
            addr_cache: LruCache::new(NonZeroUsize::new(10_000).unwrap()),
            lib_cache: HashMap::new(),
            maps_cache: LruCache::new(NonZeroUsize::new(256).unwrap()),
        }
    }

    pub fn resolve(&mut self, pid: u32, addr: u64) -> String {
        if addr == 0 {
            return "unknown".to_string();
        }
        if let Some(name) = self.addr_cache.get(&(pid, addr)) {
            return name.clone();
        }

        let name = self.resolve_uncached(pid, addr);
        self.addr_cache.put((pid, addr), name.clone());
        name
    }

    pub fn evict_pids(&mut self, pids: &[u32]) {
        for &pid in pids {
            self.maps_cache.pop(&pid);
            // Remove all addr_cache entries for this PID
            let keys_to_remove: Vec<_> = self
                .addr_cache
                .iter()
                .filter(|(&(p, _), _)| p == pid)
                .map(|(&k, _)| k)
                .collect();
            for k in keys_to_remove {
                self.addr_cache.pop(&k);
            }
        }
    }

    fn resolve_uncached(&mut self, pid: u32, addr: u64) -> String {
        if self.maps_cache.get(&pid).is_none() {
            let maps = self.parse_maps(pid);
            self.maps_cache.put(pid, maps);
        }
        let maps = self.maps_cache.get(&pid).unwrap();
        let entry = match maps.iter().find(|m| addr >= m.start && addr < m.end) {
            Some(e) => e,
            None => return format!("unknown_0x{:x}", addr),
        };

        let file_offset = addr - entry.start + entry.offset;
        let host_path = entry.host_path.clone();

        let symbols = self.load_symbols(&host_path);
        match symbols {
            Some(syms) => {
                match syms
                    .symbols
                    .binary_search_by_key(&file_offset, |&(a, _, _)| a)
                {
                    Ok(i) => syms.symbols[i].2.clone(),
                    Err(i) if i > 0 => {
                        let (sym_addr, sym_size, ref name) = syms.symbols[i - 1];
                        if sym_size == 0 || file_offset < sym_addr + sym_size {
                            name.clone()
                        } else {
                            format!("unknown_0x{:x}", file_offset)
                        }
                    }
                    _ => format!("unknown_0x{:x}", file_offset),
                }
            }
            None => format!("unknown_0x{:x}", file_offset),
        }
    }

    fn parse_maps(&self, pid: u32) -> Vec<MapEntry> {
        let maps_path = format!("{}/{}/maps", self.proc_path, pid);
        let content = match std::fs::read_to_string(&maps_path) {
            Ok(c) => c,
            Err(_) => return Vec::new(),
        };

        let mut entries = Vec::new();
        for line in content.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 6 {
                continue;
            }
            if !parts[1].contains('x') {
                continue;
            }
            let addr_parts: Vec<&str> = parts[0].split('-').collect();
            if addr_parts.len() != 2 {
                continue;
            }
            let start = u64::from_str_radix(addr_parts[0], 16).unwrap_or(0);
            let end = u64::from_str_radix(addr_parts[1], 16).unwrap_or(0);
            let offset = u64::from_str_radix(parts[2], 16).unwrap_or(0);
            let path = parts[5];
            let host_path = format!("{}/{}/root{}", self.proc_path, pid, path);
            entries.push(MapEntry {
                start,
                end,
                offset,
                host_path,
            });
        }
        entries
    }

    fn load_symbols(&mut self, host_path: &str) -> Option<&LibSymbols> {
        if !self.lib_cache.contains_key(host_path) {
            if self.lib_cache.len() > 64 {
                self.lib_cache.clear();
            }
            let symbols = Self::parse_elf_symbols(host_path);
            self.lib_cache.insert(host_path.to_string(), symbols);
        }
        self.lib_cache.get(host_path).and_then(|s| s.as_ref())
    }

    fn parse_elf_symbols(path: &str) -> Option<LibSymbols> {
        use object::{Object, ObjectSymbol};
        let data = std::fs::read(path).ok()?;
        let file = object::File::parse(&*data).ok()?;

        let mut symbols: Vec<(u64, u64, String)> = Vec::new();
        for sym in file.symbols().chain(file.dynamic_symbols()) {
            if sym.size() == 0 && symbols.iter().any(|(a, _, _)| *a == sym.address()) {
                continue;
            }
            if let Ok(name) = sym.name() {
                if !name.is_empty() && sym.address() > 0 {
                    symbols.push((sym.address(), sym.size(), name.to_string()));
                }
            }
        }
        symbols.sort_by_key(|&(a, _, _)| a);
        symbols.dedup_by_key(|s| s.0);

        if symbols.is_empty() {
            None
        } else {
            Some(LibSymbols { symbols })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Read a known static string from our own /proc/self/mem. Validates the
    /// happy path end-to-end without needing a child process.
    #[test]
    fn read_kernel_name_self_mem_roundtrip() {
        // A NUL-terminated C string embedded in .rodata.
        static NEEDLE: &[u8] = b"profi_kernel_resolve_probe\0";
        let pid = std::process::id();
        let ptr = NEEDLE.as_ptr() as u64;

        let got = read_kernel_name("/proc", pid, ptr).expect("self-mem read should succeed");
        assert_eq!(got, "profi_kernel_resolve_probe");
    }

    #[test]
    fn read_kernel_name_zero_ptr_fails() {
        let err = read_kernel_name("/proc", std::process::id(), 0).unwrap_err();
        assert!(matches!(err, KernelNameReadError::ReadFailed));
    }

    #[test]
    fn read_kernel_name_bogus_pid_fails() {
        // PID 0 never exists as a userspace process.
        let err = read_kernel_name("/proc", 0, 0xffff_ffff_ffff_0000).unwrap_err();
        assert!(matches!(
            err,
            KernelNameReadError::OpenFailed | KernelNameReadError::ReadFailed
        ));
    }

    #[test]
    fn read_kernel_name_unmapped_address_fails() {
        // Guaranteed-unmapped address (canonical kernel-half on x86-64).
        let err = read_kernel_name("/proc", std::process::id(), 0xffff_8000_0000_0000).unwrap_err();
        assert!(matches!(err, KernelNameReadError::ReadFailed));
    }

    #[test]
    fn read_kernel_name_labels_are_stable() {
        // Failure labels are emitted as Prometheus label values; protect against
        // accidental renames that would break dashboards/alerts.
        assert_eq!(KernelNameReadError::OpenFailed.as_label(), "open_failed");
        assert_eq!(KernelNameReadError::ReadFailed.as_label(), "read_failed");
        assert_eq!(KernelNameReadError::NotUtf8.as_label(), "not_utf8");
        assert_eq!(KernelNameReadError::Empty.as_label(), "empty");
    }
}
