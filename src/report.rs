// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use crate::events::*;

pub fn fmt_dur(ns: u64) -> String {
    if ns >= 1_000_000_000 {
        format!("{:.2}s", ns as f64 / 1e9)
    } else if ns >= 1_000_000 {
        format!("{:.2}ms", ns as f64 / 1e6)
    } else if ns >= 1_000 {
        format!("{:.1}\u{b5}s", ns as f64 / 1e3)
    } else {
        format!("{ns}ns")
    }
}

pub fn fmt_size(b: u64) -> String {
    if b == 0 {
        "-".into()
    } else if b >= 1 << 30 {
        format!("{:.2} GiB", b as f64 / (1u64 << 30) as f64)
    } else if b >= 1 << 20 {
        format!("{:.2} MiB", b as f64 / (1u64 << 20) as f64)
    } else if b >= 1 << 10 {
        format!("{:.1} KiB", b as f64 / (1u64 << 10) as f64)
    } else {
        format!("{b} B")
    }
}

#[derive(Default)]
pub struct TermStats {
    pub count: u64,
    pub total_ns: u64,
    pub total_bytes: u64,
    pub min_ns: u64,
    pub max_ns: u64,
}

impl TermStats {
    pub fn record(&mut self, dur_ns: u64, size: u64) {
        self.count += 1;
        self.total_ns += dur_ns;
        self.total_bytes += size;
        if self.count == 1 || dur_ns < self.min_ns {
            self.min_ns = dur_ns;
        }
        if dur_ns > self.max_ns {
            self.max_ns = dur_ns;
        }
    }

    pub fn merge(&mut self, other: &TermStats) {
        if other.count == 0 {
            return;
        }
        self.count += other.count;
        self.total_ns += other.total_ns;
        self.total_bytes += other.total_bytes;
        if self.count == other.count || other.min_ns < self.min_ns {
            self.min_ns = other.min_ns;
        }
        if other.max_ns > self.max_ns {
            self.max_ns = other.max_ns;
        }
    }
}

pub fn print_report(stats: &HashMap<(u32, u32), TermStats>) {
    println!(
        "\n{:<22} {:>8} {:>12} {:>12} {:>12} {:>12}",
        "Operation", "Count", "Total", "Avg", "Min/Max", "Data"
    );
    println!("{:-<82}", "");
    for &et in &[
        EVENT_CUDA_MALLOC,
        EVENT_CUDA_FREE,
        EVENT_CUDA_MEMCPY,
        EVENT_CUDA_MEMCPY_ASYNC,
        EVENT_CUDA_LAUNCH_KERNEL,
    ] {
        let mut keys: Vec<_> = stats.keys().filter(|(e, _)| *e == et).copied().collect();
        keys.sort();
        for key in keys {
            let s = &stats[&key];
            if s.count == 0 {
                continue;
            }
            let label = match key.0 {
                EVENT_CUDA_MALLOC => "cudaMalloc",
                EVENT_CUDA_FREE => "cudaFree",
                EVENT_CUDA_MEMCPY | EVENT_CUDA_MEMCPY_ASYNC => {
                    let prefix = if key.0 == EVENT_CUDA_MEMCPY_ASYNC {
                        "Async "
                    } else {
                        ""
                    };
                    println!(
                        "{:<22} {:>8} {:>12} {:>12} {:>12} {:>12}",
                        format!(
                            "cudaMemcpy{prefix}{}",
                            match key.1 {
                                MEMCPY_H2H => " H->H",
                                MEMCPY_H2D => " H->D",
                                MEMCPY_D2H => " D->H",
                                MEMCPY_D2D => " D->D",
                                _ => " ???",
                            }
                        ),
                        s.count,
                        fmt_dur(s.total_ns),
                        fmt_dur(s.total_ns / s.count),
                        format!("{}/{}", fmt_dur(s.min_ns), fmt_dur(s.max_ns)),
                        fmt_size(s.total_bytes),
                    );
                    continue;
                }
                EVENT_CUDA_LAUNCH_KERNEL => "cudaLaunchKernel",
                _ => "unknown",
            };
            let avg = s.total_ns / s.count;
            println!(
                "{:<22} {:>8} {:>12} {:>12} {:>12} {:>12}",
                label,
                s.count,
                fmt_dur(s.total_ns),
                fmt_dur(avg),
                format!("{}/{}", fmt_dur(s.min_ns), fmt_dur(s.max_ns)),
                fmt_size(s.total_bytes),
            );
        }
    }
    println!("{:=<82}", "");
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── fmt_dur ─────────────────────────────────────────────────────────

    #[test]
    fn fmt_dur_ns() {
        assert_eq!(fmt_dur(500), "500ns");
    }

    #[test]
    fn fmt_dur_us() {
        assert_eq!(fmt_dur(1500), "1.5\u{b5}s");
    }

    #[test]
    fn fmt_dur_ms() {
        assert_eq!(fmt_dur(5_000_000), "5.00ms");
    }

    #[test]
    fn fmt_dur_s() {
        assert_eq!(fmt_dur(2_000_000_000), "2.00s");
    }

    // ── fmt_size ────────────────────────────────────────────────────────

    #[test]
    fn fmt_size_zero() {
        assert_eq!(fmt_size(0), "-");
    }

    #[test]
    fn fmt_size_bytes() {
        assert_eq!(fmt_size(512), "512 B");
    }

    #[test]
    fn fmt_size_kib() {
        assert_eq!(fmt_size(2048), "2.0 KiB");
    }

    #[test]
    fn fmt_size_mib() {
        assert_eq!(fmt_size(1_048_576), "1.00 MiB");
    }

    #[test]
    fn fmt_size_gib() {
        assert_eq!(fmt_size(1_073_741_824), "1.00 GiB");
    }

    // ── TermStats ───────────────────────────────────────────────────────

    #[test]
    fn term_stats_single_record() {
        let mut s = TermStats::default();
        s.record(100, 200);
        assert_eq!(s.count, 1);
        assert_eq!(s.total_ns, 100);
        assert_eq!(s.total_bytes, 200);
        assert_eq!(s.min_ns, 100);
        assert_eq!(s.max_ns, 100);
    }

    #[test]
    fn term_stats_multiple_records() {
        let mut s = TermStats::default();
        s.record(100, 10);
        s.record(50, 20);
        s.record(200, 30);
        assert_eq!(s.count, 3);
        assert_eq!(s.total_ns, 350);
        assert_eq!(s.total_bytes, 60);
        assert_eq!(s.min_ns, 50);
        assert_eq!(s.max_ns, 200);
    }

    #[test]
    fn term_stats_merge_empty_into_nonempty() {
        let mut s = TermStats::default();
        s.record(100, 200);
        let empty = TermStats::default();
        s.merge(&empty);
        assert_eq!(s.count, 1);
        assert_eq!(s.total_ns, 100);
    }

    #[test]
    fn term_stats_merge_two() {
        let mut a = TermStats::default();
        a.record(100, 10);
        a.record(300, 20);
        let mut b = TermStats::default();
        b.record(50, 5);
        a.merge(&b);
        assert_eq!(a.count, 3);
        assert_eq!(a.total_ns, 450);
        assert_eq!(a.total_bytes, 35);
        assert_eq!(a.min_ns, 50);
        assert_eq!(a.max_ns, 300);
    }
}
