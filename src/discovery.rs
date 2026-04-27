// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};

#[derive(Hash, Eq, PartialEq, Clone, Debug)]
pub struct DevInode {
    pub dev: u64,
    pub ino: u64,
}

/// Scan /proc/*/maps once for all given libraries. Each library has a name
/// and a set of already-attached DevInodes. Returns new (host_path, DevInode)
/// grouped by library name.
pub fn scan_proc_for_libs(
    proc_path: &str,
    libs: &[(&str, &HashSet<DevInode>)],
    only_pids: Option<&HashSet<u32>>,
) -> HashMap<String, Vec<(String, DevInode)>> {
    let mut results: HashMap<String, Vec<(String, DevInode)>> = HashMap::new();
    let mut seen: HashMap<String, HashSet<DevInode>> = HashMap::new();
    for &(lib_name, _) in libs {
        results.insert(lib_name.to_string(), Vec::new());
        seen.insert(lib_name.to_string(), HashSet::new());
    }

    let entries = match std::fs::read_dir(proc_path) {
        Ok(e) => e,
        Err(_) => return results,
    };

    for entry in entries.flatten() {
        let name = entry.file_name();
        let pid_str = name.to_string_lossy();
        if pid_str.chars().next().is_none_or(|c| !c.is_ascii_digit()) {
            continue;
        }

        // If only_pids is set, skip PIDs not in the set
        if let Some(pids) = only_pids {
            if let Ok(pid_num) = pid_str.parse::<u32>() {
                if !pids.contains(&pid_num) {
                    continue;
                }
            }
        }

        let maps_path = format!("{proc_path}/{pid_str}/maps");
        let maps = match File::open(&maps_path) {
            Ok(m) => m,
            Err(_) => continue,
        };

        let mut remaining_libs: HashSet<&str> =
            libs.iter().map(|(lib_name, _)| *lib_name).collect();
        let mut reader = BufReader::new(maps);
        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => {}
                Err(_) => break,
            }

            for &(lib_name, attached) in libs {
                if !line.contains(lib_name) {
                    continue;
                }
                // Format: addr perms offset dev inode pathname
                let mut parts = line.split_whitespace();
                let Some(_addr) = parts.next() else {
                    continue;
                };
                let Some(_perms) = parts.next() else {
                    continue;
                };
                let Some(_offset) = parts.next() else {
                    continue;
                };
                let Some(dev_s) = parts.next() else {
                    continue;
                };
                let Some(ino_s) = parts.next() else {
                    continue;
                };
                let Some(container_path) = parts.next() else {
                    continue;
                };

                let ino: u64 = match ino_s.parse() {
                    Ok(i) => i,
                    Err(_) => continue,
                };
                let Some((maj_s, min_s)) = dev_s.split_once(':') else {
                    continue;
                };
                let major = u32::from_str_radix(maj_s, 16).unwrap_or(0);
                let minor = u32::from_str_radix(min_s, 16).unwrap_or(0);
                let dev = libc::makedev(major, minor);
                remaining_libs.remove(lib_name);

                let di = DevInode { dev, ino };
                let lib_seen = seen.get_mut(lib_name).unwrap();
                if attached.contains(&di) || lib_seen.contains(&di) {
                    continue;
                }
                lib_seen.insert(di.clone());

                let host_path = format!("{proc_path}/{pid_str}/root{container_path}");
                if std::path::Path::new(&host_path).exists() {
                    results.get_mut(lib_name).unwrap().push((host_path, di));
                }
            }

            if remaining_libs.is_empty() {
                break;
            }
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scan_proc_for_libs_finds_library_from_streamed_maps() {
        let tmp = tempfile::tempdir().unwrap();
        let proc_root = tmp.path();
        let pid_root = proc_root.join("1234");
        let container_lib = "/usr/local/cuda/lib64/libcudart.so.12";
        let host_lib = pid_root.join("root").join(&container_lib[1..]);
        std::fs::create_dir_all(host_lib.parent().unwrap()).unwrap();
        std::fs::write(&host_lib, "").unwrap();
        std::fs::write(
            pid_root.join("maps"),
            format!(
                "7f000000-7f001000 r-xp 00000000 08:01 42 {container_lib}\n\
                 7f001000-7f002000 r--p 00001000 08:01 42 {container_lib}\n"
            ),
        )
        .unwrap();

        let attached = HashSet::new();
        let found = scan_proc_for_libs(
            proc_root.to_str().unwrap(),
            &[("libcudart.so", &attached)],
            None,
        );

        let entries = found.get("libcudart.so").unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].0, host_lib.to_string_lossy());
        assert_eq!(entries[0].1.ino, 42);
    }
}
