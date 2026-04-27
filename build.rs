// SPDX-License-Identifier: Apache-2.0

use std::env;
use std::path::PathBuf;

use libbpf_cargo::SkeletonBuilder;

fn main() {
    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());

    let bpf_src = manifest_dir.join("src/bpf/profi.bpf.c");
    let bpf_include = manifest_dir.join("src/bpf");

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let skel_path = out_dir.join("profi.skel.rs");

    SkeletonBuilder::new()
        .source(&bpf_src)
        .clang_args(["-I", bpf_include.to_str().unwrap()])
        .build_and_generate(&skel_path)
        .expect("Failed to build BPF skeleton. Ensure clang ≥ 14 is in PATH.");

    println!("cargo:rerun-if-changed={}", bpf_src.display());
    println!("cargo:rerun-if-changed={}", bpf_include.display());
}
