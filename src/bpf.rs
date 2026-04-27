// SPDX-License-Identifier: Apache-2.0

// Generated BPF skeleton — written by build.rs via libbpf-cargo from
// src/bpf/profi.bpf.c. Exposes ProfiSkelBuilder, ProfiSkel, and
// typed handles for every map and program.

#![allow(clippy::all)]
#![allow(dead_code)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

include!(concat!(env!("OUT_DIR"), "/profi.skel.rs"));
