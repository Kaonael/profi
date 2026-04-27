// SPDX-License-Identifier: Apache-2.0

/// Normalize kernel name: strip C++ mangling, template args, function args, numeric suffixes.
/// `void gemm_kernel<float, 128, 64>(float*, int)` -> `gemm_kernel`
/// `triton_poi_fused_2` -> `triton_poi_fused`
/// `_ZN4cuda6detail21softmax_warp_forwardIfEvPT_` -> `softmax_warp_forward`
pub fn normalize_kernel_name(raw: &str) -> String {
    let mut name = raw;

    // Strip C++ mangled names: _Z[N]<len><name>...
    if name.starts_with("_Z") {
        // Find the last meaningful identifier: scan for sequences of letters
        let mut best = "";
        let bytes = name.as_bytes();
        let mut i = 2;
        while i < bytes.len() {
            // Skip digits (length prefixes in mangled names)
            if bytes[i].is_ascii_digit() {
                let start = i;
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                }
                let len: usize = name[start..i].parse().unwrap_or(0);
                if len > 0 && i + len <= bytes.len() {
                    let ident = &name[i..i + len];
                    if ident.len() > best.len()
                        && ident.chars().all(|c| c.is_alphanumeric() || c == '_')
                    {
                        best = ident;
                    }
                    i += len;
                    continue;
                }
            }
            i += 1;
        }
        if !best.is_empty() {
            name = best;
        }
    }

    // Strip template args <...> (including nested)
    let mut result = String::with_capacity(name.len());
    let mut depth = 0u32;
    for ch in name.chars() {
        match ch {
            '<' => depth += 1,
            '>' if depth > 0 => depth -= 1,
            _ if depth == 0 => result.push(ch),
            _ => {}
        }
    }

    // Strip function args (...)
    if let Some(paren) = result.find('(') {
        result.truncate(paren);
    }

    // Strip trailing whitespace
    let trimmed = result.trim_end();

    // Strip trailing _N numeric suffix (Triton generated: triton_poi_fused_2 -> triton_poi_fused)
    let final_name = if let Some(last_underscore) = trimmed.rfind('_') {
        let suffix = &trimmed[last_underscore + 1..];
        if !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()) {
            &trimmed[..last_underscore]
        } else {
            trimmed
        }
    } else {
        trimmed
    };

    let truncated = truncate_utf8_bytes(final_name, 64);

    if truncated.is_empty() {
        truncate_utf8_bytes(raw, 64).to_string()
    } else {
        truncated.to_string()
    }
}

fn truncate_utf8_bytes(s: &str, max_len: usize) -> &str {
    if s.len() <= max_len {
        return s;
    }

    let mut end = max_len;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

pub fn classify_kernel(name: &str) -> &'static str {
    let lower = name.to_ascii_lowercase();
    if lower.contains("flash_fwd")
        || lower.contains("flash_bwd")
        || lower.contains("paged_attention")
        || lower.contains("radix_attention")
        || lower.contains("attention")
        || lower.contains("sdpa")
    {
        "attention"
    } else if lower.contains("gemm")
        || lower.contains("cgemm")
        || lower.contains("cutlass")
        || lower.contains("linear")
        || lower.contains("matmul")
    {
        "gemm"
    } else if lower.contains("nccl")
        || lower.contains("allreduce")
        || lower.contains("allgather")
        || lower.contains("reduce_scatter")
    {
        "collective"
    } else if lower.contains("softmax")
        || lower.contains("layernorm")
        || lower.contains("rms_norm")
        || lower.contains("gelu")
        || lower.contains("silu")
    {
        "activation"
    } else if lower.contains("memset")
        || lower.contains("memcpy")
        || lower.contains("copy")
        || lower.contains("transpose")
    {
        "memory"
    } else if lower.contains("sampling") || lower.contains("topk") || lower.contains("argmax") {
        "sampling"
    } else {
        "other"
    }
}

/// Map raw NVTX range name to a stable phase label from a fixed vocabulary.
/// Prevents cardinality explosion from dynamic NVTX strings (request IDs, batch sizes, etc.).
pub fn sanitize_phase(raw: &str) -> &'static str {
    if raw.is_empty() {
        return "";
    }
    let lower = raw.to_ascii_lowercase();
    if lower.contains("prefill") || lower.contains("prompt") || lower.contains("encode") {
        "prefill"
    } else if lower.contains("decode") || lower.contains("generate") {
        "decode"
    } else if lower.contains("attention") || lower.contains("attn") {
        "attention"
    } else if lower.contains("mlp") || lower.contains("ffn") || lower.contains("expert") {
        "mlp"
    } else if lower.contains("norm") || lower.contains("embed") {
        "norm"
    } else {
        "other"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── normalize_kernel_name ───────────────────────────────────────────

    #[test]
    fn normalize_cpp_mangled() {
        // Length prefix 21 covers "softmax_warp_forwardI" (20 chars of name + template marker 'I')
        assert_eq!(
            normalize_kernel_name("_ZN4cuda6detail21softmax_warp_forwardIfEvPT_"),
            "softmax_warp_forwardI"
        );
    }

    #[test]
    fn normalize_cpp_mangled_exact() {
        // With correct length prefix 20, extracts "softmax_warp_forward"
        assert_eq!(
            normalize_kernel_name("_ZN4cuda6detail20softmax_warp_forwardIfEvPT_"),
            "softmax_warp_forward"
        );
    }

    #[test]
    fn normalize_template() {
        assert_eq!(
            normalize_kernel_name("gemm_kernel<float, 128>"),
            "gemm_kernel"
        );
    }

    #[test]
    fn normalize_nested_template() {
        assert_eq!(normalize_kernel_name("foo<bar<baz>>"), "foo");
    }

    #[test]
    fn normalize_triton_suffix() {
        assert_eq!(
            normalize_kernel_name("triton_poi_fused_2"),
            "triton_poi_fused"
        );
    }

    #[test]
    fn normalize_triton_no_suffix() {
        assert_eq!(
            normalize_kernel_name("triton_poi_fused"),
            "triton_poi_fused"
        );
    }

    #[test]
    fn normalize_function_args() {
        assert_eq!(normalize_kernel_name("my_kernel(float*, int)"), "my_kernel");
    }

    #[test]
    fn normalize_template_and_args() {
        assert_eq!(
            normalize_kernel_name("gemm<float, 128>(float*, int)"),
            "gemm"
        );
    }

    #[test]
    fn normalize_truncate_64() {
        let long_name = "a".repeat(100);
        let result = normalize_kernel_name(&long_name);
        assert_eq!(result.len(), 64);
    }

    #[test]
    fn normalize_empty() {
        let result = normalize_kernel_name("");
        assert_eq!(result, "");
    }

    #[test]
    fn normalize_unicode_fallback_truncates_to_64_bytes() {
        let result = normalize_kernel_name("<ଏ¡®ⶠ꣎0Σ\u{fffc} A0A\u{11d3c}A𞺥𐔀𑌪𞹂A𖩀ᾀ᨞0AὙ\u{2d7f}A");
        assert!(result.len() <= 64);
        assert!(result.is_char_boundary(result.len()));
    }

    #[test]
    fn normalize_bare_z_prefix() {
        let result = normalize_kernel_name("_Z");
        assert_eq!(result, "_Z");
    }

    #[test]
    fn normalize_exactly_64_chars() {
        let name = "k".repeat(64);
        let result = normalize_kernel_name(&name);
        assert_eq!(result.len(), 64);
        assert_eq!(result, name);
    }

    #[test]
    fn normalize_trailing_whitespace() {
        assert_eq!(normalize_kernel_name("kernel  "), "kernel");
    }

    #[test]
    fn normalize_only_numeric_suffix() {
        assert_eq!(normalize_kernel_name("kernel123"), "kernel123");
    }

    #[test]
    fn normalize_multiple_underscore_numbers() {
        assert_eq!(normalize_kernel_name("kernel_2_3"), "kernel_2");
    }

    #[test]
    fn normalize_real_cutlass() {
        let mangled = "_ZN7cutlass6KernelINS_4gemm6kernel14GemmUniversalINS2_24GemmIdentityThreadblockSwizzleILi1EEEfLb0EEEEE";
        let result = normalize_kernel_name(mangled);
        assert!(!result.is_empty());
        assert!(result.len() <= 64);
    }

    // ── classify_kernel ─────────────────────────────────────────────────

    #[test]
    fn classify_attention_flash() {
        assert_eq!(classify_kernel("flash_fwd_kernel"), "attention");
    }

    #[test]
    fn classify_attention_paged() {
        assert_eq!(classify_kernel("paged_attention_v2"), "attention");
    }

    #[test]
    fn classify_attention_sdpa() {
        assert_eq!(classify_kernel("sdpa_kernel"), "attention");
    }

    #[test]
    fn classify_gemm() {
        assert_eq!(classify_kernel("gemm_128x128"), "gemm");
    }

    #[test]
    fn classify_cutlass() {
        assert_eq!(classify_kernel("cutlass_gemm_kernel"), "gemm");
    }

    #[test]
    fn classify_matmul() {
        assert_eq!(classify_kernel("matmul_forward"), "gemm");
    }

    #[test]
    fn classify_collective_nccl() {
        assert_eq!(classify_kernel("nccl_allreduce_kernel"), "collective");
    }

    #[test]
    fn classify_activation_softmax() {
        assert_eq!(classify_kernel("softmax_forward"), "activation");
    }

    #[test]
    fn classify_activation_gelu() {
        assert_eq!(classify_kernel("gelu_kernel"), "activation");
    }

    #[test]
    fn classify_memory() {
        assert_eq!(classify_kernel("memset_kernel"), "memory");
    }

    #[test]
    fn classify_sampling() {
        assert_eq!(classify_kernel("topk_sampling"), "sampling");
    }

    #[test]
    fn classify_other() {
        assert_eq!(classify_kernel("my_custom_kernel"), "other");
    }

    #[test]
    fn classify_empty() {
        assert_eq!(classify_kernel(""), "other");
    }

    // ── sanitize_phase ──────────────────────────────────────────────────

    #[test]
    fn sanitize_empty() {
        assert_eq!(sanitize_phase(""), "");
    }

    #[test]
    fn sanitize_prefill() {
        assert_eq!(sanitize_phase("prefill_batch_32"), "prefill");
    }

    #[test]
    fn sanitize_prompt() {
        assert_eq!(sanitize_phase("prompt_processing"), "prefill");
    }

    #[test]
    fn sanitize_decode() {
        assert_eq!(sanitize_phase("decode_step_1"), "decode");
    }

    #[test]
    fn sanitize_attention() {
        assert_eq!(sanitize_phase("self_attention_layer"), "attention");
    }

    #[test]
    fn sanitize_attn_shorthand() {
        assert_eq!(sanitize_phase("cross_attn"), "attention");
    }

    #[test]
    fn sanitize_mlp() {
        assert_eq!(sanitize_phase("mlp_forward"), "mlp");
    }

    #[test]
    fn sanitize_norm() {
        assert_eq!(sanitize_phase("layer_norm"), "norm");
    }

    #[test]
    fn sanitize_other() {
        assert_eq!(sanitize_phase("random_thing"), "other");
    }

    #[test]
    fn sanitize_case_insensitive() {
        assert_eq!(sanitize_phase("PREFILL"), "prefill");
    }

    // ── proptest ────────────────────────────────────────────────────────

    proptest::proptest! {
        #[test]
        fn normalize_never_panics(s in "\\PC*") {
            let result = normalize_kernel_name(&s);
            assert!(result.len() <= 64);
        }

        #[test]
        fn classify_returns_known(s in "\\PC*") {
            let class = classify_kernel(&s);
            assert!(
                ["attention", "gemm", "collective", "activation", "memory", "sampling", "other"]
                    .contains(&class)
            );
        }

        #[test]
        fn sanitize_returns_known(s in "\\PC*") {
            let phase = sanitize_phase(&s);
            assert!(
                ["", "prefill", "decode", "attention", "mlp", "norm", "other"]
                    .contains(&phase)
            );
        }
    }
}
