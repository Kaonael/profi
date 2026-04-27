# Runbook: NCCL Straggler (`ProfiNCCLStraggler`)

## Symptom

Alert `ProfiNCCLStraggler` has fired. One GPU is spending `>{{ threshold }}x`
(default 1.5×) the median time of its peers inside a given NCCL collective.
Alert labels identify which GPU:

- `pid`, `gpu`, `gpu_uuid`, `operation`

Value (`$value`) is the ratio vs. the group median.

Also fires `ProfiGPUThrottling` if the cause is hardware throttling.

## Impact

**Not broken — but wasting silicon.** In data-parallel training and
tensor-parallel inference, every NCCL collective takes as long as the slowest
rank. A straggler at 1.5× steals ≈33% of effective throughput from the entire
group (`tokens/s`, `samples/s`, `TPOT`). At 2× it halves throughput. Unlike
hang, the job keeps moving — the problem shows up as slow inference, not an
outage.

## Diagnosis

### 1. Confirm which rank and how bad

```promql
topk(5, profi_nccl_straggler_ratio)
```

Read off the `pid`, `gpu`, `gpu_uuid`, and `operation` labels. The ratio
tells you the severity.

### 2. Is it the GPU, the host, or the fabric?

Hardware first — check GPU health for that specific `gpu_uuid`:

```promql
# Is this GPU throttling?
profi_gpu_throttle_active{gpu_uuid="<uuid>"} == 1

# Temperature — typical T_max before throttle: 85–90 °C
profi_gpu_temperature_celsius{gpu_uuid="<uuid>"}

# Power draw — compare to peers; a straggler at TDP while peers are at 60% is
# doing more work, which is suspicious.
profi_gpu_power_watts{gpu_uuid="<uuid>"}

# Clocks — if SM clock is pinned low, thermal or power throttling is active.
profi_gpu_clock_mhz{gpu_uuid="<uuid>", clock_type="sm"}

# ECC — pending page retirements slow a GPU silently.
profi_gpu_ecc_errors_total{gpu_uuid="<uuid>"}
```

If any of these are red, **it's the GPU** — go to mitigation section
"Hardware cause".

### 3. Is it a kernel-level imbalance, not hardware?

If GPU health is clean, the issue is at the software layer:

```promql
# If kernel-mode=full, check kernel time distribution
topk(10, profi:cuda_kernel_duration:p99:5m{gpu_uuid="<uuid>"})

# Compare against a healthy peer GPU on a sibling pod.
```

A straggler with clean hardware means one rank has more work (imbalanced batch,
uneven sharding, MoE routing hotspot, KV cache layout skew).

### 4. Is it the interconnect?

If the straggler GPU has clean kernel timings but high NCCL duration:

```promql
profi:nccl_duration:p99:5m{operation="ncclAllReduce"}
```

grouped by `gpu` — if one GPU's NCCL latency is consistently higher while its
compute kernels are fine, suspect NVLink/NIC degradation. Check host:

```bash
nvidia-smi topo -m
# If NVSwitch / NVLink shows degraded connectivity → fabric issue
```

## Mitigation

### Hardware cause

- **Thermal throttling**: verify datacenter cooling, chassis airflow. If one
  GPU is consistently hotter, inspect fans and paste.
- **Power brake**: likely PSU or PDU limit. Validate with infrastructure.
- **ECC errors accumulating**: GPU is degrading; schedule RMA.

Operationally:
```bash
kubectl cordon <node>
kubectl drain <node> --ignore-daemonsets
# Replace GPU or reboot to retire failing SMs
```

### Software cause

- **Batch imbalance** (training): switch to dynamic batching or padding-free
  attention; for HuggingFace Trainer, verify `dataloader_drop_last=True`.
- **Tensor-parallel skew** (inference): uneven shard sizes — re-check model
  loading; some models don't divide evenly across TP ranks.
- **MoE hotspot**: check `expert_load_balance` metrics if your serving engine
  exposes them; may need to enable `moe_balance_loss`.

### Fabric cause

- **NVLink flap**: host `dmesg` will show `NVLink` messages. Schedule the node
  for maintenance; replace the affected NVSwitch board if repeatable.
- **InfiniBand/Ethernet flap** (for multi-node collectives): check the NIC
  on the straggler's host; inspect fabric manager logs.

### Tuning knobs (stopgap)

- `NCCL_ALGO=Tree` (default is `Ring` for small messages, `Tree` for large).
  If messages are bimodal, forcing one can help — but benchmark first.
- `NCCL_NTHREADS=256` — raise NCCL's per-rank threads if CPU contention is
  visible.
- Reduce batch size or world size to route around the bad rank.

## References

- profi metric sources: `src/main.rs`, `src/metrics.rs`
- Tunable alert threshold: `values.yaml` → `prometheusRule.alerts.ncclStraggler.threshold`
- NCCL tuning: [NVIDIA NCCL Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- Related alerts: `ProfiGPUThrottling` (co-fires on thermal cause), `ProfiNCCLHangDetected` (escalation)
