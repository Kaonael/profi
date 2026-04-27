# profi deployment

This directory contains the production deployment artifacts for profi.

## Layout

```
deploy/
├── profi/          Helm chart (recommended install path)
├── runbooks/       Alert playbooks (linked from PrometheusRule runbook_url)
└── README.md       (this file)
```

## Install with Helm

```bash
helm install profi ./deploy/profi \
  --namespace gpu-profi --create-namespace
```

Full values reference and troubleshooting: [`profi/README.md`](profi/README.md).

## Runbooks

Every alert shipped by the chart links to a specific playbook:

- [`nccl-hang.md`](runbooks/nccl-hang.md) — distributed training/inference stuck (critical)
- [`nccl-straggler.md`](runbooks/nccl-straggler.md) — one GPU slower than peers
- [`event-loss.md`](runbooks/event-loss.md) — profi dropping events / exporter down / CUDA errors
- [`cardinality-explosion.md`](runbooks/cardinality-explosion.md) — too many label combinations

If you customize `prometheusRule.runbookBaseUrl` in `values.yaml`, set it to a
URL where your team can read these (public GitHub, internal wiki mirror, etc.).

## GPU Operator coexistence

profi and NVIDIA GPU Operator (DCGM-Exporter, driver DaemonSet, device plugin)
are complementary: DCGM sees GPU *hardware*, profi sees GPU *applications*. Both
can run on the same node without conflict. See
[`profi/README.md`](profi/README.md#gpu-operator-coexistence) for details.
