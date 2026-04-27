# Runbook: ProfiMetricsAuthFailures

Triggered when profi rejects `/metrics` requests at a sustained rate on one
or more nodes. The `reason` label on the alert narrows the cause.

## Symptom

```
ProfiMetricsAuthFailures  reason=<no_auth|tokenreview_deny|tokenreview_error|audience_mismatch|bad_cert>
```

## Impact

- **`up{job="profi"} == 0`** for the affected pod(s) → no CUDA/NCCL/NVML
  visibility for the GPUs on that node.
- Dashboards silently show "no data" while eBPF probes keep working.
- Other alerts that depend on profi series (hang detection, stragglers) go
  silent — they cannot fire if Prometheus can't read the metrics.

## Diagnosis by reason

### `reason=no_auth`

Caller connected but sent no `Authorization: Bearer …` header and no
validated client cert. Almost always a misconfigured scraper.

1. Check the ServiceMonitor / scrape config:

   ```bash
   kubectl get servicemonitor profi -n <ns> -o yaml
   ```

   In bearer mode it must reference
   `bearerTokenFile: /var/run/secrets/kubernetes.io/serviceaccount/token`
   (or a Secret via `bearerTokenSecret:`).

2. In `mtls-or-bearer` mode, Prometheus needs either a client cert under
   `tlsConfig.certFile/keyFile` **or** a token. One must be present.

### `reason=tokenreview_deny`

Token reached profi, the API server rejected it. Causes:
- Token expired or SA was deleted.
- The SA token was projected for a different cluster.
- Webhooks (OPA/Gatekeeper) rejecting the TokenReview.

Reproduce from a pod in the Prometheus namespace:

```bash
TOKEN=$(cat /var/run/secrets/kubernetes.io/serviceaccount/token)
curl -k -H "Authorization: Bearer $TOKEN" https://profi.<ns>.svc:9401/metrics -v
```

### `reason=tokenreview_error`

profi called TokenReview but got an API error. Usually RBAC:

```bash
kubectl auth can-i --as=system:serviceaccount:<profi-ns>:profi \
    create tokenreviews.authentication.k8s.io
```

Must return `yes`. If not, the Helm chart's `rbac.create: true` and
`profi.metricsAuth.mode != off` should have created the binding — check
that the release was upgraded, not just the values file edited.

Also watch for API-server outages; TokenReview calls share the control
plane's availability.

### `reason=audience_mismatch`

`--metrics-auth-audience=<aud>` is set, but the caller's token was not
projected with that audience. Update the scraper's pod spec to mount a
projected token, e.g.:

```yaml
volumes:
  - name: profi-token
    projected:
      sources:
        - serviceAccountToken:
            path: token
            audience: profi-metrics
            expirationSeconds: 3600
```

### `reason=bad_cert`

Client certificate validation failed at the TLS layer. Check that the
client cert is signed by the CA profi trusts
(`profi.metricsTls.existingSecret` key `ca.crt`, or the cert-manager managed
Secret when `profi.metricsTls.certManager.enabled=true`).
Rotate if the intermediate chain changed.

## Recovery

- Fix the scraper config or token audience as above.
- Rotate the client cert / Secret if compromised.
- In an emergency, lower to `server` + `bearer` (drops mTLS requirement)
  via `helm upgrade --set profi.metricsTls.mode=server`.

## Related metrics

- `profi_system_http_auth_success_total{method=...}` — should rise after fix.
- `profi_system_http_tokenreview_cache_total{result="hit"}` — cache efficiency.
- `profi_system_http_tokenreview_latency_seconds` — API server latency for
  TokenReview calls.
- `profi_system_http_tls_handshakes_total{result=...}` — TLS-layer failures.
