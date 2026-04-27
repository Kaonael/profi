// SPDX-License-Identifier: Apache-2.0

//! TLS and authentication for the `/metrics` HTTP server.
//!
//! Two independent dimensions, configured via CLI flags:
//!
//! * **TLS mode** (`off` / `server` / `mtls`) — controls the listener.
//! * **Auth mode** (`off` / `bearer` / `mtls-or-bearer`) — L7 middleware
//!   that either accepts a validated client certificate (provided by the
//!   rustls client verifier) or a Kubernetes ServiceAccount bearer token
//!   validated via the TokenReview API.
//!
//! `/health` and `/ready` bypass auth so kubelet probes keep working
//! (they still speak TLS to the listener when `tls.mode != off`).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use axum::extract::Request;
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use clap::Args;
use k8s_openapi::api::authentication::v1::{TokenReview, TokenReviewSpec};
use kube::api::{Api, PostParams};
use log::{debug, warn};
use moka::future::Cache;
use rustls::pki_types::CertificateDer;
use rustls::server::{ServerConfig, WebPkiClientVerifier};
use rustls::RootCertStore;
use sha2::{Digest, Sha256};

use crate::metrics::Metrics;
use crate::pem::{load_cert_chain, load_private_key};

// ── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum TlsMode {
    /// Plain HTTP listener (default, backwards compatible).
    Off,
    /// Serve TLS; accept any client.
    Server,
    /// Serve TLS and require a client certificate signed by `--metrics-tls-client-ca`.
    Mtls,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum AuthMode {
    /// No L7 authentication (default).
    Off,
    /// Require a Bearer token validated via Kubernetes TokenReview.
    Bearer,
    /// Accept either a validated client certificate or a Bearer token.
    #[clap(name = "mtls-or-bearer")]
    MtlsOrBearer,
}

/// Clap args, flattened into the top-level `Args` in `main.rs`.
#[derive(Args, Debug, Clone)]
pub struct MetricsSecurityArgs {
    /// TLS mode for the /metrics listener.
    #[arg(
        long,
        value_enum,
        default_value = "off",
        env = "PROFI_METRICS_TLS_MODE"
    )]
    pub metrics_tls_mode: TlsMode,

    /// Path to the server certificate chain (PEM). Required when tls-mode != off.
    #[arg(long, env = "PROFI_METRICS_TLS_CERT")]
    pub metrics_tls_cert: Option<PathBuf>,

    /// Path to the server private key (PEM). Required when tls-mode != off.
    #[arg(long, env = "PROFI_METRICS_TLS_KEY")]
    pub metrics_tls_key: Option<PathBuf>,

    /// Path to the trusted CA for verifying client certificates (PEM).
    /// Required when tls-mode=mtls.
    #[arg(long, env = "PROFI_METRICS_TLS_CLIENT_CA")]
    pub metrics_tls_client_ca: Option<PathBuf>,

    /// L7 authentication mode.
    #[arg(
        long,
        value_enum,
        default_value = "off",
        env = "PROFI_METRICS_AUTH_MODE"
    )]
    pub metrics_auth_mode: AuthMode,

    /// Required audience for TokenReview validation. If unset, the default
    /// API-server audience is accepted.
    #[arg(long, env = "PROFI_METRICS_AUTH_AUDIENCE")]
    pub metrics_auth_audience: Option<String>,

    /// TTL (seconds) for cached successful TokenReview results. Negative
    /// results are never cached.
    #[arg(long, default_value_t = 60, env = "PROFI_METRICS_AUTH_CACHE_TTL")]
    pub metrics_auth_cache_ttl: u64,

    /// Maximum number of cached tokens.
    #[arg(long, default_value_t = 1024, env = "PROFI_METRICS_AUTH_CACHE_SIZE")]
    pub metrics_auth_cache_size: u64,
}

impl MetricsSecurityArgs {
    pub fn tls_enabled(&self) -> bool {
        self.metrics_tls_mode != TlsMode::Off
    }

    pub fn auth_enabled(&self) -> bool {
        self.metrics_auth_mode != AuthMode::Off
    }

    pub fn needs_k8s_client(&self) -> bool {
        matches!(
            self.metrics_auth_mode,
            AuthMode::Bearer | AuthMode::MtlsOrBearer
        )
    }

    /// Validate semantic consistency between the TLS and auth modes.
    /// Called at startup so operators get a clear error instead of a
    /// runtime surprise.
    pub fn validate(&self) -> Result<()> {
        if self.tls_enabled() && (self.metrics_tls_cert.is_none() || self.metrics_tls_key.is_none())
        {
            return Err(anyhow!(
                    "--metrics-tls-cert and --metrics-tls-key are required when --metrics-tls-mode={:?}",
                    self.metrics_tls_mode
                ));
        }
        if self.metrics_tls_mode == TlsMode::Mtls && self.metrics_tls_client_ca.is_none() {
            return Err(anyhow!(
                "--metrics-tls-client-ca is required when --metrics-tls-mode=mtls"
            ));
        }
        if self.metrics_auth_mode == AuthMode::MtlsOrBearer
            && self.metrics_tls_mode != TlsMode::Mtls
        {
            return Err(anyhow!(
                "--metrics-auth-mode=mtls-or-bearer requires --metrics-tls-mode=mtls \
                 (the mTLS path needs a client verifier)"
            ));
        }
        Ok(())
    }
}

// ── TLS ServerConfig ────────────────────────────────────────────────────────

/// Build a rustls `ServerConfig` from the parsed security args.
///
/// Returns `Ok(None)` when TLS is disabled — caller should fall back to
/// plain HTTP in that case.
pub fn build_rustls_config(args: &MetricsSecurityArgs) -> Result<Option<Arc<ServerConfig>>> {
    if !args.tls_enabled() {
        return Ok(None);
    }

    let cert_path = args
        .metrics_tls_cert
        .as_ref()
        .expect("validated in MetricsSecurityArgs::validate");
    let key_path = args
        .metrics_tls_key
        .as_ref()
        .expect("validated in MetricsSecurityArgs::validate");

    let certs = load_cert_chain(cert_path).context("load server certificate chain")?;
    let key = load_private_key(key_path).context("load server private key")?;

    let builder = ServerConfig::builder();

    let cfg = match args.metrics_tls_mode {
        TlsMode::Off => unreachable!(),
        TlsMode::Server => builder
            .with_no_client_auth()
            .with_single_cert(certs, key)
            .context("build server-only TLS config")?,
        TlsMode::Mtls => {
            let ca_path = args
                .metrics_tls_client_ca
                .as_ref()
                .expect("validated in MetricsSecurityArgs::validate");
            let ca_certs: Vec<CertificateDer<'static>> =
                load_cert_chain(ca_path).context("load client CA bundle")?;

            let mut roots = RootCertStore::empty();
            for cert in ca_certs {
                roots
                    .add(cert)
                    .context("add CA cert to client verifier root store")?;
            }
            let verifier = WebPkiClientVerifier::builder(Arc::new(roots))
                .build()
                .context("build client certificate verifier")?;
            builder
                .with_client_cert_verifier(verifier)
                .with_single_cert(certs, key)
                .context("build mTLS server config")?
        }
    };

    Ok(Some(Arc::new(cfg)))
}

// ── TokenReview client + cache ──────────────────────────────────────────────

/// Thin wrapper around `kube::Api<TokenReview>` with an in-process LRU+TTL
/// cache of positive results. The cache key is `sha256(token)` so raw tokens
/// never leave memory we control.
#[derive(Clone)]
pub struct TokenReviewer {
    api: Api<TokenReview>,
    audience: Option<String>,
    cache: Cache<[u8; 32], CachedAuthn>,
    metrics: Metrics,
}

#[derive(Clone, Debug)]
struct CachedAuthn {
    username: String,
}

impl TokenReviewer {
    pub fn new(
        client: kube::Client,
        audience: Option<String>,
        cache_ttl: Duration,
        cache_size: u64,
        metrics: Metrics,
    ) -> Self {
        let api: Api<TokenReview> = Api::all(client);
        let cache = Cache::builder()
            .max_capacity(cache_size)
            .time_to_live(cache_ttl)
            .build();
        Self {
            api,
            audience,
            cache,
            metrics,
        }
    }

    fn digest(token: &str) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(token.as_bytes());
        h.finalize().into()
    }

    /// Validate a Bearer token. Returns the authenticated username on
    /// success, or an error with a reason label for the failure metric.
    pub async fn authenticate(&self, token: &str) -> Result<String, AuthFailure> {
        let key = Self::digest(token);
        if let Some(hit) = self.cache.get(&key).await {
            self.metrics
                .http_tokenreview_cache
                .with_label_values(&["hit"])
                .inc();
            return Ok(hit.username);
        }
        self.metrics
            .http_tokenreview_cache
            .with_label_values(&["miss"])
            .inc();

        let spec = TokenReviewSpec {
            token: Some(token.to_owned()),
            audiences: self.audience.as_ref().map(|a| vec![a.clone()]),
        };
        let review = TokenReview {
            spec,
            ..TokenReview::default()
        };

        let start = std::time::Instant::now();
        let result = self.api.create(&PostParams::default(), &review).await;
        self.metrics
            .http_tokenreview_latency
            .observe(start.elapsed().as_secs_f64());

        let created = match result {
            Ok(c) => c,
            Err(e) => {
                warn!("TokenReview API error: {e}");
                return Err(AuthFailure::TokenReviewError);
            }
        };

        let status = created.status.ok_or(AuthFailure::TokenReviewError)?;

        if !status.authenticated.unwrap_or(false) {
            return Err(AuthFailure::TokenReviewDeny);
        }

        if let Some(required) = &self.audience {
            let ok = status
                .audiences
                .as_ref()
                .map(|a| a.iter().any(|x| x == required))
                .unwrap_or(false);
            if !ok {
                return Err(AuthFailure::AudienceMismatch);
            }
        }

        let username = status
            .user
            .as_ref()
            .and_then(|u| u.username.clone())
            .unwrap_or_else(|| "unknown".to_owned());

        self.cache
            .insert(
                key,
                CachedAuthn {
                    username: username.clone(),
                },
            )
            .await;
        Ok(username)
    }
}

// ── Auth middleware ─────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub enum AuthFailure {
    NoAuth,
    TokenReviewDeny,
    TokenReviewError,
    AudienceMismatch,
}

impl AuthFailure {
    fn reason(self) -> &'static str {
        match self {
            AuthFailure::NoAuth => "no_auth",
            AuthFailure::TokenReviewDeny => "tokenreview_deny",
            AuthFailure::TokenReviewError => "tokenreview_error",
            AuthFailure::AudienceMismatch => "audience_mismatch",
        }
    }
}

/// State passed to the auth middleware.
#[derive(Clone)]
pub struct AuthLayerState {
    pub mode: AuthMode,
    /// True when the server is listening in TLS `mtls` mode. Any request
    /// that reaches this middleware under mTLS has already had its client
    /// cert validated by rustls — there is no way to get past the TLS
    /// handshake otherwise.
    pub mtls_listener: bool,
    pub reviewer: Option<Arc<TokenReviewer>>,
    pub metrics: Metrics,
}

/// Paths that are always unauthenticated (kubelet liveness/readiness).
fn is_unauthenticated_path(path: &str) -> bool {
    matches!(path, "/health" | "/ready")
}

/// Axum middleware. In `mtls-or-bearer` mode, a request that reached the
/// handler over an mTLS listener is already implicitly authenticated:
/// rustls's `WebPkiClientVerifier` is configured in `require-and-verify`
/// mode, so a client that got past the TLS handshake presented a cert
/// signed by the trusted CA. Otherwise, the middleware falls through to
/// Bearer-token validation via TokenReview.
pub async fn auth_middleware(
    axum::extract::State(state): axum::extract::State<Arc<AuthLayerState>>,
    req: Request,
    next: Next,
) -> Response {
    if state.mode == AuthMode::Off {
        return next.run(req).await;
    }

    let path = req.uri().path();
    if is_unauthenticated_path(path) {
        return next.run(req).await;
    }

    // In mtls-or-bearer mode, an mTLS listener itself proves the cert.
    if state.mode == AuthMode::MtlsOrBearer && state.mtls_listener {
        state
            .metrics
            .http_auth_success
            .with_label_values(&["mtls"])
            .inc();
        return next.run(req).await;
    }

    // Extract bearer token if present.
    let bearer = req
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "))
        .map(|s| s.trim().to_owned());

    let token = match bearer {
        Some(t) if !t.is_empty() => t,
        _ => {
            state
                .metrics
                .http_auth_failures
                .with_label_values(&[AuthFailure::NoAuth.reason()])
                .inc();
            return unauthorized_response();
        }
    };

    let reviewer = match state.reviewer.as_ref() {
        Some(r) => r,
        None => {
            // Shouldn't happen: if auth is enabled, main wires a reviewer.
            state
                .metrics
                .http_auth_failures
                .with_label_values(&[AuthFailure::TokenReviewError.reason()])
                .inc();
            return unauthorized_response();
        }
    };

    match reviewer.authenticate(&token).await {
        Ok(user) => {
            debug!("authenticated /metrics request for {user}");
            state
                .metrics
                .http_auth_success
                .with_label_values(&["bearer"])
                .inc();
            next.run(req).await
        }
        Err(f) => {
            state
                .metrics
                .http_auth_failures
                .with_label_values(&[f.reason()])
                .inc();
            unauthorized_response()
        }
    }
}

fn unauthorized_response() -> Response {
    (
        StatusCode::UNAUTHORIZED,
        [(axum::http::header::WWW_AUTHENTICATE, "Bearer")],
        "unauthorized\n",
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_server_mode_requires_cert_and_key() {
        let args = MetricsSecurityArgs {
            metrics_tls_mode: TlsMode::Server,
            metrics_tls_cert: None,
            metrics_tls_key: None,
            metrics_tls_client_ca: None,
            metrics_auth_mode: AuthMode::Off,
            metrics_auth_audience: None,
            metrics_auth_cache_ttl: 60,
            metrics_auth_cache_size: 1024,
        };
        assert!(args.validate().is_err());
    }

    #[test]
    fn validate_mtls_mode_requires_client_ca() {
        let args = MetricsSecurityArgs {
            metrics_tls_mode: TlsMode::Mtls,
            metrics_tls_cert: Some("cert.pem".into()),
            metrics_tls_key: Some("key.pem".into()),
            metrics_tls_client_ca: None,
            metrics_auth_mode: AuthMode::Off,
            metrics_auth_audience: None,
            metrics_auth_cache_ttl: 60,
            metrics_auth_cache_size: 1024,
        };
        assert!(args.validate().is_err());
    }

    #[test]
    fn mtls_or_bearer_requires_mtls_listener() {
        let args = MetricsSecurityArgs {
            metrics_tls_mode: TlsMode::Server,
            metrics_tls_cert: Some("cert.pem".into()),
            metrics_tls_key: Some("key.pem".into()),
            metrics_tls_client_ca: None,
            metrics_auth_mode: AuthMode::MtlsOrBearer,
            metrics_auth_audience: None,
            metrics_auth_cache_ttl: 60,
            metrics_auth_cache_size: 1024,
        };
        assert!(args.validate().is_err());
    }

    #[test]
    fn off_mode_is_valid_default() {
        let args = MetricsSecurityArgs {
            metrics_tls_mode: TlsMode::Off,
            metrics_tls_cert: None,
            metrics_tls_key: None,
            metrics_tls_client_ca: None,
            metrics_auth_mode: AuthMode::Off,
            metrics_auth_audience: None,
            metrics_auth_cache_ttl: 60,
            metrics_auth_cache_size: 1024,
        };
        assert!(args.validate().is_ok());
    }

    #[test]
    fn unauthenticated_paths_match() {
        assert!(is_unauthenticated_path("/health"));
        assert!(is_unauthenticated_path("/ready"));
        assert!(!is_unauthenticated_path("/metrics"));
        assert!(!is_unauthenticated_path("/status"));
    }

    #[test]
    fn auth_failure_reasons() {
        assert_eq!(AuthFailure::NoAuth.reason(), "no_auth");
        assert_eq!(AuthFailure::TokenReviewDeny.reason(), "tokenreview_deny");
        assert_eq!(AuthFailure::TokenReviewError.reason(), "tokenreview_error");
        assert_eq!(AuthFailure::AudienceMismatch.reason(), "audience_mismatch");
    }
}
